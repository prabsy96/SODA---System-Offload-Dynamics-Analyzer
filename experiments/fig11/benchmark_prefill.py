#!/usr/bin/env python3
"""Reproduce one Figure 11 prefill measurement on a local CUDA host.

This intentionally bypasses ModelZoo and the Cerebras Inference SDK. Figure 11
measures a local, eager PyTorch forward pass and requires CUDA kernel traces;
neither interface exposes those measurements for a public inference endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer


MODELS = {
    "gpt2": "gpt2",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--model-id", help="Override the Hugging Face model ID.")
    parser.add_argument("--batch-size", type=int, required=True, choices=(1, 2, 4, 8, 16))
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--platform-label",
        default=None,
        help="Stable host label for plots (for example intel-h100 or gh200).",
    )
    parser.add_argument("--trace", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def cuda_required() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Figure 11 reproduction requires a local CUDA GPU. "
            "Do not run this against the public Cerebras Inference SDK."
        )


def make_inputs(model: Any, batch_size: int, seq_len: int, seed: int) -> dict[str, torch.Tensor]:
    # Token IDs are deterministic and avoid tokenizer-dependent prompt length.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    vocab_size = int(getattr(model.config, "vocab_size", 0))
    if vocab_size < 2:
        raise ValueError("Model config has no usable vocab_size.")
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        generator=generator,
        dtype=torch.long,
        device="cuda",
    )
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def forward(model: Any, inputs: dict[str, torch.Tensor]) -> None:
    with torch.inference_mode():
        # This is prefill only: no generate() call and no decode token.
        output = model(**inputs, use_cache=True)
    del output


def measure_latency_ms(model: Any, inputs: dict[str, torch.Tensor], warmup: int, runs: int) -> list[float]:
    for _ in range(warmup):
        forward(model, inputs)
    torch.cuda.synchronize()

    results = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        forward(model, inputs)
        end.record()
        torch.cuda.synchronize()
        results.append(float(start.elapsed_time(end)))
    return results


def union_duration_us(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def trace_idle_metrics(model: Any, inputs: dict[str, torch.Tensor], trace_path: Path) -> dict[str, float | None]:
    """Measure trace-local idle metrics for one prefill iteration.

    GPU idle is the union-of-kernel gap inside the marked prefill range. CPU
    idle is a host-side proxy: marked range minus the union of CPU-op spans.
    It is recorded explicitly as a proxy because PyTorch traces cannot measure
    physical CPU core idleness or Cerebras appliance internals.
    """
    torch.cuda.synchronize()
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=False, profile_memory=False) as prof:
        with record_function("fig11_prefill"):
            forward(model, inputs)
            # Include the host wait for this prefill in the marked interval.
            torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace_path))

    events = json.loads(trace_path.read_text()).get("traceEvents", [])
    marker = next(
        (event for event in events if event.get("name") == "fig11_prefill" and event.get("ph") == "X"),
        None,
    )
    if marker is None:
        return {"trace_span_ms": None, "gpu_idle_ms": None, "cpu_idle_proxy_ms": None}
    start = float(marker["ts"])
    end = start + float(marker.get("dur", 0.0))
    span = max(0.0, end - start)
    gpu_intervals: list[tuple[float, float]] = []
    cpu_intervals: list[tuple[float, float]] = []
    for event in events:
        if event.get("ph") != "X" or "ts" not in event or "dur" not in event:
            continue
        event_start = float(event["ts"])
        event_end = event_start + float(event["dur"])
        if event_start < start or event_end > end:
            continue
        category = str(event.get("cat", ""))
        if "kernel" in category:
            gpu_intervals.append((event_start, event_end))
        elif "cpu_op" in category:
            cpu_intervals.append((event_start, event_end))
    return {
        "trace_span_ms": span / 1000.0,
        "gpu_idle_ms": max(0.0, span - union_duration_us(gpu_intervals)) / 1000.0,
        "cpu_idle_proxy_ms": max(0.0, span - union_duration_us(cpu_intervals)) / 1000.0,
    }


def main() -> int:
    args = parse_args()
    cuda_required()
    if args.seq_len != 512:
        print("Warning: Figure 11 uses seq_len=512; this is an extension run.", file=sys.stderr)
    if args.runs < 3:
        raise ValueError("--runs must be at least 3 for p50/p95 statistics.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    model_id = args.model_id or MODELS[args.model]
    try:
        # Tokenizer loading verifies access for gated models and records revision.
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=args.trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=args.trust_remote_code,
        ).eval().to("cuda")
    except OSError as error:
        raise RuntimeError(
            f"Could not load {model_id}. For Llama 3.2, accept the Hugging Face license "
            "and authenticate with `huggingface-cli login`, or pass --model-id to an accessible mirror."
        ) from error

    inputs = make_inputs(model, args.batch_size, args.seq_len, args.seed)
    latencies = measure_latency_ms(model, inputs, args.warmup, args.runs)
    trace_path = args.output_dir / "trace.json"
    idle_metrics: dict[str, float | None] = {}
    if args.trace:
        idle_metrics = trace_idle_metrics(model, inputs, trace_path)

    device = torch.cuda.get_device_properties(0)
    result = {
        "schema_version": 1,
        "experiment": "figure_11_prefill",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": {"label": args.model, "id": model_id, "revision": getattr(model.config, "_name_or_path", model_id)},
        "workload": {
            "mode": "eager_fp16_prefill",
            "batch_size": args.batch_size,
            "sequence_length": args.seq_len,
            "warmup_runs": args.warmup,
            "measured_runs": args.runs,
            "use_cache": True,
            "decode_tokens": 0,
        },
        "platform": {
            "hostname": os.uname().nodename,
            "gpu": torch.cuda.get_device_name(0),
            "cuda_runtime": torch.version.cuda,
            "torch": torch.__version__,
            "label": args.platform_label or torch.cuda.get_device_name(0),
            "compute_capability": [device.major, device.minor],
        },
        "latency_ms": {
            "samples": latencies,
            "mean": statistics.mean(latencies),
            "p50": statistics.median(latencies),
            "p95": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)],
        },
        "trace_metrics": idle_metrics,
        "trace_path": str(trace_path) if args.trace else None,
        "metric_notes": {
            "gpu_idle_ms": "Chrome-trace prefill span minus union of CUDA kernel intervals.",
            "cpu_idle_proxy_ms": "Chrome-trace prefill span minus union of CPU-op intervals; not physical CPU core idle time.",
        },
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"result": str(result_path), "latency_p50_ms": result["latency_ms"]["p50"], **idle_metrics}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
