#!/usr/bin/env python3
"""Measure Figure-11-compatible prefill scaling through the Cerebras Inference SDK.

The SDK accepts an array of prompts in one completions request.  Each array is
one server-side batch; this is deliberately different from issuing independent
concurrent client requests.  The server's ``time_info.prompt_time`` is used as
the prefill-latency analogue.  The SDK does not expose CPU/GPU idle time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from cerebras.cloud.sdk import Cerebras

BATCH_SIZES = (1, 2, 4, 8, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-oss-120b", help="Cerebras endpoint model ID.")
    parser.add_argument("--batch-size", type=int, required=True, choices=BATCH_SIZES)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=1, help="Keep at 1 to isolate prefill/TTFT.")
    parser.add_argument("--target-input-tokens", type=int, default=512)
    parser.add_argument("--prompt-file", type=Path, help="Use a fixed, model-compatible prompt from a file.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def make_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return args.prompt_file.read_text().strip()
    # The API reports the actual token count in usage.  It may differ by one or
    # more tokens depending on the selected model tokenizer; never assume this
    # string itself is exactly 512 tokens.
    return " " + " the" * args.target_input_tokens


def value_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


def run_batch(client: Cerebras, args: argparse.Namespace, prompt: str) -> dict[str, float | int | None]:
    wall_start = time.perf_counter()
    response = client.completions.create(
        model=args.model,
        prompt=[prompt] * args.batch_size,
        max_tokens=args.max_tokens,
        min_tokens=args.max_tokens,
        temperature=0,
        seed=args.seed,
        timeout=args.timeout,
    )
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    info = getattr(response, "time_info", None)
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    return {
        "client_wall_ms": wall_ms,
        "server_prefill_ms": value_or_none(getattr(info, "prompt_time", None)) and float(info.prompt_time) * 1000.0,
        "server_queue_ms": value_or_none(getattr(info, "queue_time", None)) and float(info.queue_time) * 1000.0,
        "server_completion_ms": value_or_none(getattr(info, "completion_time", None)) and float(info.completion_time) * 1000.0,
        "server_total_ms": value_or_none(getattr(info, "total_time", None)) and float(info.total_time) * 1000.0,
        "prompt_tokens_total": prompt_tokens,
        "prompt_tokens_per_request": (prompt_tokens / args.batch_size) if prompt_tokens is not None else None,
        "completion_tokens_total": completion_tokens,
    }


def mean_ci(values: list[float], samples: int, seed: int) -> dict[str, float]:
    if len(values) < 2:
        raise ValueError("At least two successful trials are required for a confidence interval.")
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "stddev": float(array.std(ddof=1)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n": len(values),
    }


def summarize(trials: list[dict[str, float | int | None]], args: argparse.Namespace) -> dict[str, dict[str, float] | None]:
    summary: dict[str, dict[str, float] | None] = {}
    for key in ("server_prefill_ms", "server_queue_ms", "server_completion_ms", "server_total_ms", "client_wall_ms"):
        values = [float(row[key]) for row in trials if row.get(key) is not None]
        summary[key] = mean_ci(values, args.bootstrap_samples, args.seed) if len(values) >= 2 else None
    return summary


def main() -> int:
    args = parse_args()
    if not os.environ.get("CEREBRAS_API_KEY"):
        raise RuntimeError("CEREBRAS_API_KEY is not set; cannot submit an Inference SDK experiment.")
    if args.trials < 2:
        raise ValueError("--trials must be at least 2 to compute 95% confidence intervals.")
    if args.max_tokens != 1:
        print("Warning: max_tokens != 1 includes more decode work than a prefill/TTFT measurement.", file=sys.stderr)
    prompt = make_prompt(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"])

    for _ in range(args.warmup):
        run_batch(client, args, prompt)

    trials = []
    for index in range(args.trials):
        try:
            row = run_batch(client, args, prompt)
            row["trial"] = index + 1
            trials.append(row)
            print(f"trial {index + 1}/{args.trials}: prefill={row['server_prefill_ms']} ms queue={row['server_queue_ms']} ms", flush=True)
        except Exception as error:  # preserve errors while allowing an intermittent retry-free sample set
            trials.append({"trial": index + 1, "error": f"{type(error).__name__}: {error}"})
            print(f"trial {index + 1}/{args.trials}: ERROR {error}", file=sys.stderr, flush=True)

    successes = [row for row in trials if "error" not in row]
    if len(successes) < 2:
        raise RuntimeError("Fewer than two successful trials; no valid confidence interval can be calculated.")
    result = {
        "schema_version": 1,
        "experiment": "figure_11_cerebras_sdk_prefill",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "workload": {
            "endpoint": "completions",
            "batch_size": args.batch_size,
            "batching": "one API request with an array of identical prompts",
            "target_input_tokens": args.target_input_tokens,
            "max_tokens": args.max_tokens,
            "warmup_requests": args.warmup,
            "measured_requests": args.trials,
        },
        "metrics": {
            "prefill_latency": "server_prefill_ms (SDK time_info.prompt_time)",
            "queue_delay": "server_queue_ms (SDK time_info.queue_time)",
            "hardware_idle": None,
        },
        "trials": trials,
        "summary": summarize(successes, args),
        "limitations": [
            "The SDK does not expose CS-3 CPU idle, wafer idle, kernel traces, or utilization.",
            "Actual input token count is recorded from API usage, not inferred from prompt text.",
            "Confidence intervals are non-parametric bootstrap 95% CIs for the mean across successful requests.",
        ],
    }
    output = args.output_dir / "result.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
