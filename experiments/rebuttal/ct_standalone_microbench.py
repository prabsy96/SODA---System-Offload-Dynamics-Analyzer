#!/usr/bin/env python3
"""
Standalone ΔCT Validation Microbenchmark (Reviewer D)

Validates the subtractive ΔCT estimation method against ground-truth cuBLASLt
frontend timing across 15 GEMM shapes (square, rectangular, LLM-realistic).

Methodology per shape:
  1. Default (Subtractive): Profile torch.mm via PyTorch profiler → extract
     T_xlat (ATen op start → cudaLaunchKernel). Profile matched torch.add
     baseline → compute ΔCT_default = median(T_xlat_GEMM) - median(T_xlat_base).
  2. Replay (Ground Truth): Call baremetal main_gemm_bm binary → parse
     t_setup_us + t_heur_us = cuBLASLt descriptor setup + heuristic selection.
  3. Validation: |ΔCT_default - ΔCT_replay| < 15µs.

Usage:
    python experiments/rebuttal/ct_standalone_microbench.py --output-dir DIR [options]
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.profiler import ProfilerActivity, profile


# =============================================================================
# Configuration
# =============================================================================

GEMM_SHAPES: List[Dict[str, Any]] = [
    # Square shapes (6) — verified working from prior runs
    {"m": 256, "n": 256, "k": 256, "desc": "Tiny square"},
    {"m": 512, "n": 512, "k": 512, "desc": "Small square"},
    {"m": 768, "n": 768, "k": 768, "desc": "Medium square"},
    {"m": 1024, "n": 1024, "k": 1024, "desc": "Large square"},
    {"m": 2048, "n": 2048, "k": 2048, "desc": "XL square"},
    {"m": 4096, "n": 4096, "k": 4096, "desc": "XXL square"},
    # Power-of-2 rectangular (4) — safe alignment for cuBLASLt
    {"m": 512, "n": 2048, "k": 512, "desc": "Rect P2 wide"},
    {"m": 1024, "n": 4096, "k": 1024, "desc": "Rect P2 wide2"},
    {"m": 1024, "n": 1024, "k": 2048, "desc": "Rect P2 deep"},
    {"m": 2048, "n": 4096, "k": 2048, "desc": "Rect P2 large"},
    # LLM-realistic (5) — GPT-2 + Llama shapes
    {"m": 512, "n": 768, "k": 768, "desc": "GPT-2 attn out"},
    {"m": 512, "n": 2304, "k": 768, "desc": "GPT-2 QKV"},
    {"m": 512, "n": 3072, "k": 768, "desc": "GPT-2 MLP up"},
    {"m": 512, "n": 768, "k": 3072, "desc": "GPT-2 MLP down"},
    {"m": 1024, "n": 4096, "k": 2048, "desc": "Llama-like"},
]

DEFAULT_WARMUP = 50
DEFAULT_RUNS = 200
VALIDATION_THRESHOLD_US = 15.0  # |ΔCT_default - ΔCT_replay| < 15µs


# =============================================================================
# Trace Parsing
# =============================================================================

def extract_xlat_times_from_trace(
    trace_path: Path, target_op: str
) -> List[float]:
    """
    Parse Chrome trace to extract T_xlat (op start → first cudaLaunchKernel)
    for all instances of target_op (e.g. 'aten::mm' or 'aten::add').

    Uses time-window correlation: for each ATen op, find the first
    cudaLaunchKernel within [op_ts, op_end + 100µs].
    """
    with open(trace_path) as f:
        trace_data = json.load(f)

    events = trace_data.get("traceEvents", [])

    target_ops = []
    launch_events = []

    for event in events:
        name = event.get("name", "")
        ph = event.get("ph", "")
        ts = event.get("ts", 0)
        dur = event.get("dur", 0)

        if ph != "X" or ts == 0:
            continue

        if name == target_op:
            target_ops.append({"ts": ts, "dur": dur})
        elif "cudaLaunchKernel" in name:
            launch_events.append({"ts": ts})

    target_ops.sort(key=lambda e: e["ts"])
    launch_events.sort(key=lambda e: e["ts"])

    WINDOW_US = 100
    xlat_times = []
    launch_idx = 0

    for op in target_ops:
        op_start = op["ts"]
        op_end = op_start + op["dur"]
        search_end = op_end + WINDOW_US

        # Advance launch_idx to first launch >= op_start
        while (
            launch_idx < len(launch_events)
            and launch_events[launch_idx]["ts"] < op_start
        ):
            launch_idx += 1

        if launch_idx < len(launch_events):
            launch_ts = launch_events[launch_idx]["ts"]
            if launch_ts <= search_end:
                xlat_time = launch_ts - op_start
                if xlat_time > 0:
                    xlat_times.append(xlat_time)
                launch_idx += 1  # Consume this launch

    return xlat_times


# =============================================================================
# PyTorch Profiling (Per-Shape)
# =============================================================================

def profile_torch_op(
    op_fn,
    warmup: int,
    runs: int,
    trace_path: Path,
):
    """
    Profile a single torch op: warmup, then record `runs` iterations
    under PyTorch profiler, export Chrome trace.
    """
    # Warmup
    for _ in range(warmup):
        op_fn()
    torch.cuda.synchronize()

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(runs):
            op_fn()
        torch.cuda.synchronize()

    prof.export_chrome_trace(str(trace_path))


def measure_xlat_for_shape(
    m: int, n: int, k: int,
    warmup: int, runs: int,
    device: torch.device,
) -> Tuple[Optional[float], Optional[float], Optional[float],
           Optional[float], Optional[float], Optional[float]]:
    """
    Measure T_xlat for torch.mm (GEMM) and torch.add (baseline) for a shape.

    Returns:
        (median_xlat_gemm, mean_xlat_gemm, std_xlat_gemm,
         median_xlat_base, mean_xlat_base, std_xlat_base)
    """
    # Create tensors
    A = torch.randn(m, k, device=device, dtype=torch.float32)
    B = torch.randn(k, n, device=device, dtype=torch.float32)
    # Matched-size output for baseline add
    C = torch.randn(m, n, device=device, dtype=torch.float32)

    def gemm_fn():
        torch.mm(A, B)

    def add_fn():
        torch.add(C, C)

    with tempfile.TemporaryDirectory() as tmpdir:
        gemm_trace = Path(tmpdir) / "gemm_trace.json"
        add_trace = Path(tmpdir) / "add_trace.json"

        # Profile GEMM
        profile_torch_op(gemm_fn, warmup, runs, gemm_trace)
        gemm_xlats = extract_xlat_times_from_trace(gemm_trace, "aten::mm")

        # Profile baseline (add)
        profile_torch_op(add_fn, warmup, runs, add_trace)
        add_xlats = extract_xlat_times_from_trace(add_trace, "aten::add")

    # Compute stats
    def stats(vals):
        if not vals:
            return None, None, None
        med = statistics.median(vals)
        avg = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return med, avg, std

    med_g, avg_g, std_g = stats(gemm_xlats)
    med_b, avg_b, std_b = stats(add_xlats)

    # Free GPU memory
    del A, B, C
    torch.cuda.empty_cache()

    return med_g, avg_g, std_g, med_b, avg_b, std_b


# =============================================================================
# Baremetal Replay
# =============================================================================

def find_baremetal_binary() -> Optional[Path]:
    """Find the compiled baremetal GEMM binary."""
    candidates = [
        Path(os.environ.get("SODA_ROOT", ".")) / "src" / "soda" / "microbench" / "baremetal" / "build" / "main_gemm_bm",
        Path.cwd() / "src" / "soda" / "microbench" / "baremetal" / "build" / "main_gemm_bm",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _run_baremetal_cmd(cmd: List[str], timeout: int = 60) -> Tuple[Optional[Dict], str]:
    """
    Run baremetal command and parse timing output.
    Returns (parsed_data_dict, error_message).
    """
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            error = result.stderr[:300] if result.stderr else f"Return code {result.returncode}"
            return None, error

        stdout = result.stdout

        # Parse TIMING_RESULTS_BEGIN...END JSON block
        if "TIMING_RESULTS_BEGIN" in stdout:
            start_idx = stdout.find("TIMING_RESULTS_BEGIN")
            end_idx = stdout.find("TIMING_RESULTS_END")
            if start_idx >= 0 and end_idx > start_idx:
                json_str = stdout[
                    start_idx + len("TIMING_RESULTS_BEGIN") : end_idx
                ].strip()
                try:
                    return json.loads(json_str), ""
                except json.JSONDecodeError:
                    pass

        # Fallback: direct JSON
        json_start = stdout.find("{")
        json_end = stdout.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            try:
                data = json.loads(stdout[json_start:json_end])
                if "t_setup_us" in data or "t_heur_us" in data:
                    return data, ""
            except json.JSONDecodeError:
                pass

        return None, "Could not parse timing output"

    except subprocess.TimeoutExpired:
        return None, "Timeout"
    except Exception as e:
        return None, str(e)


def profile_baremetal_gemm(
    m: int, n: int, k: int,
    binary_path: Path,
    warmup: int = 50,
    runs: int = 200,
) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    """
    Profile baremetal cuBLASLt for a specific shape.
    Returns (delta_ct_us, t_setup_us, t_heur_us, error_message).

    Tries row-major first, falls back to column-major if that fails.
    """
    if not binary_path or not binary_path.exists():
        return None, None, None, "Binary not found"

    # Row-major command (matches PyTorch default)
    cmd_row = [
        str(binary_path),
        "--m", str(m),
        "--n", str(n),
        "--k", str(k),
        "--lda", str(k),   # Row-major: lda = K
        "--ldb", str(n),   # Row-major: ldb = N
        "--ldc", str(n),   # Row-major: ldc = N
        "--dtype", "f32",
        "--warmup", str(warmup),
        "--runs", str(runs),
        "--order_a", "row",
        "--order_b", "row",
    ]

    data, error = _run_baremetal_cmd(cmd_row)
    if data is not None:
        t_setup = data.get("t_setup_us", 0)
        t_heur = data.get("t_heur_us", 0)
        return t_setup + t_heur, t_setup, t_heur, ""

    # Column-major fallback (try on any row-major failure)
    if error:
        cmd_col = [
            str(binary_path),
            "--m", str(m),
            "--n", str(n),
            "--k", str(k),
            "--lda", str(m),   # Column-major: lda = M
            "--ldb", str(k),   # Column-major: ldb = K
            "--ldc", str(m),   # Column-major: ldc = M
            "--dtype", "f32",
            "--warmup", str(warmup),
            "--runs", str(runs),
            "--order_a", "col",
            "--order_b", "col",
        ]

        data, error = _run_baremetal_cmd(cmd_col)
        if data is not None:
            t_setup = data.get("t_setup_us", 0)
            t_heur = data.get("t_heur_us", 0)
            return t_setup + t_heur, t_setup, t_heur, ""

    return None, None, None, error


# =============================================================================
# Main Validation
# =============================================================================

def run_standalone_validation(
    output_dir: Path,
    warmup: int = DEFAULT_WARMUP,
    runs: int = DEFAULT_RUNS,
    shapes: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Run standalone ΔCT validation across all GEMM shapes.
    """
    if shapes is None:
        shapes = GEMM_SHAPES

    gpu_name = torch.cuda.get_device_name(0)

    print("=" * 74)
    print("STANDALONE ΔCT VALIDATION MICROBENCHMARK (Reviewer D)")
    print("=" * 74)
    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"{len(shapes)} GEMM shapes, {warmup} warmup + {runs} runs each, dtype=float32")
    print("=" * 74)
    print()
    sys.stdout.flush()

    # Find baremetal binary
    baremetal_binary = find_baremetal_binary()
    if baremetal_binary:
        print(f"Baremetal binary: {baremetal_binary}")
    else:
        print("WARNING: Baremetal binary not found. Replay will be skipped.")
        print("Run 'build' from env.sh to compile.")
    print()
    sys.stdout.flush()

    device = torch.device("cuda")

    # Table header
    header = (
        f"{'Shape':<18} {'Description':<18} "
        f"{'T_xlat':>7} {'T_base':>7} "
        f"{'dCT_def':>7} {'dCT_rep':>7} "
        f"{'Diff':>7}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)
    sys.stdout.flush()

    results = []
    for i, shape_cfg in enumerate(shapes):
        m, n, k = shape_cfg["m"], shape_cfg["n"], shape_cfg["k"]
        desc = shape_cfg["desc"]
        shape_key = f"{m}x{n}x{k}"

        print(f"  [{i+1}/{len(shapes)}] Profiling {shape_key}...", end="", flush=True)

        # === Step 1: PyTorch profiler (subtractive method) ===
        try:
            (
                med_xlat_gemm, avg_xlat_gemm, std_xlat_gemm,
                med_xlat_base, avg_xlat_base, std_xlat_base,
            ) = measure_xlat_for_shape(m, n, k, warmup, runs, device)
        except Exception as e:
            print(f" PyTorch error: {e}")
            results.append({
                "shape": shape_key,
                "m": m, "n": n, "k": k,
                "description": desc,
                "error": f"PyTorch profiling failed: {e}",
                "validated": False,
            })
            continue

        # Compute ΔCT_default
        delta_ct_default = None
        if med_xlat_gemm is not None and med_xlat_base is not None:
            delta_ct_default = max(0.0, med_xlat_gemm - med_xlat_base)

        # === Step 2: Baremetal replay (ground truth) ===
        delta_ct_replay = None
        t_setup = None
        t_heur = None
        replay_error = ""
        if baremetal_binary:
            delta_ct_replay, t_setup, t_heur, replay_error = (
                profile_baremetal_gemm(m, n, k, baremetal_binary, warmup, runs)
            )

        # === Step 3: Validation ===
        diff = None
        validated = False
        status_char = "-"
        if delta_ct_default is not None and delta_ct_replay is not None:
            diff = delta_ct_default - delta_ct_replay
            validated = abs(diff) < VALIDATION_THRESHOLD_US
            status_char = "V" if validated else "X"
        elif delta_ct_default is not None and delta_ct_replay is None:
            status_char = "?"  # No replay
        elif replay_error:
            status_char = "!"  # Replay error

        def fmt(v):
            return f"{v:7.1f}" if v is not None else "    N/A"

        # Clear the "Profiling..." line and print result row
        print(
            f"\r{shape_key:<18} {desc:<18} "
            f"{fmt(med_xlat_gemm)} {fmt(med_xlat_base)} "
            f"{fmt(delta_ct_default)} {fmt(delta_ct_replay)} "
            f"{fmt(diff)}  {status_char}"
        )
        sys.stdout.flush()

        results.append({
            "shape": shape_key,
            "m": m, "n": n, "k": k,
            "description": desc,
            "n_gemm_samples": None,  # filled from xlat extraction
            "t_xlat_gemm_median_us": med_xlat_gemm,
            "t_xlat_gemm_mean_us": avg_xlat_gemm,
            "t_xlat_gemm_std_us": std_xlat_gemm,
            "t_xlat_base_median_us": med_xlat_base,
            "t_xlat_base_mean_us": avg_xlat_base,
            "t_xlat_base_std_us": std_xlat_base,
            "delta_ct_default_us": delta_ct_default,
            "delta_ct_replay_us": delta_ct_replay,
            "t_setup_us": t_setup,
            "t_heur_us": t_heur,
            "difference_us": diff,
            "replay_error": replay_error if replay_error else None,
            "validated": validated,
        })

    print(sep)
    print()

    # === Summary ===
    valid_results = [
        r for r in results
        if r.get("delta_ct_default_us") is not None
        and r.get("delta_ct_replay_us") is not None
    ]
    shapes_validated = sum(1 for r in valid_results if r["validated"])

    summary: Dict[str, Any] = {
        "total_shapes": len(results),
        "shapes_with_default": sum(
            1 for r in results if r.get("delta_ct_default_us") is not None
        ),
        "shapes_with_replay": sum(
            1 for r in results if r.get("delta_ct_replay_us") is not None
        ),
        "shapes_compared": len(valid_results),
        "shapes_validated": shapes_validated,
        "validation_rate": (
            shapes_validated / len(valid_results) if valid_results else 0.0
        ),
    }

    if valid_results:
        replay_vals = [r["delta_ct_replay_us"] for r in valid_results]
        default_vals = [r["delta_ct_default_us"] for r in valid_results]
        diffs = [r["difference_us"] for r in valid_results]
        abs_diffs = [abs(d) for d in diffs]

        summary.update({
            "mean_delta_ct_replay_us": statistics.mean(replay_vals),
            "std_delta_ct_replay_us": (
                statistics.stdev(replay_vals) if len(replay_vals) > 1 else 0.0
            ),
            "mean_delta_ct_default_us": statistics.mean(default_vals),
            "std_delta_ct_default_us": (
                statistics.stdev(default_vals) if len(default_vals) > 1 else 0.0
            ),
            "mean_difference_us": statistics.mean(diffs),
            "mean_abs_difference_us": statistics.mean(abs_diffs),
            "max_abs_difference_us": max(abs_diffs),
        })

        print("SUMMARY:")
        print(
            f"  cuBLAS front-end (ground truth): "
            f"{summary['mean_delta_ct_replay_us']:.1f} "
            f"+/- {summary['std_delta_ct_replay_us']:.1f} us "
            f"(N={len(valid_results)})"
        )
        print(
            f"  Subtractive estimate:            "
            f"{summary['mean_delta_ct_default_us']:.1f} "
            f"+/- {summary['std_delta_ct_default_us']:.1f} us "
            f"(N={len(valid_results)})"
        )
        print(
            f"  Mean |difference|:               "
            f"{summary['mean_abs_difference_us']:.1f} us"
        )
        print(
            f"  Validation rate:                 "
            f"{shapes_validated}/{len(valid_results)} "
            f"({summary['validation_rate']*100:.0f}%)"
        )
    else:
        print("SUMMARY: No shapes with both default and replay measurements.")

    print()
    print("=" * 74)
    if valid_results and summary["validation_rate"] >= 0.5:
        print("CONCLUSION: VALIDATED")
        print("=" * 74)
        print(
            f"  The subtractive method (dCT = T_xlat(mm) - T_xlat(add)) tracks"
        )
        print(
            f"  ground-truth cuBLASLt frontend overhead "
            f"(~{summary['mean_delta_ct_replay_us']:.1f} us) with mean error"
        )
        print(
            f"  of {summary['mean_abs_difference_us']:.1f} us across "
            f"{len(valid_results)} GEMM shapes."
        )
        print()
        print("  For Reviewer D:")
        print(
            f"    - Ground truth (baremetal cuBLASLt): "
            f"{summary['mean_delta_ct_replay_us']:.1f} us"
        )
        print(
            f"    - Subtractive estimate: "
            f"{summary['mean_delta_ct_default_us']:.1f} us"
        )
        print(
            f"    - {shapes_validated}/{len(valid_results)} shapes validated "
            f"(|diff| < {VALIDATION_THRESHOLD_US:.0f} us)"
        )
    else:
        print("CONCLUSION: NEEDS INVESTIGATION")
        print("=" * 74)
        print("  Insufficient data or low validation rate.")
    print("=" * 74)
    sys.stdout.flush()

    # === Save JSON Report ===
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "ct_standalone_report.json"

    report = {
        "title": "Standalone ΔCT Validation Microbenchmark (Reviewer D)",
        "environment": {
            "gpu": gpu_name,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "config": {
            "warmup": warmup,
            "runs": runs,
            "dtype": "float32",
            "num_shapes": len(shapes),
            "validation_threshold_us": VALIDATION_THRESHOLD_US,
        },
        "methodology": {
            "default": (
                "dCT = T_xlat(torch.mm) - T_xlat(torch.add), "
                "per-shape matched baseline"
            ),
            "replay": (
                "dCT = t_setup + t_heur from direct cuBLASLt frontend timing"
            ),
            "validation": f"|difference| < {VALIDATION_THRESHOLD_US:.0f} us",
        },
        "shapes": results,
        "summary": summary,
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nFull report saved to: {report_path}")
    sys.stdout.flush()

    return report


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Standalone ΔCT Validation Microbenchmark (Reviewer D)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/ct_standalone"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"Number of warmup iterations (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Number of measured iterations (default: {DEFAULT_RUNS})",
    )

    args = parser.parse_args()

    print(f"Standalone ΔCT Validation Microbenchmark")
    print(f"Output directory: {args.output_dir}")
    print(f"Warmup: {args.warmup}, Runs: {args.runs}")
    print()
    sys.stdout.flush()

    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        return 1

    try:
        run_standalone_validation(
            output_dir=args.output_dir,
            warmup=args.warmup,
            runs=args.runs,
        )
    except Exception as e:
        print(f"Error during validation: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
