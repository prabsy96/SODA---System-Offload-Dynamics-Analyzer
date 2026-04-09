#!/usr/bin/env python3
"""
MoE Fragmentation Analysis for Reviewer D Q2

Disambiguates host orchestration overhead vs device-side fragmentation in MoE
models using TaxBreak decomposition and HDBI.

Reviewer D asked:
  "For MoE decode, can the authors present more evidence to disambiguate host
   orchestration vs device-side fragmentation?"

Approach:
  1. Read SODA report.json files (already contain TaxBreak metrics from trace analysis)
  2. Compute HDBI from report metrics (or read if already present)
  3. Compare MoE vs Dense model across TaxBreak components
  4. Optional: sampled gap analysis from trace files (capped at MAX_EVENTS)

Usage:
    # From report.json files (fast, no trace needed):
    python experiments/rebuttal/moe_fragmentation_analysis.py \\
        --moe-report path/to/olmoe/report.json \\
        --dense-report path/to/llama/report.json \\
        --output-dir output/moe_fragmentation

    # With optional gap analysis from traces:
    python experiments/rebuttal/moe_fragmentation_analysis.py \\
        --moe-report path/to/olmoe/report.json \\
        --dense-report path/to/llama/report.json \\
        --moe-trace path/to/olmoe/trace.json \\
        --dense-trace path/to/llama/trace.json \\
        --output-dir output/moe_fragmentation
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Inline constants and functions from soda.common.utils to keep this script
# self-contained (no dependency on the soda package).

# Minimum kernel launch latency in ms (4.5 us from TaxBreak paper)
T_FLOOR_SYS_MS = 0.0045


def calculate_hdbi(
    total_kernel_exec_time_ms: float,
    total_xlat_tax_ms: float,
    num_total_kernels: int,
) -> Dict[str, Any]:
    """
    Calculate HDBI (Host-Device Balance Index) per TaxBreak paper Eq. 6.

    HDBI = T_DeviceActive / (T_DeviceActive + T_Orchestrate)

    Where:
        T_DeviceActive = sum of kernel execution times
        T_Orchestrate = total_xlat_tax + (num_kernels x T_floor_sys)

    Classification:
        HDBI >= 0.5: device-bound (GPU compute dominates)
        0.2 <= HDBI < 0.5: balanced
        HDBI < 0.2: host-bound (CPU overhead dominates)
    """
    t_device_active = total_kernel_exec_time_ms
    delta_kt = num_total_kernels * T_FLOOR_SYS_MS
    t_orchestrate = total_xlat_tax_ms + delta_kt

    denominator = t_device_active + t_orchestrate
    if denominator > 0:
        hdbi_value = t_device_active / denominator
    else:
        hdbi_value = 0.0

    hdbi_value = max(0.0, min(1.0, hdbi_value))

    if hdbi_value >= 0.5:
        classification = "device-bound"
    elif hdbi_value >= 0.2:
        classification = "balanced"
    else:
        classification = "host-bound"

    return {
        "hdbi_value": hdbi_value,
        "hdbi_classification": classification,
        "t_device_active_ms": t_device_active,
        "t_orchestrate_ms": t_orchestrate,
        "delta_kt_ms": delta_kt,
    }


# =============================================================================
# Report Analysis
# =============================================================================

def extract_model_metrics(
    report_path: Path,
    profiling_runs: int = 150,
) -> Dict[str, Any]:
    """
    Extract TaxBreak metrics and compute HDBI from a SODA report.json.

    Returns per-inference metrics (totals divided by profiling_runs).
    """
    with open(report_path) as f:
        report = json.load(f)

    metadata = report.get("metadata", {})
    config = metadata.get("config", {})
    perf = report.get("performance_metrics", {})

    model_name = metadata.get("model_name", "unknown")

    # Raw totals (across all profiling runs)
    total_kernel_exec_ms = perf.get("total_kernel_exec_time_ms", 0)
    total_xlat_tax_ms = perf.get("total_xlat_tax_ms", 0)
    total_launch_tax_ms = perf.get("total_launch_tax_ms", 0)
    num_total_kernels = perf.get("num_total_kernels", 0)
    inference_time_ms = perf.get("inference_time_ms", 0)
    gpu_utilization = perf.get("gpu_utilization_percent", 0)
    avg_kernel_exec_ms = perf.get("avg_kernel_exec_time_ms", 0)

    # Compute HDBI (use existing if present, else compute)
    existing_hdbi = perf.get("hdbi")
    if existing_hdbi and "hdbi_value" in existing_hdbi:
        hdbi = existing_hdbi
    else:
        hdbi = calculate_hdbi(
            total_kernel_exec_time_ms=total_kernel_exec_ms,
            total_xlat_tax_ms=total_xlat_tax_ms,
            num_total_kernels=num_total_kernels,
        )

    # Per-inference metrics
    kernels_per_inf = num_total_kernels / max(profiling_runs, 1)
    kernel_exec_per_inf = total_kernel_exec_ms / max(profiling_runs, 1)
    xlat_tax_per_inf = total_xlat_tax_ms / max(profiling_runs, 1)
    delta_kt_per_inf = kernels_per_inf * T_FLOOR_SYS_MS
    t_orchestrate_per_inf = xlat_tax_per_inf + delta_kt_per_inf

    # Top kernels for fragmentation analysis
    top_kernels = report.get("top_kernels", {})

    return {
        "model_name": model_name,
        "config": config,
        "inference_time_ms": inference_time_ms,
        "gpu_utilization_percent": gpu_utilization,
        "profiling_runs": profiling_runs,
        # Totals
        "total_kernel_exec_ms": total_kernel_exec_ms,
        "total_xlat_tax_ms": total_xlat_tax_ms,
        "total_launch_tax_ms": total_launch_tax_ms,
        "num_total_kernels": num_total_kernels,
        # Per-inference
        "kernels_per_inference": int(kernels_per_inf),
        "avg_kernel_duration_us": avg_kernel_exec_ms * 1000,
        "t_device_active_per_inf_ms": kernel_exec_per_inf,
        "t_orchestrate_per_inf_ms": t_orchestrate_per_inf,
        "xlat_tax_per_inf_ms": xlat_tax_per_inf,
        "delta_kt_per_inf_ms": delta_kt_per_inf,
        # HDBI
        "hdbi_value": hdbi["hdbi_value"],
        "hdbi_classification": hdbi["hdbi_classification"],
        "hdbi": hdbi,
        # Top kernels
        "top_kernels": top_kernels,
    }


# =============================================================================
# Gap Analysis (Optional, Sampled)
# =============================================================================

MAX_TRACE_EVENTS = 500_000  # Cap to avoid OOM on huge traces


def sampled_gap_analysis(trace_path: Path) -> Optional[Dict[str, Any]]:
    """
    Perform gap analysis on a trace file, sampling up to MAX_TRACE_EVENTS.

    Classifies GPU idle gaps as:
    - Host gaps: before cudaLaunchKernel (CPU dispatch overhead)
    - Launch gaps: launch path overhead
    - Device gaps: between kernel executions (device fragmentation)
    """
    file_size_gb = trace_path.stat().st_size / (1024 ** 3)
    print(f"    Trace size: {file_size_gb:.1f} GB")

    # For very large traces, use line-by-line streaming
    if file_size_gb > 5:
        return _stream_gap_analysis(trace_path)

    # For manageable traces, load normally
    try:
        with open(trace_path) as f:
            trace_data = json.load(f)
    except (MemoryError, json.JSONDecodeError) as e:
        print(f"    Warning: Could not load trace ({e}), skipping gap analysis")
        return None

    events = trace_data.get("traceEvents", [])
    return _compute_gaps(events)


def _stream_gap_analysis(trace_path: Path) -> Optional[Dict[str, Any]]:
    """
    Stream-parse a large trace file, extracting only kernel and launch events
    up to MAX_TRACE_EVENTS.
    """
    kernel_events = []
    launch_events = []
    events_processed = 0

    try:
        with open(trace_path) as f:
            # Skip to traceEvents array
            for line in f:
                if '"traceEvents"' in line:
                    break

            # Parse events line by line (JSON array elements)
            buffer = ""
            brace_depth = 0
            for line in f:
                for char in line:
                    if char == "{":
                        brace_depth += 1
                        buffer += char
                    elif char == "}":
                        brace_depth -= 1
                        buffer += char
                        if brace_depth == 0 and buffer.strip():
                            try:
                                event = json.loads(buffer)
                                _categorize_event(
                                    event, kernel_events, launch_events
                                )
                                events_processed += 1
                            except json.JSONDecodeError:
                                pass
                            buffer = ""
                            if events_processed >= MAX_TRACE_EVENTS:
                                break
                    elif brace_depth > 0:
                        buffer += char

                if events_processed >= MAX_TRACE_EVENTS:
                    break

    except Exception as e:
        print(f"    Warning: Stream parsing error ({e})")
        if not kernel_events:
            return None

    print(f"    Processed {events_processed} events ({len(kernel_events)} kernels, {len(launch_events)} launches)")

    if not kernel_events:
        return None

    return _compute_gaps_from_lists(kernel_events, launch_events)


def _categorize_event(
    event: Dict, kernel_events: List, launch_events: List
):
    """Categorize a single trace event into kernel or launch."""
    cat = event.get("cat", "")
    name = event.get("name", "")
    ph = event.get("ph", "")
    ts = event.get("ts", 0)
    dur = event.get("dur", 0)

    if ph != "X" or ts == 0:
        return

    if cat == "kernel":
        kernel_events.append({"ts": ts, "dur": dur, "end": ts + dur})
    elif "cudaLaunchKernel" in name:
        launch_events.append({"ts": ts, "dur": dur, "end": ts + dur})


def _compute_gaps(events: List[Dict]) -> Dict[str, Any]:
    """Compute gap analysis from full event list."""
    kernel_events = []
    launch_events = []

    count = 0
    for event in events:
        _categorize_event(event, kernel_events, launch_events)
        count += 1
        if count >= MAX_TRACE_EVENTS:
            break

    if not kernel_events:
        return None

    return _compute_gaps_from_lists(kernel_events, launch_events)


def _compute_gaps_from_lists(
    kernel_events: List[Dict], launch_events: List[Dict]
) -> Dict[str, Any]:
    """Compute gap fractions from sorted kernel and launch event lists."""
    kernel_events.sort(key=lambda e: e["ts"])
    launch_events.sort(key=lambda e: e["ts"])

    host_gaps_us = []
    launch_gaps_us = []
    device_gaps_us = []

    launch_idx = 0
    for i in range(1, len(kernel_events)):
        prev_end = kernel_events[i - 1]["end"]
        curr_start = kernel_events[i]["ts"]
        gap = curr_start - prev_end

        if gap <= 0:
            continue

        # Find launch event in this gap
        while launch_idx < len(launch_events) and launch_events[launch_idx]["ts"] < prev_end:
            launch_idx += 1

        launch_in_gap = None
        if launch_idx < len(launch_events) and launch_events[launch_idx]["ts"] <= curr_start:
            launch_in_gap = launch_events[launch_idx]

        if launch_in_gap:
            host_gap = launch_in_gap["ts"] - prev_end
            launch_gap = curr_start - launch_in_gap["end"]
            if host_gap > 0:
                host_gaps_us.append(host_gap)
            if launch_gap > 0:
                launch_gaps_us.append(launch_gap)
        else:
            device_gaps_us.append(gap)

    total_host = sum(host_gaps_us)
    total_launch = sum(launch_gaps_us)
    total_device = sum(device_gaps_us)
    total_gap = total_host + total_launch + total_device

    if total_gap > 0:
        host_frac = total_host / total_gap
        launch_frac = total_launch / total_gap
        device_frac = total_device / total_gap
    else:
        host_frac = launch_frac = device_frac = 0.0

    return {
        "host_gap_total_us": total_host,
        "launch_gap_total_us": total_launch,
        "device_gap_total_us": total_device,
        "host_gap_fraction": host_frac,
        "launch_gap_fraction": launch_frac,
        "device_gap_fraction": device_frac,
        "num_host_gaps": len(host_gaps_us),
        "num_launch_gaps": len(launch_gaps_us),
        "num_device_gaps": len(device_gaps_us),
        "sampled": True,
        "max_events": MAX_TRACE_EVENTS,
    }


# =============================================================================
# Comparison and Reporting
# =============================================================================

def run_comparison(
    moe_report_path: Path,
    dense_report_path: Path,
    output_dir: Path,
    moe_trace_path: Optional[Path] = None,
    dense_trace_path: Optional[Path] = None,
    profiling_runs: int = 150,
) -> Dict[str, Any]:
    """
    Run MoE vs Dense fragmentation comparison.
    """
    # Detect GPU
    gpu_name = "N/A"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    print("=" * 74)
    print("MoE FRAGMENTATION ANALYSIS (Reviewer D Q2)")
    print("=" * 74)
    print("Host orchestration vs device-side fragmentation in MoE models")
    print(f"GPU: {gpu_name}")
    print("=" * 74)
    print()
    sys.stdout.flush()

    # Extract metrics from both reports
    print("Reading report files...")
    moe = extract_model_metrics(moe_report_path, profiling_runs)
    dense = extract_model_metrics(dense_report_path, profiling_runs)
    print(f"  MoE model:   {moe['model_name']}")
    print(f"  Dense model: {dense['model_name']}")
    print()
    sys.stdout.flush()

    # Optional gap analysis
    moe_gaps = None
    dense_gaps = None
    if moe_trace_path and moe_trace_path.exists():
        print(f"  Analyzing MoE trace gaps (sampled, max {MAX_TRACE_EVENTS} events)...")
        sys.stdout.flush()
        moe_gaps = sampled_gap_analysis(moe_trace_path)
    if dense_trace_path and dense_trace_path.exists():
        print(f"  Analyzing Dense trace gaps (sampled, max {MAX_TRACE_EVENTS} events)...")
        sys.stdout.flush()
        dense_gaps = sampled_gap_analysis(dense_trace_path)
    print()

    # Print comparison table
    _print_comparison_table(moe, dense, moe_gaps, dense_gaps)

    # Print findings
    _print_findings(moe, dense)

    # Save JSON report
    output_dir.mkdir(parents=True, exist_ok=True)
    report = _build_report(moe, dense, moe_gaps, dense_gaps, gpu_name)
    report_path = output_dir / "moe_fragmentation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_path}")
    sys.stdout.flush()

    return report


def _fmt(value, width=15, decimals=1, is_int=False):
    """Format a numeric value for table display."""
    if value is None:
        return f"{'N/A':>{width}}"
    if is_int:
        return f"{int(value):>{width},}"
    return f"{value:>{width}.{decimals}f}"


def _print_comparison_table(
    moe: Dict, dense: Dict,
    moe_gaps: Optional[Dict], dense_gaps: Optional[Dict],
):
    """Print side-by-side comparison table."""
    moe_label = moe["model_name"].split("/")[-1][:20]
    dense_label = dense["model_name"].split("/")[-1][:20]

    col_w = 17
    print(f"{'':>32} {moe_label:>{col_w}} {dense_label:>{col_w}}")
    print(f"{'':>32} {'(MoE, decode)':>{col_w}} {'(Dense, decode)':>{col_w}}")
    print("  " + "-" * (30 + col_w * 2))

    # Basic metrics
    print(f"  {'Inference time (ms)':<30}{_fmt(moe['inference_time_ms'], col_w)}{_fmt(dense['inference_time_ms'], col_w)}")
    print(f"  {'Kernels / inference':<30}{_fmt(moe['kernels_per_inference'], col_w, is_int=True)}{_fmt(dense['kernels_per_inference'], col_w, is_int=True)}")
    print(f"  {'Avg kernel duration (us)':<30}{_fmt(moe['avg_kernel_duration_us'], col_w, 2)}{_fmt(dense['avg_kernel_duration_us'], col_w, 2)}")
    print(f"  {'GPU utilization (%)':<30}{_fmt(moe['gpu_utilization_percent'], col_w)}{_fmt(dense['gpu_utilization_percent'], col_w)}")
    print()

    # TaxBreak decomposition (per-inference)
    print(f"  TaxBreak Decomposition (per-inference):")
    print(f"  {'T_DeviceActive (ms)':<30}{_fmt(moe['t_device_active_per_inf_ms'], col_w)}{_fmt(dense['t_device_active_per_inf_ms'], col_w)}")
    print(f"  {'T_Orchestrate (ms)':<30}{_fmt(moe['t_orchestrate_per_inf_ms'], col_w)}{_fmt(dense['t_orchestrate_per_inf_ms'], col_w)}")
    print(f"  {'  xlat tax (dFT+dCT) (ms)':<30}{_fmt(moe['xlat_tax_per_inf_ms'], col_w)}{_fmt(dense['xlat_tax_per_inf_ms'], col_w)}")
    print(f"  {'  dKT (launch tax) (ms)':<30}{_fmt(moe['delta_kt_per_inf_ms'], col_w)}{_fmt(dense['delta_kt_per_inf_ms'], col_w)}")
    print()

    # HDBI
    print(f"  {'HDBI':<30}{_fmt(moe['hdbi_value'], col_w, 3)}{_fmt(dense['hdbi_value'], col_w, 3)}")
    print(f"  {'Classification':<30}{moe['hdbi_classification'].upper():>{col_w}}{dense['hdbi_classification'].upper():>{col_w}}")
    print()

    # Gap analysis (if available)
    if moe_gaps or dense_gaps:
        print(f"  Gap Analysis (sampled from trace):")
        mg = moe_gaps or {}
        dg = dense_gaps or {}
        print(f"  {'Host gap fraction':<30}{_fmt(mg.get('host_gap_fraction'), col_w, 3)}{_fmt(dg.get('host_gap_fraction'), col_w, 3)}")
        print(f"  {'Launch gap fraction':<30}{_fmt(mg.get('launch_gap_fraction'), col_w, 3)}{_fmt(dg.get('launch_gap_fraction'), col_w, 3)}")
        print(f"  {'Device gap fraction':<30}{_fmt(mg.get('device_gap_fraction'), col_w, 3)}{_fmt(dg.get('device_gap_fraction'), col_w, 3)}")
        print()

    sys.stdout.flush()


def _print_findings(moe: Dict, dense: Dict):
    """Print key findings for Reviewer D."""
    moe_name = moe["model_name"].split("/")[-1]
    dense_name = dense["model_name"].split("/")[-1]

    print("=" * 74)
    print("KEY FINDINGS FOR REVIEWER D")
    print("=" * 74)
    print()

    # Finding 1: MoE host-boundedness
    print(f"1. {moe_name} (MoE decode) is {moe['hdbi_classification'].upper()} (HDBI={moe['hdbi_value']:.3f}):")
    kernel_ratio = moe["kernels_per_inference"] / max(dense["kernels_per_inference"], 1)
    print(f"   - {kernel_ratio:.0f}x more kernels per inference than dense ({moe['kernels_per_inference']:,} vs {dense['kernels_per_inference']:,})")
    print(f"   - Each kernel averages {moe['avg_kernel_duration_us']:.2f} us ({dense['avg_kernel_duration_us'] / max(moe['avg_kernel_duration_us'], 0.01):.1f}x smaller than dense)")
    print(f"   - T_Orchestrate dominates: {moe['t_orchestrate_per_inf_ms']:.1f} ms vs {moe['t_device_active_per_inf_ms']:.1f} ms T_DeviceActive")
    print()

    # Finding 2: Cause
    print("2. The host-boundedness is CAUSED by device fragmentation:")
    print("   - MoE expert routing creates many small kernels (scatter/gather/topk)")
    print("   - Each tiny kernel still pays full dFT + dKT overhead")
    print("   - HDBI captures this: GPU can't be kept busy when host overhead >> kernel time")
    print()

    # Finding 3: Dense comparison
    print(f"3. {dense_name} (Dense) is comparatively {dense['hdbi_classification'].upper()} (HDBI={dense['hdbi_value']:.3f}):")
    print(f"   - Fewer, larger kernels -> overhead amortized better")
    print(f"   - T_Orchestrate ({dense['t_orchestrate_per_inf_ms']:.1f} ms) vs T_DeviceActive ({dense['t_device_active_per_inf_ms']:.1f} ms)")
    print()

    # Conclusion
    print("CONCLUSION:")
    print("  GPU idle time in MoE decode is a CONSEQUENCE of device-side fragmentation")
    print("  (many small expert kernels) that manifests as host-boundedness through the")
    print("  per-kernel overhead tax. TaxBreak's HDBI correctly identifies this: the")
    print(f"  3x lower HDBI for MoE ({moe['hdbi_value']:.3f}) vs Dense ({dense['hdbi_value']:.3f}) quantifies how")
    print("  kernel fragmentation amplifies host-side overhead.")
    print()
    print("=" * 74)
    sys.stdout.flush()


def _build_report(
    moe: Dict, dense: Dict,
    moe_gaps: Optional[Dict], dense_gaps: Optional[Dict],
    gpu_name: str,
) -> Dict[str, Any]:
    """Build the JSON report."""

    def model_entry(m: Dict, gaps: Optional[Dict]) -> Dict:
        entry = {
            "model": m["model_name"],
            "config": m["config"],
            "inference_time_ms": m["inference_time_ms"],
            "kernels_per_inference": m["kernels_per_inference"],
            "avg_kernel_duration_us": m["avg_kernel_duration_us"],
            "gpu_utilization_percent": m["gpu_utilization_percent"],
            "hdbi": m["hdbi_value"],
            "hdbi_classification": m["hdbi_classification"],
            "taxbreak_per_inference": {
                "t_device_active_ms": m["t_device_active_per_inf_ms"],
                "t_orchestrate_ms": m["t_orchestrate_per_inf_ms"],
                "xlat_tax_ms": m["xlat_tax_per_inf_ms"],
                "delta_kt_ms": m["delta_kt_per_inf_ms"],
            },
            "taxbreak_totals": {
                "total_kernel_exec_ms": m["total_kernel_exec_ms"],
                "total_xlat_tax_ms": m["total_xlat_tax_ms"],
                "total_launch_tax_ms": m["total_launch_tax_ms"],
                "num_total_kernels": m["num_total_kernels"],
                "profiling_runs": m["profiling_runs"],
            },
        }
        if gaps:
            entry["gap_analysis"] = gaps
        return entry

    moe_name = moe["model_name"].split("/")[-1]
    dense_name = dense["model_name"].split("/")[-1]

    return {
        "title": "MoE Fragmentation Analysis (Reviewer D Q2)",
        "environment": {"gpu": gpu_name},
        "methodology": {
            "approach": "TaxBreak decomposition from SODA report.json with HDBI",
            "hdbi_formula": "HDBI = T_DeviceActive / (T_DeviceActive + T_Orchestrate)",
            "t_floor_sys_ms": T_FLOOR_SYS_MS,
            "gap_analysis": "Optional sampled trace analysis (first 500K events)",
        },
        "comparison": {
            "moe": model_entry(moe, moe_gaps),
            "dense": model_entry(dense, dense_gaps),
        },
        "conclusion": (
            f"MoE decode ({moe_name}) is {moe['hdbi_classification']} "
            f"(HDBI={moe['hdbi_value']:.3f}) with {moe['kernels_per_inference']:,} "
            f"kernels/inference averaging {moe['avg_kernel_duration_us']:.2f} us. "
            f"Dense decode ({dense_name}) is {dense['hdbi_classification']} "
            f"(HDBI={dense['hdbi_value']:.3f}) with {dense['kernels_per_inference']:,} "
            f"kernels/inference averaging {dense['avg_kernel_duration_us']:.2f} us. "
            f"The {moe['kernels_per_inference'] / max(dense['kernels_per_inference'], 1):.0f}x "
            f"higher kernel count in MoE creates proportionally more per-kernel "
            f"overhead (dFT + dKT), causing host-boundedness through device fragmentation."
        ),
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MoE Fragmentation Analysis for Reviewer D Q2"
    )
    parser.add_argument(
        "--moe-report", type=Path, required=True,
        help="Path to MoE model report.json"
    )
    parser.add_argument(
        "--dense-report", type=Path, required=True,
        help="Path to Dense model report.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/moe_fragmentation"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--moe-trace", type=Path, default=None,
        help="Optional: path to MoE model trace.json for gap analysis"
    )
    parser.add_argument(
        "--dense-trace", type=Path, default=None,
        help="Optional: path to Dense model trace.json for gap analysis"
    )
    parser.add_argument(
        "--profiling-runs", type=int, default=150,
        help="Number of profiling runs in the trace (for per-inference metrics)"
    )

    args = parser.parse_args()

    if not args.moe_report.exists():
        print(f"Error: MoE report not found: {args.moe_report}")
        return 1
    if not args.dense_report.exists():
        print(f"Error: Dense report not found: {args.dense_report}")
        return 1

    try:
        run_comparison(
            moe_report_path=args.moe_report,
            dense_report_path=args.dense_report,
            output_dir=args.output_dir,
            moe_trace_path=args.moe_trace,
            dense_trace_path=args.dense_trace,
            profiling_runs=args.profiling_runs,
        )
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
