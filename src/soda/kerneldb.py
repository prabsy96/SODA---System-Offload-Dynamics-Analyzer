"""
Kernel Database Generator

Extracts unique kernels from a single profiled inference run and builds
a structured database for downstream TaxBreak analysis.
"""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch

from soda.common import utils
from soda.common.data import clean_kernel_name

# Vendor-replay eligible patterns: kernels that can be replayed via
# the baremetal cuBLAS/cuBLASLt C++ binary.
VENDOR_REPLAY_PATTERNS = [
    "cublas",
    "cublasLt",
    "cutlass",
]

# PyTorch-internal GEMM patterns: classified as GEMM but NOT vendor-replay
# eligible (no baremetal equivalent).
INTERNAL_GEMM_PATTERNS = [
    "nvjet",
    "wgmma",
    "s884gemm",
]


def _is_vendor_replayable(kernel_name: str) -> bool:
    """Check if a kernel can be replayed via the baremetal cuBLAS binary."""
    lower = kernel_name.lower()
    # Must match a vendor pattern AND not be an internal pattern
    has_vendor = any(p in lower for p in VENDOR_REPLAY_PATTERNS)
    is_internal = any(p in lower for p in INTERNAL_GEMM_PATTERNS)
    return has_vendor and not is_internal


def _extract_last_run_sequences(
    sequences: List[Dict[str, Any]],
    num_profiled_runs: int,
) -> List[Dict[str, Any]]:
    """
    Extract sequences belonging to the last profiled run only.

    The profiler captures all runs contiguously. We partition kernel
    timestamps into ``num_profiled_runs`` equal buckets and keep only
    the final bucket.
    """
    if num_profiled_runs <= 1:
        return sequences

    # Collect kernel timestamps from all valid sequences
    kernel_timestamps = []
    for seq in sequences:
        kernel = seq.get("kernel")
        if kernel and "ts" in kernel:
            kernel_timestamps.append(kernel["ts"])

    if not kernel_timestamps:
        return sequences

    ts_min = min(kernel_timestamps)
    ts_max = max(kernel_timestamps)
    total_span = ts_max - ts_min

    if total_span <= 0:
        return sequences

    # Boundary for the last run (last 1/N of the time span)
    last_run_boundary = ts_min + total_span * (num_profiled_runs - 1) / num_profiled_runs

    last_run = []
    for seq in sequences:
        kernel = seq.get("kernel")
        if kernel and kernel.get("ts", 0) >= last_run_boundary:
            last_run.append(seq)

    return last_run


def generate_kernel_database(
    tracer,
    args,
    output_path: Path,
) -> Path:
    """
    Generate an op-kernel database from a completed ModelTracer run.

    Args:
        tracer: A ModelTracer instance whose ``.run()`` has completed.
        args: Parsed CLI args (for metadata).
        output_path: Where to write ``kernel_database.json``.

    Returns:
        Path to the written database file.
    """
    sequences = tracer.sequences
    num_runs = getattr(tracer, "num_profiled_runs", 150)

    # --- Step 1: filter for valid kernel sequences and classify GEMM ---
    kernel_sequences = utils.filter_kernel_sequences(sequences)

    # --- Step 2: isolate the last profiled run ---
    last_run_seqs = _extract_last_run_sequences(kernel_sequences, num_runs)

    # --- Step 3: compute per-sequence tax metrics ---
    metrics_to_compute = ["launch_tax", "aten_xlat_tax", "py_tax"]
    utils.calculate_sequence_metrics(last_run_seqs, metrics_to_compute)

    # --- Step 4: group by identity and aggregate ---
    grouped = utils.group_sequences_by_identity(last_run_seqs)
    event_types = ["kernel", "aten_op", "cuda_launch", "torch_op"]
    aggregated = utils.aggregate_sequences(grouped, metrics_to_compute, event_types)

    # --- Step 5: sort by total kernel duration descending ---
    for entry in aggregated:
        kernel = entry.get("kernel", {})
        dur_us = kernel.get("dur", 0) or 0
        entry["_total_dur"] = dur_us * entry.get("freq", 1)

    aggregated.sort(key=lambda e: e["_total_dur"], reverse=True)

    # --- Step 6: build database entries ---
    gpu_name = ""
    all_gpu_names = []
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        num_gpus_used = getattr(args, "num_gpus", 1)
        all_gpu_names = [
            torch.cuda.get_device_name(i)
            for i in range(num_gpus_used)
        ]

    db_entries = []
    for rank, entry in enumerate(aggregated, start=1):
        kernel = entry.get("kernel", {})
        aten_op = entry.get("aten_op", {})

        raw_name = kernel.get("name", "")
        cleaned = clean_kernel_name(raw_name)

        aten_op_name = aten_op.get("name", "")
        is_gemm_op_flag = utils.is_gemm_op(aten_op_name)
        is_gemm_kernel_flag = utils.is_gemm_kernel(raw_name)
        is_gemm = is_gemm_op_flag or is_gemm_kernel_flag
        is_vendor = _is_vendor_replayable(raw_name)

        # Three-way classification:
        #   gemm     — at least one definitive GEMM signal (kernel name OR op name matches)
        #   unknown  — op name matches a GEMM op but kernel name is unrecognized
        #              (e.g. a new vendor kernel or Triton-backed mm)
        #   non_gemm — neither op nor kernel name matches any GEMM pattern
        if is_gemm_kernel_flag:
            kernel_class = "gemm"
        elif is_gemm_op_flag:
            kernel_class = "unknown"
        else:
            kernel_class = "non_gemm"

        freq = entry.get("freq", 1)

        # Duration stats: if aggregated, pull from kernel duration samples
        dur_val = kernel.get("dur", 0) or 0
        all_dur = kernel.get("all_dur")
        if all_dur and len(all_dur) > 1:
            dur_avg = sum(all_dur) / len(all_dur)
            dur_min = min(all_dur)
            dur_max = max(all_dur)
            variance = sum((x - dur_avg) ** 2 for x in all_dur) / (len(all_dur) - 1)
            dur_std = math.sqrt(variance)
            total_dur = sum(all_dur)
        else:
            dur_avg = dur_val
            dur_min = dur_val
            dur_max = dur_val
            dur_std = 0.0
            total_dur = dur_val * freq

        # Tax stats from aggregated metric summaries
        def _tax_avg(metric_name):
            m = entry.get(metric_name, {})
            if isinstance(m, dict):
                return round(m.get("avg", 0.0), 4)
            return 0.0

        db_entries.append({
            "id": f"K{rank:04d}",
            "rank": rank,
            "kernel": {
                "name": cleaned,
                "raw_name": raw_name,
                "grid": list(kernel.get("grid", [0, 0, 0])),
                "block": list(kernel.get("block", [0, 0, 0])),
                "shared_memory": kernel.get("shared_memory", 0),
                "registers_per_thread": kernel.get("registers_per_thread", None),
            },
            "aten_op": {
                "name": aten_op.get("name", ""),
                "input_dims": aten_op.get("input_dims", []),
                "input_strides": aten_op.get("input_strides", []),
                "input_type": aten_op.get("input_type", []),
                "concrete_inputs": aten_op.get("concrete_inputs", []),
            },
            "classification": {
                "is_gemm": is_gemm,
                "is_vendor_replay": is_vendor,
                # i_lib=1: kernel goes through a vendor CUDA library (cuBLAS/cuBLASLt/cutlass).
                # i_lib=0: kernel is dispatched through PyTorch ATen without a library front-end
                #          (includes nvjet, wgmma, s884gemm, all elementwise ops).
                "i_lib": 1 if is_vendor else 0,
                # kernel_class: three-way label.
                #   "gemm"     — kernel name matches a known GEMM pattern (definitive)
                #   "unknown"  — aten op is GEMM-type but kernel name is unrecognized
                #   "non_gemm" — neither op nor kernel name matches any GEMM pattern
                "kernel_class": kernel_class,
            },
            "statistics": {
                "frequency": freq,
                "total_duration_us": round(total_dur, 4),
                "avg_duration_us": round(dur_avg, 4),
                "min_duration_us": round(dur_min, 4),
                "max_duration_us": round(dur_max, 4),
                "std_duration_us": round(dur_std, 4),
            },
            "taxes": {
                "avg_launch_tax_us": _tax_avg("launch_tax"),
                "avg_aten_xlat_tax_us": _tax_avg("aten_xlat_tax"),
                "avg_py_tax_us": _tax_avg("py_tax"),
            },
        })

    # --- Step 7: build top-level database ---
    total_unique = len(db_entries)
    gemm_count = sum(1 for e in db_entries if e["classification"]["is_gemm"])

    database = {
        "version": "1.0",
        "metadata": {
            "model": args.model,
            "precision": args.precision,
            "compile_type": args.compile_type,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "max_new_tokens": args.max_new_tokens,
            "gpu_name": gpu_name,
            "num_gpus": getattr(args, "num_gpus", 1),
            "all_gpu_names": all_gpu_names,
            "timestamp": datetime.now().isoformat(),
            "num_profiled_runs": num_runs,
            "last_run_sequences": len(last_run_seqs),
        },
        "summary": {
            "total_unique_kernels": total_unique,
            "gemm_kernels": gemm_count,
            "non_gemm_kernels": total_unique - gemm_count,
            "vendor_replay_kernels": sum(
                1 for e in db_entries if e["classification"]["is_vendor_replay"]
            ),
            "total_invocations": sum(
                e["statistics"]["frequency"] for e in db_entries
            ),
            "total_kernel_exec_time_us": round(
                sum(e["statistics"]["total_duration_us"] for e in db_entries), 4
            ),
        },
        "kernels": db_entries,
    }

    # --- Step 8: write to disk ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(database, f, indent=2)

    print(f"Kernel database written to: {output_path}")
    print(
        f"  {total_unique} unique kernels "
        f"({gemm_count} GEMM, {total_unique - gemm_count} non-GEMM)"
    )

    return output_path
