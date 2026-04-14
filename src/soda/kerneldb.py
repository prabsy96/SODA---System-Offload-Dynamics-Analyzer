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

# PyTorch-internal patterns: these kernels are framework-native (I_lib=0)
# — dispatched directly by ATen/Inductor without a vendor library front-end.
FRAMEWORK_NATIVE_PATTERNS = [
    "nvjet",
    "wgmma",
    "s884gemm",
]

# Backward compat alias
INTERNAL_GEMM_PATTERNS = FRAMEWORK_NATIVE_PATTERNS


def _is_vendor_replayable(kernel_name: str) -> bool:
    """Check if a kernel can be replayed via the baremetal cuBLAS binary."""
    lower = kernel_name.lower()
    # Must match a vendor pattern AND not be a framework-native pattern
    has_vendor = any(p in lower for p in VENDOR_REPLAY_PATTERNS)
    is_fw_native = any(p in lower for p in FRAMEWORK_NATIVE_PATTERNS)
    return has_vendor and not is_fw_native


def _extract_last_run_sequences(
    sequences: List[Dict[str, Any]],
    num_profiled_runs: int,
) -> List[Dict[str, Any]]:
    """
    Extract sequences belonging to the last profiled run only.

    The profiler captures all runs contiguously.  Each run contributes
    approximately ``len(sequences) // num_profiled_runs`` sequences.

    Uses a **count-based** split: take the last ``expected_per_run``
    sequences.  This is more robust than a time-based split, which
    assumed equal run durations and failed when early runs were slower
    (JIT compilation, cold caches).

    A warning is emitted when the last bucket has fewer than 80% of the
    expected size, which indicates an unbalanced trace.
    """
    if num_profiled_runs <= 1:
        return sequences

    n = len(sequences)
    if n == 0:
        return sequences

    expected_per_run = max(1, n // num_profiled_runs)
    last_run = sequences[-expected_per_run:]

    if len(last_run) < expected_per_run * 0.8:
        print(
            f"Warning: _extract_last_run_sequences: expected ~{expected_per_run} "
            f"sequences for last run but found {len(last_run)}. "
            f"Trace may be unbalanced "
            f"(total={n}, num_runs={num_profiled_runs})."
        )

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
    metrics_to_compute = ["T_launch", "T_dispatch", "T_Py"]
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
        is_lib_mediated_op = utils.is_library_mediated_op(aten_op_name)
        is_lib_mediated_kernel = utils.is_library_mediated_kernel(raw_name)
        is_library_mediated = is_lib_mediated_op or is_lib_mediated_kernel
        is_vendor = _is_vendor_replayable(raw_name)

        # Three-way kernel_class (paper terminology):
        #   library_mediated — kernel or op routes through a vendor library (I_lib=1)
        #   unknown          — ATen op suggests a library path but kernel name
        #                      is unrecognized (e.g. new vendor kernel or Triton-backed mm)
        #   framework_native — neither op nor kernel name indicates a vendor library (I_lib=0)
        if is_lib_mediated_kernel:
            kernel_class = "library_mediated"
        elif is_lib_mediated_op:
            kernel_class = "unknown"
        else:
            kernel_class = "framework_native"

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
                "is_library_mediated": is_library_mediated,
                # Backward-compatible alias (deprecated — use is_library_mediated)
                "is_gemm": is_library_mediated,
                "is_vendor_replay": is_vendor,
                # i_lib=1: kernel goes through a vendor CUDA library (cuBLAS/cuBLASLt/cutlass).
                # i_lib=0: kernel is dispatched through PyTorch ATen without a library front-end
                #          (includes nvjet, wgmma, s884gemm, all elementwise ops).
                "i_lib": 1 if is_vendor else 0,
                # kernel_class: three-way label (paper terminology).
                #   "library_mediated" — kernel routes through a vendor library
                #   "unknown"          — ATen op suggests library path but kernel is unrecognized
                #   "framework_native" — no vendor library involvement
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
                "avg_T_launch_us": _tax_avg("T_launch"),
                "avg_T_dispatch_us": _tax_avg("T_dispatch"),
                "avg_T_Py_us": _tax_avg("T_Py"),
            },
        })

    # --- Step 7: build top-level database ---
    total_unique = len(db_entries)
    lib_mediated_count = sum(1 for e in db_entries if e["classification"]["is_library_mediated"])

    # Extract HuggingFace model config for downstream MoE classification.
    model_config_dict = None
    if hasattr(tracer, "model") and hasattr(getattr(tracer, "model", None), "config"):
        try:
            model_config_dict = tracer.model.config.to_dict()
        except Exception:
            pass

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
            "model_config": model_config_dict,
        },
        "summary": {
            "total_unique_kernels": total_unique,
            "library_mediated_kernels": lib_mediated_count,
            "framework_native_kernels": total_unique - lib_mediated_count,
            # Backward-compatible aliases (deprecated)
            "gemm_kernels": lib_mediated_count,
            "non_gemm_kernels": total_unique - lib_mediated_count,
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
        f"({lib_mediated_count} library-mediated, {total_unique - lib_mediated_count} framework-native)"
    )

    return output_path
