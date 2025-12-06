#!/usr/bin/env python3
"""
Compare PyTorch and baremetal GEMM launch tax.

Joins results from framework/pytorch/output/unique_gemm_sequences.json and
baremetal/output/baremetal_gemm_runs.json, verifies kernel matching,
computes per-kernel launch tax deltas and percentages, and emits
baremetal/output/bm_vs_framework_report.json.
"""

import json
import os
import sys
from collections import defaultdict

from soda.common import utils, print_utils
from soda.common.data import ATenOp, Kernel

def load_pytorch_results(pytorch_file):
    """
    Load PyTorch event sequences and extract per-kernel launch statistics.
    
    Returns: dict mapping job_id -> {kernel, op_signature, stats, xlat_stats, py_stats}
    """
    with open(pytorch_file, 'r') as f:
        data = json.load(f)
    
    sequences = data["sequences"]
    results = {}
    
    for idx, sequence in enumerate(sequences):
        job_id = f"{idx+1:04d}"
        
        kernel_dict = sequence["kernel"]
        aten_op_dict = sequence["aten_op"]
        launch_tax = sequence.get("launch_tax")
        xlat_tax = sequence.get("xlat_tax")
        py_tax = sequence.get("py_tax")
        
        # Convert dicts to objects
        aten_op = ATenOp.from_dict(aten_op_dict)
        kernel = Kernel.from_dict(kernel_dict)
        
        op_signature = aten_op.get_signature()
        kernel_info = kernel.get_signature()
        
        # Extract stats
        stats = launch_tax if launch_tax else None
        xlat_stats = xlat_tax if xlat_tax else None
        py_stats = py_tax if py_tax else None
        
        results[job_id] = {
            "kernel": kernel_info,
            "op_signature": op_signature,
            "stats": stats,
            "xlat_stats": xlat_stats,
            "py_stats": py_stats,
        }
    
    return results


def load_baremetal_results(baremetal_file):
    """
    Load baremetal results 
    
    Returns: dict mapping job_id -> {kernel, stats}
    """
    with open(baremetal_file, 'r') as f:
        data = json.load(f)
    
    sequences = data["sequences"]
    results = {}
    
    for sequence in sequences:
        # Skip None entries (e.g., skipped batched GEMM jobs)
        if sequence is None:
            continue

        job_id = sequence.get("job_id")

        # Skip null kernel job (0000) from comparison results
        if sequence["kernel"]["name"] == "__null__":
            continue

        results[job_id] = {
            "kernel": sequence["kernel"],
            "stats": sequence["launch_tax"],
        }
    
    return results


def compare_results(pytorch_results, baremetal_results):
    """
    Compare PyTorch and baremetal results.
    
    Returns: list of match entries
    """
    matches = []
    
    for job_id in sorted(pytorch_results.keys()):
        if job_id not in baremetal_results:
            print(f"Warning: Job {job_id} not found in baremetal results", file=sys.stderr)
            continue
        
        pytorch = pytorch_results[job_id]
        baremetal = baremetal_results[job_id]
        
        # Compute deltas (all values in microseconds)
        fw_stats = pytorch["stats"]
        bm_stats = baremetal["stats"]
        fw_xlat_stats = pytorch.get("xlat_stats")
        fw_py_stats = pytorch.get("py_stats")
        fw_avg = fw_stats["avg"]
        bm_avg = bm_stats["avg"]
        fw_xlat_avg = fw_xlat_stats["avg"] if fw_xlat_stats else None
        fw_py_avg = fw_py_stats["avg"] if fw_py_stats else None
        
        delta = fw_avg - bm_avg
        # Calculate percentage difference: (FW - BM) / FW * 100
        # Shows how much faster BM is compared to FW (base)
        delta_pct = ((fw_avg - bm_avg) / fw_avg * 100) if fw_avg > 0 else 0.0
        
        # Build match entry
        match_entry = {
            "job_id": job_id,
            "op_signature": pytorch["op_signature"],
            "kernel": {
                "name": pytorch["kernel"]["name"],
                "grid": pytorch["kernel"]["grid"],
                "block": pytorch["kernel"]["block"],
                "shared_memory": pytorch["kernel"]["shared_memory"],
            },
            "framework": fw_stats,
            "framework_xlat": fw_xlat_stats,
            "framework_py": fw_py_stats,
            "baremetal": bm_stats,
            "delta": delta,
            "delta_pct": delta_pct,
            "framework_xlat_avg": fw_xlat_avg,
            "framework_py_avg": fw_py_avg,
        }
        
        matches.append(match_entry)
    
    return matches


def print_summary(matches, baseline_tax=None):
    """Print comparison summary as compact tables."""
    per_kernel_rows = []
    for match in matches:
        kernel_name = match["kernel"]["name"]
        fw_xlat_avg = match.get("framework_xlat_avg")
        fw_py_avg = match.get("framework_py_avg")
        per_kernel_rows.append([
            match["job_id"],
            kernel_name,
            f"{fw_py_avg:.2f}" if fw_py_avg is not None else "-",
            f"{fw_xlat_avg:.2f}" if fw_xlat_avg is not None else "-",
            f"{match['framework']['avg']:.2f}",
            f"{match['baremetal']['avg']:.2f}",
            f"{match['delta_pct']:.1f}",
        ])

    if per_kernel_rows:
        title_suffix = f" | Baseline (null kernel): {baseline_tax:.2f} μs" if baseline_tax is not None else ""
        print_utils.comp_table(
            title=f"Per-Kernel Results ({len(per_kernel_rows)} kernels){title_suffix}",
            headers=["ID", "Kernel", "Tpy (μs)", "Txlat (μs)", "Tlaunch_fw (μs)", "Tlaunch_bm (μs)", "Δ(%)"],
            data=per_kernel_rows,
        )


def report():
    """
    Main comparison function.
    """
    pytorch_file = utils.get_path("PYTORCH_GEMM_SEQUENCES")
    baremetal_file = utils.get_path("BAREMETAL_GEMM_KERNELS")
    output_file = utils.get_path("FINAL_REPORT")
    
    utils.ensure_file(pytorch_file)
    utils.ensure_file(baremetal_file)
    
    print(f"Loading PyTorch sequences from {pytorch_file}")
    pytorch_results = load_pytorch_results(pytorch_file)
    print(f"Loaded {len(pytorch_results)} PyTorch sequences")
    
    print(f"Loading baremetal sequences from {baremetal_file}")
    baremetal_results = load_baremetal_results(baremetal_file)
    print(f"Loaded {len(baremetal_results)} baremetal sequences")
    
    # Extract null launch tax for baseline
    null_launch_tax = None

    baremetal_data = utils.load_json(baremetal_file)
    for sequence in baremetal_data["sequences"]:
        # Skip None entries (e.g., skipped batched GEMM jobs)
        if sequence is None:
            continue
        
        # Extract null launch tax for baseline
        if sequence["kernel"]["name"] == "__null__":
            null_launch_tax = sequence["launch_tax"]["avg"]
            break
    
    # Compare
    matches = compare_results(pytorch_results, baremetal_results)
    
    # Print summary
    print_summary(matches, baseline_tax=null_launch_tax)

    
    # Write output
    output_data = {
        "summary": {
            "total_kernels": len(matches),
        },
        "matches": matches,
    }
    
    utils.save_json(output_file, output_data)
