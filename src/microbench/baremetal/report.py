#!/usr/bin/env python3
"""
Compare PyTorch and baremetal GEMM kernel launch tax.

Joins results from framework/pytorch/output/unique_gemm_sequences.json and
baremetal/output/baremetal_gemm_runs.json, verifies kernel matching,
computes per-kernel launch tax deltas and percentages, and emits
baremetal/output/bm_vs_framework_report.json.
"""

import json
import os
import sys

from soda.common import utils, print_utils
from soda.common.data import CPUOp, Kernel

def load_pytorch_results(pytorch_file):
    """
    Load PyTorch event sequences and extract per-kernel statistics.
    
    Returns: dict mapping job_id -> {kernel, op_signature, stats}
    """
    with open(pytorch_file, 'r') as f:
        data = json.load(f)
    
    sequences = data["sequences"]
    results = {}
    
    for idx, sequence in enumerate(sequences):
        job_id = f"{idx+1:04d}"
        
        kernel_dict = sequence["kernel"]
        cpu_op_dict = sequence["cpu_op"]
        meta = sequence["meta"]
        
        # Convert dicts to objects
        cpu_op = CPUOp.from_dict(cpu_op_dict)
        kernel = Kernel.from_dict(kernel_dict)
        
        op_signature = cpu_op.get_signature()
        kernel_info = kernel.get_signature()
        
        # Extract stats
        stats = {
            "avg_kernel_tax": meta["avg_kernel_tax"],
            "min_kernel_tax": meta["min_kernel_tax"],
            "max_kernel_tax": meta["max_kernel_tax"],
            "count": meta["count"],
        }
        
        results[job_id] = {
            "kernel": kernel_info,
            "op_signature": op_signature,
            "stats": stats,
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

        job_id = sequence["meta"]["job_id"]

        # Skip null kernel job (0000) from comparison results
        if sequence["kernel"]["name"] == "__null__":
            continue

        results[job_id] = {
            "kernel": sequence["kernel"],
            "stats": sequence["meta"],  # meta contains all the stats fields
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
        fw_avg = pytorch["stats"]["avg_kernel_tax"]
        bm_avg = baremetal["stats"]["avg_kernel_tax"]
        
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
            "framework": {
                "avg_kernel_tax": fw_avg,
                "min_kernel_tax": pytorch["stats"]["min_kernel_tax"],
                "max_kernel_tax": pytorch["stats"]["max_kernel_tax"],
                "count": pytorch["stats"]["count"],
            },
            "baremetal": {
                "kernel_name": baremetal["kernel"]["name"],
                "avg_kernel_tax": bm_avg,
                "min_kernel_tax": baremetal["stats"]["min_kernel_tax"],
                "max_kernel_tax": baremetal["stats"]["max_kernel_tax"],
                "count": baremetal["stats"]["count"],
            },
            "delta": delta,
            "delta_pct": delta_pct,
        }
        
        matches.append(match_entry)
    
    return matches


def print_summary(matches, baseline_tax=None):
    """Print comparison summary as compact tables."""
    per_kernel_rows = []
    for match in matches:
        kernel_name = match["kernel"]["name"]
        per_kernel_rows.append([
            match["job_id"],
            kernel_name,
            f"{match['framework']['avg_kernel_tax']:.2f}",
            f"{match['baremetal']['avg_kernel_tax']:.2f}",
            f"{match['delta_pct']:.1f}",
        ])

    if per_kernel_rows:
        title_suffix = f" | Baseline (null kernel): {baseline_tax:.2f} μs" if baseline_tax is not None else ""
        print_utils.comp_table(
            title=f"Per-Kernel Results ({len(per_kernel_rows)} kernels){title_suffix}",
            headers=["ID", "Kernel", "Framework (μs)", "Baremetal (μs)", "Δ(%)"],
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
    
    # Extract null kernel tax for baseline
    null_kernel_tax = None

    baremetal_data = utils.load_json(baremetal_file)
    for sequence in baremetal_data["sequences"]:
        # Skip None entries (e.g., skipped batched GEMM jobs)
        if sequence is None:
            continue
        
        # Extract null kernel tax for baseline
        if sequence["kernel"]["name"] == "__null__":
            null_kernel_tax = sequence["meta"]["avg_kernel_tax"]
            break
    
    # Compare
    matches = compare_results(pytorch_results, baremetal_results)
    
    # Print summary
    print_summary(matches, baseline_tax=null_kernel_tax)
    
    # Write output
    output_data = {
        "summary": {
            "total_kernels": len(matches),
        },
        "matches": matches,
    }
    
    utils.save_json(output_file, output_data)
