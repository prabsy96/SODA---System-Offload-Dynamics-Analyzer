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
    
    # Process all baremetal results (including null kernel)
    for job_id in sorted(baremetal_results.keys()):
        baremetal = baremetal_results[job_id]
        is_null_kernel = baremetal["kernel"]["name"] == "__null_kernel__"
        
        if is_null_kernel:
            # Null kernel: no PyTorch match
            match_entry = {
                "job_id": job_id,
                "op_signature": None,
                "kernel": {
                    "name": baremetal["kernel"]["name"],
                    "grid": baremetal["kernel"].get("grid"),
                    "block": baremetal["kernel"].get("block"),
                    "shared_memory": baremetal["kernel"].get("shared_memory"),
                },
                "framework": None,
                "framework_xlat": None,
                "framework_py": None,
                "baremetal": baremetal["stats"],
                "framework_xlat_avg": None,
                "framework_py_avg": None,
                "framework_launch_avg": None,
            }
        else:
            # Regular kernel: should have PyTorch match
            if job_id not in pytorch_results:
                print(f"Warning: Job {job_id} not found in PyTorch results", file=sys.stderr)
                continue
            
            pytorch = pytorch_results[job_id]
            
            # Extract stats (all values in microseconds)
            fw_stats = pytorch["stats"]
            fw_xlat_stats = pytorch.get("xlat_stats")
            fw_py_stats = pytorch.get("py_stats")
            fw_xlat_avg = fw_xlat_stats["avg"] if fw_xlat_stats else None
            fw_py_avg = fw_py_stats["avg"] if fw_py_stats else None
            fw_launch_avg = fw_stats["avg"] if fw_stats else None
            
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
                "baremetal": baremetal["stats"],
                "framework_xlat_avg": fw_xlat_avg,
                "framework_py_avg": fw_py_avg,
                "framework_launch_avg": fw_launch_avg,
            }
        
        matches.append(match_entry)
    
    return matches


def print_summary(matches):
    """Print comparison summary as compact tables."""
    per_kernel_rows = []
    for match in matches:
        kernel_name = match["kernel"]["name"]
        fw_xlat_avg = match.get("framework_xlat_avg")
        fw_py_avg = match.get("framework_py_avg")
        fw_launch_avg = match.get("framework_launch_avg")
        per_kernel_rows.append([
            match["job_id"],
            kernel_name,
            f"{fw_py_avg:.2f}" if fw_py_avg is not None else "-",
            f"{fw_xlat_avg:.2f}" if fw_xlat_avg is not None else "-",
            f"{fw_launch_avg:.2f}" if fw_launch_avg is not None else "-",
        ])

    if per_kernel_rows:
        print_utils.comp_table(
            title=f"Per GEMM Framework Overhead (us)",
            headers=["ID", "Kernel", "py", "aten+culib", "launch"],
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
        if sequence["kernel"]["name"] == "__null_kernel__":
            null_launch_tax = sequence["launch_tax"]["avg"]
            break
    
    # Compare
    matches = compare_results(pytorch_results, baremetal_results)
    
    # Print summary
    print_summary(matches)

    
    # Write output
    output_data = {
        "summary": {
            "total_kernels": len(matches),
        },
        "matches": matches,
    }
    
    utils.save_json(output_file, output_data)
