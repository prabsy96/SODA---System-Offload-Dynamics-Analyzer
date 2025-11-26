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
from pathlib import Path

from soda import utils
from common import print_utils
# Add src to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from data import CPUOp, Kernel

def load_pytorch_results(pytorch_file):
    """
    Load PyTorch event sequences and extract per-kernel statistics.
    
    Returns: dict mapping job_id -> {kernel, op_signature, stats}
    """
    with open(pytorch_file, 'r') as f:
        data = json.load(f)
    
    sequences = data.get("sequences", [])
    results = {}
    
    for idx, sequence in enumerate(sequences):
        job_id = f"{idx+1:04d}"
        
        kernel_dict = sequence.get("kernel", {})
        cpu_op_dict = sequence.get("cpu_op", {})
        meta = sequence.get("meta", {})
        
        # Convert dicts to objects
        cpu_op = CPUOp.from_dict(cpu_op_dict) if cpu_op_dict else None
        kernel = Kernel.from_dict(kernel_dict) if kernel_dict else None
        
        op_signature = cpu_op.get_signature() if cpu_op else {}  
        kernel_info = kernel.get_signature() if kernel else {}
        
        # Extract stats
        stats = {
            "avg_kernel_tax": meta.get("avg_kernel_tax", 0.0),
            "min_kernel_tax": meta.get("min_kernel_tax", 0.0),
            "max_kernel_tax": meta.get("max_kernel_tax", 0.0),
            "count": meta.get("count", 0),
        }
        
        results[job_id] = {
            "kernel": kernel_info,
            "op_signature": op_signature,
            "stats": stats,
        }
    
    return results


def load_baremetal_results(baremetal_file):
    """
    Load baremetal results.
    
    Returns: dict mapping job_id -> {kernel, stats}
    """
    with open(baremetal_file, 'r') as f:
        data = json.load(f)
    
    kernels = data.get("kernels", [])
    results = {}
    
    for kernel_entry in kernels:
        job_id = kernel_entry["id"]
        # Skip null kernel job (0000) from comparison results
        if kernel_entry.get("target_kernel") == "__null__":
            continue
        results[job_id] = {
            "kernel": kernel_entry["kernel"],
            "stats": kernel_entry["stats"],
            "target_kernel": kernel_entry.get("target_kernel", ""),
        }
    
    return results


def verify_kernel_match(pytorch_kernel, baremetal_kernel):
    """
    Verify that kernels match by name and optionally by config.
    Uses Kernel.compare() for normalized comparison.
    
    Returns: (name_match, config_match)
    """
    pytorch_kernel_obj = Kernel.from_dict(pytorch_kernel) if pytorch_kernel else None
    baremetal_kernel_obj = Kernel.from_dict(baremetal_kernel) if baremetal_kernel else None
    if not pytorch_kernel_obj or not baremetal_kernel_obj:
        return False, False

    compare_result = pytorch_kernel_obj.compare(
        baremetal_kernel_obj, show_table=False, full=False
    )

    name_match = compare_result.get("name", False)
    grid_match = compare_result.get("grid", [])
    block_match = compare_result.get("block", [])
    shared_match = compare_result.get("shared_memory", False)
    config_match = bool(grid_match and all(grid_match) and block_match and all(block_match) and shared_match)

    return name_match, config_match


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
        
        # Verify kernel match
        name_match, config_match = verify_kernel_match(pytorch["kernel"], baremetal["kernel"])
        
        # Compute deltas (all values in microseconds)
        fw_avg = pytorch["stats"]["avg_kernel_tax"]
        bm_avg = baremetal["stats"]["avg_kernel_tax"]
        
        delta = fw_avg - bm_avg
        # Calculate percentage difference: (FW - BM) / FW * 100
        # Shows how much faster BM is compared to FW (base)
        delta_pct = ((fw_avg - bm_avg) / fw_avg * 100) if fw_avg > 0 else 0.0
        
        # Build match entry
        match_entry = {
            "id": job_id,
            "op_signature": pytorch["op_signature"],
            "kernel": {
                "name": pytorch["kernel"]["name"],
                "grid": pytorch["kernel"]["grid"],
                "block": pytorch["kernel"]["block"],
                "shared_memory": pytorch["kernel"]["shared_memory"],
            },
            "match_status": {
                "name_match": name_match,
                "config_match": config_match,
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
        name_match = "✓" if match["match_status"]["name_match"] else "✗"
        config_match = "✓" if match["match_status"]["config_match"] else "✗"
        kernel_name = match["kernel"].get("name") or "__null_kernel__"
        per_kernel_rows.append([
            match["id"],
            kernel_name,
            f"{match['framework']['avg_kernel_tax']:.2f}",
            f"{match['baremetal']['avg_kernel_tax']:.2f}",
            f"{match['delta_pct']:.1f}",
            f"{name_match}{config_match}",
        ])

    if per_kernel_rows:
        title_suffix = f" | Baseline (null kernel): {baseline_tax:.2f} μs" if baseline_tax is not None else ""
        print_utils.comp_table(
            title=f"Per-Kernel Results ({len(per_kernel_rows)} kernels){title_suffix}",
            headers=["ID", "Kernel", "FW(μs)", "BM(μs)", "Δ(%)", "Match"],
            data=per_kernel_rows,
        )

    name_matches = sum(1 for m in matches if m["match_status"]["name_match"])
    config_matches = sum(1 for m in matches if m["match_status"]["config_match"])
    exact_matches = sum(
        1 for m in matches if m["match_status"]["name_match"] and m["match_status"]["config_match"]
    )
    stats_rows = [
        ["Exact matches (name+config)", f"{exact_matches}/{len(matches)}"],
        ["Name matches", f"{name_matches}/{len(matches)}"],
        ["Config matches", f"{config_matches}/{len(matches)}"],
    ]
    print_utils.comp_table(
        title="Match Statistics",
        headers=["Metric", "Count"],
        data=stats_rows,
    )


def compare():
    """
    Main comparison function.
    """
    pytorch_file = utils.get_path("PYTORCH_GEMM_SEQUENCES")
    baremetal_file = utils.get_path("BAREMETAL_GEMM_KERNELS")
    output_file = utils.get_path("FINAL_REPORT")
    
    utils.ensure_file(pytorch_file)
    utils.ensure_file(baremetal_file)
    
    print(f"Loading PyTorch results from {pytorch_file}")
    pytorch_results = load_pytorch_results(pytorch_file)
    print(f"Loaded {len(pytorch_results)} PyTorch event sequences")
    
    print(f"Loading baremetal results from {baremetal_file}")
    baremetal_results = load_baremetal_results(baremetal_file)
    print(f"Loaded {len(baremetal_results)} baremetal kernels")
    
    # Extract null kernel tax (baseline launch tax)
    null_kernel_tax = None
    with open(baremetal_file, 'r') as f:
        data = json.load(f)
    for kernel_entry in data.get("kernels", []):
        if kernel_entry.get("target_kernel") == "__null__":
            null_kernel_tax = kernel_entry["stats"]["avg_kernel_tax"]
            break
    
    # Compare
    matches = compare_results(pytorch_results, baremetal_results)
    
    # Print summary
    print_summary(matches, baseline_tax=null_kernel_tax)
    
    # Write output
    output_data = {
        "summary": {
            "total_kernels": len(matches),
            "exact_matches": sum(1 for m in matches if m["match_status"]["name_match"] and m["match_status"]["config_match"]),
            "name_matches": sum(1 for m in matches if m["match_status"]["name_match"]),
        },
        "matches": matches,
    }
    
    utils.save_json(output_file, output_data)