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
    Uses same normalization as PyTorch verify_replayed_kernels.py.
    
    Returns: (name_match, config_match)
    """
    # Name match
    pytorch_name = pytorch_kernel["name"]
    baremetal_name = baremetal_kernel["name"]
    
    # Exact match or normalized match
    name_match = (pytorch_name == baremetal_name or 
                  utils.clean_kernel_name(pytorch_name) == utils.clean_kernel_name(baremetal_name))
    
    # Config match (grid, block, shared_memory) - use normalized comparison
    pytorch_kernel_obj = Kernel.from_dict(pytorch_kernel) if pytorch_kernel else None
    baremetal_kernel_obj = Kernel.from_dict(baremetal_kernel) if baremetal_kernel else None
    pytorch_config = pytorch_kernel_obj.get_signature() if pytorch_kernel_obj else {}
    baremetal_config = baremetal_kernel_obj.get_signature() if baremetal_kernel_obj else {}
    
    # Require exact match 
    config_match = (
        pytorch_config["grid"] == baremetal_config["grid"] and
        pytorch_config["block"] == baremetal_config["block"] and
        pytorch_config["shared_memory"] == baremetal_config["shared_memory"]
    )
    
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


def print_summary(matches):
    """Print comparison summary to stdout."""
    print("\n" + "="*80)
    print("PyTorch vs Baremetal GEMM Launch Tax Comparison")
    print("="*80)
    
    # Per-kernel summary
    print(f"\nPer-Kernel Results ({len(matches)} kernels):")
    print(f"{'ID':<6} {'Kernel':<35} {'FW(μs)':<9} {'BM(μs)':<9} {'Δ(%)':<8} {'Match':<6}")
    print("-"*80)
    
    for match in matches:
        kernel_short = match["kernel"]["name"][:38]
        fw_avg = match["framework"]["avg_kernel_tax"]
        bm_avg = match["baremetal"]["avg_kernel_tax"]
        delta = match["delta"]
        delta_pct = match["delta_pct"]
        
        name_match = "✓" if match["match_status"]["name_match"] else "✗"
        config_match = "✓" if match["match_status"]["config_match"] else "✗"
        match_str = f"{name_match}{config_match}"
        
        print(f"{match['id']:<6} {kernel_short:<35} {fw_avg:<9.2f} {bm_avg:<9.2f} {delta_pct:<8.1f} {match_str:<6}")
    print("="*80)
    
    # Match statistics
    name_matches = sum(1 for m in matches if m["match_status"]["name_match"])
    config_matches = sum(1 for m in matches if m["match_status"]["config_match"])
    exact_matches = sum(1 for m in matches if m["match_status"]["name_match"] and m["match_status"]["config_match"])
    
    print(f"\nMatch Statistics:")
    print(f"\tExact matches (name + config): {exact_matches}/{len(matches)}")
    print(f"\tName matches:                  {name_matches}/{len(matches)}")
    print(f"\tConfig matches:                {config_matches}/{len(matches)}")


def compare():
    """
    Main comparison function.
    """
    pytorch_file = utils.get_path("PYTORCH_GEMM_SEQUENCES")
    baremetal_file = utils.get_path("BAREMETAL_GEMM_KERNELS")
    output_file = utils.get_path("FINAL_REPORT")
    
    utils.ensure_file(pytorch_file)
    utils.ensure_file(baremetal_file)
    
    print(f"Loading PyTorch results from {pytorch_file}...")
    pytorch_results = load_pytorch_results(pytorch_file)
    print(f"Loaded {len(pytorch_results)} PyTorch event sequences")
    
    print(f"Loading baremetal results from {baremetal_file}...")
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
    print("Comparing results...")
    matches = compare_results(pytorch_results, baremetal_results)
    
    # Print summary
    if null_kernel_tax is not None:
        print(f"\nBaseline launch tax (null kernel): {null_kernel_tax:.2f} μs")
    
    print_summary(matches)
    
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
    
    print(f"\nComparison results written to {output_file}")

