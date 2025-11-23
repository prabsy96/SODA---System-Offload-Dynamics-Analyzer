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

# Module-level variable for microbench directory (set in __main__)
microbench_dir = None




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
        
        kernel = sequence.get("kernel", {})
        cpu_op = sequence.get("cpu_op", {})
        meta = sequence.get("meta", {})
        
        # Extract op signature
        op_signature = {
            "op": cpu_op.get("name", ""),
            "input_dims": cpu_op.get("input_dims", []),
            "input_strides": cpu_op.get("input_strides", []),
            "input_type": cpu_op.get("input_type", []),
            "concrete_inputs": cpu_op.get("concrete_inputs", []),
        }
        
        # Extract kernel info
        kernel_info = {
            "name": kernel.get("name", ""),
            "grid": kernel.get("grid", []),
            "block": kernel.get("block", []),
            "shared_memory": kernel.get("shared_memory", 0),
        }
        
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
    
    runs = data.get("runs", [])
    results = {}
    
    for run in runs:
        job_id = run["id"]
        # Skip null kernel job (0000) from comparison results
        if run.get("target_kernel") == "__null__":
            continue
        results[job_id] = {
            "kernel": run["kernel"],
            "stats": run["stats"],
            "target_kernel": run.get("target_kernel", ""),
        }
    
    return results


def _to_tuple_int(x):
    """Convert list/tuple to tuple of ints (same as PyTorch verify_replayed_kernels.py)."""
    if isinstance(x, (list, tuple)):
        try:
            return tuple(int(v) for v in x)
        except Exception:
            return tuple()
    return tuple()

def _norm_shared_mem(v):
    """Normalize shared memory (same as PyTorch verify_replayed_kernels.py)."""
    if v in (None, '0'):
        return 0
    try:
        return int(v)
    except Exception:
        return 0

def extract_config(kernel):
    """Extract normalized config (same as PyTorch verify_replayed_kernels.py)."""
    return {
        "grid": _to_tuple_int(kernel.get("grid") or ()),
        "block": _to_tuple_int(kernel.get("block") or ()),
        "shared_memory": _norm_shared_mem(kernel.get("shared_memory")),
    }

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
    pytorch_config = extract_config(pytorch_kernel)
    baremetal_config = extract_config(baremetal_kernel)
    
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


def compare(pytorch_file, baremetal_file, output_file):
    """
    Main comparison function.
    """
    rel_pytorch = os.path.relpath(pytorch_file, microbench_dir) if microbench_dir else pytorch_file
    print(f"Loading PyTorch results from {rel_pytorch}...", end=" ")
    pytorch_results = load_pytorch_results(pytorch_file)
    print(f"Loaded {len(pytorch_results)} PyTorch event sequences")
    
    rel_baremetal = os.path.relpath(baremetal_file, microbench_dir) if microbench_dir else baremetal_file
    print(f"Loading baremetal results from {rel_baremetal}...", end=" ")
    baremetal_results = load_baremetal_results(baremetal_file)
    print(f"Loaded {len(baremetal_results)} baremetal runs")
    
    # Extract null kernel tax (baseline launch tax)
    null_kernel_tax = None
    with open(baremetal_file, 'r') as f:
        data = json.load(f)
    for run in data.get("runs", []):
        if run.get("target_kernel") == "__null__":
            null_kernel_tax = run["stats"]["avg_kernel_tax"]
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
    
    rel_output = os.path.relpath(output_file, microbench_dir) if microbench_dir else output_file
    print(f"\nComparison results written to {rel_output}")


def entry_point(experiment_dir: Path) -> None:
    # Check if env.sh has been sourced
    if not os.environ.get("SODA_ENV_LOADED"):
        print("Error: SODA environment not loaded.", file=sys.stderr)
        print("Please run: source env.sh", file=sys.stderr)
        sys.exit(1)
    
    # Get paths from environment
    pytorch_file = utils.get_path("UNIQUE_GEMM_SEQUENCES", base_path=experiment_dir)
    baremetal_file = utils.get_path("BAREMETAL_RUNS", base_path=experiment_dir)
    output_file = utils.get_path("BAREMETAL_REPORT", base_path=experiment_dir)
    global microbench_dir
    microbench_dir = os.environ.get("MICROBENCH_DIR")
    
    # Check if input files exist
    utils.ensure_file(pytorch_file)
    utils.ensure_file(baremetal_file)
    
    compare(pytorch_file, baremetal_file, output_file)

