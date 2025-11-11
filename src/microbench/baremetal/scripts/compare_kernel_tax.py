#!/usr/bin/env python3
"""
Compare PyTorch and baremetal GEMM kernel launch tax.

Joins results from framework/pytorch/output/unique_gemm_kernel_chains.json and
baremetal/output/baremetal_gemm_runs.json, verifies kernel matching,
computes per-kernel launch tax deltas and percentages, and emits
baremetal/output/bm_vs_framework_report.json.
"""

import json
import os
import sys

# Module-level variable for microbench directory (set in __main__)
microbench_dir = None


def normalize_kernel_name(name):
    """Extract short kernel name for easier comparison."""
    # Remove 'void ' prefix and template parameters for cleaner comparison
    name = name.replace("void ", "")
    # For very long names, just use the first part
    if "<" in name:
        base = name.split("<")[0]
        return base
    return name


def load_pytorch_results(pytorch_file):
    """
    Load PyTorch kernel chains and extract per-kernel statistics.
    
    Returns: dict mapping job_id -> {kernel, op_signature, stats}
    """
    with open(pytorch_file, 'r') as f:
        data = json.load(f)
    
    chains = data.get("causal_chains", [])
    results = {}
    
    for idx, chain in enumerate(chains):
        job_id = f"{idx+1:04d}"
        
        kernel = chain.get("kernel", {})
        cpu_op = chain.get("cpu_op", {})
        meta = chain.get("meta", {})
        
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
            "avg_kernel_tax_us": meta.get("avg_kernel_tax_us", 0.0),
            "min_kernel_tax_us": meta.get("min_kernel_tax_us", 0.0),
            "max_kernel_tax_us": meta.get("max_kernel_tax_us", 0.0),
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
                  normalize_kernel_name(pytorch_name) == normalize_kernel_name(baremetal_name))
    
    # Config match (grid, block, shared_memory) - use normalized comparison
    pytorch_config = extract_config(pytorch_kernel)
    baremetal_config = extract_config(baremetal_kernel)
    
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
        
        # Compute deltas
        fw_avg = pytorch["stats"]["avg_kernel_tax_us"]
        bm_avg = baremetal["stats"]["avg_kernel_tax_us"]
        
        delta_us = fw_avg - bm_avg
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
                "avg_kernel_tax_us": fw_avg,
                "min_kernel_tax_us": pytorch["stats"]["min_kernel_tax_us"],
                "max_kernel_tax_us": pytorch["stats"]["max_kernel_tax_us"],
                "count": pytorch["stats"]["count"],
            },
            "baremetal": {
                "kernel_name": baremetal["kernel"]["name"],
                "avg_kernel_tax_us": bm_avg,
                "min_kernel_tax_us": baremetal["stats"]["min_kernel_tax_us"],
                "max_kernel_tax_us": baremetal["stats"]["max_kernel_tax_us"],
                "count": baremetal["stats"]["count"],
            },
            "delta_us": delta_us,
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
        fw_avg = match["framework"]["avg_kernel_tax_us"]
        bm_avg = match["baremetal"]["avg_kernel_tax_us"]
        delta = match["delta_us"]
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
    print(f"  Exact matches (name + config): {exact_matches}/{len(matches)}")
    print(f"  Name matches:                  {name_matches}/{len(matches)}")
    print(f"  Config matches:                {config_matches}/{len(matches)}")


def compare(pytorch_file, baremetal_file, output_file):
    """
    Main comparison function.
    """
    rel_pytorch = os.path.relpath(pytorch_file, microbench_dir) if microbench_dir else pytorch_file
    print(f"Loading PyTorch results from {rel_pytorch}...")
    pytorch_results = load_pytorch_results(pytorch_file)
    print(f"  Loaded {len(pytorch_results)} PyTorch kernel chains")
    
    rel_baremetal = os.path.relpath(baremetal_file, microbench_dir) if microbench_dir else baremetal_file
    print(f"Loading baremetal results from {rel_baremetal}...")
    baremetal_results = load_baremetal_results(baremetal_file)
    print(f"  Loaded {len(baremetal_results)} baremetal runs")
    
    # Compare
    print("Comparing results...")
    matches = compare_results(pytorch_results, baremetal_results)
    
    # Print summary
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
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    rel_output = os.path.relpath(output_file, microbench_dir) if microbench_dir else output_file
    print(f"\nComparison results written to {rel_output}")


if __name__ == "__main__":
    # Check if env.sh has been sourced
    if not os.environ.get("SODA_ENV_LOADED"):
        print("Error: SODA environment not loaded.", file=sys.stderr)
        print("Please run: source env.sh", file=sys.stderr)
        sys.exit(1)
    
    # Get paths from environment
    pytorch_file = os.environ["PYTORCH_UNIQUE_KERNELS"]
    baremetal_file = os.environ["BAREMETAL_RUNS"]
    output_file = os.environ["BAREMETAL_REPORT"]
    microbench_dir = os.environ.get("MICROBENCH_DIR")
    
    if not os.path.exists(pytorch_file):
        print(f"Error: PyTorch results not found: {pytorch_file}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(baremetal_file):
        print(f"Error: Baremetal results not found: {baremetal_file}", file=sys.stderr)
        print("Run run_bm_suite.py first", file=sys.stderr)
        sys.exit(1)
    
    compare(pytorch_file, baremetal_file, output_file)

