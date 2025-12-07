#!/usr/bin/env python3
"""
Search for cuBLASLt algorithm indices that produce PyTorch kernel configurations (offline).

Reads jobs from baremetal/output/jobs.json, searches for cuBLASLt heuristic
indices that produce the same kernel configurations as PyTorch, and updates
jobs.json with heur_idx field.

This is an offline process that can be run separately from profiling.
"""

import json
import os
import subprocess
import re
import sqlite3
from typing import Optional, Tuple, List

from soda.common import utils, print_utils
from soda.common.data import Kernel
from soda.microbench.baremetal.utils import (
    nsys_profile_to_sql, 
    extract_kernels_from_trace, 
    extract_launches_from_trace,
    build_binary,
    build_base_args
)


def get_heuristic_algo_count(job):
    """
    Query how many heuristic algorithms are available for this problem.
    Uses run_gemm with warmup=0 runs=0 to query without executing.
    
    Args:
        job: Job dictionary with GEMM parameters
    
    Returns: total_count
    Raises: RuntimeError if query fails
    """
    base_args = build_base_args(job) + ["--list_heuristics"]
    
    try:
        result = subprocess.run(base_args, capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Timeout querying algorithm count for job {job['id']}")
    except Exception as e:
        raise RuntimeError(f"Failed to query algorithm count for job {job['id']}: {e}")

    if result.returncode != 0:
        raise RuntimeError(f"Query failed for job {job['id']} with return code {result.returncode}:\n{result.stderr}")
    
    # Parse "Total algorithms = Y"
    total = None
    for line in result.stdout.split('\n'):
        if line.startswith("Total algorithms"):
            match = re.search(r'Total algorithms\s*=\s*(\d+)', line)
            if match:
                total = int(match.group(1))
                break
            else: 
                # Keep looking for the total
                pass
    
    assert total is not None, f"Could not parse algorithm count from output for job {job['id']}:\n{result.stdout}"
    return total
    
def sweep_cublas_algos(job, max_count=200):
    """Search for cuBLASLt algorithm index that produces target kernel config.

    With PyTorch-matched settings, algorithm index 0 should produce the same kernel.
    Falls back to searching other algorithm indices if index 0 doesn't match.

    Returns: algorithm index or None if no match found.
    """
    traces_dir = utils.get_path("BAREMETAL_TRACES")
    utils.ensure_dir(traces_dir)

    # Initialize result variables
    match_algo_idx = None

    # Get target kernel from job
    target_kernel = Kernel.from_dict(job)

    # Force cublas algo search to fail (used for testing)
    if os.getenv("FORCE_NO_MATCH") == "1":
        target_kernel.grid = [999, 999, 999]
    target_kernel.print()

    # Sweep through algorithm indices
    for algo_idx in range(max_count):

        # Build trace path and args
        trace_file_name = f"match_{job['id']}_algo{algo_idx}"
        args = build_base_args(job) + [
            "--warmup", "0", 
            "--runs", "1", 
            "--heuristic_index", str(algo_idx)
        ]

        # Test cuBLAS algo index 
        success, trace_file_sql, message = nsys_profile_to_sql(
            trace_file_name, args, timeout=100, cleanup=True
        )

        if not success:
            print(message)
            continue
        else: 
            actual_kernel = None

            # Extract kernels from trace
            kernels = extract_kernels_from_trace(trace_file_sql, cleanup=True)
            if not kernels:
                # Failed to extract any kernel at all 
                raise RuntimeError("Failed to extract kernel from trace")
            else: 
                if len(kernels) != 1:
                    raise RuntimeError(f"Multiple kernels found in trace: {[k.name for k in kernels]}")
                else: 
                    actual_kernel = kernels[0]

            # Compare actual kernel with target kernel
            match_result = actual_kernel.compare(
                target_kernel,
                show_table=True,
                title=f"Algorithm #{algo_idx}",
                full=False,
            )

            if match_result["match"]:
                match_algo_idx = algo_idx
                break
            else: 
                # Kernel didn't match for this algorithm index
                # Let's try the next one
                pass 

    # Report outcome
    if match_algo_idx is None:
        print(f"No cuBLAS algorithm found.")
    else:
        print(f"Found cuBLAS algorithm @ index {match_algo_idx}")

    return match_algo_idx


def search_cublas_algos_offline():
    """
    Search for algorithm indices for all jobs and update jobs.json.
    """
    jobs_file = utils.get_path("BAREMETAL_JOBS")
    
    # Load jobs
    jobs_data = utils.load_json(jobs_file)
    jobs = jobs_data["jobs"]
    print(f"Loaded {len(jobs)}")
    
    # Build binary
    build_binary()
    binary_path = utils.get_path("BAREMETAL_BINARY")
    utils.ensure_file(binary_path)
    
    # Counters
    num_algos_found = 0
    num_algos_not_found = 0
    num_jobs_searched = 0
    
    for job in jobs:

        # Skip null kernel jobs 
        if job["name"] == "__null_kernel__":
            job["heur_idx"] = None
            continue
        
        # Get algorithm count
        algo_count = get_heuristic_algo_count(job)
        algo_info = f"({algo_count} algorithm(s))"
        print_utils.iter_start(f"Job {job['id']} {algo_info}")
        
        # Search for algorithm index
        algo_idx = sweep_cublas_algos(job, max_count=algo_count)

        # Update heur_idx; algo_idx is None if no match found
        job["heur_idx"] = algo_idx

        # Update counters
        if algo_idx is None:
            num_algos_not_found += 1
        else:
            num_algos_found += 1
        num_jobs_searched += 1

    # Update jobs.json with heur_idx and matching summary
    jobs_data["jobs"] = jobs

    # Update summary
    jobs_data["summary"]["offline_cublas_algo_search"] = {
        "algos_found": num_algos_found,
        "algos_not_found": num_algos_not_found,
        "total_jobs": num_jobs_searched
    }

    assert num_jobs_searched == num_algos_found + num_algos_not_found, "Total jobs != Algos found + Algos not found"
    
    utils.save_json(jobs_file, jobs_data)
    
    # Print summary table
    print_utils.comp_table(
        title="Summary",
        headers=["Metric", "Count"],
        data=[
            ["Jobs searched", f"{num_jobs_searched}"],
            ["Algorithms found", f"{num_algos_found}"],
            ["No algorithm found", f"{num_algos_not_found}"],
        ]
    )
