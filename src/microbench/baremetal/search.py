#!/usr/bin/env python3
"""
Search for cuBLASLt algorithm indices that produce PyTorch kernel configurations (offline).

Reads jobs from baremetal/output/jobs.json, searches for cuBLASLt algorithm
indices that produce the same kernel configurations as PyTorch, and updates
jobs.json with matched_algo_index field.

This is an offline process that can be run separately from profiling.
"""

import json
import os
import sys
import subprocess
import re
import sqlite3
from pathlib import Path
from soda import utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from common import print_utils
from data import Kernel
from microbench.baremetal import nsys_utils


def build_binary():
    """Build the C++ binary using cmake."""
    baremetal_dir = utils.get_path("BAREMETAL_MICROBENCH_DIR")
    print("Building C++ binary...")
    build_dir = baremetal_dir / "build"
    
    # Configure
    result = subprocess.run(
        ["cmake", "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"],
        cwd=str(baremetal_dir),
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"CMake configure failed:\n{result.stderr}")
    
    # Build
    result = subprocess.run(
        ["cmake", "--build", str(build_dir)],
        cwd=str(baremetal_dir),
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Build failed:\n{result.stderr}")
    
    print("Build successful")


def get_algo_count(job):
    """
    Query how many algorithms are available for this problem using --query_algo_count flag.
    
    Args:
        job: Job dictionary with GEMM parameters
    
    Returns: total_count
    Raises: RuntimeError if query fails
    """
    base_args = build_base_args(job) + ["--query_algo_count"]
    
    try:
        result = subprocess.run(base_args, capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Timeout querying algorithm count for job {job.get('id')}")
    except Exception as e:
        raise RuntimeError(f"Failed to query algorithm count for job {job.get('id')}: {e}")

    if result.returncode != 0:
        raise RuntimeError(f"Query failed for job {job.get('id')} with return code {result.returncode}:\n{result.stderr}")
    
    # Parse "Available algorithms: 0-X (total: Y)" format
    for line in result.stdout.split('\n'):
        if "Available algorithms:" in line:
            match = re.search(r'Available algorithms: 0-(\d+)(?: \(total: (\d+)\))?', line)
            if match:
                max_idx = int(match.group(1))
                total = int(match.group(2)) if match.group(2) else max_idx + 1
                return total
    
    raise RuntimeError(f"Could not parse algorithm count from output for job {job.get('id')}:\n{result.stdout}")


def extract_kernel_from_trace(sqlite_path):
    """Extract kernel from nsys sqlite trace.
    
    Returns: Kernel object or None if no kernel found
    """

    kernel = None
    try:
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.cursor()
        
        # Query for kernel events (GEMM kernels only) with all available fields
        cursor.execute("""
            SELECT k.start, k.end, k.correlationId, s.value, 
                   k.gridX, k.gridY, k.gridZ,
                   k.blockX, k.blockY, k.blockZ,
                   k.staticSharedMemory, k.dynamicSharedMemory,
                   k.deviceId, k.contextId, k.streamId,
                   k.registersPerThread
            FROM CUPTI_ACTIVITY_KIND_KERNEL as k
            JOIN StringIds as s ON k.demangledName = s.id
            WHERE LOWER(s.value) LIKE '%gemm%'
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            start_ns, end_ns, corr_id, name, gx, gy, gz, bx, by, bz, static_smem, dyn_smem, device_id, context_id, stream_id, regs = row
            kernel = Kernel(
                name=name,
                grid=[gx, gy, gz],
                block=[bx, by, bz],
                shared_memory=static_smem + dyn_smem,
                correlation=corr_id,
                ts=start_ns / 1000.0,  # Convert nanoseconds to microseconds
                dur=(end_ns - start_ns) / 1000.0,  # Convert nanoseconds to microseconds
                device=device_id,
                context=context_id,
                stream=stream_id,
                registers_per_thread=regs
            )
    except Exception as e:
        print(f"Error extracting kernel from trace: {e}")
        return None

    _cleanup_test_trace(sqlite_path)
    return kernel

def build_base_args(job):
    """Build base command line arguments for the C++ binary.
    
    Args:
        job: Job dictionary with GEMM parameters
    
    Returns:
        List of command line arguments 
    """
    binary_path = utils.get_path("BAREMETAL_BINARY")
    
    return [
        str(binary_path),
        "--m", str(job["m"]),
        "--n", str(job["n"]),
        "--k", str(job["k"]),
        "--lda", str(job["lda"]),
        "--ldb", str(job["ldb"]),
        "--ldc", str(job["ldc"]),
        "--order_a", job.get("order_a", "row"),
        "--order_b", job.get("order_b", "row"),
        "--trans_a", job.get("trans_a", "N"),
        "--trans_b", job.get("trans_b", "N"),
        "--dtype", job["dtype"],
        "--alpha", str(job["alpha"]),
        "--beta", str(job["beta"]),
    ]
    
    
def _run_nsys_for_algorithm(job, algo_idx):
    """
    Run nsys profiling for a specific algorithm index, extract kernel, and clean up.
    
    Args:
        job: Job dictionary with 'id' field
        algo_idx: Algorithm index to test
    
    Returns:
        (success: bool, kernel: Kernel|None, message: str)
        
        Two cases:
        1. Success: (True, kernel, "success")
        2. Failure: (False, None, error_message)
    """
    
    trace_dir = nsys_utils.get_trace_dir()
    trace_path = trace_dir / f"match_{job['id']}_algo{algo_idx}.nsys-rep"
    
    base_args = build_base_args(job)
    args = base_args + ["--warmup", "0", "--runs", "3", "--algo_index", str(algo_idx)]
    success, sqlite_path, message = nsys_utils.nsys_profile_to_sqlite(
        trace_path, args, timeout=100, clean_trace=True
    )
    
    if not success:
        if message and "Invalid algorithm index" in message:
            match = re.search(r'available: 0-(\d+)', message)
            if match:
                max_available = int(match.group(1))
                return (False, None, f"Invalid algorithm index {algo_idx} (available: 0-{max_available})")
        return (False, None, message or "nsys profiling failed")
    
    # Extract kernel from trace
    kernel = extract_kernel_from_trace(sqlite_path)
    
    # Clean up sqlite file
    utils.remove_file(sqlite_path)
    
    if kernel is None:
        return (False, None, "Failed to extract kernel from trace")
    
    return (True, kernel, "success")


def find_matching_algo_idx(job, max_idx=200):
    """
    Search for cuBLASLt algorithm index that produces target kernel config.
    
    With PyTorch-matched settings, algorithm index 0 should produce the same kernel.
    Falls back to searching other algorithm indices if index 0 doesn't match.
    
    Returns: algorithm index or None if no match found
    """
    algo_idx = None

    # Build target kernel configuration
    target_kernel = Kernel(
        name=job.get("target_kernel"),
        grid=[999, 999, 999] if os.getenv("FORCE_NO_MATCH") == "1" else job.get("target_grid"),
        block=job.get("target_block"),
        shared_memory=job.get("target_shared_mem"),
        registers_per_thread=job.get("target_registers_per_thread")
    )
    target_kernel.print()
    
    
    # Search through algorithm indices
    for algo_idx in range(max_idx):
        # Run nsys profiling for this algorithm and extract kernel
        success, actual_kernel, message = _run_nsys_for_algorithm(job, algo_idx)
        
        if not success:
            # Check if we've exhausted available algorithms
            if "Invalid algorithm index" in message:
                print(f"Algorithm index {algo_idx}: {message}")
                print(f"No matching algorithm found after exhausting available indices")
                print_utils.iter_end()
                return None
            print(f"Algorithm index {algo_idx}: {message}")
            continue
        
        # Compare actual kernel with target kernel
        match_result = actual_kernel.compare(
            target_kernel, 
            show_table=True, 
            title=f"Algorithm #{algo_idx}", 
            full=False
        )
        
        if match_result["match"]:
            print(f"Search completed @ algo index {algo_idx}")
            print_utils.iter_end()
            return algo_idx
    
    # No match found after searching all algorithms
    if algo_idx is not None:
        print(f"No matching algorithm found after trying {algo_idx + 1} algorithm(s)")
    else:
        print(f"No matching algorithm found")
    print_utils.iter_end()
    return None


def search_algorithms_offline():
    """
    Search for algorithm indices for all jobs and update jobs.json.
    """
    jobs_file = utils.get_path("BAREMETAL_JOBS")
    
    # Load jobs
    jobs_data = utils.load_json(jobs_file)
    
    jobs = jobs_data.get("jobs", [])
    print(f"Loaded {len(jobs)} jobs from {jobs_file}")
    
    # Build binary
    build_binary()
    binary_path = utils.get_path("BAREMETAL_BINARY")
    utils.ensure_file(binary_path)
    
    # Search for algorithm indices for each job
    algorithms_found = 0
    total_searched = 0
    
    for job in jobs:

        # Skip null kernel jobs 
        if job.get("null_kernel"):
            job["matched_algo_index"] = None
            continue
        
        total_searched += 1
        
        # Get algorithm count
        total_algos = get_algo_count(job)
        algo_info = f" ({total_algos} algorithm{'s' if total_algos != 1 else ''})"
        print_utils.iter_start(f"Job {job['id']}{algo_info}")
        
        # Search for algorithm index
        algo_idx = find_matching_algo_idx(job, max_idx=total_algos)
        
        if algo_idx is not None:
            job["matched_algo_index"] = algo_idx
            algorithms_found += 1
        else:
            job["matched_algo_index"] = None  # Explicitly set to None when no match found
    
    # Update jobs.json with matched_algo_index 
    jobs_data["jobs"] = jobs
    if "matching_summary" not in jobs_data:
        jobs_data["matching_summary"] = {}
    jobs_data["matching_summary"]["matches_found"] = algorithms_found
    jobs_data["matching_summary"]["no_matches"] = total_searched - algorithms_found
    jobs_data["matching_summary"]["total_jobs"] = total_searched
    
    utils.save_json(jobs_file, jobs_data)
    
    # Print summary table
    print_utils.comp_table(
        title="Summary",
        headers=["Metric", "Count"],
        data=[
            ["Total jobs", f"{total_searched}"],
            ["Algorithms found", f"{algorithms_found}"],
            ["No algorithm found", f"{total_searched - algorithms_found}"],
        ]
    )
