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
    
    Returns: (max_index, total_count) or (None, None) if query fails
    """
    base_args = build_base_args(job) + ["--query_algo_count"]
    
    try:
        result = subprocess.run(base_args, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return None, None
        
        # Parse "Available algorithms: 0-X (total: Y)" format
        for line in result.stdout.split('\n'):
            if "Available algorithms:" in line:
                match = re.search(r'Available algorithms: 0-(\d+)(?: \(total: (\d+)\))?', line)
                if match:
                    max_idx = int(match.group(1))
                    total = int(match.group(2)) if match.group(2) else max_idx + 1
                    return max_idx, total
    except Exception:
        pass
    
    return None, None


def extract_kernel_from_trace(sqlite_path):
    """Extract kernel from nsys sqlite trace.
    
    Returns: Kernel object or None if no kernel found
    """
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
            return Kernel(
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
    except Exception:
        pass
    
    return None

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
    
    
def _run_nsys_for_algorithm(job, algo_idx, base_args):
    """
    Run nsys profiling for a specific algorithm index and export to sqlite.
    Cleans up trace file after export (only sqlite is needed).
    
    Args:
        job: Job dictionary with 'id' field
        algo_idx: Algorithm index to test
        base_args: Base command line arguments (from build_base_args)
    
    Returns:
        (success: bool, sqlite_path: str|None, message: str)
        
        Two cases:
        1. Success: (True, sqlite_path, "success")
        2. Invalid index: (False, None, "Invalid algorithm index X (available: 0-Y)")
    """
    job_id = job["id"]
    
    trace_dir = nsys_utils.get_trace_dir()
    trace_path = trace_dir / f"match_test_{job_id}_algo{algo_idx}.nsys-rep"
    
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
    
    return (True, sqlite_path, "success")

def _cleanup_test_trace(sqlite_path):
    """Clean up temporary sqlite trace file."""
    try:
        if sqlite_path:
            sqlite_path_obj = Path(sqlite_path)
            if sqlite_path_obj.exists():
                sqlite_path_obj.unlink()
    except:
        pass


def search_algorithm_index(job, max_algorithms=200):
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
    
    # Determine max algorithms to try
    max_algo_idx, total_algo_count = get_algo_count(job)
    if max_algo_idx is not None:
        max_algorithms = min(max_algorithms, max_algo_idx + 1)
    
    # Setup base arguments
    base_args = build_base_args(job)
    
    # Search through algorithm indices
    for algo_idx in range(max_algorithms):
        # Run nsys profiling for this algorithm
        success, sqlite_path, message = _run_nsys_for_algorithm(job, algo_idx, base_args)
        
        if not success:
            # Check if we've exhausted available algorithms
            if "Invalid algorithm index" in message:
                print(f"Algorithm index {algo_idx}: {message}")
                print(f"No matching algorithm found after exhausting available indices")
                print_utils.iter_end()
                return None
            print(f"Algorithm index {algo_idx}: {message}")
            continue
        
        # Extract actual kernel from trace
        actual_kernel = extract_kernel_from_trace(sqlite_path)
        
        # Clean up sqlite file immediately after extraction (we only need the kernel object)
        _cleanup_test_trace(sqlite_path)
        
        if not actual_kernel:
            print(f"Algorithm index {algo_idx}: No kernel found in trace")
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


def search_algorithm_indices():
    """
    Search for algorithm indices for all jobs and update jobs.json.
    """
    jobs_file = utils.get_path("BAREMETAL_JOBS")
    
    # Check if jobs file exists
    utils.ensure_file(jobs_file)
    
    # Load jobs
    with open(jobs_file, 'r') as f:
        jobs_data = json.load(f)
    
    jobs = jobs_data.get("jobs", [])
    print(f"Loaded {len(jobs)} jobs from {jobs_file}")
    
    # Build binary
    build_binary()
    binary_path = utils.get_path("BAREMETAL_BINARY")
    if not binary_path.exists():
        raise RuntimeError(f"Binary not found: {binary_path}")
    
    # Search for algorithm indices for each job
    algorithms_found = 0
    searchable_jobs = []
    
    for job in jobs:
        # Skip if no target kernel
        if not job.get("target_kernel") or job["target_kernel"] == "unknown":
            continue

        # Skip null kernel jobs (no GEMM arguments to search)
        if job.get("null_kernel"):
            job["matched_algo_index"] = None
            continue
        
        searchable_jobs.append(job)
    
    for job in searchable_jobs:
        job_id = job["id"]
        
        # Query available algorithm count for iter_start message
        max_algo_idx, total_algo_count = get_algo_count(job)
        if max_algo_idx is not None:
            num_algos = max_algo_idx + 1
            if num_algos == 0:
                algo_info = f" (Brute forcing 0-200)"
            else:
                algo_info = f" ({num_algos} algorithm{'s' if num_algos != 1 else ''})"
        else:
            algo_info = f" (Brute forcing 0-200)"
        
        print_utils.iter_start(f"Job {job_id}{algo_info}")
        
        # Search for algorithm index
        algo_idx = search_algorithm_index(job, max_algorithms=200)
        
        if algo_idx is not None:
            job["matched_algo_index"] = algo_idx
            algorithms_found += 1
        else:
            job["matched_algo_index"] = None  # Explicitly set to None when no match found
    
    # Update jobs.json with matched_algo_index 
    jobs_data["jobs"] = jobs
    if "matching_summary" not in jobs_data:
        jobs_data["matching_summary"] = {}
    total_searched = len(searchable_jobs)
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
