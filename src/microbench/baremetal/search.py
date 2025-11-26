#!/usr/bin/env python3
"""
Search for cuBLASLt algorithm indices that produce PyTorch kernel configurations (offline).

Reads jobs from baremetal/output/jobs.json, searches for cuBLASLt algorithm
indices that produce the same kernel configurations as PyTorch, and updates
jobs.json with cublas_index field.

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
    print("Building C++ binary")
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


def get_kernel_from_job(job):
    """Construct a Kernel object from a job dict."""
    return Kernel(
        name=job.get("target_kernel"),
        grid=job.get("target_grid"),
        block=job.get("target_block"),
        shared_memory=job.get("target_shared_mem"),
        registers_per_thread=job.get("target_registers_per_thread")
    )

def get_max_algo_idx(job):
    """
    Query how many algorithms are available for this problem.
    Uses run_gemm with warmup=0 runs=0 to query without executing.
    
    Args:
        job: Job dictionary with GEMM parameters
    
    Returns: total_count
    Raises: RuntimeError if query fails
    """
    base_args = build_base_args(job) + ["--warmup", "0", "--runs", "0", "--algo_index", "0"]
    
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
                total = int(match.group(2)) 
                assert max_idx == total - 1
                return max_idx
    
    raise RuntimeError(f"Could not parse algorithm count from output for job {job.get('id')}:\n{result.stdout}")

def extract_kernel_from_trace(sqlite_path):
    """Extract kernel from nsys sqlite trace.
    
    Returns: Kernel object or None if no kernel found
    """

    kernel = None
    try:
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.cursor()
        
        # Query for all kernel events with all available fields
        cursor.execute("""
            SELECT k.start, k.end, k.correlationId, s.value, 
                   k.gridX, k.gridY, k.gridZ,
                   k.blockX, k.blockY, k.blockZ,
                   k.staticSharedMemory, k.dynamicSharedMemory,
                   k.deviceId, k.contextId, k.streamId,
                   k.registersPerThread
            FROM CUPTI_ACTIVITY_KIND_KERNEL as k
            JOIN StringIds as s ON k.demangledName = s.id
        """)
        
        kernel = None
        # Fetch all rows and filter in Python
        for row in cursor.fetchall():
            start_ns, end_ns, corr_id, name, gx, gy, gz, bx, by, bz, static_smem, dyn_smem, device_id, context_id, stream_id, regs = row

            if kernel is not None and "gemm" in name.lower(): 
                raise RuntimeError(f"Multiple kernels found in trace: {kernel.name} and {name}")
            
            if "gemm" in name.lower() or "null" in name.lower(): 
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
            
        conn.close()
    except Exception as e:
        print(f"Error extracting kernel from trace: {e}")
        return None

    utils.remove_file(sqlite_path)
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
    
    
def test_cublas_algo(job, algo_idx):
    """
    Run nsys profiling for a specific algorithm index, extract kernel, and clean up.
    Handles errors internally and returns a simple context message.
    
    Args:
        job: Job dictionary with 'id' field
        algo_idx: Algorithm index to test
    
    Returns:
        (success: bool, kernel: Kernel|None, message: str)
    """
    
    trace_dir = nsys_utils.get_trace_dir()
    trace_path = trace_dir / f"match_{job['id']}_algo{algo_idx}.nsys-rep"
    
    base_args = build_base_args(job)
    args = base_args + ["--warmup", "0", "--runs", "1", "--algo_index", str(algo_idx)]
    success, sqlite_path, message = nsys_utils.nsys_profile_to_sqlite(
        trace_path, args, timeout=100, clean_trace=True
    )
    
    if not success:
        return (False, None, message)
    
    # Extract kernel from trace
    kernel = extract_kernel_from_trace(sqlite_path)
    
    # Clean up sqlite file
    utils.remove_file(sqlite_path)
    
    if kernel is None:
        return (False, None, "Failed to extract kernel from trace")
    
    # Successful profiling and kernel extraction
    return (True, kernel, "success")


def sweep_cublas_algos(job, max_idx=200):
    """Search for cuBLASLt algorithm index that produces target kernel config.

    With PyTorch-matched settings, algorithm index 0 should produce the same kernel.
    Falls back to searching other algorithm indices if index 0 doesn't match.

    Returns: algorithm index or None if no match found.
    """
    # Initialize result variables
    match_algo_idx = None

    # Get target kernel from job
    target_kernel = get_kernel_from_job(job)

    # Force cublas algo search to fail (used for testing)
    if os.getenv("FORCE_NO_MATCH") == "1" or True:
        target_kernel.grid = [999, 999, 999]
    target_kernel.print()

    # Sweep through algorithm indices
    for algo_idx in range(max_idx +1):
        # Test cuBLAS algo index and extract generated kernel
        success, actual_kernel, message = test_cublas_algo(job, algo_idx)

        if not success:
            print(message)
            continue

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
    jobs = jobs_data.get("jobs", [])
    print(f"Loaded {len(jobs)} jobs from {jobs_file}")
    
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
        if job.get("null_kernel"):
            job["cublas_index"] = None
            continue
        
        # Get algorithm count
        max_algo_idx = get_max_algo_idx(job)
        algo_info = f"(0-{max_algo_idx} algorithms)"
        print_utils.iter_start(f"Job {job['id']} {algo_info}")
        
        # Search for algorithm index
        algo_idx = sweep_cublas_algos(job, max_idx=max_algo_idx)

        # Update cublas_index; algo_idx is None if no match found
        job["cublas_index"] = algo_idx

        # Update counters
        if algo_idx is None:
            num_algos_not_found += 1
        else:
            num_algos_found += 1
        num_jobs_searched += 1

        print_utils.iter_end()
    
    # Update jobs.json with cublas_index and matching summary
    jobs_data["jobs"] = jobs
    if "matching_summary" not in jobs_data:
        jobs_data["matching_summary"] = {}
    jobs_data["matching_summary"]["algos_found"] = num_algos_found
    jobs_data["matching_summary"]["algos_not_found"] = num_algos_not_found
    jobs_data["matching_summary"]["total_jobs"] = num_jobs_searched

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
