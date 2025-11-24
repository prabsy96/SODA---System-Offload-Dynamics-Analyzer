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
from pathlib import Path
from soda import utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from common import print_utils


def build_binary():
    """Build the C++ binary using cmake."""
    baremetal_dir = utils.get_path("BAREMETAL_MICROBENCH_DIR")
    print("Building C++ binary...")
    build_dir = os.path.join(baremetal_dir, "build")
    
    # Configure
    result = subprocess.run(
        ["cmake", "-B", build_dir, "-DCMAKE_BUILD_TYPE=Release"],
        cwd=baremetal_dir,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"CMake configure failed:\n{result.stderr}")
    
    # Build
    result = subprocess.run(
        ["cmake", "--build", build_dir],
        cwd=baremetal_dir,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Build failed:\n{result.stderr}")
    
    print("Build successful")


def get_available_algorithm_count(job, binary_path):
    """
    Query how many algorithms are available for this problem using --query_algo_count flag.
    
    Returns: (max_index, total_count) or (None, None) if query fails
    """
    base_args = [
        binary_path,
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
        "--query_algo_count",  # Query count without running benchmark
    ]
    
    try:
        # Run without nsys (faster) - just to get the algorithm count
        result = subprocess.run(base_args, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            # Parse "Available algorithms: 0-X (total: Y)" or "Available algorithms: 0-X" from stdout
            for line in result.stdout.split('\n'):
                if "Available algorithms:" in line:
                    # Try to match with (total: Y) first
                    match = re.search(r'Available algorithms: 0-(\d+) \(total: (\d+)\)', line)
                    if match:
                        max_idx = int(match.group(1))
                        total = int(match.group(2))
                        return max_idx, total
                    # Fallback: match without (total: Y)
                    match = re.search(r'Available algorithms: 0-(\d+)', line)
                    if match:
                        max_idx = int(match.group(1))
                        total = max_idx + 1
                        return max_idx, total
        else:
            # Query failed - binary might not support --query_algo_count or other error
            pass
    except Exception as e:
        # Query failed due to exception
        pass
    
    return None, None


def get_kernel_config_from_trace(sqlite_path):
    """Extract kernel config from nsys sqlite trace."""
    import sqlite3
    try:
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.cursor()
        
        # Query for kernel events (GEMM kernels only)
        cursor.execute("""
            SELECT k.start, k.end, k.correlationId, s.value, 
                   k.gridX, k.gridY, k.gridZ,
                   k.blockX, k.blockY, k.blockZ,
                   k.staticSharedMemory, k.dynamicSharedMemory
            FROM CUPTI_ACTIVITY_KIND_KERNEL as k
            JOIN StringIds as s ON k.demangledName = s.id
            WHERE LOWER(s.value) LIKE '%gemm%'
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            start_ns, end_ns, corr_id, name, gx, gy, gz, bx, by, bz, static_smem, dyn_smem = row
            return {
                "name": name,
                "grid": [gx, gy, gz],
                "block": [bx, by, bz],
                "shared_memory": static_smem + dyn_smem,
            }
    except Exception:
        pass
    
    return None


def _extract_target_config(job):
    """Extract target kernel configuration from job."""
    target_kernel = job["target_kernel"]
    target_grid = job.get("target_grid", [0, 0, 0])
    target_block = job.get("target_block", [256, 1, 1])
    target_shared_mem = job.get("target_shared_mem", 0)
    
    # For testing: set FORCE_NO_MATCH=1 to test no-match scenario
    if os.getenv("FORCE_NO_MATCH") == "1":
        target_grid = [999, 999, 999]  # Impossible grid to force no match
    
    return target_kernel, target_grid, target_block, target_shared_mem


def _print_target_config(target_kernel, target_grid, target_block, target_shared_mem):
    """Print target kernel configuration as a table."""
    data = [
        ["name", target_kernel],
        ["grid", str(target_grid)],
        ["block", str(target_block)],
        ["shared_mem", str(target_shared_mem)],
    ]
    print_utils.comp_table("Target kernel", ["Field", "Value"], data)
    

def _build_base_args(job, binary_path):
    """Build base command line arguments for the C++ binary."""
    return [
        binary_path,
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
        "--warmup", "0",
        "--runs", "3",
    ]
    
    
def _run_nsys_for_algorithm(job, binary_path, algo_idx, base_args, trace_dir):
    """
    Run nsys profiling for a specific algorithm index.
    
    Returns: (success, test_trace_path, sqlite_path) or (False, None, None) on failure
    """
    test_trace_path = os.path.join(trace_dir, f"match_test_{job['id']}_algo{algo_idx}.nsys-rep")
    test_args = base_args + ["--algo_index", str(algo_idx)]
    
    nsys_cmd = [
        "nsys", "profile",
        "--trace=cuda,osrt",
        "--output", test_trace_path,
        "--force-overwrite=true",
    ] + test_args
    
    try:
        result = subprocess.run(nsys_cmd, capture_output=True, text=True, timeout=100)
        if result.returncode != 0:
            # Check if we've hit the end of available algorithms
            if result.stderr:
                if "Invalid algorithm index" in result.stderr:
                    # Extract max available index from error message (actual runtime count)
                    match = re.search(r'available: 0-(\d+)', result.stderr)
                    if match:
                        max_available = int(match.group(1))
                        print(f"Index {algo_idx} is invalid. Only {max_available + 1} algos found at runtime. Fallback: index 0")
                        return (False, None, None, "exhausted")  # Signal to stop searching
                # Show error message for other failures
                error_msg = result.stderr.strip().split('\n')[-1]
                if error_msg:
                    print(f"Algorithm index {algo_idx}: {error_msg}")
                else:
                    print(f"Algorithm index {algo_idx}: nsys failed (code {result.returncode})")
            else:
                print(f"Algorithm index {algo_idx}: nsys failed (code {result.returncode})")
            return (False, None, None, None)
    except subprocess.TimeoutExpired:
        print(f"Algorithm index {algo_idx}: nsys timeout")
        return (False, None, None, None)
    except Exception as e:
        print(f"Algorithm index {algo_idx}: Exception: {e}")
        return (False, None, None, None)
    
    if not os.path.exists(test_trace_path):
        print(f"Algorithm index {algo_idx}: Trace file not created")
        return (False, None, None, None)
    
    # Export to sqlite
    sqlite_path = test_trace_path.replace(".nsys-rep", ".sqlite")
    if os.path.exists(sqlite_path):
        os.remove(sqlite_path)
    export_cmd = ["nsys", "export", "--type=sqlite", "--output", sqlite_path, test_trace_path]
    export_result = subprocess.run(export_cmd, capture_output=True, text=True)
    if export_result.returncode != 0 or not os.path.exists(sqlite_path):
        print(f"Algorithm index {algo_idx}: Trace export failed")
        return (False, None, None, None)
    
    return (True, test_trace_path, sqlite_path, None)








def _cleanup_test_trace(test_trace_path, sqlite_path):
    """Clean up temporary trace files."""
    try:
        if os.path.exists(test_trace_path):
            os.remove(test_trace_path)
        if os.path.exists(sqlite_path):
            os.remove(sqlite_path)
    except:
        pass


def search_algorithm_index(job, binary_path, output_dir, max_algorithms=200):
    """
    Search for cuBLASLt algorithm index that produces target kernel config.
    
    With PyTorch-matched settings, algorithm index 0 should produce the same kernel.
    Falls back to searching other algorithm indices if index 0 doesn't match.
    
    Returns: algorithm index or None if no match found
    """
    if not job.get("target_kernel") or job["target_kernel"] == "unknown":
        return None
    
    target_kernel, target_grid, target_block, target_shared_mem = _extract_target_config(job)
    _print_target_config(target_kernel, target_grid, target_block, target_shared_mem)
    
    # Build target kernel dict
    target_kernel_dict = {
        "name": target_kernel or "",
        "grid": target_grid,
        "block": target_block,
        "shared_memory": target_shared_mem,
    }
    
    # Query available algorithm count upfront
    max_available_idx, total_available = get_available_algorithm_count(job, binary_path)
    if max_available_idx is not None:
        max_algorithms = min(max_algorithms, max_available_idx + 1)
    
    base_args = _build_base_args(job, binary_path)
    trace_dir = os.path.join(output_dir, "traces")
    os.makedirs(trace_dir, exist_ok=True)
    
    algorithms_tried = 0
    
    # Try algorithm 0 first (PyTorch's default - should match)
    for algo_idx in range(max_algorithms):
        algorithms_tried += 1
        
        success, test_trace_path, sqlite_path, status = _run_nsys_for_algorithm(
            job, binary_path, algo_idx, base_args, trace_dir
        )
        
        if status == "exhausted":
            # All algorithms exhausted
            break
        
        if not success:
            continue
        
        # Parse trace to get kernel config
        kernel_config = get_kernel_config_from_trace(sqlite_path)
        
        if not kernel_config:
            print(f"Algorithm index {algo_idx}: No kernel config found in trace")
            _cleanup_test_trace(test_trace_path, sqlite_path)
            continue
        
        # Compare with target using utils.compare_kernels (prints table automatically)
        match_result = utils.compare_kernels(kernel_config, target_kernel_dict, show_table=True, title=f"Algorithm #{algo_idx}")
        
        if match_result["match"]:
            print(f"Search completed @ algo index {algo_idx}")
            _cleanup_test_trace(test_trace_path, sqlite_path)
            return algo_idx
        
        # Clean up after checking (no match)
        _cleanup_test_trace(test_trace_path, sqlite_path)
    
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
    
    baremetal_dir = utils.get_path("BAREMETAL_MICROBENCH_DIR")
    binary_path = baremetal_dir / "build" / "main_gemm_bm"
    if not os.path.exists(binary_path):
        raise RuntimeError(f"Binary not found: {binary_path}")
    
    # Create temporary output dir for test traces
    output_dir = os.path.dirname(jobs_file)
    trace_dir = os.path.join(output_dir, "traces")
    os.makedirs(trace_dir, exist_ok=True)
    
    # Only count jobs that actually require a search (exclude null/unknown targets)
    searchable_jobs = [
        job for job in jobs
        if job.get("target_kernel") and job["target_kernel"] != "unknown" and not job.get("null_kernel")
    ]
    total_searchable = len(searchable_jobs)
    
    # Search for algorithm indices for each job
    algorithms_found = 0
    no_algorithm = []
    
    for job in jobs:
        job_id = job["id"]
        
        # Skip if no target kernel
        if not job.get("target_kernel") or job["target_kernel"] == "unknown":
            continue

        # Skip null kernel jobs (no GEMM arguments to search)
        if job.get("null_kernel"):
            job["matched_algo_index"] = None
            continue
        
        # Query available algorithm count for iter_start message
        max_available_idx, total_available = get_available_algorithm_count(job, binary_path)
        if max_available_idx is not None:
            num_algos = max_available_idx + 1
            if num_algos == 0:
                algo_info = f" (Brute forcing 0-200)"
            else:
                algo_info = f" ({num_algos} algorithm{'s' if num_algos != 1 else ''})"
        else:
            algo_info = f" (Brute forcing 0-200)"
        
        print_utils.iter_start(f"Job {job_id}{algo_info}")
        
        # Search for algorithm index
        matched_algo_idx = search_algorithm_index(job, binary_path, output_dir, max_algorithms=200)
        
        if matched_algo_idx is not None:
            job["matched_algo_index"] = matched_algo_idx
            algorithms_found += 1
        else:
            job["matched_algo_index"] = None  # Explicitly set to None when no match found
            no_algorithm.append(job_id)
    
    # Update jobs.json with matched_algo_index
    jobs_data["jobs"] = jobs
    if "matching_summary" not in jobs_data:
        jobs_data["matching_summary"] = {}
    jobs_data["matching_summary"]["matches_found"] = algorithms_found
    jobs_data["matching_summary"]["no_matches"] = no_algorithm
    jobs_data["matching_summary"]["total_jobs"] = total_searchable
    
    utils.save_json(jobs_file, jobs_data)
    
    print(f"\n==============================================")
    print(f"Summary")

    
    denominator = total_searchable if total_searchable > 0 else len(jobs)
    print(f"- Algorithms found: {algorithms_found}/{denominator}")
    if no_algorithm:
        # Format job IDs: remove one leading zero (0002 -> 002) and limit to 5
        no_algo_ids = []
        for job_id in no_algorithm[:5]:
            # Remove one leading zero if present (0002 -> 002, 0001 -> 001)
            if job_id.startswith('0') and len(job_id) > 1:
                no_algo_ids.append(job_id[1:])
            else:
                no_algo_ids.append(job_id)
        no_algo_str = ', '.join(no_algo_ids)
        if len(no_algorithm) > 5:
            no_algo_str += f" ({len(no_algorithm) - 5} more)"
        print(f"- No algorithm found: {no_algo_str}")
    print(f"- Updated {jobs_file}")
    print(f"==============================================")

