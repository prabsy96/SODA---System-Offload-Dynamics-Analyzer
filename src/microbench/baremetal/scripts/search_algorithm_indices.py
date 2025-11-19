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
from soda import SodaProfiler

# Module-level variable for microbench directory (set in __main__)
microbench_dir = None


def build_binary(baremetal_dir):
    """Build the C++ binary using cmake."""
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
        print(f"CMake configure failed:\n{result.stderr}", file=sys.stderr)
        return False
    
    # Build
    result = subprocess.run(
        ["cmake", "--build", build_dir],
        cwd=baremetal_dir,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}", file=sys.stderr)
        return False
    
    print("Build successful")
    return True


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
    except:
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
    """Print target kernel configuration."""
    print(f"\t* Target kernel and config")
    print(f"\t\t** name: {target_kernel}")
    print(f"\t\t** grid: {target_grid}")
    print(f"\t\t** block: {target_block}")
    print(f"\t\t** shared_mem: {target_shared_mem}")


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
        result = subprocess.run(nsys_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            # Check if we've hit the end of available algorithms
            if result.stderr:
                if "Invalid algorithm index" in result.stderr:
                    # Extract max available index from error message
                    match = re.search(r'available: 0-(\d+)', result.stderr)
                    if match:
                        max_available = int(match.group(1))
                        print(f"\t\t** Algorithm index {algo_idx}: Invalid (available: 0-{max_available})")
                        return (False, None, None, "exhausted")  # Signal to stop searching
                # Show error message for other failures
                error_msg = result.stderr.strip().split('\n')[-1]  # Get last line of error
                if error_msg:
                    print(f"\t\t** Algorithm index {algo_idx}: {error_msg}")
                else:
                    print(f"\t\t** Algorithm index {algo_idx}: nsys failed (code {result.returncode})")
            else:
                print(f"\t\t** Algorithm index {algo_idx}: nsys failed (code {result.returncode})")
            return (False, None, None, None)
    except subprocess.TimeoutExpired:
        print(f"\t\t** Algorithm index {algo_idx}: nsys timeout")
        return (False, None, None, None)
    except Exception as e:
        print(f"\t\t** Algorithm index {algo_idx}: Exception: {e}")
        return (False, None, None, None)
    
    if not os.path.exists(test_trace_path):
        print(f"\t\t** Algorithm index {algo_idx}: Trace file not created")
        return (False, None, None, None)
    
    # Export to sqlite
    sqlite_path = test_trace_path.replace(".nsys-rep", ".sqlite")
    if os.path.exists(sqlite_path):
        os.remove(sqlite_path)
    export_cmd = ["nsys", "export", "--type=sqlite", "--output", sqlite_path, test_trace_path]
    export_result = subprocess.run(export_cmd, capture_output=True, text=True)
    if export_result.returncode != 0 or not os.path.exists(sqlite_path):
        print(f"\t\t** Algorithm index {algo_idx}: Trace export failed")
        return (False, None, None, None)
    
    return (True, test_trace_path, sqlite_path, None)


def _compare_kernel_config(kernel_config, target_kernel, target_grid, target_block, target_shared_mem):
    """
    Compare kernel config with target config.
    
    Returns: (name_match, grid_match, block_match, shared_match, matches_dict)
    """
    kernel_name = kernel_config.get("name", "")
    grid = kernel_config.get("grid", [0, 0, 0])
    block = kernel_config.get("block", [256, 1, 1])
    shared_mem = kernel_config.get("shared_memory", 0)
    
    name_match = target_kernel.lower() == kernel_name.lower()
    grid_match = (grid == target_grid)
    block_match = (block == target_block)
    shared_match = (shared_mem == target_shared_mem)
    
    # Per-dimension matches for display
    grid_x_match = len(grid) > 0 and len(target_grid) > 0 and grid[0] == target_grid[0]
    grid_y_match = len(grid) > 1 and len(target_grid) > 1 and grid[1] == target_grid[1]
    grid_z_match = len(grid) > 2 and len(target_grid) > 2 and grid[2] == target_grid[2]
    block_x_match = len(block) > 0 and len(target_block) > 0 and block[0] == target_block[0]
    block_y_match = len(block) > 1 and len(target_block) > 1 and block[1] == target_block[1]
    block_z_match = len(block) > 2 and len(target_block) > 2 and block[2] == target_block[2]
    
    matches = {
        "name": name_match,
        "grid_match": grid_match,
        "grid_x": grid_x_match,
        "grid_y": grid_y_match,
        "grid_z": grid_z_match,
        "block_match": block_match,
        "block_x": block_x_match,
        "block_y": block_y_match,
        "block_z": block_z_match,
        "shared": shared_match,
        "kernel_name": kernel_name,
        "grid": grid,
        "block": block,
        "shared_mem": shared_mem,
    }
    
    return name_match, grid_match, block_match, shared_match, matches


def _print_algorithm_result(algo_idx, matches):
    """Print the comparison result for an algorithm."""
    kernel_name = matches["kernel_name"]
    grid = matches["grid"]
    block = matches["block"]
    shared_mem = matches["shared_mem"]
    
    clean_kernel_name = SodaProfiler.get_clean_kernel_name(kernel_name)
    
    grid_ticks = f"[{'✓' if matches['grid_x'] else '✗'} {'✓' if matches['grid_y'] else '✗'} {'✓' if matches['grid_z'] else '✗'}]"
    block_ticks = f"[{'✓' if matches['block_x'] else '✗'} {'✓' if matches['block_y'] else '✗'} {'✓' if matches['block_z'] else '✗'}]"
    
    print(f"\t\t** Algorithm index {algo_idx}")
    print(f"\t\t\t*** kernel_name={clean_kernel_name}\t[{'✓' if matches['name'] else '✗'}]")
    print(f"\t\t\t*** grid={grid}\t{grid_ticks}")
    print(f"\t\t\t*** block={block}\t{block_ticks}")
    print(f"\t\t\t*** shared_mem={shared_mem}\t[{'✓' if matches['shared'] else '✗'}]")


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
    
    # Query available algorithm count upfront
    max_available_idx, total_available = get_available_algorithm_count(job, binary_path)
    if max_available_idx is not None:
        print(f"\t* Available algorithms: 0-{max_available_idx}")
        max_algorithms = min(max_algorithms, max_available_idx + 1)
    
    print(f"\t* Starting algorithm search...")
    
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
            print(f"\t\t** Algorithm index {algo_idx}: No kernel config found in trace")
            _cleanup_test_trace(test_trace_path, sqlite_path)
            continue
        
        # Compare with target
        name_match, grid_match, block_match, shared_match, matches = _compare_kernel_config(
            kernel_config, target_kernel, target_grid, target_block, target_shared_mem
        )
        
        _print_algorithm_result(algo_idx, matches)
        
        if name_match and grid_match and block_match and shared_match:
            print(f"\t* ✓ Found algorithm index {algo_idx}")
            _cleanup_test_trace(test_trace_path, sqlite_path)
            return algo_idx
        
        # Clean up after checking (no match)
        _cleanup_test_trace(test_trace_path, sqlite_path)
    
    print(f"\t* ✗ No algorithm index found (tried {algorithms_tried} algorithm{'s' if algorithms_tried != 1 else ''})")
    return None


def search_algorithm_indices(jobs_file, baremetal_dir):
    """
    Search for algorithm indices for all jobs and update jobs.json.
    """
    # Load jobs
    with open(jobs_file, 'r') as f:
        jobs_data = json.load(f)
    
    jobs = jobs_data.get("jobs", [])
    rel_path = os.path.relpath(jobs_file, microbench_dir) if microbench_dir else jobs_file
    print(f"Loaded {len(jobs)} jobs from {rel_path}")
    
    # Build binary
    if not build_binary(baremetal_dir):
        print("Build failed, exiting", file=sys.stderr)
        return False
    
    binary_path = os.path.join(baremetal_dir, "build", "main_gemm_bm")
    if not os.path.exists(binary_path):
        print(f"Binary not found: {binary_path}", file=sys.stderr)
        return False
    
    # Create temporary output dir for test traces
    output_dir = os.path.dirname(jobs_file)
    trace_dir = os.path.join(output_dir, "traces")
    os.makedirs(trace_dir, exist_ok=True)
    
    # Search for algorithm indices for each job
    algorithms_found = 0
    no_algorithm = []
    
    for job in jobs:
        job_id = job["id"]
        print(f"\nSearching algorithm for job {job_id}")
        
        # Skip if no target kernel
        if not job.get("target_kernel") or job["target_kernel"] == "unknown":
            print(f"\t* Skipping (no target kernel)")
            continue
        
        # Search for algorithm index
        matched_algo_idx = search_algorithm_index(job, binary_path, output_dir, max_algorithms=200)
        
        if matched_algo_idx is not None:
            job["matched_algo_index"] = matched_algo_idx
            algorithms_found += 1
            # Don't print again - search_algorithm_index already printed the result
        else:
            job["matched_algo_index"] = None  # Explicitly set to None when no match found
            no_algorithm.append(job_id)
            # Don't print again - search_algorithm_index already printed the no-match message
    
    # Update jobs.json with matched_algo_index
    jobs_data["jobs"] = jobs
    if "matching_summary" not in jobs_data:
        jobs_data["matching_summary"] = {}
    jobs_data["matching_summary"]["matches_found"] = algorithms_found
    jobs_data["matching_summary"]["no_matches"] = no_algorithm
    jobs_data["matching_summary"]["total_jobs"] = len(jobs)
    
    os.makedirs(os.path.dirname(jobs_file), exist_ok=True)
    with open(jobs_file, 'w') as f:
        json.dump(jobs_data, f, indent=2)
    
    rel_path = os.path.relpath(jobs_file, microbench_dir) if microbench_dir else jobs_file
    print(f"\n==============================================")
    print(f"Summary")
    print(f"- Algorithms found: {algorithms_found}/{len(jobs)}")
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
    print(f"- Updated {rel_path}")
    print(f"==============================================")
    return True


if __name__ == "__main__":
    # Check if env.sh has been sourced
    if not os.environ.get("SODA_ENV_LOADED"):
        print("Error: SODA environment not loaded.", file=sys.stderr)
        print("Please run: source env.sh", file=sys.stderr)
        sys.exit(1)
    
    # Get paths from environment
    baremetal_dir = os.environ["BAREMETAL_MICROBENCH_DIR"]
    jobs_file = os.environ["BAREMETAL_JOBS"]
    microbench_dir = os.environ.get("MICROBENCH_DIR")
    
    if not os.path.exists(jobs_file):
        print(f"Error: Jobs file not found: {jobs_file}", file=sys.stderr)
        print("Run gen_bm_jobs.py first", file=sys.stderr)
        sys.exit(1)
    
    success = search_algorithm_indices(jobs_file, baremetal_dir)
    sys.exit(0 if success else 1)

