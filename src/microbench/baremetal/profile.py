#!/usr/bin/env python3
"""
Profile baremetal GEMM kernels using matched algorithms (profiling phase).

Reads jobs from baremetal/output/jobs.json, uses matched_algo_index from
algorithm matching phase, runs full nsys profiling, computes kernel launch
tax statistics, and emits baremetal/output/baremetal_gemm_runs.json.

This phase assumes search_algorithm_indices.py has already been run.
"""

import json
import sys
import subprocess
import re
from microbench.baremetal.search import build_binary, build_base_args
from microbench.baremetal import nsys_utils
from soda import utils
from common import print_utils


def run_job_under_nsys(job):
    """
    Run a single job under nsys profiling using matched algorithm index.
    """
    job_id = job["id"]
    
    # Skip batched GEMMs for now (non-contiguous layout issue)
    if job.get("batch", 1) > 1:
        print(f"Skipping job {job_id} (batched GEMM with non-standard layout)")
        return None
    
    print(f"Running job {job_id}...")
    
    binary_path = utils.get_path("BAREMETAL_BINARY")
    trace_dir = nsys_utils.get_trace_dir()

    # Handle null kernel job
    if job.get("null_kernel"):
        args = [
            str(binary_path),
            "--null_kernel",
            "--warmup", str(job["warmup"]),
            "--runs", str(job["runs"]),
        ]
    else:
        matched_algo_idx = job.get("matched_algo_index")
        
        # Build command line args 
        args = build_base_args(job) + [
            "--warmup", str(job["warmup"]),
            "--runs", str(job["runs"]),
        ]
        
        if "batch" in job:
            args = args + ["--batch", str(job["batch"])]
        
        if matched_algo_idx is not None:
            args = args + ["--algo_index", str(matched_algo_idx)]
    
    trace_output = trace_dir / f"trace_{job_id}"
    success, sqlite_path, message = nsys_utils.nsys_profile_to_sqlite(
        trace_output, args
    )

    if not success:
        print(f"nsys profiling failed for job {job_id}:\n{message}", file=sys.stderr)
        return None

    return sqlite_path


def parse_trace_and_compute_stats(sqlite_path, runs, warmup=0):
    """
    Parse nsys sqlite trace and compute kernel launch tax statistics.
    
    Uses same event linking logic as PyTorch extract_kernel_sequences.py:
    - Find cudaLaunchKernel events (cuda_runtime)
    - Find kernel events (kernel category)
    - Link via correlation ID
    - Compute kernel_tax = kernel.ts - cudaLaunchKernel.ts (microseconds)
    - Aggregate statistics
    
    Args:
        sqlite_path: Path to nsys sqlite export
        runs: Expected number of measurement runs
        warmup: Number of warmup runs to skip
    
    Returns: dict with kernel info and stats
    """
    import sqlite3
    
    # Connect to sqlite database
    try:
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Error opening sqlite database {sqlite_path}: {e}", file=sys.stderr)
        return None
    
    # Query for CUDA runtime API calls (cudaLaunchKernel)
    cuda_launches = {}
    try:
        cursor.execute("""
            SELECT start, end, correlationId
            FROM CUPTI_ACTIVITY_KIND_RUNTIME
            WHERE nameId IN (SELECT id FROM StringIds WHERE value LIKE 'cudaLaunchKernel%')
        """)
        for row in cursor.fetchall():
            start_ns, end_ns, corr_id = row
            cuda_launches[corr_id] = {
                "ts": start_ns / 1000.0,  # Convert ns to us
                "dur": (end_ns - start_ns) / 1000.0,
                "correlation": corr_id,
            }
    except Exception as e:
        print(f"Warning: Could not query CUDA runtime events: {e}", file=sys.stderr)
    
    # Query for kernel events
    # For null kernel, get any kernel; for GEMM, filter by name
    kernels = []
    try:
        # Query all kernels (will filter by name later if needed)
        cursor.execute("""
            SELECT k.start, k.end, k.correlationId, s.value, 
                   k.gridX, k.gridY, k.gridZ,
                   k.blockX, k.blockY, k.blockZ,
                   k.staticSharedMemory, k.dynamicSharedMemory
            FROM CUPTI_ACTIVITY_KIND_KERNEL as k
            JOIN StringIds as s ON k.demangledName = s.id
        """)
        for row in cursor.fetchall():
            start_ns, end_ns, corr_id, name, gx, gy, gz, bx, by, bz, static_smem, dyn_smem = row
            # For GEMM jobs, only include kernels with 'gemm' in name
            # For null kernel, include all kernels
            if "gemm" not in name.lower() and "null" not in name.lower():
                continue
            kernels.append({
                "name": "__null__" if "null" in name.lower() else name,
                "ts": start_ns / 1000.0,  # Convert ns to us
                "dur": (end_ns - start_ns) / 1000.0,
                "correlation": corr_id,
                "grid": [gx, gy, gz],
                "block": [bx, by, bz],
                "shared_memory": static_smem + dyn_smem,
            })
    except Exception as e:
        print(f"Warning: Could not query kernel events: {e}", file=sys.stderr)
    
    conn.close()
    
    if not cuda_launches:
        print(f"Warning: No CUDA launch events found in {sqlite_path}", file=sys.stderr)
        return None
    
    if not kernels:
        print(f"Warning: No kernel events found in {sqlite_path}", file=sys.stderr)
        return None
    
    # Link kernels to cuda launches and compute kernel tax
    # Skip warmup iterations (first N kernel launches)
    kernel_tax_values = []
    kernel_info = None
    skipped = 0
    
    for kernel in kernels:
        correlation = kernel["correlation"]
        cuda_launch = cuda_launches.get(correlation)
        
        if cuda_launch and kernel["ts"] is not None and cuda_launch["ts"] is not None:
            # Skip warmup iterations
            if skipped < warmup:
                skipped += 1
                continue
            
            kernel_tax = kernel["ts"] - cuda_launch["ts"]
            assert kernel_tax >= 0, f"Negative kernel tax detected: kernel.ts={kernel['ts']}, cudaLaunchKernel.ts={cuda_launch['ts']}, tax={kernel_tax}"
            kernel_tax_values.append(kernel_tax)
            
            # Capture kernel info from first kernel
            if kernel_info is None:
                kernel_info = {
                    "name": kernel["name"],
                    "grid": kernel["grid"],
                    "block": kernel["block"],
                    "shared_memory": kernel["shared_memory"],
                }
    
    # Compute statistics
    if not kernel_tax_values:
        print(f"Warning: No matched kernel-launch pairs found in {sqlite_path}", file=sys.stderr)
        return None
    
    stats = utils.calculate_avg_min_max(kernel_tax_values, "kernel_tax")
    stats["count"] = len(kernel_tax_values)
    
    return {
        "kernel": kernel_info,
        "stats": stats,
    }


def collect_gpu_info():
    """Collect GPU and CUDA environment information."""
    env = {}
    
    # Get GPU name
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            env["gpu"] = result.stdout.strip()
    except Exception:
        env["gpu"] = "unknown"
    
    # Get CUDA version
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            # Extract version from output
            match = re.search(r"release (\d+\.\d+)", result.stdout)
            if match:
                env["cuda_version"] = match.group(1)
    except Exception:
        env["cuda_version"] = "unknown"
    
    return env


def profile_baremetal_gemm_kernels():
    """
    Profile baremetal GEMM kernels for all jobs using matched algorithms.
    """
    jobs_file = utils.get_path("BAREMETAL_JOBS")
    output_file = utils.get_path("BAREMETAL_GEMM_KERNELS")
    
    # Check if jobs file exists
    utils.ensure_file(jobs_file)
    
    # Load jobs
    with open(jobs_file, 'r') as f:
        jobs_data = json.load(f)
    
    jobs = jobs_data.get("jobs", [])
    print(f"Loaded {len(jobs)} jobs from {jobs_file}")
    
    # Check if matching has been done
    matching_summary = jobs_data.get("matching_summary", {})
    matches_found = matching_summary.get("matches_found", 0)
    if matches_found == 0:
        print("Warning: No algorithm matches found in jobs.json", file=sys.stderr)
        print("Run search_algorithm_indices.py first to search for algorithm indices", file=sys.stderr)
    else:
        print(f"Using {matches_found} matched algorithms")
    
    # Build binary
    build_binary()
    binary_path = utils.get_path("BAREMETAL_BINARY")
    if not binary_path.exists():
        raise RuntimeError(f"Binary not found: {binary_path}")
    
    # Run each job
    kernels = []
    
    for job in jobs:
        sqlite_path = run_job_under_nsys(job)
        
        if sqlite_path is None:
            continue
        
        # Parse trace and compute stats
        result = parse_trace_and_compute_stats(sqlite_path, job["runs"], job.get("warmup", 0))
        
        if result is None:
            print(f"Warning: No kernel events found for job {job['id']}", file=sys.stderr)
            continue
        
        # Add job ID and target kernel to result
        kernel_entry = {
            "id": job["id"],
            "target_kernel": job.get("target_kernel", "unknown"),
            "kernel": result["kernel"],
            "stats": result["stats"],
        }
        
        kernels.append(kernel_entry)
    
    # Collect environment info
    env = collect_gpu_info()
    
    # Write output
    output_data = {
        "summary": {
            "count": len(kernels),
            "source": str(jobs_file),
        },
        "kernels": kernels,
        "env": env,
    }
    
    utils.save_json(output_file, output_data)
    
    # Print a compact table summary
    table_rows = []
    for entry in kernels:
        table_rows.append([
            entry["id"],
            next((job["cpu_op"] for job in jobs if job["id"] == entry["id"]), "unknown"),
            entry["kernel"]["name"],
            f"{entry['stats']['avg_kernel_tax']:.2f} μs",
        ])

    if table_rows:
        print_utils.comp_table(
            title="Profile Summary",
            headers=["Job", "CPU Op", "Kernel", "Avg Kernel Tax"],
            data=table_rows,
        )
    
    print(f"\nCompleted {len(kernels)} jobs -> {output_file}")
