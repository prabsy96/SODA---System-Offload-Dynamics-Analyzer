#!/usr/bin/env python3
"""
Profile baremetal GEMM kernels using matched algorithms (profiling phase).

Reads jobs from baremetal/output/jobs.json, uses cublas_index from
algorithm matching phase, runs full nsys profiling, computes kernel launch
tax statistics, and emits baremetal/output/baremetal_gemm_runs.json.

This phase assumes search_cublas_algos_offline.py has already been run.
"""

import sqlite3
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
    
    print(f"Running job {job_id}")
    
    binary_path = utils.get_path("BAREMETAL_BINARY")
    trace_dir = nsys_utils.get_trace_dir()

    # Handle null kernel job
    if job.get("name") == "__null__":
        args = [
            str(binary_path),
            "--null_kernel",
            "--warmup", str(job["warmup"]),
            "--runs", str(job["runs"]),
        ]
    else:
        matched_algo_idx = job.get("cublas_index")
        
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
    
    # Query for kernel events (with all available fields)
    # For null kernel, get any kernel; for GEMM, filter by name
    kernels = []
    try:
        # Query all kernels (will filter by name later if needed)
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
        for row in cursor.fetchall():
            start_ns, end_ns, corr_id, name, gx, gy, gz, bx, by, bz, static_smem, dyn_smem, device_id, context_id, stream_id, regs = row
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
                "device": device_id,
                "context": context_id,
                "stream": stream_id,
                "registers_per_thread": regs,
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

def profile_baremetal_gemm_kernels():
    """
    Profile baremetal GEMM kernels for all jobs using matched algorithms.
    """ 

    # Load jobs
    jobs_file = utils.get_path("BAREMETAL_JOBS")
    jobs_data = utils.load_json(jobs_file)
    jobs = jobs_data.get("jobs", [])
    print(f"Loaded {len(jobs)} jobs.")
    
    # Check if offline cublas algorithm search has been completed
    offline_cublas_search = jobs_data.get("summary", {}).get("offline_cublas_search", {})
    algos_found = offline_cublas_search.get("algos_found", None)
    if algos_found is None:
        print("Warning: No cublas algorithm indices found in jobs.json", file=sys.stderr)
    else:
        print(f"Using {algos_found} matched algorithms")
    
    # Build binary
    build_binary()
    binary_path = utils.get_path("BAREMETAL_BINARY")
    utils.ensure_file(binary_path)
    
    # Run each job
    sequences = []
    
    for job in jobs:

        # Skip batched GEMMs for now due to non-contiguous layout issue
        if job.get("batch", 1) > 1:
            print(f"Skipping job {job['id']} (FIXME batched GEMM)")
            continue
        
        sqlite_path = run_job_under_nsys(job)
        
        if sqlite_path is None:
            continue
        
        # Parse trace and compute stats
        result = parse_trace_and_compute_stats(sqlite_path, job["runs"], job.get("warmup", 0))
        
        if result is None:
            print(f"Warning: No kernel events found for job {job['id']}", file=sys.stderr)
            continue
        
        # Build sequence entry matching PyTorch format
        sequence_entry = {
            "cpu_op": job.get("cpu_op") if job.get("name") != "__null__" else {},
            "kernel": result["kernel"],
            "meta": {
                "job_id": job["id"],
                **result["stats"]  # avg_kernel_tax, min_kernel_tax, max_kernel_tax, count
            }
        }
        
        # Capture null kernel baseline tax for delta calculations
        if job["name"] == "__null__":
            null_kernel_tax = result["stats"]["avg_kernel_tax"]
        
        sequences.append(sequence_entry)
    
    # Write output in PyTorch format
    output_data = {
        "summary": {"count": len(sequences)},
        "sequences": sequences,
    }

    output_file = utils.get_path("BAREMETAL_GEMM_KERNELS")
    utils.save_json(output_file, output_data)
    
    print(f"\nCompleted {len(sequences)} jobs -> {output_file}")
    
    # Print a summary table with % delta over null kernel 
    print_summary(sequences, null_kernel_tax)

def print_summary(sequences, null_kernel_tax=None):
    """Print a compact table summary of profiled sequences, with % delta over null kernel."""
    table_data = []
    for sequence in sequences:
        avg = sequence["meta"]["avg_kernel_tax"]
        delta_pct = (
            (avg - null_kernel_tax) / null_kernel_tax * 100
            if null_kernel_tax else 0.0
        )
        table_data.append([
            sequence["meta"]["job_id"],
            sequence.get("cpu_op", {}).get("name", "unknown"),
            sequence["kernel"]["name"],
            f"{avg:.2f} μs",
            f"{delta_pct:.1f}%",
        ])

    if table_data:
        print_utils.comp_table(
            title="Profile Summary",
            headers=["Job", "CPU Op", "Kernel", "Avg Kernel Tax", "Δ(%)"],
            data=table_data,
        )