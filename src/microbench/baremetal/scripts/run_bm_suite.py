#!/usr/bin/env python3
"""
Run baremetal GEMM suite under nsys profiling and aggregate results.

Reads jobs from baremetal/output/jobs.json, builds C++ binary, runs each job
under nsys profiling, parses traces, computes kernel launch tax statistics,
and emits baremetal/output/baremetal_gemm_runs.json.
"""

import json
import os
import sys
import subprocess
import re
from pathlib import Path

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


def run_job_under_nsys(job, binary_path, output_dir):
    """
    Run a single job under nsys profiling.
    
    Returns: (qdrep_path, json_path)
    """
    job_id = job["id"]
    
    # Skip batched GEMMs for now (non-contiguous layout issue)
    if job.get("batch", 1) > 1:
        print(f"\nSkipping job {job_id} (batched GEMM with non-standard layout)")
        return None, None
    
    print(f"\nRunning job {job_id}...")
    
    # Build command line args
    args = [
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
        "--warmup", str(job["warmup"]),
        "--runs", str(job["runs"]),
    ]
    
    # Add batch if present
    if "batch" in job:
        args.extend(["--batch", str(job["batch"])])
    
    # nsys command
    trace_dir = os.path.join(output_dir, "traces")
    os.makedirs(trace_dir, exist_ok=True)
    
    qdrep_path = os.path.join(trace_dir, f"trace_{job_id}")
    
    nsys_cmd = [
        "nsys", "profile",
        "--trace=cuda,osrt",
        f"--output={qdrep_path}",
        "--force-overwrite=true",
    ] + args
    
    # Run under nsys
    result = subprocess.run(nsys_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"nsys profiling failed for job {job_id}:\n{result.stderr}", file=sys.stderr)
        return None, None
    
    # Export to sqlite (more reliable than JSON for programmatic parsing)
    sqlite_path = qdrep_path + ".sqlite"
    export_cmd = [
        "nsys", "export",
        "--type=sqlite",
        f"--output={sqlite_path}",
        "--force-overwrite=true",
        qdrep_path + ".nsys-rep"
    ]
    
    result = subprocess.run(export_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"nsys export failed for job {job_id}:\n{result.stderr}", file=sys.stderr)
        return None, None
    
    # Print relative path
    rel_path = os.path.relpath(sqlite_path, microbench_dir) if microbench_dir else sqlite_path
    print(f"\t** Job {job_id} completed, trace exported to {rel_path}")
    return qdrep_path, sqlite_path


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
        rel_path = os.path.relpath(sqlite_path, microbench_dir) if microbench_dir else sqlite_path
        print(f"Error opening sqlite database {rel_path}: {e}", file=sys.stderr)
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
    # Filter for GEMM kernels only 
    # Kernel name must contain 'gemm' to exclude other kernels (e.g., splitKreduce_kernel, etc.)
    kernels = []
    try:
        cursor.execute("""
            SELECT k.start, k.end, k.correlationId, s.value, 
                   k.gridX, k.gridY, k.gridZ,
                   k.blockX, k.blockY, k.blockZ,
                   k.staticSharedMemory, k.dynamicSharedMemory
            FROM CUPTI_ACTIVITY_KIND_KERNEL as k
            JOIN StringIds as s ON k.demangledName = s.id
            WHERE LOWER(s.value) LIKE '%gemm%'
        """)
        for row in cursor.fetchall():
            start_ns, end_ns, corr_id, name, gx, gy, gz, bx, by, bz, static_smem, dyn_smem = row
            kernels.append({
                "name": name,
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
        rel_path = os.path.relpath(sqlite_path, microbench_dir) if microbench_dir else sqlite_path
        print(f"Warning: No CUDA launch events found in {rel_path}", file=sys.stderr)
        return None
    
    if not kernels:
        rel_path = os.path.relpath(sqlite_path, microbench_dir) if microbench_dir else sqlite_path
        print(f"Warning: No kernel events found in {rel_path}", file=sys.stderr)
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
            
            # Capture kernel info from first kernel (all should be same)
            if kernel_info is None:
                kernel_info = {
                    "name": kernel["name"],
                    "grid": kernel["grid"],
                    "block": kernel["block"],
                    "shared_memory": kernel["shared_memory"],
                }
    
    # Compute statistics
    if not kernel_tax_values:
        rel_path = os.path.relpath(sqlite_path, microbench_dir) if microbench_dir else sqlite_path
        print(f"Warning: No matched kernel-launch pairs found in {rel_path}", file=sys.stderr)
        return None
    
    stats = {
        "avg_kernel_tax": sum(kernel_tax_values) / len(kernel_tax_values),
        "min_kernel_tax": min(kernel_tax_values),
        "max_kernel_tax": max(kernel_tax_values),
        "count": len(kernel_tax_values),
    }
    
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


def run_suite(jobs_file, baremetal_dir, output_file):
    """
    Run the entire baremetal GEMM suite.
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
    
    # Run each job
    output_dir = os.path.dirname(output_file)
    runs = []
    
    for job in jobs:
        qdrep_path, sqlite_path = run_job_under_nsys(job, binary_path, output_dir)
        
        if sqlite_path is None:
            continue
        
        # Parse trace and compute stats
        result = parse_trace_and_compute_stats(sqlite_path, job["runs"], job.get("warmup", 0))
        
        if result is None:
            print(f"Warning: No kernel events found for job {job['id']}", file=sys.stderr)
            continue
        
        # Add job ID and target kernel to result
        run_entry = {
            "id": job["id"],
            "target_kernel": job.get("target_kernel", "unknown"),
            "kernel": result["kernel"],
            "stats": result["stats"],
        }
        
        runs.append(run_entry)
        print(f"\t** Kernel: {result['kernel']['name'][:60]}...")
        print(f"\t** Avg kernel tax: {result['stats']['avg_kernel_tax']:.2f} μs")
    
    # Collect environment info
    env = collect_gpu_info()
    
    # Write output
    output_data = {
        "summary": {
            "jobs": len(runs),
            "source": jobs_file,
        },
        "runs": runs,
        "env": env,
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    rel_path = os.path.relpath(output_file, microbench_dir) if microbench_dir else output_file
    print(f"\nCompleted {len(runs)} jobs -> {rel_path}")
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
    output_file = os.environ["BAREMETAL_RUNS"]
    microbench_dir = os.environ.get("MICROBENCH_DIR")
    
    if not os.path.exists(jobs_file):
        print(f"Error: Jobs file not found: {jobs_file}", file=sys.stderr)
        print("Run gen_bm_jobs.py first", file=sys.stderr)
        sys.exit(1)
    
    success = run_suite(jobs_file, baremetal_dir, output_file)
    sys.exit(0 if success else 1)

