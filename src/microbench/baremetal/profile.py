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
from typing import Dict, Any
from microbench.baremetal.utils import build_binary, build_base_args, nsys_profile_to_sql, extract_kernels_from_trace, extract_launches_from_trace

from soda.common import utils, print_utils
from soda.common.data import CPUOp


def run_job(job):
    """
    Run a single job under nsys profiling and emit sqlite trace.
    """
    job_id = job["id"]
    
    print(f"Running job {job_id}")
    
    binary_path = utils.get_path("BAREMETAL_BINARY")
    utils.ensure_file(binary_path)

    # Handle null kernel job
    if job["name"] == "__null__":
        args = [
            str(binary_path),
            "--null_kernel",
            "--warmup", str(job["warmup"]),
            "--runs", str(job["runs"]),
        ]
    else:

        matched_algo_idx = job["cublas_index"]
        args = build_base_args(job) + [
            "--warmup", str(job["warmup"]),
            "--runs", str(job["runs"]),
        ]
        
        if "batch" in job:
            args = args + ["--batch", str(job["batch"])]
        
        if matched_algo_idx is not None:
            args = args + ["--algo_index", str(matched_algo_idx)]
    
    trace_file_name = f"trace_{job_id}"
    success, trace_file_sql, message = nsys_profile_to_sql(
        trace_file_name, args, timeout=600, cleanup=True
    )

    if not success:
        print(f"nsys profiling failed for job {job_id}:\n{message}", file=sys.stderr)
        return None

    return trace_file_sql


def parse_trace_and_compute_stats(job, trace_file_sql, runs, warmup=0):
    """
    Parse nsys sqlite trace and compute kernel launch tax statistics.
    
    Uses same event linking and aggregation logic as PyTorch pipeline:
    - Find cu(da)LaunchKernel events
    - Find kernel events (kernel category)
    - Link via correlation ID (utils.link_sequences)
    - Compute kernel_tax = kernel.ts - cu(da)LaunchKernel.ts (utils.calculate_per_seq_launch_tax)
    - Aggregate multiple runs (utils.deduplicate_and_aggregate)
    
    Args:
        job: Job dictionary with cpu_op, job_id, and config
        trace_file_sql: Path to nsys sqlite export
        runs: Expected number of measurement runs
        warmup: Number of warmup runs to skip
    
    Returns: Aggregated sequence dict with keys: kernel, meta, cuda_launch, cpu_op
    """
    # Extract CUDA launch events and kernel events
    cuda_launches = extract_launches_from_trace(trace_file_sql)
    kernels = extract_kernels_from_trace(trace_file_sql, cleanup=False)

    if not cuda_launches:
        raise RuntimeError(f"No CUDA launch events found in {trace_file_sql}")
    if not kernels:
        raise RuntimeError(f"No kernel events found in {trace_file_sql}")
    
    # All kernels in this trace belong to the same job/cpu_op
    # Handle null kernel vs regular GEMM jobs differently
    external_id = job["id"]
    if job["name"] == "__null__":
        # Null kernel job (hack to ensure we have a cpu_op)
        # cpu_op is null in jobs.json so use a dummy __nop__ cpu_op
        # and use job_id as external_id
        cpu_op = CPUOp(
            name="__nop__",
            external_id=external_id,
        ).get_signature(full=True)
    else:
        # Regular GEMM job: cpu_op was extracted from PyTorch ptrace
        # Use its original external_id to preserve linking
        cpu_op = job["cpu_op"]
        external_id = cpu_op["external_id"]
    
    # Convert to events structure for shared utilities
    # This adapts baremetal data to the format expected by utils.link_sequences()
    events = {
        "cpu": {
            "ops": {
                # Map external_id -> cpu_op (preserves original linking)
                external_id: cpu_op
            },
            "launches": cuda_launches  # Dict[corr_id -> launch_dict]
        },
        "gpu": {
            "kernels": [
                {
                    "name": kernel.name,
                    "external_id": external_id,
                    "ts": kernel.ts,
                    "dur": kernel.dur,
                    "correlation": kernel.correlation,
                    "grid": kernel.grid,
                    "block": kernel.block,
                    "shared_memory": kernel.shared_memory,
                    "device": kernel.device,
                    "context": kernel.context,
                    "stream": kernel.stream,
                    "registers_per_thread": kernel.registers_per_thread,
                }
                for kernel in kernels
            ]
        }
    }
    
    # Same approach as PyTorch microbench (see microbench/framework/pytorch/profile.py)
    linked_sequences = utils.link_sequences(events)
    linked_sequences_with_tax = utils.calculate_per_seq_launch_tax(linked_sequences)
    
    # Skip warmup iterations by slicing
    sequences_after_warmup = linked_sequences_with_tax[warmup:]
    
    if not sequences_after_warmup:
        raise RuntimeError(f"No sequences remaining after warmup in {trace_file_sql}")
    
    # Aggregate using shared utility (same as PyTorch pipeline)
    # This deduplicates by kernel signature and aggregates kernel_tax stats into meta
    aggregated_sequences = utils.deduplicate_and_aggregate(sequences_after_warmup)
    
    # Baremetal runs same kernel config multiple times, so should have exactly 1 unique kernel
    if len(aggregated_sequences) != 1:
        kernel_names = [seq["kernel"]["name"] for seq in aggregated_sequences]
        raise RuntimeError(
            f"Expected 1 unique kernel after aggregation, found {len(aggregated_sequences)}: {kernel_names}"
        )
    
    # Return the single aggregated sequence
    # Format: {meta: {avg_kernel_tax, ...}, kernel: {...}, cuda_launch: {...}, cpu_op: {...}}
    return aggregated_sequences[0]

def profile_baremetal_gemm_kernels() -> Dict[str, Any]:
    """
    Profile baremetal GEMM kernels for all jobs using matched algorithms.
    
    Returns:
        Dictionary with profiled baremetal GEMM sequences data (same format as saved JSON).
    """ 

    # Load jobs
    jobs_file = utils.get_path("BAREMETAL_JOBS")
    jobs_data = utils.load_json(jobs_file)
    jobs = jobs_data["jobs"]
    print(f"Loaded {len(jobs)} jobs.")
    
    # Check if offline cublas algorithm search has been completed
    if "offline_cublas_search" in jobs_data["summary"]:
        offline_cublas_search = jobs_data["summary"]["offline_cublas_search"]
        algos_found = offline_cublas_search["algos_found"]
        print(f"Using {algos_found} matched algorithms")
    else:
        raise RuntimeError("Offline cublas algorithm search has not been completed.")
    
    # Build binary
    build_binary()
    binary_path = utils.get_path("BAREMETAL_BINARY")
    utils.ensure_file(binary_path)
    
    # Run each job and collect sequences
    sequences = []
    null_kernel_tax = None
    
    for job in jobs:

        # Skip batched GEMMs for now due to non-contiguous layout issue
        if "batch" in job and job["batch"] > 1:
            print(f"Skipping job {job['id']} (TODO: batched GEMM)")
            sequences.append(None)  # Append None to maintain alignment
            continue
        
        trace_file_sql = run_job(job)
        if trace_file_sql is None:
            raise Exception(f"No trace file found for job {job['id']}")
        
        # Parse trace and compute stats (returns aggregated sequence with cpu_op already included)
        sequence = parse_trace_and_compute_stats(job, trace_file_sql, job["runs"], job["warmup"])
        
        if sequence is None:
            raise Exception(f"No kernel events found for job {job['id']}")
        
        # Add job_id to meta (cpu_op already included from parse_trace_and_compute_stats)
        sequence["meta"]["job_id"] = job["id"]
        
        # Capture null kernel baseline tax for delta calculations
        if job["name"] == "__null__":
            null_kernel_tax = sequence["meta"]["avg_kernel_tax"]
        
        sequences.append(sequence)
    
    baremetal_gemm_sequences_file = utils.get_path("BAREMETAL_GEMM_KERNELS")
    baremetal_gemm_sequences_data = {
        "summary": {"count": len(sequences)},
        "sequences": sequences,
    }
    utils.save_json(baremetal_gemm_sequences_file, baremetal_gemm_sequences_data)
    print(f"Saved {baremetal_gemm_sequences_data['summary']['count']} baremetal GEMM sequences to {baremetal_gemm_sequences_file}")
    
    # Print a summary table with % delta over null kernel 
    print_summary(sequences, null_kernel_tax)
    
    return baremetal_gemm_sequences_data

def print_summary(sequences, null_kernel_tax=None):
    """Print a compact table summary of profiled sequences, with % delta over null kernel."""
    table_data = []
    for sequence in sequences:

        # Skip None entries (skipped baremetal jobs)
        if sequence is None:
            continue

        avg = sequence["meta"]["avg_kernel_tax"]
        delta_pct = (
            (avg - null_kernel_tax) / null_kernel_tax * 100
            if null_kernel_tax else 0.0
        )
        table_data.append([
            sequence["meta"]["job_id"],
            sequence["cpu_op"]["name"],
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
