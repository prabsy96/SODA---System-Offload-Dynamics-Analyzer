#!/usr/bin/env python3
"""
Profile baremetal GEMM kernels using matched algorithms (profiling phase).

Reads jobs from baremetal/output/jobs.json, uses cublas_index from
algorithm matching phase, runs full nsys profiling, computes kernel launch
tax statistics, and emits baremetal/output/baremetal_gemm_runs.json.

By default, this phase assumes search_cublas_algos_offline.py has already
been run, but the caller can choose to skip that requirement.
"""

import sqlite3
import json
import sys
import subprocess
import re
from typing import Dict, Any
from microbench.baremetal.utils import build_binary, build_base_args, nsys_profile_to_sql, extract_kernels_from_trace, extract_launches_from_trace

from soda.common import utils, print_utils
from soda.common.data import ATenOp


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

        args = build_base_args(job) + [
            "--warmup", str(job["warmup"]),
            "--runs", str(job["runs"]),
        ]
        
        if "batch" in job:
            args = args + ["--batch", str(job["batch"])]

        if "cublas_index" not in job:
            # Offline cuBLASLt algorithm search metadata not present for this job
            # This means the offline search step was skipped for this run.
            # In this case, we rely on cuBLASLt heuristic algorithm selection.
            # This is implicitly done by the binary main_gemm_bm.cpp
            pass
        else:
            # Offline cuBLASLt algorithm search has been completed for this job.
            matched_algo_idx = job["cublas_index"]
            if matched_algo_idx is None:
                # Offline search ran but found no matching algorithm index.
                # Fall back to heuristic algorithm selection in main_gemm_bm.cpp.
                pass
            else: 
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
    Parse nsys sqlite trace and compute launch tax statistics.
    
    Uses same event linking and aggregation logic as PyTorch pipeline:
    - Find cu(da)LaunchKernel events
    - Find kernel events (kernel category)
    - Link via correlation ID (utils.link_sequences)
    - Compute launch_tax = kernel.ts - cu(da)LaunchKernel.ts (utils.calculate_sequence_metrics)
    - Aggregate multiple runs (utils.aggregate_sequences)
    
    Args:
        job: Job dictionary with aten_op, job_id, and config
        trace_file_sql: Path to nsys sqlite export
        runs: Expected number of measurement runs
        warmup: Number of warmup runs to skip
    
    Returns: Aggregated sequence dict with keys: kernel, launch_tax, cuda_launch, aten_op
    """
    # Extract CUDA launch events and kernel events
    cuda_launches = extract_launches_from_trace(trace_file_sql)
    kernels = extract_kernels_from_trace(trace_file_sql, cleanup=False)
    nvtx_ranges = extract_nvtx_ranges_from_trace(trace_file_sql)

    if not cuda_launches:
        raise RuntimeError(f"No CUDA launch events found in {trace_file_sql}")
    if not kernels:
        raise RuntimeError(f"No kernel events found in {trace_file_sql}")
    
    # All kernels in this trace belong to the same job/aten_op
    # Handle null kernel vs regular GEMM jobs differently
    external_id = job["id"]
    if job["name"] == "__null__":
        # Null kernel job (hack to ensure we have an aten_op)
        # aten_op is null in jobs.json so use a dummy __nop__ op
        # and use job_id as external_id
        aten_op = ATenOp(
            name="__nop__",
            ts=0.0,
            dur=0.0,
            external_id=external_id,
        ).get_signature(full=True)
    else:
        # Regular GEMM job: aten_op was extracted from PyTorch ptrace
        # Use its original external_id to preserve linking
        aten_op = job["aten_op"]
        # HACK: jobs.json stores aggregated duration; reset to raw 0.0 so re-aggregation works.
        aten_op["dur"] = 0.0
        aten_op["ts"] = 0.0
        external_id = aten_op["external_id"]
    
    # Convert to events structure for shared utilities
    # This adapts baremetal data to the format expected by utils.link_sequences()
    events = {
        "cpu": {
            "aten_ops": {
                # Map external_id -> aten_op (preserves original linking)
                external_id: aten_op
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
    linked_sequences_with_tax = utils.calculate_sequence_metrics(linked_sequences, metrics=["launch_tax"])
    
    # Skip warmup iterations by slicing
    sequences_after_warmup = linked_sequences_with_tax[warmup:]
    
    if not sequences_after_warmup:
        raise RuntimeError(f"No sequences remaining after warmup in {trace_file_sql}")
    
    # Aggregate using shared utility (same as PyTorch pipeline)
    # Deduplicate + aggregate by kernel signature (adds launch_tax stats)
    grouped_seqs_by_id_dict = utils.group_sequences_by_identity(sequences_after_warmup)
    aggregated_sequences = utils.aggregate_sequences(
        grouped_seqs_by_id_dict,
        metrics=["launch_tax"],
        event_types=["kernel", "aten_op", "cuda_launch"],
    )
    
    # Baremetal runs same kernel config multiple times, so should have exactly 1 unique kernel
    if len(aggregated_sequences) != 1:
        kernel_names = [seq["kernel"]["name"] for seq in aggregated_sequences]
        raise RuntimeError(
            f"Expected 1 unique kernel after aggregation, found {len(aggregated_sequences)}: {kernel_names}"
        )
    
    # Return the single aggregated sequence
    # Format: {kernel_tax: {...}, kernel: {...}, cuda_launch: {...}, cpu_op: {...}}
    return aggregated_sequences[0]

def profile_baremetal_gemm_kernels(
    skip_offline_cublas_algo_search: bool = False,
) -> Dict[str, Any]:
    """
    Profile baremetal GEMM kernels for all jobs using matched algorithms.
    
    Args:
        skip_offline_cublas_algo_search:
            If True, allow profiling even when offline cuBLASLt algorithm
            search metadata is missing; baremetal will then rely on heuristic
            algorithm selection instead of matched cublas_index.
    
    Returns:
        Dictionary with profiled baremetal GEMM sequences data (same format as saved JSON).
    """ 
    
    # Load jobs
    jobs_file = utils.get_path("BAREMETAL_JOBS")
    jobs_data = utils.load_json(jobs_file)
    jobs = jobs_data["jobs"]
    print(f"Loaded {len(jobs)} jobs.")
    
    # Check if offline cublas algorithm search has been completed
    if "offline_cublas_algo_search" in jobs_data["summary"]:
        offline_cublas_algo_search = summary["offline_cublas_algo_search"]
        algos_found = offline_cublas_algo_search["algos_found"]
        print(f"Using {algos_found} matched algorithms")
    else:
        if skip_offline_cublas_algo_search:
            print(
                "Skipping offline cuBLASLt algorithm search metadata; "
                "profiling baremetal kernels with heuristic cuBLASLt algorithms.",
                file=sys.stderr,
            )
        else:
            raise RuntimeError("Offline cublas algorithm search has not been completed.")
    
    # Build binary
    build_binary()
    binary_path = utils.get_path("BAREMETAL_BINARY")
    utils.ensure_file(binary_path)
    
    # Run each job and collect sequences
    sequences = []
    null_launch_tax = None

    for job in jobs:

        # Skip batched GEMMs for now due to non-contiguous layout issue
        if "batch" in job and job["batch"] > 1:
            print(f"Skipping job {job['id']} (TODO: batched GEMM)")
            sequences.append(None)  # Append None to maintain alignment
            continue
        
        trace_file_sql = run_job(job)
        if trace_file_sql is None:
            raise Exception(f"No trace file found for job {job['id']}")
        
        # Parse trace and compute stats (returns aggregated sequence with aten_op already included)
        sequence = parse_trace_and_compute_stats(job, trace_file_sql, job["runs"], job["warmup"])
        
        if sequence is None:
            raise Exception(f"No kernel events found for job {job['id']}")
        
        # Attach job_id for downstream reporting
        sequence["job_id"] = job["id"]
        
        # Capture null kernel baseline tax for delta calculations
        if job["name"] == "__null__":
            null_launch_tax = sequence["launch_tax"]["avg"]
        
        sequences.append(sequence)
    
    baremetal_gemm_sequences_file = utils.get_path("BAREMETAL_GEMM_KERNELS")
    baremetal_gemm_sequences_data = {
        "summary": {"count": len(sequences)},
        "sequences": sequences,
    }
    utils.save_json(baremetal_gemm_sequences_file, baremetal_gemm_sequences_data)
    print(f"Saved {baremetal_gemm_sequences_data['summary']['count']} baremetal GEMM sequences to {baremetal_gemm_sequences_file}")
    
    # Print a summary table with % delta over null kernel 
    print_summary(sequences, null_launch_tax)
    
    return baremetal_gemm_sequences_data

def print_summary(sequences, null_launch_tax=None):
    """Print a compact table summary of profiled sequences, with % delta over null kernel."""
    table_data = []
    for sequence in sequences:

        # Skip None entries (skipped baremetal jobs)
        if sequence is None:
            continue

        avg = sequence["launch_tax"]["avg"]
        delta_pct = (
            (avg - null_launch_tax) / null_launch_tax * 100
            if null_launch_tax else 0.0
        )
        table_data.append([
            sequence["job_id"],
            sequence["aten_op"]["name"],
            sequence["kernel"]["name"],
            f"{avg:.2f} μs",
            f"{delta_pct:.1f}%",
        ])

    if table_data:
        print_utils.comp_table(
            title="Profile Summary",
            headers=["Job", "CPU Op", "Kernel", "Avg Launch Tax", "Δ(%)"],
            data=table_data,
        )
