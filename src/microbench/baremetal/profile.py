#!/usr/bin/env python3
"""
Profile baremetal GEMM kernels using matched algorithms (profiling phase).

Reads jobs from baremetal/output/jobs.json, uses heur_idx from
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
from microbench.baremetal.utils import build_binary, build_base_args, nsys_profile, extract_kernels_sql, extract_launches_sql, extract_nvtx_ranges_sql as extract_nvtx_markers_sql

from soda.common import utils, print_utils
from soda.common.data import ATenOp


def run_job(job, warmup: int, runs: int):
    """
    Run a single job under nsys profiling and emit sqlite trace.
    """
    job_id = job["id"]
    
    print(f"Running job {job_id}")
    
    binary_path = utils.get_path("BAREMETAL_BINARY")
    utils.ensure_file(binary_path)

    # Handle null kernel job
    if job["name"] == "__null_kernel__":
        args = [
            str(binary_path),
            "--null_kernel",
            "--warmup", str(warmup),
            "--runs", str(runs),
        ]
    else:

        args = build_base_args(job) + [
            "--warmup", str(warmup),
            "--runs", str(runs),
        ]
        
        if "batch" in job:
            args = args + ["--batch", str(job["batch"])]

        if "heur_idx" in job:
            matched_algo_idx = job["heur_idx"]
            if matched_algo_idx is not None:
                args = args + ["--heuristic_index", str(matched_algo_idx)]
    
    trace_file_name = f"trace_{job_id}"
    success, trace_file_sql, message = nsys_profile(
        trace_file_name, args, timeout=600, cleanup=False
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
    # Extract CUDA launch events, kernel events, and NVTX ranges
    cuda_launches = extract_launches_sql(trace_file_sql)
    kernels = extract_kernels_sql(trace_file_sql)
    nvtx_markers = extract_nvtx_markers_sql(trace_file_sql)
    # utils.remove_file(trace_file_sql) # DEBUG 

    assert cuda_launches, f"No CUDA launch events found in {trace_file_sql}"
    assert kernels, f"No kernel events found in {trace_file_sql}"
    assert nvtx_markers, f"No NVTX markers found in {trace_file_sql}"
    
    # All kernels in this trace belong to the same job/aten_op
    # Handle null kernel vs regular GEMM jobs differently
    external_id = job["id"]
    if job["name"] == "__null_kernel__":
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

    # Attach cuBLASLt NVTX markers (culib) to each sequence when applicable,
    # then compute per-sequence taxes.
    if job["name"] != "__null_kernel__":
        culib_sequences = link_culib_sequences(nvtx_markers)
        annotate_sequences_with_culib_markers(
            linked_sequences,
            culib_sequences,
            expected_num_sequences=(warmup + runs),
        )
        metrics = ["launch_tax", "culib_xlat_tax", "culib_shim_tax"]
    else: 
        metrics = ["launch_tax"]

    linked_sequences_with_tax = utils.calculate_sequence_metrics(
        linked_sequences, metrics=metrics
    )
    
    # Skip warmup iterations by slicing
    sequences_after_warmup = linked_sequences_with_tax[warmup:]
    if not sequences_after_warmup:
        raise RuntimeError(f"No sequences remaining after warmup in {trace_file_sql}")
    
    # Aggregate using shared utility (same as PyTorch pipeline)
    # Deduplicate + aggregate by kernel signature (adds launch/cublib stats).
    grouped_seqs_by_id_dict = utils.group_sequences_by_identity(sequences_after_warmup)

    if job["name"] == "__null_kernel__":
        agg_metrics = ["launch_tax"]
        event_types = ["kernel", "aten_op", "cuda_launch"]
    else:
        agg_metrics = ["launch_tax", "culib_xlat_tax", "culib_shim_tax"]
        event_types = ["kernel", "aten_op", "cuda_launch", "culib"]

    aggregated_sequences = utils.aggregate_sequences(
        grouped_seqs_by_id_dict,
        metrics=agg_metrics,
        event_types=event_types,
    )
    
    # Baremetal runs same kernel config multiple times, so should have exactly 1 unique kernel
    if len(aggregated_sequences) != 1:
        kernel_names = [seq["kernel"]["name"] for seq in aggregated_sequences]
        raise RuntimeError(
            f"Expected 1 unique kernel after aggregation, found {len(aggregated_sequences)}: {kernel_names}"
        )
    
    # Return the single aggregated sequence
    # Format: {launch_tax: {...}, kernel: {...}, cuda_launch: {...}, aten_op: {...}}
    return aggregated_sequences[0]


def annotate_sequences_with_culib_markers(sequences, culib_sequences, expected_num_sequences):
    """
    Attach pre-linked cuBLASLt NVTX triplets (culib) to each sequence.
    """
    # Sanity: expect warmup + measured runs, and 1-1 mapping.
    assert (
        len(culib_sequences) == len(sequences) == expected_num_sequences
    ), f"culib/sequences length mismatch: {len(culib_sequences)} vs {len(sequences)} vs {expected_num_sequences}"

    for seq, culib in zip(sequences, culib_sequences):
        seq["culib"] = culib
        seq["culib_temp"] = culib["temperature"]

def get_culib_phase(name: str):
    """
    Parse a cuBLASLt NVTX name like 'lib:setup:cold' into (phase, temp).
    """
    _, phase, temp = name.split(":")
    return phase, temp


def link_culib_sequences(nvtx_markers):
    """
    Link per-run lib markers (setup, heur, run) by temperature and order.
    Returns a list of culib dicts aligned by time.
    """
    phases = ("setup", "heur", "run")
    markers_by_phase = {phase: [] for phase in phases}

    # Walk all markers once and bucket by phase.
    for marker in nvtx_markers:
        phase, _ = get_culib_phase(marker["name"])
        markers_by_phase[phase].append(marker)

    # Sort each phase by time.
    for phase in phases:
        markers_by_phase[phase].sort(key=lambda m: m["ts"])

    setup_markers = markers_by_phase["setup"]
    heur_markers = markers_by_phase["heur"]
    run_markers = markers_by_phase["run"]

    def strip(marker):
        return {
            "ts": marker["ts"],
            "dur_us": marker["dur_us"],
        }

    culib_sequences = []
    for setup, heur, run in zip(
        markers_by_phase["setup"], 
        markers_by_phase["heur"], 
        markers_by_phase["run"]
    ):
        _, temp = get_culib_phase(run["name"])

        # Basic per-range sanity.
        for phase, marker in (("setup", setup), ("heur", heur), ("run", run)):
            assert marker["dur_us"] >= 0.0, f"{phase} has negative duration: {marker['dur_us']}"

        # Ordering sanity: setup fully before heur, heur fully before run.
        setup_end = setup["ts"] + setup["dur_us"]
        heur_end = heur["ts"] + heur["dur_us"]
        run_end = run["ts"] + run["dur_us"]
        assert setup_end <= heur["ts"] <= heur_end <= run["ts"] <= run_end, (
            f"Unexpected cuBLASLt phase ordering: "
            f"setup=({setup['ts']},{setup_end}), "
            f"heur=({heur['ts']},{heur_end}), "
            f"run=({run['ts']},{run_end})"
        )

        culib_sequences.append({
            "temperature": temp,
            "setup": strip(setup),
            "heur": strip(heur),
            "run": strip(run),
        })

    return culib_sequences

def profile_baremetal_gemm_kernels(
    warmup: int,
    runs: int,
    skip_offline_cublas_algo_search: bool = False,
) -> Dict[str, Any]:
    """
    Profile baremetal GEMM kernels for all jobs using matched algorithms.
    
    Args:
        warmup: Number of warmup iterations to skip per job.
        runs: Number of measured iterations per job.
        skip_offline_cublas_algo_search:
            If True, allow profiling even when offline cuBLASLt algorithm
            search metadata is missing; baremetal will then rely on heuristic
            algorithm selection instead of matched heur_idx.
    
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
        offline_cublas_algo_search = jobs_data["summary"]["offline_cublas_algo_search"]
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
        
        trace_file_sql = run_job(job, warmup, runs)
        assert trace_file_sql is not None, f"No trace file found for job {job['id']}"
        
        # Parse trace and compute stats (returns aggregated sequence with aten_op already included)
        sequence = parse_trace_and_compute_stats(job, trace_file_sql, runs, warmup)
        assert sequence is not None, f"No kernel events found for job {job['id']}"

        # Attach job_id for downstream reporting
        sequence["job_id"] = job["id"]
        
        # Capture null kernel baseline tax for delta calculations
        null_launch_tax = sequence["launch_tax"]["avg"] if job["name"] == "__null_kernel__" else None
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
