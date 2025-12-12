#!/usr/bin/env python3
"""
Profile baremetal GEMM kernels using matched algorithms (profiling phase).

Reads jobs from baremetal/output/jobs.json, uses heur_idx from
algorithm matching phase, runs full nsys profiling, computes kernel launch
tax statistics, and emits baremetal/output/baremetal_gemm_runs.json.

By default, this phase assumes search_cublas_algos_offline.py has already
been run, but the caller can choose to skip that requirement.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from torch.profiler import profile, ProfilerActivity, record_function as record

from soda.microbench.baremetal.utils import (
    build_binary,
    build_base_args,
    nsys_profile,
    extract_kernels_sql,
    extract_launches_sql,
    extract_culib_markers_sql,
    link_culib_sequences,
    annotate_sequences_with_culib_phases,
)

from soda.common import utils, print_utils
from soda.common.data import ATenOp, clean_kernel_name

SUPPORTED_OPS = {
    # GEMM operations
    "aten::addmm": lambda inputs: torch.addmm(inputs[0], inputs[1], inputs[2]) if len(inputs) >= 3 else None,
    "aten::mm": lambda inputs: torch.mm(inputs[0], inputs[1]) if len(inputs) >= 2 else None,
    "aten::bmm": lambda inputs: torch.bmm(inputs[0], inputs[1]) if len(inputs) >= 2 else None,
    "aten::matmul": lambda inputs: torch.matmul(inputs[0], inputs[1]) if len(inputs) >= 2 else None,
    "aten::linear": lambda inputs: torch.nn.functional.linear(inputs[0], inputs[1], inputs[2] if len(inputs) > 2 else None) if len(inputs) >= 2 else None,
    
    # Elementwise operations
    "aten::add": lambda inputs: torch.add(inputs[0], inputs[1]) if len(inputs) >= 2 else inputs[0],
    "aten::mul": lambda inputs: torch.mul(inputs[0], inputs[1]) if len(inputs) >= 2 else inputs[0],
    "aten::div": lambda inputs: torch.div(inputs[0], inputs[1]) if len(inputs) >= 2 else inputs[0],
    "aten::sub": lambda inputs: torch.sub(inputs[0], inputs[1]) if len(inputs) >= 2 else inputs[0],
    
    # Reduction operations
    "aten::sum": lambda inputs: torch.sum(inputs[0]),
    "aten::mean": lambda inputs: torch.mean(inputs[0].float()).to(inputs[0].dtype),
    "aten::max": lambda inputs: torch.max(inputs[0]),
    "aten::min": lambda inputs: torch.min(inputs[0]),
    
    # Normalization
    "aten::layer_norm": lambda inputs: torch.nn.functional.layer_norm(inputs[0], inputs[0].shape[-1:]) if len(inputs) >= 1 else None,
    "aten::softmax": lambda inputs: torch.softmax(inputs[0], dim=-1) if len(inputs) >= 1 else None,
    "aten::_softmax": lambda inputs: torch.softmax(inputs[0], dim=-1) if len(inputs) >= 1 else None,
    
    # Activation functions
    "aten::relu": lambda inputs: torch.relu(inputs[0]),
    "aten::gelu": lambda inputs: torch.nn.functional.gelu(inputs[0]),
    "aten::silu": lambda inputs: torch.nn.functional.silu(inputs[0]),
    "aten::tanh": lambda inputs: torch.tanh(inputs[0]),
    "aten::sigmoid": lambda inputs: torch.sigmoid(inputs[0]),
    
    # Reshape operations
    "aten::view": lambda inputs: inputs[0].view(-1) if len(inputs) >= 1 else None,
    "aten::reshape": lambda inputs: inputs[0].reshape(-1) if len(inputs) >= 1 else None,
    "aten::transpose": lambda inputs: inputs[0].transpose(0, 1) if len(inputs) >= 1 and inputs[0].dim() >= 2 else inputs[0],
    "aten::permute": lambda inputs: inputs[0].permute(*range(inputs[0].dim()-1, -1, -1)) if len(inputs) >= 1 else None,
    "aten::contiguous": lambda inputs: inputs[0].contiguous(),
    "aten::flatten": lambda inputs: inputs[0].flatten(),
    
    # Copy/memory operations
    "aten::copy_": lambda inputs: inputs[0].clone() if len(inputs) >= 1 else None,
    "aten::clone": lambda inputs: inputs[0].clone(),
    "aten::to": lambda inputs: inputs[0].clone(),
    "aten::_to_copy": lambda inputs: inputs[0].clone(),
    
    # Embedding operations
    "aten::embedding": lambda inputs: torch.nn.functional.embedding(inputs[0].long() % 1000, torch.randn(1000, inputs[1].shape[-1] if len(inputs) > 1 else 768, device=inputs[0].device)) if len(inputs) >= 1 else None,
    
    # Attention-related (common on H100/H200)
    "aten::scaled_dot_product_attention": lambda inputs: torch.nn.functional.scaled_dot_product_attention(inputs[0], inputs[1], inputs[2]) if len(inputs) >= 3 else None,
    "aten::_scaled_dot_product_flash_attention": lambda inputs: torch.nn.functional.scaled_dot_product_attention(inputs[0], inputs[1], inputs[2]) if len(inputs) >= 3 else None,
}


def is_op_supported(aten_op_name: str) -> bool:
    """Check if an ATen operation is supported for replay."""
    if aten_op_name in SUPPORTED_OPS:
        return True
    # Try base name match (strip aten:: and trailing _)
    op_base = aten_op_name.split("::")[-1].rstrip("_")
    return any(op_base in supported_op for supported_op in SUPPORTED_OPS)


def execute_any_operation(
    aten_op_name: str,
    inputs: List[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Execute any supported ATen operation."""
    if aten_op_name in SUPPORTED_OPS:
        with record(f"torch_op:{aten_op_name}"):
            return SUPPORTED_OPS[aten_op_name](inputs)
    
    # Fallback: try base name match
    op_base = aten_op_name.split("::")[-1].rstrip("_")
    for supported_op, func in SUPPORTED_OPS.items():
        if op_base in supported_op:
            with record(f"torch_op:{aten_op_name}"):
                return func(inputs)
    
    raise ValueError(f"Unsupported operation: {aten_op_name}")


def profile_any_operation(
    aten_op_name: str,
    inputs: List[torch.Tensor],
    warmup: int,
    runs: int,
    trace_file: Path,
) -> None:
    """Profile any supported PyTorch operation N times."""
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            try:
                execute_any_operation(aten_op_name, inputs)
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    # Profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with torch.no_grad():
            for _ in range(runs):
                try:
                    execute_any_operation(aten_op_name, inputs)
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
    
    prof.export_chrome_trace(str(trace_file))


def replay_all_sequences_from_aten_ops(
    sequences: List[Dict[str, Any]], 
    warmup: int,
    runs: int
) -> List[Dict[str, Any]]:
    """Replay all event sequences (GEMM and non-GEMM)."""
    kernel_traces_dir = utils.get_path("PYTORCH_TRACES")
    utils.ensure_dir(kernel_traces_dir)

    sequence_by_idx = {}
    supported_count = 0
    skipped_count = 0
    
    print(f"Profiling {len(sequences)} PyTorch kernels with {runs} run{'s' if runs > 1 else ''} each (warmup={warmup})")
    
    for i, event_sequence in enumerate(sequences):
        aten_op = event_sequence["aten_op"]
        kernel = event_sequence["kernel"]
        aten_op_name = aten_op["name"]
        expected_kernel = clean_kernel_name(kernel["name"])
        seq_idx = i + 1
        is_gemm = event_sequence.get("is_gemm", False)
        
        # Check if operation is supported
        if not is_op_supported(aten_op_name):
            print(f"[{seq_idx}/{len(sequences)}] SKIP (unsupported): {aten_op_name} -> {expected_kernel}")
            skipped_count += 1
            continue
        
        kernel_type = "GEMM" if is_gemm else "other"
        print(f"[{seq_idx}/{len(sequences)}] [{kernel_type}] {aten_op_name} -> {expected_kernel}")
        
        # Generate trace filename
        trace_file_name = utils.format_sequence_filename(
            seq_idx, 
            aten_op['name'], 
            expected_kernel, 
            extension="json"
        )
        trace_file = kernel_traces_dir / trace_file_name

        try:
            # Create input tensors
            inputs = create_input_tensors(aten_op)
            
            # Profile the operation
            profile_any_operation(
                aten_op_name=aten_op_name,
                inputs=inputs,
                warmup=warmup,
                runs=runs,
                trace_file=trace_file,
            )

            # Load and process trace
            trace_data = utils.load_json(trace_file)
            events = utils.collect_events(trace_data)
            linked_sequences = utils.link_sequences(events)
            linked_sequences_with_tax = utils.calculate_sequence_metrics(
                linked_sequences, 
                metrics=["launch_tax", "aten_xlat_tax", "py_tax"]
            )

            grouped_seqs_by_id_dict = utils.group_sequences_by_identity(linked_sequences_with_tax)
            agg_sequence = utils.aggregate_sequences(
                grouped_seqs_by_id_dict,
                metrics=["launch_tax", "aten_xlat_tax", "py_tax"],
                event_types=["kernel", "aten_op", "cuda_launch", "torch_op"],
            )
            
            # Mark GEMM status
            for seq in agg_sequence:
                seq["is_gemm"] = is_gemm
            
            sequence_by_idx[i] = agg_sequence
            supported_count += 1
            
        except Exception as e:
            print(f"  Error profiling {aten_op_name}: {e}")
            skipped_count += 1
            continue

    # Flatten results
    all_replayed_sequences = []
    for kernel_idx in sorted(sequence_by_idx.keys()):
        all_replayed_sequences.extend(sequence_by_idx[kernel_idx])

    print(f"Successfully profiled {supported_count} operations, skipped {skipped_count} unsupported")
    
    if all_replayed_sequences:
        utils.validate_sequences(all_replayed_sequences)
    
    return all_replayed_sequences


def profile_pytorch_all_sequences(
    target_sequences: Dict[str, Any],
    warmup: int,
    runs: int
) -> Dict[str, Any]:
    """Profile all PyTorch kernel sequences (GEMM and non-GEMM)."""
    kernel_traces_dir = utils.get_path("PYTORCH_TRACES")
    utils.ensure_dir(kernel_traces_dir, cleanup=True)
    
    replayed_sequences = replay_all_sequences_from_aten_ops(
        target_sequences["sequences"],
        warmup=warmup, 
        runs=runs
    )
    
    # Match frequencies from target sequences
    target_seqs = target_sequences["sequences"]
    for i, replayed_seq in enumerate(replayed_sequences):
        if i < len(target_seqs):
            replayed_seq["freq"] = target_seqs[i].get("count", 1)

    # Count GEMM vs non-GEMM
    gemm_count = sum(1 for s in replayed_sequences if s.get("is_gemm", False))
    non_gemm_count = len(replayed_sequences) - gemm_count

    pytorch_all_sequences_file = utils.get_path("PYTORCH_ALL_SEQUENCES")
    pytorch_all_sequences_data = {
        "summary": {
            "count": len(replayed_sequences),
            "gemm_count": gemm_count,
            "non_gemm_count": non_gemm_count,
        },
        "sequences": replayed_sequences
    }
    utils.save_json(pytorch_all_sequences_file, pytorch_all_sequences_data)
    print(f"Saved {len(replayed_sequences)} PyTorch sequences to {pytorch_all_sequences_file}")
    print(f"  - {gemm_count} GEMM sequences")
    print(f"  - {non_gemm_count} non-GEMM sequences")
    
    return pytorch_all_sequences_data

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
    ################################################################################
    # Extract CUDA launch events, kernel events, and NVTX ranges
    cuda_launches = extract_launches_sql(trace_file_sql)
    kernels = extract_kernels_sql(trace_file_sql)
    culib_markers = extract_culib_markers_sql(trace_file_sql)
    # utils.remove_file(trace_file_sql) # DEBUG 

    if not kernels:
        print(f"Warning: No kernel events found in {trace_file_sql}. This job may have failed to execute.")
        print(f"  - Job ID: {job.get('id')}")
        print(f"  - heur_idx: {job.get('heur_idx')}")
        print(f"  - This can happen if no matching cuBLAS algorithm was found for this kernel.")
        return None

    assert cuda_launches, f"No CUDA launch events found in {trace_file_sql}"
    assert kernels, f"No kernel events found in {trace_file_sql}"
    assert culib_markers, f"No cuBLASLt markers found in {trace_file_sql}"
    
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
        marker_phases = ["setup", "heur", "run"]
        metrics = ["launch_tax", "culib_xlat_tax", "shim_tax"]
    else: 
        marker_phases = ["run"]
        # Null kernel job: no cuBLASLt library translation
        # NOTE: The notion of shim tax is still valid, however, its defined as 
        # shim_tax = launch - run where run is the time __null_kernel__ is called.
        metrics = ["launch_tax", "shim_tax"]

    # Link cuBLASLt markers to create culib sequences
    culib_sequences = link_culib_sequences(culib_markers, marker_phases)

    # Attach culib sequences to recently profiled sequences
    annotate_sequences_with_culib_phases(
        linked_sequences,
        culib_sequences,
        expected_num_sequences=(warmup + runs),
    )

    # Compute per-sequence taxes
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
        agg_metrics = ["launch_tax", "shim_tax"]
        event_types = ["kernel", "aten_op", "cuda_launch", "culib"]
    else:
        agg_metrics = ["launch_tax", "culib_xlat_tax", "shim_tax"]
        event_types = ["kernel", "aten_op", "cuda_launch", "culib"]

    aggregated_sequences = utils.aggregate_sequences(
        grouped_seqs_by_id_dict,
        metrics=agg_metrics,
        event_types=event_types,
    )

    # Baremetal runs same kernel config multiple times, so should have exactly 1 unique kernel
    assert len(aggregated_sequences) == 1, f"Expected 1 unique kernel after aggregation, found {len(aggregated_sequences)}"
    
    # Return the single aggregated sequence
    # Format: {launch_tax: {...}, kernel: {...}, cuda_launch: {...}, aten_op: {...}}
    return aggregated_sequences[0]

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
    
    matched_jobs = [j for j in jobs if j.get("heur_idx") is not None and j.get("name") != "__null_kernel__"]
    print(f"Using {len(matched_jobs)} matched algorithms")
    
    # Build binary
    print("Building C++ binary")
    build_binary()
    print("Build successful")
    
    results = []
    binary_path = utils.get_path("BAREMETAL_BINARY")
    utils.ensure_file(binary_path)
    
    # Run each job and collect sequences
    sequences = []
    for job in jobs:

        # Skip batched GEMMs for now due to non-contiguous layout issue
        # FIXME: Implement batched GEMM support in cublaslt backend 
        if "batch" in job and job["batch"] > 1:
            print(f"Skipping job {job['id']} (TODO: batched GEMM)")
            sequences.append(None)  # Append None to maintain alignment
            continue

        # FIX: Check heur_idx BEFORE running the job, not after
        # This prevents running jobs that will produce empty traces
        if job.get("heur_idx") is None and job.get("name") != "__null_kernel__":
            print(f"Skipping job {job['id']}: No cuBLAS algorithm found (PyTorch internal kernel).")
            print(f"  - Kernel '{job.get('name')}' is not a cuBLAS kernel.")
            print(f"  - This is common on H100/H200/GB200 where PyTorch uses optimized internal kernels.")
            sequences.append(None)  # Append None to maintain alignment
            continue
        
        # Now safe to run the job
        trace_file_sql = run_job(job, warmup, runs)
        if trace_file_sql is None:
            print(f"Warning: No trace file generated for job {job['id']}. Skipping.")
            sequences.append(None)
            continue
        
        # Parse trace and compute stats (returns aggregated sequence with aten_op already included)
        sequence = parse_trace_and_compute_stats(job, trace_file_sql, runs, warmup)
        if sequence is None:
            print(f"Warning: Could not parse trace for job {job['id']}. Skipping.")
            sequences.append(None)
            continue

        # Attach job_id for downstream reporting
        sequence["job_id"] = job["id"]

        sequences.append(sequence)
    
    # Filter out None entries for the saved output
    valid_sequences = [s for s in sequences if s is not None]
    
    baremetal_gemm_sequences_file = utils.get_path("BAREMETAL_GEMM_KERNELS")
    baremetal_gemm_sequences_data = {
        "summary": {"count": len(valid_sequences)},
        "sequences": valid_sequences,
    }
    utils.save_json(baremetal_gemm_sequences_file, baremetal_gemm_sequences_data)
    print(f"Saved {baremetal_gemm_sequences_data['summary']['count']} baremetal GEMM sequences to {baremetal_gemm_sequences_file}")
    
    # Print a summary table with % delta over null kernel 
    print_summary(valid_sequences)
    
    return baremetal_gemm_sequences_data



def print_summary(sequences):
    """Print a compact table summary of profiled sequences, with % delta over null kernel."""
    # Derive null launch tax from sequences if present.
    null_launch_tax = None
    for seq in sequences:
        if seq["kernel"]["name"] == "__null_kernel__":
            null_launch_tax = seq["launch_tax"]["avg"]
            break

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
