"""
SODA utility functions
"""

import argparse
import json
import os
import sys
import re
import copy
import torch
from pathlib import Path
from typing import Any, Dict
from collections import defaultdict
import numpy as np
import shutil

def calculate_avg_min_max(values, base_name=None):
    """
    Calculate avg/min/max from a list of values. If base_name is provided,
    the keys are suffixed with it (e.g., avg_kernel_tax); otherwise the keys
    are plain avg/min/max.
    """
    if not values:
        return {}

    avg = sum(values) / len(values)
    min_val = min(values)
    max_val = max(values)

    suffix = f"_{base_name}" if base_name else ""

    result = {
        f"avg{suffix}": avg,
        f"min{suffix}": min_val,
        f"max{suffix}": max_val,
    }

    return result

def clean_kernel_name(kernel_name: str) -> str:
    """
    Extract a clean kernel name from the full signature.
    
    Args:
        kernel_name: Full kernel name (may be a C++ function signature).
    
    Returns:
        Clean kernel name (just the kernel name, no namespace or template parameters).
    
    Examples:
        "void at::native::vectorized_elementwise_kernel<4, ...>" 
        -> "vectorized_elementwise_kernel"
        
        "void at::native::(anonymous namespace)::elementwise_kernel<...>"
        -> "elementwise_kernel"
        
        "turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn"
        -> "turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn"
    """
    # Extract everything before '<' (removes template parameters)
    # This handles cases where '(' appears in template params like "(anonymous namespace)"
    if '<' in kernel_name:
        clean_kernel_name = kernel_name.split('<')[0].strip()
    elif '(' in kernel_name:
        # If no '<' but has '(', extract before '(' (function parameters)
        clean_kernel_name = kernel_name.split('(')[0].strip()
    else:
        clean_kernel_name = kernel_name
    
    # Remove 'void' prefix if present
    clean_kernel_name = clean_kernel_name.replace('void', '').strip()
    
    # Extract just the kernel name (last part after '::')
    if '::' in clean_kernel_name:
        clean_kernel_name = clean_kernel_name.split('::')[-1]
    
    return clean_kernel_name.strip()

def us_to_ms(microseconds: float) -> float:
    """
    Convert a duration from microseconds to milliseconds.
    
    Args:
        microseconds: Duration expressed in microseconds.
    
    Returns:
        The same duration expressed in milliseconds.
    """
    return microseconds / 1000.0

def get_path(env_var: str) -> Path:
    """
    Get path from environment variable.
    
    Args:
        env_var: Environment variable name.
    
    Returns:
        Path object from environment variable.
    """
    return Path(os.environ[env_var])

def ensure_dir(path) -> None:
    """
    Ensure directory exists, creating parent directories if needed.
    
    Args:
        path: Path or string to directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

def load_json(file_path) -> Dict[str, Any]:
    """
    Load JSON file.
    
    Args:
        file_path: Path to JSON file.
    
    Returns:
        Dictionary loaded from JSON file.
    
    Raises:
        FileNotFoundError: If file does not exist.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File does not exist: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(file_path: str, data: Dict[str, Any], indent: int = 2) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        file_path: Path to output file.
        data: Dictionary to save.
        indent: JSON indentation (default: 2).
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

# GEMM operations to extract
GEMM_OPS = ['aten::addmm', 'aten::mm', 'aten::bmm']

def filter_gemm_sequences(event_sequences):
    """Filter for GEMM kernels only."""
    gemm_sequences = []
    for e in event_sequences:
        # Must be from a GEMM operation
        if e.get('cpu_op') and e['cpu_op']['name'] in GEMM_OPS:
            # Kernel name must contain 'gemm' (eg: volta_sgemm_64x32_sliced1x4_nn)
            if 'gemm' in e.get('kernel', {}).get('name', '').lower():
                gemm_sequences.append(copy.deepcopy(e))
    return gemm_sequences

def make_kernel_identity_key(kernel, input_dims):
    """
    Build a stable identity key for a kernel + its originating CPU op.
    Components:
      - kernel name
      - grid dims
      - block dims
      - shared memory
      - input dims (tuplized to make hashable)
    """
    kernel_name = kernel.get("name")
    grid_tuple = tuple(kernel.get("grid", []))
    block_tuple = tuple(kernel.get("block", []))
    shared_mem = kernel.get("shared_memory", 0)
    input_dims_tuple = tuple(tuple(d) if isinstance(d, list) else d for d in input_dims)
    return (kernel_name, grid_tuple, block_tuple, shared_mem, input_dims_tuple)

def group_sequences_by_identity(sequences):
    """
    Group event sequences by kernel identity key.
    Returns a dict: key -> list[sequence]
    """
    grouped = defaultdict(list)
    for e in sequences:
        if e.get("kernel") and e.get("cpu_op"):
            kernel = e["kernel"]
            input_dims = e["cpu_op"]["input_dims"]
            key = make_kernel_identity_key(kernel, input_dims)
            grouped[key].append(e)
    return grouped

def aggregate_execution_metrics(instances):
    """
    Aggregate execution metrics across a group of identical event sequences.
    All time values in microseconds.
    Produces avg/min/max for:
      - kernel.dur
      - cpu_op.dur
      - cuda_launch.dur
      - kernel_tax
    Returns a deep-copied, aggregated base instance.
    """
    base_instance = copy.deepcopy(instances[0])
    
    # Aggregate kernel duration
    kernel_durations = [
        inst.get("kernel", {}).get("dur")
        for inst in instances
        if inst.get("kernel", {}).get("dur") is not None
    ]
    if kernel_durations:
        base_instance["kernel"].update(calculate_avg_min_max(kernel_durations, "dur"))
        base_instance["kernel"]["all_dur"] = kernel_durations
    
    # Aggregate cpu_op duration
    cpu_op_durations = [
        inst.get("cpu_op", {}).get("dur")
        for inst in instances
        if inst.get("cpu_op", {}).get("dur") is not None
    ]
    if cpu_op_durations:
        base_instance["cpu_op"].update(calculate_avg_min_max(cpu_op_durations, "dur"))
        base_instance["cpu_op"]["all_dur"] = cpu_op_durations
    
    # Aggregate cuda_launch duration
    cuda_launch_durations = [
        inst.get("cuda_launch", {}).get("dur")
        for inst in instances
        if inst.get("cuda_launch", {}).get("dur") is not None
    ]
    if cuda_launch_durations:
        base_instance["cuda_launch"].update(calculate_avg_min_max(cuda_launch_durations, "dur"))
        base_instance["cuda_launch"]["all_dur"] = cuda_launch_durations
    
    # Aggregate kernel_tax
    kernel_tax_values = [
        inst.get("kernel_tax")
        for inst in instances
        if inst.get("kernel_tax") is not None
    ]

    meta = {"count": len(instances), "all_kernel_tax": kernel_tax_values}
    meta.update(calculate_avg_min_max(kernel_tax_values, "kernel_tax"))
    base_instance["meta"] = meta

    # Clean up unneeded fields
    base_instance.pop("kernel_tax")
    base_instance["kernel"].pop("dur")
    base_instance["cpu_op"].pop("dur")
    base_instance["cuda_launch"].pop("dur")
    base_instance["kernel"].pop("ts")
    base_instance["cpu_op"].pop("ts")
    base_instance["cuda_launch"].pop("ts")
    
    return base_instance

def deduplicate_and_aggregate(sequences):
    """Deduplicate sequences by identity key, validate static properties, and aggregate metrics."""
    # Group by identity
    grouped_sequences = group_sequences_by_identity(sequences)
    
    unique_gemm_sequences = []
    for key, instances in grouped_sequences.items():
        
        # Verify static properties
        base_instance = copy.deepcopy(instances[0])
        base_kernel = base_instance['kernel']
        
        # Verify static properties
        static_props = ['shared_memory', 'registers_per_thread', 'occupancy', 'blocks_per_SM', 'warps_per_SM', 'stream', 'device', 'context']
        for prop in static_props:
            values = [inst['kernel'].get(prop) for inst in instances if inst['kernel'].get(prop) is not None]
            if values and len(set(values)) > 1:
                kernel_name = clean_kernel_name(key[0])
                raise AssertionError(f"Static property '{prop}' inconsistent for kernel {kernel_name}: {set(values)} (across {len(instances)} instances)")

        # Aggregate execution metrics (kernel tax, cpu op duration, cuda launch duration, kernel duration)
        base_instance = aggregate_execution_metrics(instances)
        
        # Remove timestamp
        base_instance['kernel'].pop('ts', None)
        
        # Construct output structure 
        sequence_entry = {
            "meta": base_instance.get('meta'),
            "kernel": base_instance['kernel'],
            "cuda_launch": base_instance.get('cuda_launch'),
            "cpu_op": base_instance.get('cpu_op')
        }
        
        unique_gemm_sequences.append(sequence_entry)
    
    return unique_gemm_sequences

def get_args_parser() -> argparse.ArgumentParser:
    """Create and return argument parser."""
    parser = argparse.ArgumentParser(
        description="SODA: System Offload Dynamics Analyzer. Analyze CPU–GPU dynamics of PyTorch models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Hugging Face model name or path for profiling and analysis.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        dest="output_dir",
        default=Path(os.environ.get("SODA_OUTPUT", ".")),
        help="Output directory for analysis artifacts (traces, reports, etc.)",
    )
    parser.add_argument(
        "-c",
        "--compile-type",
        dest="compile_type",
        default="eager",
        choices=["eager", "torch.compile", "flash-attention"],
        help="Execution mode for the model.",
    )
    parser.add_argument(
        "-d", "--device", default="cuda", choices=["cpu", "cuda"], 
        help="Device to run the model on."
    )
    parser.add_argument(
        "-p",
        "--precision",
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Precision for model weights and operations",
    )
    parser.add_argument(
        "-sl", "--seq-len", dest="seq_len", type=int, default=512, 
        help="Sequence length for synthetic input."
    )
    parser.add_argument(
        "-bs", "--batch-size", dest="batch_size", type=int, default=1, 
        help="Batch size for synthetic input."
    )
    parser.add_argument(
        "-f",
        "--fusion",
        nargs="+",
        type=int,
        help="List of kernel chain lengths to analyze for fusion opportunities.",
    )
    parser.add_argument(
        "-ps",
        "--prox-score",
        dest="prox_score",
        type=float,
        default=1.0,
        help="Proximity score threshold (0.0 to 1.0) for fusion recommendations.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s 0.1.0"
    )
    
    return parser

def parse_and_validate_args(args=None) -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = get_args_parser()
    parsed_args = parser.parse_args(args)
    
    # Validate arguments
    if parsed_args.device == "cpu" and parsed_args.precision == "float16":
        print("Warning: float16 is not supported on CPU. Forcing float32.")
        parsed_args.precision = "float32"

    if not torch.cuda.is_available() and parsed_args.device == "cuda":
        print("Error: CUDA is not available. Please select --device cpu.", file=sys.stderr)
        sys.exit(1)
    
    return parsed_args

def setup_deterministic_mode(seed=1234):
    """
    Lock down all non-determinism knobs for reproducible kernel selection.
    Sets PyTorch flags and environment variables to minimize randomness.
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Core determinism flags
    torch.backends.cudnn.benchmark = False  # Disable autotuner
    torch.backends.cudnn.deterministic = True  # Force deterministic algos
    
    # Disable TF32 for bitwise reproducibility
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    # Set matmul precision
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass
    
    # Use deterministic algorithms
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        # Some ops don't have deterministic variants, continue anyway
        pass
    
    # Disable FP16 reduced precision reduction
    try:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    except AttributeError:
        pass
    
    # Set environment variables for deterministic behavior
    os.environ["PYTORCH_JIT"] = "0"  # Disable JIT fusion randomness
    os.environ["PYTORCH_DISABLE_NVFUSER"] = "1"  # Disable nvFuser
    os.environ["TORCH_COMPILE_DISABLE"] = "1"  # Disable Torch.compile
    
    # Deterministic cuBLAS
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["CUBLASLT_ALLOW_TF32"] = "0"
    
    # cuDNN algo finder
    os.environ["CUDNN_FIND_MODE"] = "DEFAULT"
    
    # CUDA device connections
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "8"  # Default, but explicit
    
    # Clear cache for consistent memory layout
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def collect_env_metadata():
    """
    Collect only metadata that directly affects kernel selection determinism.
    """
    metadata = {
        # Version info for verification (same stack = same kernels)
        "torch_version": torch.__version__,
        "cuda_toolkit_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }
    
    if torch.cuda.is_available():
        metadata.update({
            "device_name": torch.cuda.get_device_name(0),
            "sm_capability": torch.cuda.get_device_capability(0),
        })
        # Driver/runtime versions can affect JIT/PTX compilation
        try:
            metadata["driver_version"] = torch.cuda.driver_version()
        except (AttributeError, RuntimeError):
            metadata["driver_version"] = None
        try:
            metadata["runtime_version"] = torch.cuda.runtime_version()
        except (AttributeError, RuntimeError):
            metadata["runtime_version"] = None
    else:
        metadata.update({
            "driver_version": None,
            "runtime_version": None,
        })
    
    # Backend settings that directly affect kernel selection
    metadata["cudnn"] = {
        "benchmark": torch.backends.cudnn.benchmark,
        "deterministic": torch.backends.cudnn.deterministic,
        "allow_tf32": torch.backends.cudnn.allow_tf32,
        "algo_finder": os.environ.get("CUDNN_FIND_MODE", "DEFAULT"),
    }
    
    # Matmul precision (affects kernel selection)
    try:
        metadata["matmul_precision"] = torch.get_float32_matmul_precision()
    except AttributeError:
        metadata["matmul_precision"] = None
    
    # Matmul TF32 setting (critical for GEMM kernel selection on Ampere+)
    try:
        metadata["matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    except AttributeError:
        metadata["matmul_allow_tf32"] = None
    
    # cuBLAS/cuBLASLt configuration (affects kernel selection)
    metadata["blas"] = {
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cublaslt_allow_tf32": os.environ.get("CUBLASLT_ALLOW_TF32"),
    }
    
    # AMP/autocast (can silently flip dtypes and thus kernels)
    metadata["amp"] = {
        "autocast_enabled": torch.is_autocast_enabled() if hasattr(torch, "is_autocast_enabled") else False,
    }
    
    # Determinism and reproducibility
    metadata["seeds"] = {
        "torch_manual_seed": torch.initial_seed(),
        "cuda_deterministic_algorithms": torch.are_deterministic_algorithms_enabled() if hasattr(torch, "are_deterministic_algorithms_enabled") else None,
    }
    
    # Environment variables that affect kernel selection
    metadata["env"] = {
        "PYTORCH_JIT": os.environ.get("PYTORCH_JIT"),
        "PYTORCH_DISABLE_NVFUSER": os.environ.get("PYTORCH_DISABLE_NVFUSER"),
        "TORCH_COMPILE_DISABLE": os.environ.get("TORCH_COMPILE_DISABLE"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    
    return metadata


def generate_summary(data, **extra):
    """Generate default summary from event sequences data.
    
    Args:
        data: List of event sequences.
        **extra: Additional summary fields to include.
    
    Returns:
        Summary dictionary with default fields plus any additional fields.
    """
    summary = {
        "total_kernels": len([c for c in data if c.get("kernel")]),
        "total_cpu_ops": len([c for c in data if c.get("cpu_op")]),
        "total_cuda_launches": len([c for c in data if c.get("cuda_launch")]),
        "total_sequences": len(data)
    }
    summary.update(extra)  # Unwrap additional fields
    return summary

def run_extraction_pipeline(trace_file, event_sequences):
    """
    Main pipeline: extract -> save, filter -> save, deduplicate -> save.
    """
    trace_file = Path(trace_file)
    
    # Save model trace to traces/model_trace folder
    model_trace_dir = get_path("PYTORCH_MODEL_TRACE_DIR")
    ensure_dir(model_trace_dir)

    
    model_trace_file = get_path("PYTORCH_MODEL_TRACE_FILE")
    shutil.copy2(trace_file, model_trace_file)
    
    # Collect environment metadata once
    env_metadata = collect_env_metadata()
    
    # Save env_metadata separately
    env_metadata_file = get_path("PYTORCH_ENV_METADATA")
    ensure_dir(env_metadata_file.parent)
    save_json(env_metadata_file, env_metadata)
    
    # Step 1: Save all event sequences 
    save_json(get_path("PYTORCH_ALL_KERNELS"), {
        "summary": generate_summary(event_sequences),
        "sequences": event_sequences
    })
    
    # Step 2: Filter GEMM event sequences
    gemm_sequences = filter_gemm_sequences(event_sequences)
    save_json(get_path("PYTORCH_GEMM_KERNELS"), {
        "summary": generate_summary(gemm_sequences),
        "sequences": gemm_sequences
    })
    
    # Step 3: Create unique event sequences
    unique_gemm_sequences = deduplicate_and_aggregate(gemm_sequences)
    save_json(get_path("PYTORCH_UNIQUE_KERNELS"), {
        "summary": generate_summary(
            unique_gemm_sequences,
            original_count=len(gemm_sequences),
            deduplication_ratio=f"{len(unique_gemm_sequences)}/{len(gemm_sequences)}"
        ),
        "sequences": unique_gemm_sequences
    })
    
    print_summary()
    
    return {
        "sequences": event_sequences,
        "env_metadata": env_metadata
    }

def print_summary():
    """Print extraction summary from saved JSON files."""
    all_kernels_file = get_path("PYTORCH_ALL_KERNELS")
    all_data = load_json(all_kernels_file)
    print(f"Extracted {all_data['summary']['total_kernels']} kernels")
    print(f"Linked {all_data['summary']['total_sequences']} event sequences")
    print(f"Saved to {all_kernels_file}")
    
    gemm_kernels_file = get_path("PYTORCH_GEMM_KERNELS")
    gemm_data = load_json(gemm_kernels_file)
    print(f"Saved {gemm_data['summary']['total_kernels']} GEMM event sequences to {gemm_kernels_file}")
    
    unique_kernels_file = get_path("PYTORCH_UNIQUE_KERNELS")
    unique_data = load_json(unique_kernels_file)
    print(f"Saved {unique_data['summary']['total_kernels']} unique GEMM event sequences to {unique_kernels_file}")
    print(f"\t* Uniqueness key: kernel_name + grid + block + shared_memory + input_dims")

# This module is now a library - extraction is integrated into SodaProfiler.analyze()
# Functions exported for use by other modules:
# - collect_env_metadata(): Collect environment metadata
# - setup_deterministic_mode(): Setup deterministic mode (used by replay)
# - generate_summary(): Generate summary from event sequences (used by replay)
# - run_extraction_pipeline(): Main extraction pipeline (called from SodaProfiler.analyze())
# - print_summary(): Print extraction summary from saved JSON files