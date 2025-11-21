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
                kernel_name = utils.clean_kernel_name(key[0])
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
