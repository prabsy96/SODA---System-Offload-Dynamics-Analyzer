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
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque
import numpy as np
import shutil
from . import print_utils
from .data import Kernel, ATenOp, Sequence, clean_kernel_name

def calculate_avg_min_max(values, base_name=None):
    """
    Calculate avg/min/max from a list of values. If base_name is provided,
    the keys are suffixed with it (e.g., avg_launch_tax); otherwise the keys
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

def summarize_metric(values: List[float]) -> Dict[str, Any]:
    """
    Build a summary dict with count, all samples, and avg/min/max stats.
    Assumes values is non-empty.
    """
    summary = {
        "count": len(values),
        "all": list(values),
    }
    summary.update(calculate_avg_min_max(values))
    return summary

def format_sequence_filename(index: int, op_name: str, kernel_name: str, extension: str = "png") -> str:
    """
    Format a filename for a sequence file (trace, plot, etc.) using index, op name, and kernel name.
    
    Args:
        index: Sequence index (1-based).
        op_name: CPU operation name (e.g., "aten::addmm").
        kernel_name: Kernel name.
        extension: File extension (default: "png").
    
    Returns:
        Formatted filename: "{index:02d}_{op_short}_{kernel_short}.{extension}"
    """
    op_short = op_name.replace("::", "_")
    kernel_short = clean_kernel_name(kernel_name).strip()
    return f"{index:02d}_{op_short}_{kernel_short}.{extension}"


def parse_dtype_to_cublaslt(dtype_str: str) -> str:
    """Map PyTorch dtype strings to cuBLASLt dtype codes.
    
    Args:
        dtype_str: PyTorch dtype string (e.g., "float32", "float16", "half")
    
    Returns:
        cuBLASLt dtype code (e.g., "f32", "f16", "bf16", "f64")
    """
    dtype_map = {
        "float": "f32",
        "float32": "f32",
        "half": "f16",
        "float16": "f16",
        "bfloat16": "bf16",
        "double": "f64",
        "float64": "f64",
        "float8_e4m3fn": "f8",
    }

    # NOTE: Hack for c10::BFloat16 type strings
    dtype_str = dtype_str.replace("c10::", "").lower()

    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported data type: '{dtype_str}'. Supported types: {list(dtype_map.keys())}")
    
    return dtype_map[dtype_str.lower()]

def parse_dtype_to_torch(dtype_str: str):
    """Map dtype strings to torch.dtype objects.
    
    Args:
        dtype_str: Dtype string (e.g., "float32", "float16", "half", "int32")
    
    Returns:
        torch.dtype object
    """

    dtype_map = {
        "float": torch.float32,
        "float32": torch.float32,
        "half": torch.float16,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float64": torch.float64,
        "double": torch.float64,
        "int32": torch.int32,
        "int64": torch.int64,
        "long": torch.int64,
    }

    # Add FP8 E4M3 mapping if available in this torch build
    if hasattr(torch, "float8_e4m3fn"):
        dtype_map["float8_e4m3fn"] = torch.float8_e4m3fn
    elif dtype_str.replace("c10::", "").lower() == "float8_e4m3fn":
        raise ValueError("Unsupported data type: 'float8_e4m3fn'. Please upgrade to PyTorch 2.1+ for FP8 support.")
    
    # NOTE: Hack for c10::BFloat16 type strings
    dtype_str = dtype_str.replace("c10::", "").lower()
    
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported data type: '{dtype_str}'. Supported types: {list(dtype_map.keys())}")
    
    return dtype_map[dtype_str]

def get_sequence_str(sequence: Dict[str, Any]) -> str:
    """
    Build a sequence string from a sequence dictionary.
    
    Args:
        sequence: Dictionary containing 'aten_op' and 'kernel' keys.
    
    Returns:
        Formatted sequence string: "{op_name} -> {kernel_name}".
    """
    aten_op = sequence['aten_op']
    kernel = sequence['kernel']
    return f"{aten_op['name']} -> {kernel['name']}"

def ms_to_us(milliseconds: float) -> float:
    """
    Convert a duration from milliseconds to microseconds.
    
    Args:
        milliseconds: Duration in milliseconds.
    
    Returns:
        Duration in microseconds.
    """
    return milliseconds * 1000.0

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
    
    - If the path is absolute (e.g., source code paths, tool paths), returns it as-is.
    - If the path is relative (e.g., output files), resolves it against EXPERIMENT_DIR.
    
    Use this function for ALL environment variable paths. It automatically handles
    both absolute and relative paths correctly.
    
    Args:
        env_var: Environment variable name.
    
    Returns:
        Path object from environment variable, resolved if relative.
    """
    path = Path(os.environ[env_var])
    experiment_dir = Path(os.environ["EXPERIMENT_DIR"])
    
    # If path is relative, resolve against EXPERIMENT_DIR
    if not path.is_absolute():
        path = experiment_dir / path
    
    return path

def ensure_file(file_path: Path) -> None:
    """
    Check if a file exists at the given path, raise error if not found.
    
    Args:
        file_path: Path to file.
    
    Raises:
        FileNotFoundError: If file does not exist.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path.name} not found at {file_path}")

def ensure_dir(path, cleanup: bool = False) -> None:
    """
    Ensure directory exists, creating parent directories if needed.
    
    Args:
        path: Path or string to directory.
        cleanup: If True, remove existing directory before creating.
    """
    path = Path(path)
    if cleanup and path.is_dir():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def remove_file(file_path: str | Path) -> None:
    """
    Remove a file if it exists. Does nothing if file doesn't exist.
    
    Args:
        file_path: Path to file to remove (str or Path object).
    """
    Path(file_path).unlink(missing_ok=True)

def load_json(file_path: str | Path) -> Dict[str, Any]:
    """
    Load JSON file.
    
    Args:
        file_path: Path to JSON file (str or Path object).
    
    Returns:
        Dictionary loaded from JSON file.
    
    Raises:
        FileNotFoundError: If file does not exist.
    """
    file_path = Path(file_path)
    ensure_file(file_path)
    
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(file_path: str | Path, data: Dict[str, Any], indent: int = 2) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        file_path: Path to output file (str or Path object).
        data: Dictionary to save.
        indent: JSON indentation (default: 2).
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def _parse_scalar(value, default):
    if value is None or value == '':
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def extract_alpha_beta(concrete_inputs: List[Any], default_alpha: float = 1.0, default_beta: float = 1.0) -> Tuple[float, float]:
    """Extract alpha and beta scalars from concrete_inputs for addmm operations.
    
    Args:
        concrete_inputs: List of concrete input values from aten_op
        default_alpha: Default alpha value if not found (default: 1.0)
        default_beta: Default beta value if not found (default: 1.0)
    
    Returns:
        Tuple of (alpha, beta) floats
    """    
    # Alpha is at index 3, beta is at index 4
    if len(concrete_inputs) >= 5:
        alpha = _parse_scalar(concrete_inputs[3], default_alpha)
        beta = _parse_scalar(concrete_inputs[4], default_beta)
    else:
        alpha = default_alpha
        beta = default_beta
    
    return alpha, beta



def validate_sequences(sequences: List[Dict[str, Any]]) -> None:
    """Validate that all sequences have required fields (kernel, aten_op, cuda_launch).
    
    Args:
        sequences: List of event sequences.
    
    Raises:
        AssertionError: If any sequence is missing required fields.
    """
    num_sequences = len(sequences)
    assert all(c['kernel'] for c in sequences), f"Some sequences missing kernel (total: {num_sequences})"
    assert all(c['aten_op'] for c in sequences), f"Some sequences missing aten_op (total: {num_sequences})"
    assert all(c['cuda_launch'] for c in sequences), f"Some sequences missing cuda_launch (total: {num_sequences})"

def validate_kernel_static_props(sequences: List[Dict[str, Any]]) -> None:
    """Validate that static kernel properties are consistent across sequences.
    
    Static properties (shared_memory, registers_per_thread, occupancy, etc.) should be
    identical for all sequences of the same kernel, as they are determined by the kernel
    code and launch configuration, not execution timing.
    
    Args:
        sequences: List of event sequences that should have identical static properties.
    
    Raises:
        AssertionError: If any static property has inconsistent values across sequences.
    """
    # Note: Some fields (occupancy, blocks_per_SM etc.) are only available in PyTorch trace, and not in SQLite traces 
    static_properties = [
        'shared_memory', 
        'registers_per_thread', 
        'occupancy', 
        'blocks_per_SM', 
        'warps_per_SM', 
        'stream', 
        'device', 
        'context',
        'queued'
    ]

    for prop in static_properties:
        values = []
        for seq in sequences:
            prop_value = seq['kernel'].get(prop)
            if prop_value is not None:
                values.append(prop_value)
        
        if values and len(set(values)) > 1:
            kernel_name = clean_kernel_name(sequences[0]['kernel']['name'])
            raise AssertionError(f"Static property '{prop}' inconsistent for kernel {kernel_name}: {set(values)} (across {len(sequences)} sequences)")

def filter_gemm_sequences(sequences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter for GEMM kernels only."""
    # GEMM operations to extract
    gemm_ops = ['aten::addmm', 'aten::mm', 'aten::bmm']

    gemm_sequences = []
    for seq in sequences:
        aten_op_name = seq['aten_op']['name']
        kernel_name = seq['kernel']['name']
        # FIXME: Clean up
        # if aten_op_name in gemm_ops and 'gemm' in kernel_name.lower():
        # Identifying gemm kernels is enough
        # Its tedious to identify all aten ops that will produce a gemm kernel
        # if aten_op_name in gemm_ops and 'gemm' in kernel_name.lower():
        # Alternatively, just check if the kernel name contains 'mm' in aten op name
        # Must be from a GEMM operation
        if 'gemm' in kernel_name.lower():
            gemm_sequences.append(copy.deepcopy(seq))
                
    validate_sequences(gemm_sequences)
    return gemm_sequences

def make_kernel_identity_key(kernel, aten_op):
    """
    Build a stable identity key for a kernel + its originating ATen op.
    Components:
      - kernel name
      - grid dims
      - block dims
      - shared memory
      - input dims (tuplized to make hashable)
    
    Args:
        kernel: Kernel object
        aten_op: ATenOp object
    """
    return (
        kernel.name, 
        tuple(kernel.grid), 
        tuple(kernel.block), 
        kernel.shared_memory, 
        tuple(tuple(d) if isinstance(d, list) else d for d in aten_op.input_dims)
    )

def group_sequences_by_identity(sequences):
    """
    Group event sequences by kernel identity key.
    
    Args:
        sequences: List of sequence dictionaries
    
    Returns: dict: key -> list[dict]
    """
    grouped = defaultdict(list)
    for seq in sequences:
        kernel = seq["kernel"]
        aten_op = seq["aten_op"]
        if kernel and aten_op:
            kernel_obj = Kernel.from_dict(kernel)
            aten_op_obj = ATenOp.from_dict(aten_op)
            if kernel_obj and aten_op_obj:
                key = make_kernel_identity_key(kernel_obj, aten_op_obj)
                grouped[key].append(seq)
    for seq_group in grouped.values():
        validate_kernel_static_props(seq_group)
    return grouped

def agg_event_metric(seq_group, event_type: str, metric: str):
    """Aggregate event-level metrics (e.g., duration) if available."""
    first_value = seq_group[0][event_type][metric]
    # FIXME: Clean up 
    # Too defensive; we should handle None values.
    # if first_value is None:
    #     return None
    values = [seq[event_type][metric] for seq in seq_group]
    return summarize_metric(values)

def agg_seq_metric(seq_group, metric_name: str):
    """Aggregate sequence-level metrics such as launch tax."""
    values = [seq[metric_name] for seq in seq_group]
    return summarize_metric(values)

def aggregate_sequences(grouped_sequences, metrics: List[str], event_types: List[str]):
    """
    Aggregate grouped sequences into unique GEMM sequences.

    Args:
        grouped_sequences: Dict mapping identity key -> list[sequence dict]
        metrics: Sequence-level metrics to summarize (e.g., ["launch_tax", "xlat_tax"])
        event_types: Event types to aggregate (e.g., ["kernel", "aten_op", "cuda_launch", "torch_op"])
    """
    unique_sequences = []
    for key, seq_group in grouped_sequences.items():
        # Use first sequence as template for aggregation.
        first_seq = seq_group[0]

        # Create sequence level template.
        agg_seq = {"count": len(seq_group)}
        for event_type in event_types:
            agg_seq[event_type] = dict(first_seq[event_type])

        # Aggregate sequence-level metrics (e.g., launch/xlat tax).
        for metric in metrics:
            if metric in first_seq:
                agg_seq[metric] = agg_seq_metric(seq_group, metric)

        # Aggregate event-level metrics (e.g., per event durations).
        event_metrics = ["dur"]
        for event_type in event_types:
            for metric in event_metrics:
                agg_seq[event_type][metric] = agg_event_metric(seq_group, event_type, metric)

        # Clean up; ts has no meaning after aggregation.
        for event_type in event_types:
            agg_seq[event_type]["ts"] = None

        # Save aggregated unique sequence.
        unique_sequences.append(agg_seq)

    # Validate the aggregated sequences.
    validate_sequences(unique_sequences)
    return unique_sequences

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
        default="bfloat16",
        choices=["float32", "float16", "bfloat16", "float8_e4m3fn"],
        help="Precision for model weights and operations",
    )
    parser.add_argument(
        "-sl", "--seq-len", dest="seq_len", type=int, default=128, 
        help="Sequence length for synthetic input."
    )
    parser.add_argument(
        "--max-new-tokens",
        dest="max_new_tokens",
        type=int,
        default=1,
        help="Number of new tokens to generate during decoder profiling.",
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
        "--microbench",
        action="store_true",
        help="Enable deterministic setup for microbench reproducibility.",
    )
    parser.add_argument(
        "--skip-offline-cublas-algo-search",
        dest="skip_offline_cublas_algo_search",
        action="store_true",
        help="Skip offline cuBLASLt algorithm search in the microbench pipeline (use heuristic algorithms).",
    )
    parser.add_argument(
        "--skip-pytorch-profile",
        dest="skip_pytorch_profile",
        action="store_true",
        help="Skip PyTorch GEMM kernel profiling in the microbench pipeline.",
    )
    parser.add_argument(
        "--skip-baremetal-profile",
        dest="skip_baremetal_profile",
        action="store_true",
        help="Skip baremetal GEMM kernel profiling in the microbench pipeline.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of times to replay each kernel for microbenchmarking.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations before profiling.",
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
    if parsed_args.device == "cpu" and parsed_args.precision in ["float16", "float8_e4m3fn", "float8_e5m2", "bfloat16"]:
        print(f"Warning: {parsed_args.precision} is not supported on CPU. Forcing float32.")
        parsed_args.precision = "float32"

    if not torch.cuda.is_available() and parsed_args.device == "cuda":
        print("Error: CUDA is not available. Please select --device cpu.", file=sys.stderr)
        sys.exit(1)

    if parsed_args.max_new_tokens < 1:
        print("Error: --max-new-tokens must be >= 1.", file=sys.stderr)
        sys.exit(1)

    if parsed_args.precision == "float8_e4m3fn":
        if not hasattr(torch, "float8_e4m3fn"):
            print("Error: FP8 requires PyTorch 2.1+ with float8 support.", file=sys.stderr)
            sys.exit(1)

        if parsed_args.device == "cuda":
            capability = torch.cuda.get_device_capability()
            if capability[0] < 9 and not (capability[0] == 8 and capability[1] >= 9):
                print(f"Warning: FP8 typically requires SM89+ (Ada/Hopper). Detected SM{capability[0]}{capability[1]}.", file=sys.stderr)
                print("FP8 may not be hardware-accelerated on this device.", file=sys.stderr)
    
    return parsed_args

def setup_deterministic_mode():
    """
    Lock down all non-determinism knobs for reproducible kernel selection.
    Sets PyTorch flags and environment variables to minimize randomness.
    """
    # Core determinism flags
    torch.backends.cudnn.benchmark = False  # Disable autotuner
    torch.backends.cudnn.deterministic = True  # Force deterministic algos
    
    # Toggle TF32 via whichever API the current torch build exposes.
    cuda_matmul_backend = torch.backends.cuda.matmul
    if hasattr(cuda_matmul_backend, "fp32_precision"):
        cuda_matmul_backend.fp32_precision = "ieee"
    elif hasattr(cuda_matmul_backend, "allow_tf32"):
        cuda_matmul_backend.allow_tf32 = False

    cudnn_backend = torch.backends.cudnn
    if hasattr(cudnn_backend, "conv") and hasattr(cudnn_backend.conv, "fp32_precision"):
        cudnn_backend.conv.fp32_precision = "ieee"
    elif hasattr(cudnn_backend, "allow_tf32"):
        cudnn_backend.allow_tf32 = False
    
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
    
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    
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
        "algo_finder": os.environ.get("CUDNN_FIND_MODE", "DEFAULT"),
    }
    cudnn_backend = torch.backends.cudnn
    if hasattr(cudnn_backend, "conv") and hasattr(cudnn_backend.conv, "fp32_precision"):
        metadata["cudnn"]["conv_fp32_precision"] = cudnn_backend.conv.fp32_precision
    elif hasattr(cudnn_backend, "allow_tf32"):
        metadata["cudnn"]["allow_tf32"] = cudnn_backend.allow_tf32
    else:
        metadata["cudnn"]["conv_fp32_precision"] = None
    
    # Matmul precision (affects kernel selection)
    try:
        metadata["matmul_precision"] = torch.get_float32_matmul_precision()
    except AttributeError:
        metadata["matmul_precision"] = None
    
    # Matmul TF32 setting (critical for GEMM kernel selection on Ampere+)
    cuda_matmul_backend = torch.backends.cuda.matmul
    if hasattr(cuda_matmul_backend, "fp32_precision"):
        metadata["matmul_fp32_precision"] = cuda_matmul_backend.fp32_precision
    elif hasattr(cuda_matmul_backend, "allow_tf32"):
        metadata["matmul_allow_tf32"] = cuda_matmul_backend.allow_tf32
    else:
        metadata["matmul_fp32_precision"] = None
    
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

def collect_events(trace: Dict[str, Any]) -> Dict[str, Any]:
    """
    Collects all events from trace organized by category.
    
    Args:
        trace: Chrome trace format dictionary with "traceEvents" key.
    
    Returns:
        Dictionary with hierarchical structure:
        - cpu: Dict with keys:
            - aten_ops: Dict[external_id, aten_op_dict] - ATen operations
            - launches: Dict[correlation_id, cuda_launch_dict] - CUDA launch events (cu(da)LaunchKernel)
        - gpu: Dict with keys:
            - kernels: List of kernel events
            - memory: List of memcpy/memset events
            - all: List of all GPU events
    """
    aten_op_events_by_ext_id = {}
    torch_op_events_by_ext_id = {}
    torch_op_buffer = []
    cuda_launch_events_by_corr = {}
    kernel_events = []
    gpu_mem_events = []
    
    for event in trace["traceEvents"]:

        name = event.get("name", "")
        cat = event.get("cat", "")
        args = event.get("args", {})
        external_id = args.get("External id", None)
        correlation = args.get("correlation", None)

        if external_id is not None:
            op_events_by_ext_id[external_id] = {
                "name": name,
                "ts": event["ts"],
                "dur": event.get("dur", 0),
                "args": args,
                "input_dims": args.get("Input Dims", []),
                "external_id": external_id
            }
         
        if cat == "cpu_op" and external_id is not None:
            op_events_by_ext_id[external_id] = {
                "type": "cpu_op",
                "name": name,
                "external_id": external_id,
                # FIX: Use .get() to avoid KeyError on T4
                "input_dims": args.get("Input Dims", []),
                "input_strides": args.get("Input Strides", []),
                "input_type": args.get("Input type", ""),
                "concrete_inputs": args.get("Concrete Inputs", []),
                "ts": event["ts"],
                "dur": event["dur"]
            }
            if torch_op_buffer:
                # NOTE: This is a hack: we pair the oldest buffered torch_op with the next ATen op.
                torch_event = torch_op_buffer.pop(0)
                torch_event["external_id"] = external_id
                assert external_id not in torch_op_events_by_ext_id, "Duplicate torch_op for external_id"
                torch_op_events_by_ext_id[external_id] = torch_event
        elif cat == "user_annotation" and name.startswith("torch_op"):
            torch_op_buffer.append({
                "type": "torch_op",
                "name": name,
                "ts": event["ts"],
                "dur": event["dur"],
            })
        elif (cat == "cuda_runtime" and name == "cudaLaunchKernel") or \
             (cat == "cuda_driver" and name == "cuLaunchKernel"):
            if external_id is not None and correlation is not None:
                cuda_launch_events_by_corr[correlation] = {
                    "type": "cuda_launch",
                    "name": name,
                    "external_id": external_id,
                    "correlation": correlation,
                    "ts": event["ts"],
                    "dur": event["dur"],
                    "cbid": args.get("cbid")  # Driver API may not have cbid
                }
        elif cat == "kernel" and external_id is not None and correlation is not None:
            kernel_events.append({
                "type": "kernel",
                "name": clean_kernel_name(name),
                "external_id": external_id,
                "correlation": correlation,
                "grid": args["grid"],
                "block": args["block"],
                "shared_memory": args["shared memory"],
                "registers_per_thread": args["registers per thread"],
                "blocks_per_SM": args["blocks per SM"],
                "warps_per_SM": args["warps per SM"],
                "occupancy": args["est. achieved occupancy %"],
                "stream": args["stream"],
                "device": args["device"],
                "context": args["context"],
                "queued": args["queued"],
                "dur": event["dur"],
                "ts": event["ts"]
            })
        elif cat == "gpu_memcpy" or cat == "gpu_memset":
            gpu_mem_events.append({
                "type": cat,
                "name": name,
                "correlation": correlation,
                "stream": args["stream"],
                "device": args["device"],
                "context": args["context"],
                "ts": event["ts"],
                "dur": event["dur"],
                "bytes": args["bytes"],
                "memory_bandwidth_gbs": args["memory bandwidth (GB/s)"] if cat == "gpu_memcpy" else None
            })
    
    # Create hierarchical structure
    assert not torch_op_buffer, "Unmatched torch_op events remaining after trace scan"
    events = {
        "cpu": {
            "aten_ops": aten_op_events_by_ext_id,
            "torch_ops": torch_op_events_by_ext_id,
            "launches": cuda_launch_events_by_corr
        },
        "gpu": {
            "kernels": kernel_events,
            "memory": gpu_mem_events,
            "all": kernel_events + gpu_mem_events
        }
    }
    return events


def link_sequences(events: Dict[str, Any]) -> List[Dict]:
    """
    Get event sequences linking CPU operations, CUDA launches, and kernels.
    
    Args:
        events: Dictionary with hierarchical structure from collect_events.

    Returns:
        List of event sequence dictionaries with keys: kernel, cuda_launch, aten_op.
    """

    import logging
    logger = logging.getLogger("soda")

    
    gpu_events = events["gpu"]
    aten_ops = events["cpu"]["aten_ops"]
    cuda_launches = events["cpu"]["launches"]
    torch_ops = events["cpu"].get("torch_ops", {})
    kernel_events = gpu_events["kernels"]

    sequences = []
    skipped_kernels = []

    for kernel in kernel_events:
        external_id = kernel["external_id"]
        correlation = kernel["correlation"]
        aten_op = aten_ops[external_id]
        cuda_launch = cuda_launches[correlation]
        torch_op = torch_ops.get(external_id)

        sequences.append({
            "kernel": kernel,
            "cuda_launch": cuda_launch,
            "aten_op": aten_op,
            "torch_op": torch_op,
        })
    if skipped_kernels:
        logger.debug(
            f"Skipped {len(skipped_kernels)} kernels with unmatched events: "
            f"{skipped_kernels[:5]}{'...' if len(skipped_kernels) > 5 else ''}"
        )
    validate_sequences(sequences)
    return sequences


def calculate_sequence_metrics(sequences: List[Dict], metrics: List[str]) -> List[Dict]:
    """
    Calculates per-sequence metrics (e.g., launch_tax, xlat_tax) and adds them to the sequence dict.

    Args:
        sequences: List of event sequence dictionaries.
        metrics: Metrics to compute (e.g., ["launch_tax", "xlat_tax"])

    Returns:
        Modified event sequences with requested metric keys added to each.
    """
    for seq in sequences:
        kernel = seq["kernel"]
        aten_op = seq["aten_op"]
        cuda_launch = seq["cuda_launch"]
        t_op = aten_op["ts"]
        t_api = cuda_launch["ts"]

        if "xlat_tax" in metrics:
            # Delta xlat captures pre-launch translation from PyTorch op boundary to CUDA launch (t_op -> t_api).
            xlat_tax = t_api - t_op
            assert xlat_tax >= 0, f"Negative xlat tax detected: cu(da)LaunchKernel.ts={t_api}, aten_op.ts={t_op}, tax={xlat_tax}"
            seq["xlat_tax"] = xlat_tax

        if "launch_tax" in metrics:
            # Delta launch captures post-launch execution from CUDA launch to kernel execution (t_api -> t_kernel).
            # Caveat: This might also include queue latency if the kernel is not immediately executed.
            launch_tax = kernel["ts"] - t_api
            assert launch_tax >= 0, f"Negative launch tax detected: kernel.ts={kernel['ts']}, cu(da)LaunchKernel.ts={t_api}, tax={launch_tax}"
            seq["launch_tax"] = launch_tax

        if "py_tax" in metrics:
            torch_op = seq["torch_op"]
            assert torch_op is not None, "torch_op missing for py_tax"
            t_py = torch_op["ts"]
            py_tax = t_op - t_py
            assert py_tax >= 0, f"Negative py_tax detected: aten_op.ts={t_op}, torch_op.ts={t_py}, tax={py_tax}"
            seq["py_tax"] = py_tax

    return sequences


def calculate_total_tax(sequences: List[Dict], tax_type: str) -> float:
    """
    Calculates total for a given sequence-level tax metric (e.g., launch or xlat).

    Args:
        sequences: List of event sequence dictionaries with the metric key.
        tax_type: Metric type without or with the "_tax" suffix (e.g., "launch", "launch_tax").

    Returns:
        Total tax in microseconds.
    """
    metric_key = tax_type if tax_type.endswith("_tax") else f"{tax_type}_tax"
    total_tax = 0.0
    for seq in sequences:
        tax_value = seq[metric_key]
        if tax_value is not None:
            total_tax += tax_value
    return total_tax


def calculate_avg_tax(sequences: List[Dict], tax_type: str) -> float:
    """
    Calculates average for a given sequence-level tax metric across all sequences.

    Args:
        sequences: List of event sequence dictionaries with the metric key.
        tax_type: Metric type without or with the "_tax" suffix (e.g., "launch", "launch_tax").

    Returns:
        Average tax in microseconds.
    """
    if not sequences:
        return 0.0
    
    total_tax = calculate_total_tax(sequences, tax_type)
    num_kernels = len(sequences)
    return total_tax / num_kernels if num_kernels > 0 else 0.0


def get_average_kernel_duration(events: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculates the average execution duration (aka operational intensity) for each unique kernel.
    
    Aggregates all instances of each kernel and computes the mean duration.
        
    Args:
        events: Dictionary with hierarchical structure from collect_events.
        
    Returns:
        Dictionary mapping kernel name to average duration.
    """
    kernel_events = events["gpu"]["kernels"]
    
    kernel_stats = defaultdict(lambda: {"total_duration": 0.0, "count": 0})

    for kernel in kernel_events:
        kernel_name = kernel["name"]
        kernel_stats[kernel_name]["total_duration"] += kernel["dur"]
        kernel_stats[kernel_name]["count"] += 1
    
    avg_durations = {}
    for name, stat in kernel_stats.items():
        if stat["count"] > 0:
            avg_durations[name] = stat["total_duration"] / stat["count"]
        else:
            avg_durations[name] = 0.0
    
    return avg_durations

def get_top_k_kernels(events: Dict[str, Any], k: int = 3) -> Dict[str, List[Tuple[str, Dict]]]:
    """
    Calculates the top-k most frequent and time-consuming kernels.
    
    Args:
        events: Dictionary with hierarchical structure from collect_events.
        k: Number of top kernels to return.
        
    Returns:
        Dictionary with keys:
        - "by_frequency": List of (kernel_name, stats_dict) tuples, sorted by frequency
        - "by_duration": List of (kernel_name, stats_dict) tuples, sorted by duration
        Each kernel_stats dict contains: frequency, duration
        Returns empty lists if no kernel events.
    """
    kernel_events = events["gpu"]["kernels"]
    
    if not kernel_events:
        return {"by_frequency": [], "by_duration": []}
    
    kernel_stats = defaultdict(lambda: {"frequency": 0, "duration": 0.0})
    
    for kernel in kernel_events:
        kernel_name = kernel["name"]
        kernel_stats[kernel_name]["frequency"] += 1
        kernel_stats[kernel_name]["duration"] += float(kernel["dur"])
    
    # Top k by frequency
    top_k_by_freq = sorted(
        kernel_stats.items(), 
        key=lambda item: item[1]["frequency"], 
        reverse=True
    )[:k]
    
    # Top k by duration
    top_k_by_dur = sorted(
        kernel_stats.items(), 
        key=lambda item: item[1]["duration"], 
        reverse=True
    )[:k]

    return {
        "by_frequency": top_k_by_freq,
        "by_duration": top_k_by_dur
    }

def generate_experiment_name(
    model: str,
    compile_type: str,
    precision: str,
    batch_size: int,
    seq_len: int,
    max_new_tokens: int,
) -> str:
    """
    Generates a unique experiment directory name from arguments.
    
    Args:
        model: Model name (e.g., "gpt2" or "meta-llama/Llama-3.2-3B").
        compile_type: Compilation type (e.g., "eager", "torch.compile").
        precision: Precision string (e.g., "bfloat16", "float16").
        batch_size: Batch size.
        seq_len: Sequence length.
        max_new_tokens: Number of new tokens generated during tracing.
        
    Returns:
        Experiment directory name string.
    """
    return (
        f"{model.replace('/', '_')}_{compile_type}_{precision}"
        f"_bs{batch_size}_sl{seq_len}_mt{max_new_tokens}"
    )


def calculate_total_inference_time(trace: Dict[str, Any]) -> float:
    """
    Calculates total wall-clock inference time from ALL trace events.
    
    Includes CPU ops, CUDA launch events, and GPU execution.
    Only considers complete events (ph="X") with timestamps and durations.
    Excludes flow markers, metadata, and instant events.
        
    Args:
        trace: Chrome trace format dictionary with "traceEvents" key.
        
    Returns:
        Total inference time in microseconds (max_end - min_start).
    """
    all_timestamps = []
    
    for event in trace["traceEvents"]:
        if event["ph"] == "X":  # Only complete events (excludes flow markers)
            start_time = float(event["ts"])
            duration = float(event["dur"])
            end_time = start_time + duration
            all_timestamps.append((start_time, end_time))
    
    if not all_timestamps:
        return 0.0
    
    min_start = min(start_time for start_time, _ in all_timestamps)
    max_end = max(end_time for _, end_time in all_timestamps)
    
    return max_end - min_start

def calculate_total_gpu_time_span(events: Dict[str, Any]) -> float:
    """
    Calculates the end-to-end GPU time span by finding min start and max end
    across GPU execution events (kernel, gpu_memcpy, gpu_memset).
    
    Measures the extreme time window of GPU execution (from first to last GPU event).
    Excludes cu(da)LaunchKernel (CPU-side calls).
        
    Args:
        events: Dictionary with hierarchical structure from collect_events.
        
    Returns:
        Time span in microseconds (max_end - min_start).
    """
    gpu_events = events["gpu"]["all"]
    
    if not gpu_events:
        return 0.0
    
    gpu_event_intervals = []
    for event in gpu_events:
        start_time = float(event["ts"])
        end_time = start_time + float(event["dur"])
        gpu_event_intervals.append((start_time, end_time))
    
    min_start = min(start_time for start_time, _ in gpu_event_intervals)
    max_end = max(end_time for _, end_time in gpu_event_intervals)
    
    return max_end - min_start

def calculate_kernel_exec_time(events: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculates total and average kernel execution time.
        
    Args:
        events: Dictionary with hierarchical structure from collect_events.
        
    Returns:
        Dictionary with "total" and "avg" keys (in microseconds).
    """
    kernel_events = events["gpu"]["kernels"]
    num_kernels = len(kernel_events)
    
    total_kernel_exec_time = 0.0
    for kernel in kernel_events:
        total_kernel_exec_time += float(kernel["dur"])
    
    avg_kernel_exec_time = total_kernel_exec_time / num_kernels if num_kernels > 0 else 0.0
    
    return {
        "total": total_kernel_exec_time,
        "avg": avg_kernel_exec_time,
    }

def calculate_true_gpu_busy_time(events: Dict[str, Any]) -> float:
    """
    Calculates GPU busy time by merging overlapping GPU event intervals.
    
    Accounts for concurrent GPU execution across all streams by merging
    overlapping time intervals. Includes kernel, gpu_memcpy, and gpu_memset
    events. If events run concurrently on different streams, their overlapping 
    time is counted once.
        
    Args:
        events: Dictionary with hierarchical structure from collect_events.
        
    Returns:
        Merged GPU busy time in microseconds.
    """
    gpu_events = events.get("gpu", {}).get("all", [])
    
    # Edge case: no GPU events
    if not gpu_events:
        return 0.0
    
    # Extract GPU event intervals
    gpu_event_intervals = []
    for event in gpu_events:
        start_time = float(event["ts"])
        end_time = start_time + float(event["dur"])
        gpu_event_intervals.append((start_time, end_time))
    
    # Sort by start time
    gpu_event_intervals = sorted(gpu_event_intervals)
    
    # Merge overlapping GPU event intervals
    merged_intervals = [gpu_event_intervals[0]]
    for current_start, current_end in gpu_event_intervals[1:]:
        last_start, last_end = merged_intervals[-1]
        if current_start < last_end:
            # Overlapping: merge intervals
            merged_intervals[-1] = (last_start, max(last_end, current_end))
        else:
            # Non-overlapping: add new interval
            merged_intervals.append((current_start, current_end))
    
    # Calculate total GPU busy time: sum of durations of all merged intervals
    # Each merged interval represents a continuous period of GPU activity
    true_gpu_busy_time = 0.0
    for start_time, end_time in merged_intervals:
        true_gpu_busy_time += (end_time - start_time)
    
    return true_gpu_busy_time

def calculate_gpu_utilization(events: Dict[str, Any]) -> float:
    """
    Calculates GPU utilization percentage.
        
    Args:
        events: Dictionary with hierarchical structure from collect_events.
        
    Returns:
        GPU utilization as a percentage (0.0 to 100.0).
    """
    # Calculate denominator: time span of GPU execution events only 
    total_gpu_time_span = calculate_total_gpu_time_span(events)
    
    # Avoid division by zero
    if total_gpu_time_span == 0.0:
        return 0.0
    
    # Calculate numerator: non overlapping busy time of GPU execution events
    true_gpu_busy_time = calculate_true_gpu_busy_time(events)
    
    # Calculate GPU utilization percentage
    gpu_utilization = (true_gpu_busy_time / total_gpu_time_span)
    gpu_utilization = gpu_utilization * 100.0

    return gpu_utilization

def calculate_framework_tax(
    inference_time_us: float,
    gpu_busy_time_us: float
) -> Dict[str, float]:
    """
    Calculate framework tax - the CPU-side time not spent on GPU computation.
    
    Definitions:
    - T_total: Total wall-clock inference time (inference_time_us)
    - T_gpu_busy: True GPU active time, accounting for concurrency (gpu_busy_time_us)
    - T_exposed: Exposed Framework Tax = T_total - T_gpu_busy
    
    Args:
        inference_time_us: Total inference time in microseconds (T_total).
        gpu_busy_time_us: GPU busy time in microseconds (T_gpu_busy).
    
    Returns:
        Dictionary containing:
        - T_exposed: Exposed framework tax in microseconds
        - T_exposed_ms: Exposed framework tax in milliseconds
        - T_exposed_percent: T_exposed as a percentage of T_total
        - T_gpu_busy_percent: T_gpu_busy as a percentage of T_total
        - is_framework_bound: Boolean flag (True if T_exposed > 50%)
    """
    # T_exposed = T_total - T_gpu_busy
    # Clamp to 0 to handle potential measurement noise where GPU time > CPU time slightly
    t_exposed_us = max(0.0, inference_time_us - gpu_busy_time_us)
    
    # Calculate percentages
    if inference_time_us > 0:
        t_exposed_percent = (t_exposed_us / inference_time_us) * 100.0
        t_gpu_busy_percent = (gpu_busy_time_us / inference_time_us) * 100.0
    else:
        t_exposed_percent = 0.0
        t_gpu_busy_percent = 0.0
        
    # Heuristic: If exposed tax > 50%, we are framework bound
    is_framework_bound = t_exposed_percent > 50.0

    return {
        "T_exposed": t_exposed_us,
        "T_exposed_ms": us_to_ms(t_exposed_us),
        "T_exposed_percent": t_exposed_percent,
        "T_gpu_busy_percent": t_gpu_busy_percent,
    #    "is_framework_bound": is_framework_bound
    }

def analyze_per_stream(events: Dict[str, Any]) -> Dict:
    """
    Analyzes GPU events grouped by stream.
    
    For each stream, calculates:
    - Total operations and kernel count
    - Total kernel execution time (sum of durations)
    - True GPU busy time (merged overlapping intervals)
        
    Args:
        events: Dictionary with hierarchical structure from collect_events.
        
    Returns:
        Dictionary mapping stream_id to stream metrics.
    """
    gpu_events = events["gpu"]["all"]
    
    stream_info = defaultdict(lambda: {
        "ops": [], "total_kernel_exec_time": 0.0, "true_gpu_busy_time": 0.0,
        "op_count": 0, "kernel_count": 0
    })
    
    for op in gpu_events:
        stream_id = op.get("stream", "unknown_stream")
        stream_info[stream_id]["ops"].append(op)
    
    for stream_id, data in stream_info.items():
        ops_on_stream = sorted(data["ops"], key=lambda x: float(x["ts"]))
        stream_info[stream_id]["ops"] = ops_on_stream
        stream_info[stream_id]["op_count"] = len(ops_on_stream)
        
        stream_kernels = [op for op in ops_on_stream if op["type"] == "kernel"]
        stream_info[stream_id]["kernel_count"] = len(stream_kernels)
        stream_info[stream_id]["total_kernel_exec_time"] = sum(
            float(k["dur"]) for k in stream_kernels
        )
        
        if stream_kernels:
            stream_intervals = sorted([
                (float(k["ts"]), float(k["ts"]) + float(k["dur"]))
                for k in stream_kernels
            ])
            s_merged = [stream_intervals[0]]
            for s_start, s_end in stream_intervals[1:]:
                sl_start, sl_end = s_merged[-1]
                if s_start < sl_end:
                    s_merged[-1] = (sl_start, max(sl_end, s_end))
                else:
                    s_merged.append((s_start, s_end))
            stream_info[stream_id]["true_gpu_busy_time"] = sum(
                end - start for start, end in s_merged
            )
    
    return dict(stream_info)

def analyze_kernel_fusion_candidates(sequences: List[Dict], exact_length: int, prox_score_threshold: float, logger=None) -> Optional[Dict]:
    """
    Analyzes kernel launch sequences to find opportunities for fusion.

    Args:
        sequences: List of event sequence dictionaries.
        exact_length: The exact length of sequences to analyze.
        prox_score_threshold: The proximity score required to recommend a fusion (e.g., 1.0 for deterministic).
        logger: Optional logger instance for logging results. If None, no logging is performed.
        
    Returns:
        Dictionary with fusion analysis results, or None if no candidates found or invalid input.
    """
    if exact_length < 2:
        if logger:
            logger.warning("Sequence length must be at least 2.")
        return None

    all_segments = []
    current_segment = []
    # Separate event sequences by synchronization points
    for seq in sequences:
        cuda_launch = seq["cuda_launch"]
        if cuda_launch and 'cudaStreamSynchronize' in cuda_launch["name"]:
            if current_segment:
                all_segments.append(current_segment)
            current_segment = []
        current_segment.append(seq)
    if current_segment:
        all_segments.append(current_segment)

    unique_fusion_candidates: Set[Tuple[str, ...]] = set()
    
    # Process each segment to find sequences
    for segment in all_segments:
        current_chain = deque(maxlen=exact_length)
        for seq in segment:
            kernel = seq["kernel"]
            if kernel:
                current_chain.append(kernel["name"])
                if len(current_chain) == exact_length:
                    chain_tuple = tuple(current_chain)
                    unique_fusion_candidates.add(chain_tuple)

    # Calculate proximity scores
    fusion_recommendations = []
    kernel_freq: DefaultDict[str, int] = defaultdict(int)
    for seq in sequences:
        kernel = seq["kernel"]
        if kernel:
            kernel_freq[kernel["name"]] += 1
        
    for chain in unique_fusion_candidates:
        # Count occurrences of this exact chain
        count = 0
        for segment in all_segments:
            segment_kernels = [seq["kernel"]["name"] for seq in segment if seq["kernel"]]
            for i in range(len(segment_kernels) - exact_length + 1):
                if tuple(segment_kernels[i:i+exact_length]) == chain:
                    count += 1
        
        starting_kernel = chain[0]
        total_occurrences = kernel_freq[starting_kernel]
        
        if total_occurrences > 0:
            proximity_score = count / total_occurrences
            if proximity_score >= prox_score_threshold:
                fusion_recommendations.append((chain, count, proximity_score))
    
    # Report findings
    if logger:
        logger.info(f"=== Fusion Analysis (Length={exact_length}, Threshold={prox_score_threshold}) ===")
    if not fusion_recommendations:
        if logger:
            logger.info("\t* No kernel chains met the fusion criteria.")
        return None

    sorted_recommendations = sorted(fusion_recommendations, key=lambda x: x[1], reverse=True)
    if logger:
        logger.info(f"\t* Found {len(sorted_recommendations)} potential fusion candidates:")
        for idx, (chain, count, score) in enumerate(sorted_recommendations, 1):
            logger.info(f"\t* Chain {idx}\tFound {count} times\tProx. Score = {score:.2f}")
            for kernel in chain:
                logger.info(f"\t\t** {kernel}")
        logger.info("")
    
    # Return structured results
    return {
        "length": exact_length,
        "threshold": prox_score_threshold,
        "candidates": [
            {
                "chain": list(chain),
                "count": count,
                "proximity_score": score
            }
            for chain, count, score in sorted_recommendations
        ]
    }

def generate_synthetic_inputs(tokenizer, device: torch.device, batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
    """
    Generates synthetic tokenized inputs for profiling.
    
    Args:
        tokenizer: Tokenizer providing vocab_size.
        device: Torch device for the tensors.
        batch_size: Batch size for the inputs.
        seq_len: Sequence length for the inputs.
        
    Returns:
        Dictionary with 'input_ids' and 'attention_mask' tensors.
    """
    # FIXME: Stale code; explained below
    # input_ids = torch.randint(
    #     1, 
    #     tokenizer.vocab_size, 
    #     size=(batch_size, seq_len), 
    #     device=device,
    # )
    # atten_mask = torch.ones(
    #     batch_size, 
    #     seq_len, 
    #     device=device,
    # )

    # Build tensors on CPU, then move them, so TorchScript/ONNX traces don't "lock" a device
    # Pass the shape positionally as a tuple of ints to avoid dtype/device parsing quirks
    input_ids = torch.randint(
        1,
        tokenizer.vocab_size,
        (batch_size, seq_len),
        dtype=torch.long
    ).to(device)

    atten_mask = torch.ones(
        (batch_size, seq_len),
        dtype=torch.long
    ).to(device)

    return {
        "input_ids": input_ids,
        "attention_mask": atten_mask,
    }
