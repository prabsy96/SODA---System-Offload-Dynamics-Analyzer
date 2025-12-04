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
from .data import Kernel, CPUOp, Sequence, clean_kernel_name

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
        sequence: Dictionary containing 'cpu_op' and 'kernel' keys.
    
    Returns:
        Formatted sequence string: "{op_name} -> {kernel_name}".
    """
    cpu_op = sequence['cpu_op']
    kernel = sequence['kernel']
    return f"{cpu_op['name']} -> {kernel['name']}"

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
        concrete_inputs: List of concrete input values from cpu_op
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
    """Validate that all sequences have required fields (kernel, cpu_op, cuda_launch).
    
    Args:
        sequences: List of event sequences.
    
    Raises:
        AssertionError: If any sequence is missing required fields.
    """
    num_sequences = len(sequences)
    assert all(c['kernel'] for c in sequences), f"Some sequences missing kernel (total: {num_sequences})"
    assert all(c['cpu_op'] for c in sequences), f"Some sequences missing cpu_op (total: {num_sequences})"
    assert all(c['cuda_launch'] for c in sequences), f"Some sequences missing cuda_launch (total: {num_sequences})"

def validate_static_props(sequences: List[Dict[str, Any]]) -> None:
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
        cpu_op_name = seq['cpu_op']['name']
        kernel_name = seq['kernel']['name']
        # FIXME: Clean up
        # if cpu_op_name in gemm_ops and 'gemm' in kernel_name.lower():
        # Identifying gemm kernels is enough
        # Its tedious to identify all aten ops that will produce a gemm kernel
        # if cpu_op_name in gemm_ops and 'gemm' in kernel_name.lower():
        # Alternatively, just check if the kernel name contains 'mm' in aten op name
        # Must be from a GEMM operation
        if 'gemm' in kernel_name.lower():
            gemm_sequences.append(copy.deepcopy(seq))
                
    validate_sequences(gemm_sequences)
    return gemm_sequences

def make_kernel_identity_key(kernel, cpu_op):
    """
    Build a stable identity key for a kernel + its originating CPU op.
    Components:
      - kernel name
      - grid dims
      - block dims
      - shared memory
      - input dims (tuplized to make hashable)
    
    Args:
        kernel: Kernel object
        cpu_op: CPUOp object
    """
    return (
        kernel.name, 
        tuple(kernel.grid), 
        tuple(kernel.block), 
        kernel.shared_memory, 
        tuple(tuple(d) if isinstance(d, list) else d for d in cpu_op.input_dims)
    )

def group_by_identity(sequences):
    """
    Group event sequences by kernel identity key.
    
    Args:
        sequences: List of sequence dictionaries
    
    Returns: dict: key -> list[dict]
    """
    grouped = defaultdict(list)
    for seq in sequences:
        kernel = seq["kernel"]
        cpu_op = seq["cpu_op"]
        if kernel and cpu_op:
            kernel_obj = Kernel.from_dict(kernel)
            cpu_op_obj = CPUOp.from_dict(cpu_op)
            if kernel_obj and cpu_op_obj:
                key = make_kernel_identity_key(kernel_obj, cpu_op_obj)
                grouped[key].append(seq)
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
        inst["kernel"]["dur"]
        for inst in instances
        if inst["kernel"]["dur"] is not None
    ]
    if kernel_durations:
        base_instance["kernel"].update(calculate_avg_min_max(kernel_durations, "dur"))
        base_instance["kernel"]["all_dur"] = kernel_durations
    
    # Aggregate cpu_op duration
    cpu_op_durations = [
        inst["cpu_op"]["dur"]
        for inst in instances
        if inst["cpu_op"]["dur"] is not None
    ]
    if cpu_op_durations:
        base_instance["cpu_op"].update(calculate_avg_min_max(cpu_op_durations, "dur"))
        base_instance["cpu_op"]["all_dur"] = cpu_op_durations
    
    # Aggregate cuda_launch duration
    cuda_launch_durations = [
        inst["cuda_launch"]["dur"]
        for inst in instances
        if inst["cuda_launch"]["dur"] is not None
    ]
    if cuda_launch_durations:
        base_instance["cuda_launch"].update(calculate_avg_min_max(cuda_launch_durations, "dur"))
        base_instance["cuda_launch"]["all_dur"] = cuda_launch_durations
    
    # Aggregate kernel_tax
    kernel_tax_values = [
        inst["kernel_tax"]
        for inst in instances
        if inst["kernel_tax"] is not None
    ]

    meta = {"count": len(instances), "all_kernel_tax": kernel_tax_values}
    meta.update(calculate_avg_min_max(kernel_tax_values, "kernel_tax"))
    base_instance["meta"] = meta

    
    # TODO: Remove old code
    # base_instance.pop("kernel_tax")
    # base_instance["kernel"].pop("dur")
    # base_instance["cpu_op"].pop("dur")
    # base_instance["cuda_launch"].pop("dur")
    # base_instance["kernel"].pop("ts")
    # base_instance["cpu_op"].pop("ts")
    # base_instance["cuda_launch"].pop("ts")
    
    # Null these fields since they dont mean anything after aggregation
    # Kernel tax of the first sequence (base instance)
    base_instance["kernel_tax"] = None
    # Duration of each event of the first sequence (base instance)
    base_instance["kernel"]["dur"] = None
    base_instance["cpu_op"]["dur"] = None
    base_instance["cuda_launch"]["dur"] = None
    # Timestamp of each event of the first sequence (base instance)
    base_instance["kernel"]["ts"] = None
    base_instance["cpu_op"]["ts"] = None
    base_instance["cuda_launch"]["ts"] = None

    return base_instance

def deduplicate_and_aggregate(sequences):
    """Deduplicate sequences by identity key, validate static properties, and aggregate metrics."""
    # Group by identity
    grouped_sequences = group_by_identity(sequences)
    
    unique_gemm_sequences = []
    for key, instances in grouped_sequences.items():
        
        # Verify static properties are consistent across instances
        validate_static_props(instances)

        # Aggregate execution metrics (kernel tax, cpu op duration, cuda launch duration, kernel duration)
        base_instance = aggregate_execution_metrics(instances)
        
        # Construct output structure 
        # TODO: Cleanup
        # sequence_entry = {
        #     "meta": base_instance["meta"],
        #     "kernel": base_instance['kernel'],
        #     "cuda_launch": base_instance["cuda_launch"],
        #     "cpu_op": base_instance["cpu_op"]
        # }
        sequence_entry = base_instance
        
        unique_gemm_sequences.append(sequence_entry)
    
    validate_sequences(unique_gemm_sequences)
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
    TODO: refactor - Move to SodaTraceProcessor.collect_events()
    
    Collects all events from trace organized by category.
    
    Args:
        trace: Chrome trace format dictionary with "traceEvents" key.
    
    Returns:
        Dictionary with hierarchical structure:
        - cpu: Dict with keys:
            - ops: Dict[external_id, cpu_op_dict] - CPU operations
            - launches: Dict[correlation_id, cuda_launch_dict] - CUDA launch events (cu(da)LaunchKernel)
        - gpu: Dict with keys:
            - kernels: List of kernel events
            - memory: List of memcpy/memset events
            - all: List of all GPU events
    """
    op_events_by_ext_id = {}
    cuda_launch_events_by_corr = {}
    kernel_events = []
    gpu_mem_events = []
    
    for event in trace["traceEvents"]:

        cat = event.get("cat", "")
        args = event.get("args", {})
        external_id = args.get("External id", None)
        correlation = args.get("correlation", None)
         
        if cat == "cpu_op" and external_id is not None:
            op_events_by_ext_id[external_id] = {
                "type": "cpu_op",
                "name": event["name"],
                "external_id": external_id,
                "input_dims": args["Input Dims"],
                "input_strides": args["Input Strides"],
                "input_type": args["Input type"],
                "concrete_inputs": args["Concrete Inputs"],
                "ts": event["ts"],
                "dur": event["dur"]
            }
        elif (cat == "cuda_runtime" and event["name"] == "cudaLaunchKernel") or \
             (cat == "cuda_driver" and event["name"] == "cuLaunchKernel"):
            if external_id is not None and correlation is not None:
                cuda_launch_events_by_corr[correlation] = {
                    "type": "cuda_launch",
                    "name": event["name"],
                    "external_id": external_id,
                    "correlation": correlation,
                    "ts": event["ts"],
                    "dur": event["dur"],
                    "cbid": args.get("cbid")  # Driver API may not have cbid
                }
        elif cat == "kernel" and external_id is not None and correlation is not None:
            kernel_events.append({
                "type": "kernel",
                "name": clean_kernel_name(event["name"]),
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
                "name": event["name"],
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
    events = {
        "cpu": {
            "ops": op_events_by_ext_id,
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
    TODO: refactor - Move to SodaTraceProcessor.link_sequences()
    
    Get event sequences linking CPU operations, CUDA launches, and kernels.
    
    Args:
        events: Dictionary with hierarchical structure from collect_events.

    Returns:
        List of event sequence dictionaries with keys: kernel, cuda_launch, cpu_op.
    """

    import logging
    logger = logging.getLogger("soda")

    
    gpu_events = events["gpu"]
    cpu_ops = events["cpu"]["ops"]
    cuda_launches = events["cpu"]["launches"]
    kernel_events = gpu_events["kernels"]

    sequences = []
    skipped_kernels = []

    for kernel in kernel_events:
        external_id = kernel["external_id"]
        correlation = kernel["correlation"]

        # Skip kernels without matching CPU op
        if external_id not in cpu_ops:
            skipped_kernels.append((kernel["name"], f"missing cpu_op (external_id={external_id})"))
            continue
        
        # Skip kernels without matching CUDA launch event
        if correlation not in cuda_launches:
            skipped_kernels.append((kernel["name"], f"missing cuda_launch (correlation={correlation})"))
            continue
        
        cpu_op = cpu_ops[external_id] 
        cuda_launch = cuda_launches[correlation] 
        
        sequences.append({
            "kernel": kernel,
            "cuda_launch": cuda_launch,
            "cpu_op": cpu_op,
        })
    if skipped_kernels:
        logger.debug(
            f"Skipped {len(skipped_kernels)} kernels with unmatched events: "
            f"{skipped_kernels[:5]}{'...' if len(skipped_kernels) > 5 else ''}"
        )
    validate_sequences(sequences)
    return sequences


def calculate_per_seq_launch_tax(sequences: List[Dict]) -> List[Dict]:
    """
    Calculates launch tax for each sequence and adds it to the sequence dict.

    Args:
        sequences: List of event sequence dictionaries.

    Returns:
        Modified event sequences with "kernel_tax" key added to each.
    """
    for seq in sequences:
        kernel = seq["kernel"]
        cuda_launch = seq["cuda_launch"]
        launch_tax = kernel["ts"] - cuda_launch["ts"]
        assert launch_tax >= 0, f"Negative launch tax detected: kernel.ts={kernel['ts']}, cu(da)LaunchKernel.ts={cuda_launch['ts']}, tax={launch_tax}"
        seq["kernel_tax"] = launch_tax

    return sequences


def calculate_total_launch_tax(sequences: List[Dict]) -> float:
    """
    Calculates total launch tax across all sequences.

    Args:
        sequences: List of event sequence dictionaries with "kernel_tax" key.

    Returns:
        Total launch tax in microseconds.
    """
    total_tax = 0.0
    for seq in sequences:
        kernel_tax = seq["kernel_tax"]
        if kernel_tax is not None:
            total_tax += kernel_tax
    return total_tax


def calculate_avg_launch_tax(sequences: List[Dict]) -> float:
    """
    Calculates average launch tax across all sequences.

    Args:
        sequences: List of event sequence dictionaries with "kernel_tax" key.

    Returns:
        Average launch tax in microseconds.
    """
    if not sequences:
        return 0.0
    
    total_tax = calculate_total_launch_tax(sequences)
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
    gpu_events = events["gpu"]["all"]
    
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
        interval_duration = end_time - start_time
        true_gpu_busy_time += interval_duration
    
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
    
    Args:
        inference_time_us: Total inference time in microseconds.
        gpu_busy_time_us: GPU busy time in microseconds.
    
    Returns:
        Dictionary containing:
        - framework_tax_ms: Framework tax in milliseconds
        - framework_tax_percent: Framework tax as percentage of inference time
        - gpu_busy_time_percent: GPU busy time as percentage of inference time
        - is_framework_bound: Flag indicating if the framework is bound by CPU-side latency
    """
    # Framework tax = everything except GPU compute
    framework_tax_us = max(0.0, inference_time_us - gpu_busy_time_us)

    is_framework_bound = framework_tax_us > gpu_busy_time_us
    
    # Calculate percentages
    if inference_time_us > 0:
        framework_tax_percent = (framework_tax_us / inference_time_us) * 100
        gpu_busy_time_percent = (gpu_busy_time_us / inference_time_us) * 100
    else:
        framework_tax_percent = 0.0
        gpu_busy_time_percent = 0.0
    
    return {
        "framework_tax_ms": us_to_ms(framework_tax_us),
        "framework_tax_percent": framework_tax_percent,
        "gpu_busy_time_percent": gpu_busy_time_percent,
        "is_framework_bound": is_framework_bound,
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
