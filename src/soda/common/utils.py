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
import subprocess
from . import print_utils
from .data import Kernel, ATenOp, Sequence, clean_kernel_name


# =============================================================================
# TaxBreak Constants (from paper)
# =============================================================================

# T_floor_sys: Legacy fallback constant — used only by baremetal/report.py.
# Do NOT use inside calculate_hdbi(); pass t_sys_us explicitly instead.
# Source: TaxBreak paper, measured on H100 (~4.5µs)
T_FLOOR_SYS_MS = 0.0045  # 4.5 microseconds in milliseconds

# =============================================================================
# GPU Clock Frequency Reporting (read-only, no root required)
# =============================================================================

def get_gpu_clock_info(device_id: int = 0) -> Dict[str, Any]:
    """
    Query current GPU clock frequencies for reproducibility reporting.
    Does not require root access.

    Returns:
        Dictionary with clock info, or empty dict on failure.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--id={device_id}",
             "--query-gpu=clocks.current.graphics,clocks.current.memory,clocks.max.graphics,clocks.max.memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 4:
                return {
                    "graphics_clock_mhz": int(parts[0]),
                    "memory_clock_mhz": int(parts[1]),
                    "max_graphics_clock_mhz": int(parts[2]),
                    "max_memory_clock_mhz": int(parts[3]),
                }
    except Exception as e:
        print(f"Warning: Could not query GPU clocks: {e}", file=sys.stderr)
    return {}


def report_gpu_clocks(device_id: int = 0, context: str = "") -> None:
    """
    Print current GPU clock frequencies for experiment reproducibility.

    Args:
        device_id: CUDA device index.
        context: Optional label (e.g., "before profiling", "after warmup").
    """
    info = get_gpu_clock_info(device_id)
    if info:
        prefix = f"[{context}] " if context else ""
        print(f"{prefix}GPU {device_id} clocks: "
              f"graphics={info['graphics_clock_mhz']}/{info['max_graphics_clock_mhz']} MHz, "
              f"memory={info['memory_clock_mhz']}/{info['max_memory_clock_mhz']} MHz")


def is_gemm_op(aten_op_name: str) -> bool:
    """Check if an ATen op is a GEMM operation."""
    gemm_ops = [
        "aten::mm",
        "aten::bmm", 
        "aten::addmm",
        "aten::matmul",
        "aten::linear",
        "aten::_scaled_mm",
        "aten::_scaled_dot_product",
    ]
    return any(op in aten_op_name for op in gemm_ops)


def is_gemm_kernel(kernel_name: str) -> bool:
    """Check if a kernel name indicates a GEMM kernel (cuBLAS, Cutlass, internal)."""
    gemm_patterns = [
        # cuBLAS patterns
        "cublas",
        "cublasLt",
        # Cutlass patterns
        "cutlass",
        # PyTorch internal GEMM patterns (H100/Hopper)
        "nvjet",
        "wgmma",
        "s884gemm",
        "gemm",
        "Gemm",
        "GEMM",
        # Flash attention (also GEMM-like)
        "flash",
        "fmha",
    ]
    return any(pattern in kernel_name for pattern in gemm_patterns)


def filter_kernel_sequences(sequences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter for sequences that have both a kernel and aten_op.
    Marks each sequence with is_gemm classification using both op name AND kernel name.
    
    Args:
        sequences: List of event sequences
    
    Returns:
        Filtered sequences with is_gemm field added
    """
    kernel_sequences = []
    for seq in sequences:
        kernel = seq.get("kernel")
        aten_op = seq.get("aten_op")
        cuda_launch = seq.get("cuda_launch")
        
        # Must have kernel, aten_op, and cuda_launch
        if not kernel or not aten_op or not cuda_launch:
            continue
        
        # Get names safely
        aten_name = aten_op.get("name", "") if isinstance(aten_op, dict) else ""
        kernel_name = kernel.get("name", "") if isinstance(kernel, dict) else ""
        
        # Check BOTH ATen op name AND kernel name for GEMM classification
        # This catches PyTorch internal GEMM kernels (nvjet, cutlass, wgmma) on H100
        seq["is_gemm"] = is_gemm_op(aten_name) or is_gemm_kernel(kernel_name)
        
        kernel_sequences.append(seq)

    return kernel_sequences

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
    Build a summary dict with count, all samples, avg/min/max/std stats.
    Assumes values is non-empty.
    """
    import math
    n = len(values)
    avg = sum(values) / n

    # Compute sample standard deviation
    if n > 1:
        variance = sum((x - avg) ** 2 for x in values) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0

    summary = {
        "count": n,
        "all": list(values),
        "avg": avg,
        "min": min(values),
        "max": max(values),
        "std": std,
    }
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


def write_log(log_key: str, log: str) -> None:
    """Write a log message to the specified log file."""
    try:
        log_path = get_path(log_key)
        # Ensure parent directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log + "\n")
    except Exception as e:
        # Fallback: print to stderr if we can't write to file
        import sys
        print(f"WARNING: Could not write to {log_key}: {e}", file=sys.stderr)
        print(f"LOG: {log}", file=sys.stderr)


def check_assert(condition: bool, message: str, excuse: str = "") -> bool:
    """
    Check assertion and log if failed. Returns condition result.
    Does NOT raise - just logs and continues.
    """
    if not condition:
        log = f"ASSERT FAILED: {message}"
        if excuse:
            log += f" | Excuse: {excuse}"
        try:
            write_log("ASSERT_LOG", log)
        except Exception:
            import sys
            print(log, file=sys.stderr)
    return condition

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

def calculate_hsb_metrics(
    inference_time_us: float,
    gpu_busy_time_us: float,
    sequences: List[Dict[str, Any]],
    taxbreak_lut: Dict[str, float],
) -> Dict[str, float]:
    """
    Calculate HSB (Hardware-Software Inversion) metrics.
    
    HSB = 1 - (T_Exposed / T_Structural)
    
    Args:
        inference_time_us: Total inference time in microseconds
        gpu_busy_time_us: GPU busy time in microseconds
        sequences: List of kernel sequences
        taxbreak_lut: Lookup table mapping kernel_name -> T_fo
    
    Returns:
        Dictionary with HSB metrics:
        - t_exposed_us: Exposed framework overhead (GPU idle time)
        - t_structural_us: Structural framework overhead (sum of T_fo)
        - hsb: Hardware-Software Inversion metric
        - hsb_classification: Human-readable classification
    """
    # Calculate T_Exposed (GPU idle time)
    t_exposed = max(0.0, inference_time_us - gpu_busy_time_us)
    
    # Calculate T_Structural (sum of per-kernel framework overheads)
    t_structural = 0.0
    
    if taxbreak_lut:
        avg_t_fo = sum(taxbreak_lut.values()) / len(taxbreak_lut)
    else:
        avg_t_fo = 0.0
    
    for seq in sequences:
        kernel = seq.get("kernel", {})
        kernel_name = kernel.get("name", "") if isinstance(kernel, dict) else ""
        
        if kernel_name in taxbreak_lut:
            t_fo = taxbreak_lut[kernel_name]
        else:
            t_fo = avg_t_fo
        
        freq = seq.get("freq", 1) or 1
        t_structural += t_fo * freq
    
     # Calculate HSB
    epsilon = 1e-6
    if t_structural < epsilon:
        hsb = 1.0 if t_exposed < epsilon else -10.0
    else:
        hsb = 1.0 - (t_exposed / t_structural)
    
    # Classify HSB
    if hsb >= 0.5:
        classification = "hardware-bound"
    elif hsb >= 0:
        classification = "balanced"
    else:
        classification = "framework-bound"
    
    return {
        "t_exposed_us": t_exposed,
        "t_structural_us": t_structural,
        "hsb": hsb,
        "hsb_classification": classification,
    }

def to_hashable(obj: Any) -> Any:
    """
    Recursively convert an object to a hashable type.
    
    - Lists become tuples
    - Dicts become tuples of (key, value) pairs
    - Other types are returned as-is
    
    Args:
        obj: Any Python object
    
    Returns:
        Hashable version of the object
    """
    if isinstance(obj, list):
        return tuple(to_hashable(item) for item in obj)
    elif isinstance(obj, dict):
        return tuple((k, to_hashable(v)) for k, v in sorted(obj.items()))
    elif isinstance(obj, set):
        return frozenset(to_hashable(item) for item in obj)
    else:
        return obj

def make_kernel_identity_key(kernel, aten_op):
    """
    Create a hashable identity key from kernel and aten_op dicts.
    
    Args:
        kernel: Kernel dict with name, grid, block, shared_memory
        aten_op: ATen op dict with input_dims
    
    Returns:
        Hashable tuple key
    """
    kernel_name = clean_kernel_name(kernel.get("name", ""))
    grid = to_hashable(kernel.get("grid", [0, 0, 0]))
    block = to_hashable(kernel.get("block", [0, 0, 0]))
    shared_mem = kernel.get("shared_memory", 0)
    input_dims = to_hashable(aten_op.get("input_dims", []))
    
    return (kernel_name, grid, block, shared_mem, input_dims)

def group_sequences_by_identity(sequences: List[Dict[str, Any]]) -> Dict[tuple, List[Dict[str, Any]]]:
    """
    Group sequences by their identity key (kernel config + input dims).
    
    The identity key is a tuple of:
    - kernel name (cleaned)
    - grid dimensions (as tuple)
    - block dimensions (as tuple)
    - shared memory
    - input dimensions (as nested tuple)
    
    Args:
        sequences: List of sequence dictionaries
    
    Returns:
        Dictionary mapping identity keys to lists of sequences
    """
    grouped = defaultdict(list)
    
    for seq in sequences:
        kernel = seq.get("kernel", {})
        aten_op = seq.get("aten_op", {})
        key = make_kernel_identity_key(kernel, aten_op)
        grouped[key].append(seq)
    
    return grouped
def agg_event_metric(seq_group, event_type: str, metric: str):
    """Aggregate event-level metrics (e.g., duration) if available."""
    values = [seq[event_type][metric] for seq in seq_group]
    return summarize_metric(values)

def agg_seq_metric(seq_group, metric_name: str):
    """Aggregate sequence-level metrics such as launch tax."""
    values = [seq[metric_name] for seq in seq_group]
    return summarize_metric(values)

def get_args_parser() -> argparse.ArgumentParser:
    """Create and return argument parser."""
    parser = argparse.ArgumentParser(
        description="SODA: System Offload Dynamics Analyzer. Analyze CPU–GPU dynamics of PyTorch models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model",
        default="",
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
        default=150,
        help="Number of times to replay each kernel for microbenchmarking.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup iterations before profiling (helps stabilize GPU clocks).",
    )
    parser.add_argument(
        "--direct-trace",
        dest="direct_trace",
        action="store_true",
        default=True,
        help="Use direct trace analysis (no replay). This is the default mode.",
    )
    parser.add_argument(
        "--replay",
        dest="direct_trace",
        action="store_false",
        help="Use replay-based analysis instead of direct trace.",
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s 0.1.0"
    )
    parser.add_argument(
        "--kernel-db",
        dest="kernel_db",
        action="store_true",
        help="Generate op-kernel database after profiling.",
    )
    parser.add_argument(
        "--taxbreak",
        dest="taxbreak",
        action="store_true",
        help="Run enhanced TaxBreak pipeline (requires --kernel-db-path).",
    )
    parser.add_argument(
        "--kernel-db-path",
        dest="kernel_db_path",
        type=str,
        default=None,
        help="Path to kernel_database.json for --taxbreak mode.",
    )
    parser.add_argument(
        "--ncu",
        dest="ncu",
        action="store_true",
        help="Run ncu profiling on top-N kernels (use with --taxbreak).",
    )
    parser.add_argument(
        "--ncu-top-n",
        dest="ncu_top_n",
        type=int,
        default=10,
        help="Number of top kernels (by duration) to profile with ncu (default: 10).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print full expert-level output (per-kernel tables, HDBI decomposition, "
             "derivation details). Default shows only the compact layperson summary.",
    )
    parser.add_argument(
        "--carbon-intensity",
        dest="carbon_intensity",
        type=float,
        default=400.0,
        help="Grid carbon intensity in gCO2eq/kWh for carbon footprint estimation "
             "(default: 400.0). Regional presets (approx): US=386, EU=295, FR=58, "
             "DE=380, CN=581, global=475.",
    )
    parser.add_argument(
        "--pue",
        dest="pue",
        type=float,
        default=1.1,
        help="Data-center Power Usage Effectiveness for carbon estimation "
             "(default: 1.1). Values: 1.0=bare GPU server, 1.1=efficient DC, "
             "1.5=average DC.",
    )

    return parser

def parse_and_validate_args(args=None) -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = get_args_parser()
    parsed_args = parser.parse_args(args)
    
    # Validate arguments
    # --model is required unless running in --taxbreak mode (Stage 2)
    if not getattr(parsed_args, "taxbreak", False) and not parsed_args.model:
        parser.error("the following arguments are required: -m/--model")

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
            - torch_ops: Dict[external_id, torch_op_dict] - Torch operations (Python level)
            - aten_ops: Dict[external_id, aten_op_dict] - ATen operations
            - launches: Dict[correlation_id, cuda_launch_dict] - CUDA launch events
        - gpu: Dict with keys:
            - kernels: List of kernel events
            - memory: List of memcpy/memset events
            - all: List of all GPU events
    """
    aten_op_events_by_ext_id = {}
    torch_op_events_by_ext_id = {}
    cuda_launch_events_by_corr = {}
    kernel_events = []
    gpu_mem_events = []
    
    for event in trace["traceEvents"]:
        # Skip non-complete events
        if event.get("ph") != "X":
            continue
        
        cat = event.get("cat", "")
        name = event.get("name", "")
        
        # GPU kernel events
        if cat == "kernel":
            kernel_events.append({
                "name": name,
                "ts": event["ts"],
                "dur": event.get("dur", 0),
                "correlation": event.get("args", {}).get("correlation"),
                "external_id": event.get("args", {}).get("External id"),
                "grid": event.get("args", {}).get("grid", [0, 0, 0]),
                "block": event.get("args", {}).get("block", [0, 0, 0]),
                "shared_memory": event.get("args", {}).get("shared memory", 0),
            })
        
        # GPU memory events
        elif cat in ("gpu_memcpy", "gpu_memset"):
            gpu_mem_events.append({
                "name": name,
                "ts": event["ts"],
                "dur": event.get("dur", 0),
                "cat": cat,
            })
        
        # CUDA launch events (CPU side)
        elif cat == "cuda_runtime" and "LaunchKernel" in name:
            corr = event.get("args", {}).get("correlation")
            if corr:
                cuda_launch_events_by_corr[corr] = {
                    "name": name,
                    "ts": event["ts"],
                    "dur": event.get("dur", 0),
                    "correlation": corr,
                    "external_id": event.get("args", {}).get("External id"),
                }
        
        # ATen operations (C++ level)
        elif cat == "cpu_op" and name.startswith("aten::"):
            ext_id = event.get("args", {}).get("External id")
            if ext_id:
                aten_op_events_by_ext_id[ext_id] = {
                    "name": name,
                    "ts": event["ts"],
                    "dur": event.get("dur", 0),
                    "external_id": ext_id,
                    "input_dims": event.get("args", {}).get("Input Dims", []),
                    "input_type": event.get("args", {}).get("Input type", []),
                    "input_strides": event.get("args", {}).get("Input Strides", []),
                    "concrete_inputs": event.get("args", {}).get("Concrete Inputs", []),
                }
        
        # Torch operations (Python level) - capture nn.Module calls and torch.* calls
        elif cat == "python_function" or (cat == "cpu_op" and not name.startswith("aten::")):
            ext_id = event.get("args", {}).get("External id")
            if ext_id:
                # Only add if not already present (prefer first occurrence)
                if ext_id not in torch_op_events_by_ext_id:
                    torch_op_events_by_ext_id[ext_id] = {
                        "name": name,
                        "ts": event["ts"],
                        "dur": event.get("dur", 0),
                        "external_id": ext_id,
                    }
    
    # Create hierarchical structure
    events = {
        "cpu": {
            "torch_ops": torch_op_events_by_ext_id,
            "aten_ops": aten_op_events_by_ext_id,
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
        List of event sequence dictionaries with keys: kernel, cuda_launch, aten_op, torch_op.
    """
    torch_ops = events["cpu"].get("torch_ops", {})
    aten_ops = events["cpu"]["aten_ops"]
    cuda_launches = events["cpu"]["launches"]
    kernel_events = events["gpu"]["kernels"]

    sequences = []
    orphan_kernels = []

    for kernel in kernel_events:
        corr = kernel.get("correlation")
        ext_id = kernel.get("external_id")
        
        cuda_launch = cuda_launches.get(corr)
        aten_op = aten_ops.get(ext_id)
        
        # Try to find torch_op by external_id
        torch_op = torch_ops.get(ext_id)
        
        # If no direct match, try to find enclosing torch_op by timestamp
        if torch_op is None and aten_op is not None:
            aten_ts = aten_op["ts"]
            # Find torch_op that started before aten_op and is still active
            for tid, t_op in torch_ops.items():
                t_start = t_op["ts"]
                t_end = t_start + t_op.get("dur", 0)
                if t_start <= aten_ts <= t_end:
                    torch_op = t_op
                    break
        
        if cuda_launch and aten_op:
            sequences.append({
                "kernel": kernel,
                "cuda_launch": cuda_launch,
                "aten_op": aten_op,
                "torch_op": torch_op,  # May still be None
            })
        else:
            orphan_kernels.append(kernel)

    # Log orphan count but don't fail
    if orphan_kernels:
        print(f"Warning: {len(orphan_kernels)} orphan kernels (missing aten/launch events)")

    return sequences

def calculate_tklqt(sequences: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate TKLQT (Total Kernel Launch + Queue Time).
    
    TKLQT = sum of (kernel_start_ts - cuda_launch_start_ts) for all sequences.
    This measures the total time from CPU launch API call to GPU kernel execution start.
    
    Args:
        sequences: List of event sequence dictionaries with kernel and cuda_launch events.
    
    Returns:
        Dictionary with total, avg, min, max TKLQT in microseconds.
    """
    tklqt_values = []
    
    for seq in sequences:
        kernel = seq.get("kernel")
        cuda_launch = seq.get("cuda_launch")
        
        if not kernel or not cuda_launch:
            continue
        
        # Get timestamps - handle both dict and direct value formats
        if isinstance(kernel, dict):
            kernel_start = kernel.get("ts", 0)
        else:
            continue
            
        if isinstance(cuda_launch, dict):
            launch_start = cuda_launch.get("ts", 0)
        else:
            continue
        
        if kernel_start > 0 and launch_start > 0:
            # Time from launch API call to kernel execution start (in microseconds)
            lqt = kernel_start - launch_start
            if lqt >= 0:  # Sanity check - kernel should start after launch
                tklqt_values.append(lqt)
            elif lqt > -10:  # Small negative values are measurement noise, clamp to 0
                tklqt_values.append(0.0)
    
    if not tklqt_values:
        return {"total": 0.0, "avg": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    
    return {
        "total": sum(tklqt_values),
        "avg": sum(tklqt_values) / len(tklqt_values),
        "min": min(tklqt_values),
        "max": max(tklqt_values),
        "count": len(tklqt_values),
    }

def calculate_t_orchestrator(sequences: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate T_orchestrator (total CPU-side dispatch overhead).
    
    Per TaxBreak paper:
        T_orchestrator = Σ (T_py + T_aten + T_lib + T_sys) for all kernels
    
    Where for each kernel invocation:
        - T_py: Python dispatch overhead
        - T_aten: ATen translation overhead  
        - T_lib: Library (cuBLAS etc) overhead
        - T_sys: System/launch overhead (includes TKLQT)
    
    For kernels without microbench breakdown, we use:
        T_orchestrator ≈ launch_tax + aten_xlat_tax
    
    Args:
        sequences: List of event sequences with timing metrics.
    
    Returns:
        Dictionary with total, avg T_orchestrator in microseconds.
    """
    t_orch_values = []
    
    for seq in sequences:
        # Try to get detailed breakdown first
        launch_tax = seq.get("launch_tax", 0)
        aten_xlat_tax = seq.get("aten_xlat_tax", 0)
        
        # Handle dict format (from aggregated sequences)
        if isinstance(launch_tax, dict):
            launch_tax = launch_tax.get("avg", 0)
        if isinstance(aten_xlat_tax, dict):
            aten_xlat_tax = aten_xlat_tax.get("avg", 0)
        
        # Ensure non-negative (clamp measurement noise)
        launch_tax = max(0.0, float(launch_tax or 0))
        aten_xlat_tax = max(0.0, float(aten_xlat_tax or 0))
        
        # T_orchestrator for this kernel = launch_tax + aten_xlat_tax
        t_orch = launch_tax + aten_xlat_tax
        
        if t_orch > 0:
            t_orch_values.append(t_orch)
    
    if not t_orch_values:
        return {"total": 0.0, "avg": 0.0, "count": 0}
    
    return {
        "total": sum(t_orch_values),
        "avg": sum(t_orch_values) / len(t_orch_values),
        "count": len(t_orch_values),
    }

def calculate_sequence_metrics(sequences: List[Dict], metrics: List[str]) -> List[Dict]:
    """
    Calculates per-sequence metrics (e.g., launch_tax, aten_xlat_tax) and adds them to the sequence dict.

    Args:
        sequences: List of event sequence dictionaries.
        metrics: Metrics to compute (e.g., ["launch_tax", "aten_xlat_tax", "py_tax"])

    Returns:
        Modified event sequences with requested metric keys added to each.
    """
    for seq in sequences:
        kernel = seq.get("kernel")
        cuda_launch = seq.get("cuda_launch")
        aten_op = seq.get("aten_op")
        torch_op = seq.get("torch_op")  # May be None
        
        # Skip sequences missing required events
        if not kernel or not cuda_launch or not aten_op:
            continue
        
        # launch_tax: time from cuda_launch start to kernel start
        if "launch_tax" in metrics:
            launch_tax = kernel["ts"] - cuda_launch["ts"]
            seq["launch_tax"] = launch_tax
        
        # aten_xlat_tax: time from aten_op start to cuda_launch start
        if "aten_xlat_tax" in metrics:
            aten_xlat_tax = cuda_launch["ts"] - aten_op["ts"]
            seq["aten_xlat_tax"] = aten_xlat_tax
        
        # py_tax: time from torch_op start to aten_op start (requires torch_op)
        if "py_tax" in metrics:
            if torch_op is not None and "ts" in torch_op:
                py_tax = aten_op["ts"] - torch_op["ts"]
                seq["py_tax"] = py_tax
            else:
                # Fallback: set py_tax to 0 if torch_op not available
                seq["py_tax"] = 0.0

    return sequences

def aggregate_sequences(grouped_sequences, metrics: List[str], event_types: List[str]):
    """
    Aggregate grouped sequences into unique GEMM sequences.

    Args:
        grouped_sequences: Dict mapping identity key -> list[sequence dict]
        metrics: Sequence-level metrics to summarize (e.g., ["launch_tax", "aten_xlat_tax"])
        event_types: Event types to aggregate (e.g., ["kernel", "aten_op", "cuda_launch", "torch_op"])
    """
    unique_sequences = []
    for key, seq_group in grouped_sequences.items():
        # Count frequency as number of occurrences in the group
        freq = len(seq_group)
        
        # Take first sequence as representative
        first_seq = seq_group[0]
        
        aggregated = {
            "freq": freq,  # This is the count of how many times this kernel was invoked
        }
        
        # Aggregate event-level data (use first occurrence as representative)
        for event_type in event_types:
            if first_seq.get(event_type):
                aggregated[event_type] = first_seq[event_type].copy() if isinstance(first_seq[event_type], dict) else first_seq[event_type]
        
        # Aggregate sequence-level metrics with statistics
        for metric in metrics:
            values = []
            for seq in seq_group:
                val = seq.get(metric)
                if val is not None:
                    # Handle both raw values and dict values
                    if isinstance(val, dict):
                        if "avg" in val:
                            values.append(val["avg"])
                    else:
                        values.append(val)
            
            if values:
                aggregated[metric] = summarize_metric(values)
            else:
                aggregated[metric] = {"avg": 0.0, "min": 0.0, "max": 0.0, "count": 0, "all": []}
        
        # Preserve is_gemm if present
        if "is_gemm" in first_seq:
            aggregated["is_gemm"] = first_seq["is_gemm"]
        
        unique_sequences.append(aggregated)

    # Validate the aggregated sequences
    validate_sequences(unique_sequences)
    return unique_sequences

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


def calculate_hdbi(
    total_kernel_exec_time_ms: float,
    total_xlat_tax_ms: float,
    num_total_kernels: int,
    t_sys_us: float = T_FLOOR_SYS_MS * 1000.0,
) -> Dict[str, Any]:
    """
    Calculate HDBI (Host-Device Balance Index) per TaxBreak paper Eq. 6.

    HDBI = T_DeviceActive / (T_DeviceActive + T_Orchestrate)

    Where:
        T_DeviceActive = Σ(t_k) = sum of kernel execution times
        T_Orchestrate = Σ(ΔFT + I_lib·ΔCT + ΔKT)
                      = total_xlat_tax + (num_kernels × t_sys)

    Classification:
        HDBI ≥ 0.5: device-bound (GPU compute dominates)
        0.2 ≤ HDBI < 0.5: balanced
        HDBI < 0.2: host-bound (CPU overhead dominates)

    Args:
        total_kernel_exec_time_ms: Sum of kernel durations (T_DeviceActive)
        total_xlat_tax_ms: Sum of translation overhead (ΔFT + ΔCT)
        num_total_kernels: Number of kernel invocations (for ΔKT calculation)
        t_sys_us: System floor in microseconds from dynamic null-kernel measurement.
                  Defaults to the H100 hardcoded value (T_FLOOR_SYS_MS × 1000).
                  Always pass the dynamically-measured value from TaxBreak pipeline.

    Returns:
        Dictionary containing HDBI metrics and classification.
    """
    # T_DeviceActive = sum of kernel execution times
    t_device_active = total_kernel_exec_time_ms

    # ΔKT = num_kernels × T_sys (using dynamically-measured floor)
    delta_kt = num_total_kernels * (t_sys_us / 1000.0)

    # T_Orchestrate = xlat_tax (ΔFT + ΔCT) + ΔKT
    t_orchestrate = total_xlat_tax_ms + delta_kt

    # HDBI = T_DeviceActive / (T_DeviceActive + T_Orchestrate)
    denominator = t_device_active + t_orchestrate
    if denominator > 0:
        hdbi_value = t_device_active / denominator
    else:
        hdbi_value = 0.0

    # Clamp to valid range [0, 1]
    hdbi_value = max(0.0, min(1.0, hdbi_value))

    # Classification per TaxBreak paper
    if hdbi_value >= 0.5:
        classification = "device-bound"
    elif hdbi_value >= 0.2:
        classification = "balanced"
    else:
        classification = "host-bound"

    return {
        "hdbi_value": hdbi_value,
        "hdbi_classification": classification,
        "t_device_active_ms": t_device_active,
        "t_orchestrate_ms": t_orchestrate,
        "delta_kt_ms": delta_kt,
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
        
        stream_kernels = [op for op in ops_on_stream if op.get("type") == "kernel"]
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

def compute_kernel_fragmentation(events: Dict[str, Any], total_output_tokens: int) -> Dict[str, Any]:
    """
    Compute kernel launch fragmentation metrics for MoE and dense model analysis.

    MoE models dispatch many small kernels per output token (routing, gating, expert
    selection).  This function quantifies that fragmentation signal.

    Args:
        events: Dictionary from collect_events() with hierarchical GPU event structure.
        total_output_tokens: Number of output (decode) tokens in the profiled run.

    Returns:
        Dictionary with fragmentation metrics:
            total_kernel_launches: Raw kernel count in this trace.
            kernels_per_output_token: Normalized launch rate (None if tokens==0).
            unique_kernel_count: Number of distinct kernel names.
            kernel_diversity_ratio: unique_kernel_count / total_kernel_launches.
    """
    kernels = events.get("gpu", {}).get("kernels", [])
    n = len(kernels)
    unique_names = len(set(k.get("name", "") for k in kernels))
    return {
        "total_kernel_launches": n,
        "kernels_per_output_token": round(n / total_output_tokens, 2) if total_output_tokens > 0 else None,
        "unique_kernel_count": unique_names,
        "kernel_diversity_ratio": round(unique_names / n, 4) if n > 0 else 0.0,
    }


def generate_synthetic_inputs(
    tokenizer, 
    device: torch.device, 
    batch_size: int, 
    seq_len: int,
    model_config=None  # Add model config parameter
) -> Dict[str, torch.Tensor]:
    """
    Generates synthetic tokenized inputs for profiling.
    """
    # Ensure pad_token is set (GPT-2 doesn't have one by default)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Get max position embeddings from model config (most reliable)
    max_pos = seq_len  # default to requested
    if model_config is not None:
        # Different models use different attribute names
        if hasattr(model_config, 'max_position_embeddings'):
            max_pos = model_config.max_position_embeddings
        elif hasattr(model_config, 'n_positions'):
            max_pos = model_config.n_positions
        elif hasattr(model_config, 'n_ctx'):
            max_pos = model_config.n_ctx
    
    # Clamp seq_len to model's max position embeddings
    if seq_len > max_pos:
        print(f"Warning: seq_len {seq_len} exceeds model max_position_embeddings {max_pos}. Clamping.")
        seq_len = max_pos

    # Generate random token IDs within valid vocab range
    input_ids = torch.randint(
        1,
        tokenizer.vocab_size - 1,
        (batch_size, seq_len),
        dtype=torch.long,
        device=device
    )

    attention_mask = torch.ones(
        (batch_size, seq_len),
        dtype=torch.long,
        device=device
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }