import torch
import json
import re
import sys
import os
import copy
import shutil
from collections import defaultdict
from pathlib import Path
from soda import ModelHandler, SodaProfiler, SodaTraceProcessor
from soda import utils


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

def setup_deterministic_mode(seed=1234):
    """
    Lock down all non-determinism knobs for reproducible kernel selection.
    Sets PyTorch flags and environment variables to minimize randomness.
    """
    # Set random seeds
    torch.manual_seed(seed)
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

def run_extraction_pipeline(trace_file):
    """
    Main pipeline: extract -> save, filter -> save, deduplicate -> save.
    """
    trace_file = Path(trace_file)
    traces_dir = utils.get_path("PYTORCH_TRACES")
    
    # Save model trace to traces/model_trace folder
    model_trace_dir = utils.get_path("PYTORCH_MODEL_TRACE_DIR")
    utils.ensure_dir(model_trace_dir)
    model_trace_file = utils.get_path("PYTORCH_MODEL_TRACE_FILE")
    shutil.copy2(trace_file, model_trace_file)
    
    # Collect environment metadata once
    env_metadata = collect_env_metadata()
    
    # Save env_metadata separately
    env_metadata_file = utils.get_path("PYTORCH_ENV_METADATA")
    utils.ensure_dir(env_metadata_file.parent)
    utils.save_json(env_metadata_file, env_metadata)
    
    # Step 1: Extract all event sequences
    trace = utils.load_json(trace_file)
    events = SodaProfiler.collect_events_from_trace(trace)
    event_sequences = SodaProfiler.get_linked_event_sequences(events)
    event_sequences = SodaProfiler.calculate_per_seq_launch_tax(event_sequences)
    utils.save_json(utils.get_path("PYTORCH_ALL_KERNELS"), {
        "summary": generate_summary(event_sequences),
        "sequences": event_sequences
    })
    
    # Step 2: Filter GEMM event sequences
    gemm_sequences = utils.filter_gemm_sequences(event_sequences)
    utils.save_json(utils.get_path("PYTORCH_GEMM_KERNELS"), {
        "summary": generate_summary(gemm_sequences),
        "sequences": gemm_sequences
    })
    
    # Step 3: Create unique event sequences
    unique_gemm_sequences = utils.deduplicate_and_aggregate(gemm_sequences)
    utils.save_json(utils.get_path("PYTORCH_UNIQUE_KERNELS"), {
        "summary": generate_summary(
            unique_gemm_sequences,
            original_count=len(gemm_sequences),
            deduplication_ratio=f"{len(unique_gemm_sequences)}/{len(gemm_sequences)}"
        ),
        "sequences": unique_gemm_sequences
    })
    
    return {
        "sequences": event_sequences,
        "env_metadata": env_metadata
    }

def print_summary():
    """Print extraction summary from saved files."""
    all_kernels_file = utils.get_path("PYTORCH_ALL_KERNELS")
    gemm_kernels_file = utils.get_path("PYTORCH_GEMM_KERNELS")
    unique_kernels_file = utils.get_path("PYTORCH_UNIQUE_KERNELS")
    
    all_data = utils.load_json(all_kernels_file)
    gemm_data = utils.load_json(gemm_kernels_file)
    unique_data = utils.load_json(unique_kernels_file)
    
    print(f"Extracted {all_data['summary']['total_kernels']} kernels")
    print(f"Linked {all_data['summary']['total_sequences']} event sequences")
    print(f"Saved to {all_kernels_file}")
    print(f"Saved {gemm_data['summary']['total_kernels']} GEMM event sequences to {gemm_kernels_file}")
    print(f"Saved {unique_data['summary']['total_kernels']} unique GEMM event sequences to {unique_kernels_file}")
    print(f"\t* Uniqueness key: kernel_name + grid + block + shared_memory + input_dims")

if __name__ == "__main__":
    args = utils.parse_and_validate_args()
    
    os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/tmp/hf")
    
    print("Setting up deterministic mode...")
    setup_deterministic_mode(seed=1234)
    
    model_handler = ModelHandler(
        model_name=args.model,
        device=args.device,
        compile_type=args.compile_type,
        precision=args.precision,
    )
    
    model_inputs = model_handler.generate_synthetic_inputs(
        args.batch_size, args.seq_len
    )
    profiler = SodaProfiler(model_handler=model_handler, args=args, log_console=False, log_file=False)
    profiler.trace_file_path = Path("temp_trace.json")
    trace_file = profiler.profile_forward_pass(model_inputs)
    run_extraction_pipeline(trace_file)
    os.remove(trace_file)
    
    print_summary()