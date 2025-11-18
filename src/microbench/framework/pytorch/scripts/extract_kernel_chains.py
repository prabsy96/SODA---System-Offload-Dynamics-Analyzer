import torch
import json
import re
import sys
import os
import copy
import shutil
from collections import defaultdict
from torch.profiler import profile, ProfilerActivity
from soda import ModelHandler, SodaProfiler

# GEMM operations to extract
GEMM_OPS = ['aten::addmm', 'aten::mm', 'aten::bmm']

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

def calculate_avg_min_max(values, base_name):
    """Calculate avg/min/max from a list of values and return dict with base_name."""
    if not values:
        return {}
    return {
        f"avg_{base_name}": sum(values) / len(values),
        f"min_{base_name}": min(values),
        f"max_{base_name}": max(values)
    }

def make_kernel_identity_key(kernel, cpu_op):
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
    input_dims = cpu_op.get("input_dims", []) if cpu_op else []
    input_dims_tuple = tuple(tuple(d) if isinstance(d, list) else d for d in input_dims)
    return (kernel_name, grid_tuple, block_tuple, shared_mem, input_dims_tuple)

def group_chains_by_identity(chains):
    """
    Group kernel chains by kernel identity key.
    Returns a dict: key -> list[chain]
    """
    grouped = defaultdict(list)
    for e in chains:
        if e.get("kernel") and e.get("cpu_op"):
            key = make_kernel_identity_key(e["kernel"], e["cpu_op"])
            grouped[key].append(e)
    return grouped

def aggregate_execution_metrics(instances):
    """
    Aggregate execution metrics across a group of identical kernel chains.
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

def collect_events_from_trace(trace):
    """First pass: collect all events by type from trace."""
    cpu_ops = {}
    cuda_launches = {}
    kernels = []
    
    for event in trace.get("traceEvents", []):
        cat = event.get("cat")
        args = event.get("args", {})
        external_id = args.get("External id")
        correlation = args.get("correlation")
        
        if cat == "cpu_op" and external_id is not None:
            cpu_ops[external_id] = {
                "type": "cpu_op",
                "name": event.get("name", ""),
                "external_id": external_id,
                "input_dims": args.get("Input Dims", []),
                "input_strides": args.get("Input Strides", []),
                "input_type": args.get("Input type", []),
                "concrete_inputs": args.get("Concrete Inputs", []),
                "ts": event.get("ts"),
                "dur": event.get("dur")
            }
        elif cat == "cuda_runtime" and event.get("name") == "cudaLaunchKernel":
            if external_id is not None and correlation is not None:
                cuda_launches[correlation] = {
                    "type": "cuda_launch",
                    "external_id": external_id,
                    "correlation": correlation,
                    "ts": event.get("ts"),
                    "dur": event.get("dur"),
                    "cbid": args.get("cbid")
                }
        elif cat == "kernel" and external_id is not None and correlation is not None:
            kernels.append({
                "type": "kernel",
                "name": event.get("name", ""),
                "external_id": external_id,
                "correlation": correlation,
                "grid": args.get("grid"),
                "block": args.get("block"),
                "shared_memory": args.get("shared memory"),
                "registers_per_thread": args.get("registers per thread"),
                "blocks_per_SM": args.get("blocks per SM"),
                "warps_per_SM": args.get("warps per SM"),
                "occupancy": args.get("est. achieved occupancy %"),
                "stream": args.get("stream"),
                "device": args.get("device"),
                "context": args.get("context"),
                "queued": args.get("queued"),
                "dur": event.get("dur"),
                "ts": event.get("ts")
            })
    
    return cpu_ops, cuda_launches, kernels

def build_kernel_chains(kernels, cpu_ops, cuda_launches):
    """Second pass: build linked kernel causal structure starting from kernels."""
    kernel_chains = []
    
    for kernel in kernels:
        external_id = kernel["external_id"]
        correlation = kernel["correlation"]
        cpu_op = cpu_ops.get(external_id)
        cuda_launch = cuda_launches.get(correlation)
        
        kernel_tax = None
        if kernel.get('ts') is not None and cuda_launch and cuda_launch.get('ts') is not None:
            kernel_tax = kernel['ts'] - cuda_launch['ts']
            assert kernel_tax >= 0, f"Negative kernel tax detected: kernel.ts={kernel['ts']}, cudaLaunchKernel.ts={cuda_launch['ts']}, tax={kernel_tax}"
        
        kernel_chains.append({
            "kernel": kernel,
            "cuda_launch": cuda_launch,
            "cpu_op": cpu_op,
            "kernel_tax": kernel_tax
        })
    
    return kernel_chains

def filter_gemm_kernel_chains(kernel_chains):
    """Filter for GEMM kernels only."""
    gemm_chains = []
    for e in kernel_chains:
        # Must be from a GEMM operation
        if e.get('cpu_op') and e['cpu_op']['name'] in GEMM_OPS:
            # Kernel name must contain 'gemm' (eg: volta_sgemm_64x32_sliced1x4_nn)
            if 'gemm' in e.get('kernel', {}).get('name', '').lower():
                gemm_chains.append(copy.deepcopy(e))
    return gemm_chains

def save_output(output_dir, filename, data, env_metadata, summary=None):
    """Save data to JSON file. All time values in microseconds."""
    if summary is None:
        summary = {
            "total_kernels": len([c for c in data if c.get("kernel")]),
            "total_cpu_ops": len([c for c in data if c.get("cpu_op")]),
            "total_cuda_launches": len([c for c in data if c.get("cuda_launch")]),
            "total_chains": len(data)
        }
    
    output = {
        "summary": summary,
        "causal_chains": data,
        "env_metadata": env_metadata
    }
    
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)

def create_unique_kernels(gemm_chains):
    """Create unique GEMM kernels with averaged execution metrics."""
    unique_groups = group_chains_by_identity(gemm_chains)
    
    unique_gemm_chains = []
    for key, instances in unique_groups.items():
        
        # Verify static properties 
        base_instance = copy.deepcopy(instances[0])
        base_kernel = base_instance['kernel']
        
        # Verify static properties
        static_props = ['shared_memory', 'registers_per_thread', 'occupancy', 'blocks_per_SM', 'warps_per_SM', 'stream', 'device', 'context']
        for prop in static_props:
            values = [inst['kernel'].get(prop) for inst in instances if inst['kernel'].get(prop) is not None]
            if values and len(set(values)) > 1:
                print(f"Warning: {prop} inconsistent for kernel {key[0][:50]}: {set(values)}")
        # Aggregate execution metrics (kernel tax, cpu op duration, cuda launch duration, kernel duration)
        base_instance = aggregate_execution_metrics(instances)
        
        # Remove timestamp
        base_instance['kernel'].pop('ts', None)
        
        # Construct output structure 
        chain_entry = {
            "meta": base_instance.get('meta'),
            "kernel": base_instance['kernel'],
            "cuda_launch": base_instance.get('cuda_launch'),
            "cpu_op": base_instance.get('cpu_op')
        }
        
        unique_gemm_chains.append(chain_entry)
    
    return unique_gemm_chains

def extract_kernel_chains(trace_file):
    """
    Extract all kernel causal chains from trace file.
    Returns: kernel_chains
    """
    with open(trace_file, "r") as f:
        trace = json.load(f)
    
    cpu_ops, cuda_launches, kernels = collect_events_from_trace(trace)
    kernel_chains = build_kernel_chains(kernels, cpu_ops, cuda_launches)
    
    return kernel_chains

def save_all_kernel_chains(kernel_chains, env_metadata, output_dir):
    """Save all extracted kernel causal chains."""
    save_output(output_dir, "all_kernel_chains.json", kernel_chains, env_metadata)

def save_gemm_kernel_chains(gemm_chains, env_metadata, output_dir):
    """Save filtered GEMM kernel causal chains."""
    save_output(output_dir, "gemm_kernel_chains.json", gemm_chains, env_metadata)

def save_unique_gemm_kernel_chains(unique_chains, gemm_chains, env_metadata, output_dir):
    """Save deduplicated unique GEMM kernel causal chains."""
    summary = {
        "total_kernels": len(unique_chains),
        "total_cpu_ops": len(unique_chains),
        "total_cuda_launches": len(unique_chains),
        "total_chains": len(unique_chains),
        "original_count": len(gemm_chains),
        "deduplication_ratio": f"{len(unique_chains)}/{len(gemm_chains)}"
    }
    save_output(output_dir, "unique_gemm_kernel_chains.json", unique_chains, env_metadata, summary=summary)

def run_extraction_pipeline(trace_file):
    """
    Main pipeline: extract -> save, filter -> save, deduplicate -> save.
    """
    output_dir = os.environ.get("PYTORCH_OUTPUT", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model trace to traces/model_trace folder
    model_trace_dir = os.path.join(output_dir, "traces", "model_trace")
    os.makedirs(model_trace_dir, exist_ok=True)
    model_trace_path = os.path.join(model_trace_dir, "model_trace.json")
    shutil.copy2(trace_file, model_trace_path)
    
    # Collect environment metadata once
    env_metadata = collect_env_metadata()
    
    # Step 1: Extract all kernel chains
    kernel_chains = extract_kernel_chains(trace_file)
    save_all_kernel_chains(kernel_chains, env_metadata, output_dir)
    
    # Step 2: Filter GEMM kernel chains
    gemm_chains = filter_gemm_kernel_chains(kernel_chains)
    save_gemm_kernel_chains(gemm_chains, env_metadata, output_dir)
    
    # Step 3: Create unique kernel chains
    unique_gemm_chains = create_unique_kernels(gemm_chains)
    save_unique_gemm_kernel_chains(unique_gemm_chains, gemm_chains, env_metadata, output_dir)
    
    return {
        "chains": kernel_chains,
        "env_metadata": env_metadata
    }

def sanitize_trace_file(trace_file):
    """Sanitize trace file by removing control characters."""
    with open(trace_file, "rb") as f:
        content = f.read().decode('utf-8', errors='replace')
    sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', content)
    with open(trace_file, "w") as f:
        f.write(sanitized)

def generate_trace(model_handler, model_inputs):
    """Generate and sanitize trace file."""
    print("Generating trace...")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with torch.no_grad():
            _ = model_handler.pytorch_model(**model_inputs)
    
    trace_file = "temp_trace.json"
    prof.export_chrome_trace(trace_file)
    sanitize_trace_file(trace_file)
    
    return trace_file

def print_summary():
    """Print extraction summary from saved files."""
    output_dir = os.environ.get("PYTORCH_OUTPUT", "output")
    with open(os.path.join(output_dir, 'all_kernel_chains.json'), 'r') as f:
        all_data = json.load(f)
    with open(os.path.join(output_dir, 'gemm_kernel_chains.json'), 'r') as f:
        gemm_data = json.load(f)
    with open(os.path.join(output_dir, 'unique_gemm_kernel_chains.json'), 'r') as f:
        unique_data = json.load(f)
    
    print(f"Extracted {all_data['summary']['total_kernels']} kernels")
    print(f"Linked {all_data['summary']['total_chains']} kernel chains")
    print(f"Saved to {os.path.join(output_dir, 'all_kernel_chains.json')}")
    print(f"Saved {gemm_data['summary']['total_kernels']} GEMM kernel chains to {os.path.join(output_dir, 'gemm_kernel_chains.json')}")
    print(f"Saved {unique_data['summary']['total_kernels']} unique GEMM kernel chains to {os.path.join(output_dir, 'unique_gemm_kernel_chains.json')}")
    print(f"\t* Uniqueness key: kernel_name + grid + block + shared_memory + input_dims")

if __name__ == "__main__":
    args = SodaProfiler.parse_and_validate_args()
    
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
    
    trace_file = generate_trace(model_handler, model_inputs)
    run_extraction_pipeline(trace_file)
    os.remove(trace_file)
    
    print_summary()