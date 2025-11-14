import torch
import json
import sys
import os
import argparse
import shutil
import copy
from torch.profiler import profile, ProfilerActivity

profiling_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, profiling_dir)
from extract_kernel_chains import setup_deterministic_mode, collect_env_metadata, sanitize_trace_file, save_output, extract_kernel_chains, filter_gemm_kernel_chains, calculate_avg_min_max, make_kernel_identity_key, group_chains_by_identity, aggregate_execution_metrics
from verify_replayed_kernels import get_clean_kernel_name

def restore_environment(metadata):
    torch.manual_seed(metadata["seeds"]["torch_manual_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(metadata["seeds"]["torch_manual_seed"])
    
    torch.backends.cudnn.benchmark = metadata["cudnn"]["benchmark"]
    torch.backends.cudnn.deterministic = metadata["cudnn"]["deterministic"]
    torch.backends.cudnn.allow_tf32 = metadata["cudnn"]["allow_tf32"]
    
    if metadata.get("matmul_precision"):
        try:
            torch.set_float32_matmul_precision(metadata["matmul_precision"])
        except AttributeError:
            pass
    
    if metadata["blas"]["cublas_workspace_config"]:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = metadata["blas"]["cublas_workspace_config"]
    if metadata["blas"]["cublaslt_allow_tf32"]:
        os.environ["CUBLASLT_ALLOW_TF32"] = metadata["blas"]["cublaslt_allow_tf32"]
    
    for key, value in metadata["env"].items():
        if value is not None:
            os.environ[key] = value
    
    if metadata["seeds"]["cuda_deterministic_algorithms"]:
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            pass

def recreate_tensor(dims, dtype_str, strides=None, device="cuda"):
    """Create a tensor with specified dimensions, dtype, and strides."""
    if not dims or dims == []:
        return None
    
    # Map dtype strings to torch dtypes
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
    dtype = dtype_map.get(dtype_str.lower(), torch.float32)
    tensor = torch.randn(*dims, dtype=dtype, device=device)
    
    if strides and len(strides) == len(dims):
        default_strides = []
        stride = 1
        for dim in reversed(dims):
            default_strides.insert(0, stride)
            stride *= dim
        
        if strides != default_strides:
            max_size = max(s * d for s, d in zip(strides, dims) if d > 0)
            storage = torch.empty(max_size + 100, dtype=dtype, device=device)
            tensor = torch.as_strided(storage, dims, strides)
            source = torch.randn(*dims, dtype=dtype, device=device)
            tensor.copy_(source)
    
    return tensor

def execute_pytorch_operation(op_name, inputs):
    """Execute a GEMM operation (addmm, mm, or bmm)."""
    if op_name == "aten::addmm" and len(inputs) >= 3:
        return torch.addmm(inputs[0], inputs[1], inputs[2])
    elif op_name == "aten::mm" and len(inputs) >= 2:
        return torch.mm(inputs[0], inputs[1])
    elif op_name == "aten::bmm" and len(inputs) >= 2:
        return torch.bmm(inputs[0], inputs[1])
    else:
        raise ValueError(f"Unsupported operation: {op_name}")

def create_input_tensors(cpu_op, seed):
    """Create input tensors from CPU operation metadata."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    input_dims = cpu_op["input_dims"]
    input_types = cpu_op.get("input_type", [])
    input_strides = cpu_op.get("input_strides", [])
    
    inputs = []
    for i, dims in enumerate(input_dims):
        if dims == []:
            continue
        dtype_str = input_types[i] if i < len(input_types) else "float"
        strides = input_strides[i] if i < len(input_strides) else None
        tensor = recreate_tensor(dims, dtype_str, strides)
        if tensor is not None:
            inputs.append(tensor)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    return inputs

def profile_operation(op_name, inputs, runs=1, trace_filename=None, warmup_runs=100):
    """Profile a PyTorch operation N times and return trace file path."""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    if trace_filename is None:
        trace_file = os.path.join(output_dir, "temp_replay_trace.json")
    else:
        trace_file = trace_filename
    
    # Warmup: execute without profiling to stabilize kernels/caches
    with torch.no_grad():
        for _ in range(max(0, warmup_runs)):
            execute_pytorch_operation(op_name, inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with torch.no_grad():
            for _ in range(runs):
                execute_pytorch_operation(op_name, inputs)
                # Synchronize to ensure each kernel completes before launching the next
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
    
    prof.export_chrome_trace(trace_file)
    sanitize_trace_file(trace_file)
    
    return trace_file

def aggregate_replayed_chains(replayed_chains, runs):
    """Aggregate statistics across N runs, grouping by kernel identity."""
    if runs == 1 or len(replayed_chains) == 0:
        return replayed_chains
    
    # Group chains by kernel identity 
    grouped_chains = group_chains_by_identity(replayed_chains)
    
    aggregated_kernel_chains = []
    for identity_key, instances in grouped_chains.items():
        aggregated_instance = aggregate_execution_metrics(instances)
        aggregated_kernel_chains.append(aggregated_instance)
    
    return aggregated_kernel_chains

def replay_kernel_from_cpu_op(cpu_op, exp_kernel_name, metadata, kernel_idx, runs=1, warmup_runs=100):
    """Replay the operation N times and extract resulting kernel chains from trace."""
    seed = metadata["seeds"]["torch_manual_seed"]
    inputs = create_input_tensors(cpu_op, seed)
    
    # Save trace file in kernel_traces folder
    output_dir = "output"
    kernel_traces_dir = os.path.join(output_dir, "traces", "kernel_traces")
    os.makedirs(kernel_traces_dir, exist_ok=True)
    
    # Generate trace filename based on operation name and expected kernel name
    op_name_short = cpu_op.get("name", "unknown").replace("::", "_")
    trace_filename = os.path.join(kernel_traces_dir, f"{kernel_idx:02d}_{op_name_short}_{exp_kernel_name}.json")
    
    # Profile the operation and extract kernel chains from the trace file
    trace_file = profile_operation(cpu_op["name"], inputs, runs=runs, trace_filename=trace_filename, warmup_runs=warmup_runs)
    kernel_chains = extract_kernel_chains(trace_file)
    
    # Aggregate statistics across N runs
    agg_kernel_chains = aggregate_replayed_chains(kernel_chains, runs)
    
    return agg_kernel_chains

def replay_all_kernel_chains(kernel_chains, env_metadata, runs=1, warmup_runs=100):
    """Replay all kernel chains, optionally running each N times for microbenchmarking."""
    seed = env_metadata["seeds"]["torch_manual_seed"]
    all_replayed_kernel_chains = []
    
    print(f"Replaying {len(kernel_chains)} kernels with {runs} run{'s' if runs > 1 else ''} each (warmup={warmup_runs})")
    
    # Replay each kernel chain
    for i, kernel_chain in enumerate(kernel_chains):
        cpu_op = kernel_chain.get("cpu_op")
        assert cpu_op is not None, f"CPU operation is None for kernel chain {i}"
        
        exp_kernel_name = get_clean_kernel_name(kernel_chain['kernel']['name'])
        print(f"* [{i+1}/{len(kernel_chains)}] {cpu_op['name']} -> {exp_kernel_name}")
        
        replayed_chains = replay_kernel_from_cpu_op(cpu_op, exp_kernel_name, env_metadata, i+1, runs=runs, warmup_runs=warmup_runs)
        all_replayed_kernel_chains.extend(replayed_chains)
    
    return all_replayed_kernel_chains

def save_replayed_kernel_chains(kernel_chains, env_metadata, output_dir):
    """Save replayed kernel causal chains."""
    save_output(output_dir, "replayed_gemm_kernel_chains.json", kernel_chains, env_metadata)

def run_replay_pipeline(input_file, runs=1, warmup_runs=100):
    """
    Main pipeline: load -> setup -> replay all -> save.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Clean previous kernel traces 
    kernel_traces_dir = os.path.join(output_dir, "traces", "kernel_traces")
    if os.path.isdir(kernel_traces_dir):
        shutil.rmtree(kernel_traces_dir)
    
    # Collect environment metadata once
    env_metadata = collect_env_metadata()
    
    # Step 1: Load kernel chains
    if os.path.isabs(input_file) or os.path.exists(input_file):
        file_path = input_file
    else:
        file_path = os.path.join(output_dir, input_file)
    
    with open(file_path, "r") as f:
        data = json.load(f)
    
    # Step 2: Setup environment
    restore_environment(data["env_metadata"])
    setup_deterministic_mode(seed=data["env_metadata"]["seeds"]["torch_manual_seed"])
    
    # Step 3: Replay all kernel chains (each N times if runs > 1)
    replayed_kernel_chains = replay_all_kernel_chains(data["causal_chains"], data["env_metadata"], runs=runs, warmup_runs=warmup_runs)
    
    # Step 4: Filter GEMM kernel chains
    replayed_gemm_chains = filter_gemm_kernel_chains(replayed_kernel_chains)
    save_replayed_kernel_chains(replayed_gemm_chains, env_metadata, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay kernel chains from extracted metadata")
    parser.add_argument("input_file", help="Input JSON file with kernel chains (e.g., unique_gemm_kernel_chains.json)")
    parser.add_argument("--runs", type=int, default=1, help="Number of times to replay each kernel (default: 1, use >1 for microbenchmarking)")
    parser.add_argument("--warmup", type=int, default=100, help="Number of warmup iterations before profiling (default: 100)")
    
    args = parser.parse_args()
    run_replay_pipeline(args.input_file, runs=args.runs, warmup_runs=args.warmup)
