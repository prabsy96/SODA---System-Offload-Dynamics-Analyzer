import torch
import json
import sys
import os
import argparse
import shutil
import copy
from pathlib import Path
from torch.profiler import profile, ProfilerActivity

profiling_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, profiling_dir)
from extract_kernel_sequences import setup_deterministic_mode, collect_env_metadata, sanitize_trace_file, save_output, filter_gemm_sequences, calculate_avg_min_max, make_kernel_identity_key, group_sequences_by_identity, aggregate_execution_metrics
from soda import SodaProfiler

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

def profile_operation(op_name, inputs, trace_filename, runs=1, warmup_runs=100):
    """Profile a PyTorch operation N times and return trace file path."""
    trace_file = Path(trace_filename)
    
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
    
    prof.export_chrome_trace(str(trace_file))
    sanitize_trace_file(str(trace_file))
    
    return trace_file

def aggregate_replayed_sequences(replayed_sequences, runs):
    """Aggregate statistics across N runs, grouping by kernel identity."""
    if runs == 1 or len(replayed_sequences) == 0:
        return replayed_sequences
    
    # Group sequences by kernel identity 
    grouped_sequences = group_sequences_by_identity(replayed_sequences)
    
    aggregated_sequences = []
    for identity_key, instances in grouped_sequences.items():
        aggregated_instance = aggregate_execution_metrics(instances)
        aggregated_sequences.append(aggregated_instance)
    
    return aggregated_sequences

def replay_kernel_from_cpu_op(cpu_op, exp_kernel_name, metadata, kernel_idx, runs=1, warmup_runs=100):
    """Replay the operation N times and extract resulting event sequences from trace."""
    seed = metadata["seeds"]["torch_manual_seed"]
    inputs = create_input_tensors(cpu_op, seed)
    
    # Save trace file in kernel_traces folder
    kernel_traces_dir = SodaProfiler.get_path("PYTORCH_KERNEL_TRACES_DIR")
    SodaProfiler.ensure_dir(kernel_traces_dir)
    
    # Generate trace filename based on operation name and expected kernel name
    op_name_short = cpu_op.get("name", "unknown").replace("::", "_")
    trace_filename = kernel_traces_dir / f"{kernel_idx:02d}_{op_name_short}_{exp_kernel_name}.json"
    
    # Profile the operation and extract event sequences from the trace file
    trace_file = profile_operation(cpu_op["name"], inputs, runs=runs, trace_filename=trace_filename, warmup_runs=warmup_runs)
    trace = SodaProfiler.load_json(trace_file)
    events = SodaProfiler.collect_events_from_trace(trace)
    event_sequences = SodaProfiler.get_linked_event_sequences(events)
    event_sequences = SodaProfiler.calculate_per_seq_tklqt(event_sequences)
    
    # Aggregate statistics across N runs
    agg_sequences = aggregate_replayed_sequences(event_sequences, runs)
    
    return agg_sequences

def replay_all_event_sequences(event_sequences, env_metadata, runs=1, warmup_runs=100):
    """Replay all event sequences, optionally running each N times for microbenchmarking."""
    seed = env_metadata["seeds"]["torch_manual_seed"]
    all_replayed_sequences = []
    
    print(f"Replaying {len(event_sequences)} kernels with {runs} run{'s' if runs > 1 else ''} each (warmup={warmup_runs})")
    
    # Replay each event sequence
    for i, event_sequence in enumerate(event_sequences):
        cpu_op = event_sequence.get("cpu_op")
        assert cpu_op is not None, f"CPU operation is None for event sequence {i}"
        
        exp_kernel_name = SodaProfiler.get_clean_kernel_name(event_sequence['kernel']['name'])
        print(f"* [{i+1}/{len(event_sequences)}] {cpu_op['name']} -> {exp_kernel_name}")
        
        replayed_sequences = replay_kernel_from_cpu_op(cpu_op, exp_kernel_name, env_metadata, i+1, runs=runs, warmup_runs=warmup_runs)
        all_replayed_sequences.extend(replayed_sequences)
    
    return all_replayed_sequences


def run_replay_pipeline(runs=1, warmup_runs=100):
    """
    Main pipeline: load -> setup -> replay all -> save.
    """
    # Clean previous kernel traces 
    kernel_traces_dir = SodaProfiler.get_path("PYTORCH_KERNEL_TRACES_DIR")
    if kernel_traces_dir.is_dir():
        shutil.rmtree(kernel_traces_dir)
    
    # Collect environment metadata once
    env_metadata = collect_env_metadata()
    
    # Step 1: Load event sequences 
    input_file = SodaProfiler.get_path("PYTORCH_UNIQUE_KERNELS")
    data = SodaProfiler.load_json(input_file)
    
    # Step 2: Setup environment
    restore_environment(data["env_metadata"])
    setup_deterministic_mode(seed=data["env_metadata"]["seeds"]["torch_manual_seed"])
    
    # Step 3: Replay all event sequences (each N times if runs > 1)
    replayed_sequences = replay_all_event_sequences(data["sequences"], data["env_metadata"], runs=runs, warmup_runs=warmup_runs)
    
    # Step 4: Filter GEMM event sequences
    replayed_gemm_sequences = filter_gemm_sequences(replayed_sequences)
    save_output(SodaProfiler.get_path("PYTORCH_REPLAYED_KERNELS"), replayed_gemm_sequences, env_metadata)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay event sequences from extracted metadata")
    parser.add_argument("--runs", type=int, default=1, help="Number of times to replay each kernel (default: 1, use >1 for microbenchmarking)")
    parser.add_argument("--warmup", type=int, default=100, help="Number of warmup iterations before profiling (default: 100)")
    
    args = parser.parse_args()
    run_replay_pipeline(runs=args.runs, warmup_runs=args.warmup)
