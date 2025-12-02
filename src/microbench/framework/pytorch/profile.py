from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
from torch.profiler import profile, ProfilerActivity
from soda.common import utils

# Helper functions
def create_input_tensors(cpu_op: Dict[str, Any]) -> List[torch.Tensor]:
    """
    Create input tensors from CPU operation metadata.
    """
    input_dims = cpu_op["input_dims"]
    input_types = cpu_op["input_type"]
    input_strides = cpu_op["input_strides"]
    
    inputs = []
    for i, dims in enumerate(input_dims):
        if dims == []:
            continue
        dtype_str = input_types[i] if i < len(input_types) else "float"
        strides = input_strides[i] if i < len(input_strides) else None
        tensor = create_tensor(dims, dtype_str, strides)
        if tensor is not None:
            inputs.append(tensor)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    return inputs

def create_tensor(
    dims: List[int],
    dtype_str: str,
    strides: Optional[List[int]] = None,
    device: str = "cuda"
) -> Optional[torch.Tensor]:
    """
    Create a tensor with specified dimensions, dtype, and strides.
    """
    if not dims or dims == []:
        return None
    
    # Map dtype strings to torch dtypes
    dtype = utils.parse_dtype_to_torch(dtype_str)
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

def execute_operation(
    op_name: str,
    inputs: List[torch.Tensor]
) -> torch.Tensor:
    """
    Execute a GEMM operation (addmm, mm, or bmm).
    """
    if op_name == "aten::addmm" and len(inputs) >= 3:
        return torch.addmm(inputs[0], inputs[1], inputs[2])
    elif op_name == "aten::mm" and len(inputs) >= 2:
        return torch.mm(inputs[0], inputs[1])
    elif op_name == "aten::bmm" and len(inputs) >= 2:
        return torch.bmm(inputs[0], inputs[1])
    else:
        raise ValueError(f"Unsupported operation: {op_name}")

def profile_operation(
    op_name: str,
    inputs: List[torch.Tensor],
    warmup: int,
    runs: int,
    trace_file: Path
) -> None:
    """
    Profile a PyTorch operation N times and return trace file path.
    """
    # Warmup: execute without profiling to stabilize kernels/caches
    with torch.no_grad():
        for _ in range(warmup):
            execute_operation(op_name, inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with torch.no_grad():
            for _ in range(runs):
                execute_operation(op_name, inputs)
                # Synchronize to ensure each kernel completes before launching the next
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
    
    prof.export_chrome_trace(str(trace_file))

def replay_sequences_from_cpu_ops(
    sequences: List[Dict[str, Any]], 
    warmup: int,
    runs: int
) -> List[Dict[str, Any]]:
    """
    Replay all event sequences
    """
    # Save trace file in kernel_traces folder
    kernel_traces_dir = utils.get_path("PYTORCH_TRACES")
    utils.ensure_dir(kernel_traces_dir)

    # Store sequences per replay index
    sequence_by_idx = {}
    
    print(f"Profiling {len(sequences)} PyTorch GEMM kernels with {runs} run{'s' if runs > 1 else ''} each (warmup={warmup})")
    
    # Replay each event sequence
    for i, event_sequence in enumerate(sequences):
        cpu_op = event_sequence["cpu_op"]
        kernel = event_sequence["kernel"]

        # FIXME: Clean up
        # assert kernel is not None, f"Kernel is None for event sequence {i}"
        # assert cpu_op is not None, f"CPU operation is None for event sequence {i}"
        expected_kernel = utils.clean_kernel_name(event_sequence['kernel']['name'])
        seq_idx = i+1 
        
        print(f"[{seq_idx}/{len(sequences)}] {cpu_op['name']} -> {expected_kernel}")
        
        # Generate trace filename based on operation name and expected kernel name
        trace_file_name = utils.format_sequence_filename(
            seq_idx, 
            cpu_op['name'], 
            expected_kernel, 
            extension="json"
        )
        trace_file = kernel_traces_dir / trace_file_name

        # Profile the operation 'runs' times with 'warmup' warmup runs
        profile_operation(
            op_name=cpu_op["name"], 
            inputs=create_input_tensors(cpu_op), 
            warmup=warmup, 
            runs=runs, 
            trace_file=trace_file
        )

        # Load trace data from trace file
        trace_data = utils.load_json(trace_file)

        # Collect and link events into sequences from trace data
        events = utils.collect_events(trace_data)
        linked_sequences = utils.link_sequences(events)

        # Calculate kernel tax for event sequences
        linked_sequences_with_tax = utils.calculate_per_seq_launch_tax(linked_sequences)

        agg_sequence = utils.deduplicate_and_aggregate(linked_sequences_with_tax)
        sequence_by_idx[i] = agg_sequence

    # Extend all sequences at the end
    all_replayed_sequences = []
    for kernel_idx in sorted(sequence_by_idx.keys()):
        all_replayed_sequences.extend(sequence_by_idx[kernel_idx])

    utils.validate_sequences(all_replayed_sequences)
    return all_replayed_sequences

def profile_pytorch_gemm_sequences(
    target_gemm_sequences: Dict[str, Any],
    warmup: int,
    runs: int
) -> Dict[str, Any]:
    """
    Profile PyTorch GEMM sequences to measure kernel tax.
    
    Args:
        target_gemm_sequences: Dictionary with target GEMM sequences data.
        warmup: Number of warmup runs.
        runs: Number of measurement runs.
    
    Returns:
        Dictionary with profiled PyTorch GEMM sequences data (same format as saved JSON).
    """
    # Clean previous kernel traces 
    kernel_traces_dir = utils.get_path("PYTORCH_TRACES")
    utils.ensure_dir(kernel_traces_dir, cleanup=True)
    
    # Replay all event sequences from cpu ops
    replayed_sequences = replay_sequences_from_cpu_ops(
        target_gemm_sequences["sequences"],
        warmup=warmup, 
        runs=runs
    )
    
    # Some cpu ops can produce non GEMM kernels as side effects, filter them out
    pytorch_gemm_sequences = utils.filter_gemm_sequences(replayed_sequences)

    pytorch_gemm_sequences_file = utils.get_path("PYTORCH_GEMM_SEQUENCES")
    pytorch_gemm_sequences_data = {
        "summary": {"count": len(pytorch_gemm_sequences)},
        "sequences": pytorch_gemm_sequences
    }
    utils.save_json(pytorch_gemm_sequences_file, pytorch_gemm_sequences_data)
    print(f"Saved {pytorch_gemm_sequences_data['summary']['count']} PyTorch GEMM sequences to {pytorch_gemm_sequences_file}")
    
    return pytorch_gemm_sequences_data
