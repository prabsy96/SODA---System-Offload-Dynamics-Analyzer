"""
Utility functions for trace data collection and processing.
"""
from typing import Any, Dict


def get_clean_kernel_name(kernel_name: str) -> str:
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


def collect_events_from_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
    """
    Collects all events from trace organized by category.
    
    Args:
        trace: Chrome trace format dictionary with "traceEvents" key.
    
    Returns:
        Dictionary with hierarchical structure:
        - cpu: Dict with keys:
            - ops: Dict[external_id, cpu_op_dict] - CPU operations
            - launches: Dict[correlation_id, cuda_launch_dict] - CUDA runtime launches
        - gpu: Dict with keys:
            - kernels: List of kernel events
            - memory: List of memcpy/memset events
            - all: List of all GPU events
    """
    op_events_by_ext_id = {}
    cuda_launch_events_by_corr = {}
    kernel_events = []
    gpu_mem_events = []
    
    for event in trace.get("traceEvents", []):
        cat = event.get("cat")
        args = event.get("args", {})
        external_id = args.get("External id")
        correlation = args.get("correlation")
        
        if cat == "cpu_op" and external_id is not None:
            op_events_by_ext_id[external_id] = {
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
                cuda_launch_events_by_corr[correlation] = {
                    "type": "cuda_launch",
                    "name": event.get("name", ""),
                    "external_id": external_id,
                    "correlation": correlation,
                    "ts": event.get("ts"),
                    "dur": event.get("dur"),
                    "cbid": args.get("cbid")
                }
        elif cat == "kernel" and external_id is not None and correlation is not None:
            kernel_events.append({
                "type": "kernel",
                "name": get_clean_kernel_name(event.get("name", "")),
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
        elif cat == "gpu_memcpy" or cat == "gpu_memset":
            gpu_mem_events.append({
                "type": cat,
                "name": event.get("name", ""),
                "correlation": correlation,
                "stream": args.get("stream"),
                "device": args.get("device"),
                "context": args.get("context"),
                "ts": event.get("ts"),
                "dur": event.get("dur"),
                "bytes": args.get("bytes"),
                "memory_bandwidth_gbs": args.get("memory bandwidth (GB/s)") if cat == "gpu_memcpy" else None
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

