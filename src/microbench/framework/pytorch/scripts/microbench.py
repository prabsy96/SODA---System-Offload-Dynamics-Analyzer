from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import torch
import shutil
from torch.profiler import profile, ProfilerActivity
from soda import utils

class SodaMicrobench:
    
    def __init__(self, tracer: 'ModelTracer', args):
        self.tracer = tracer
        self.args = args
        # Experiment directory is the base for all paths
        self.experiment_dir = tracer.output_dir
    
    def extract_unique_gemm_sequences(self) -> None:
        """
        Main pipeline: extract -> save, filter -> save, deduplicate -> save.
        """
        # Calculate kernel tax for event sequences
        event_sequences = utils.calculate_per_seq_launch_tax(list(self.tracer.event_sequences))

        # Save all sequences    
        all_sequences_file = utils.get_path("ALL_SEQUENCES", base_path=self.experiment_dir)
        all_sequences_data = {
            "summary": generate_summary(event_sequences),
            "sequences": event_sequences
        }
        utils.save_json(all_sequences_file, all_sequences_data)
        print(f"Saved {all_sequences_data['summary']['total_sequences']} sequences to {all_sequences_file}")

        # Filter and save gemm sequences
        gemm_sequences = utils.filter_gemm_sequences(event_sequences)
        all_gemm_sequences_file = utils.get_path("ALL_GEMM_SEQUENCES", base_path=self.experiment_dir)
        all_gemm_sequences_data = {
            "summary": generate_summary(gemm_sequences),
            "sequences": gemm_sequences
        }
        utils.save_json(all_gemm_sequences_file, all_gemm_sequences_data)
        print(f"Saved {all_gemm_sequences_data['summary']['total_kernels']} GEMM sequences to {all_gemm_sequences_file}")
        
        unique_gemm_sequences = utils.deduplicate_and_aggregate(gemm_sequences)
        unique_gemm_sequences_file = utils.get_path("UNIQUE_GEMM_SEQUENCES", base_path=self.experiment_dir)
        unique_gemm_sequences_data = {
            "summary": generate_summary(
                unique_gemm_sequences,
                original_count=len(gemm_sequences),
                deduplication_ratio=f"{len(unique_gemm_sequences)}/{len(gemm_sequences)}"
            ),
            "sequences": unique_gemm_sequences
        }
        utils.save_json(unique_gemm_sequences_file, unique_gemm_sequences_data)
        print(f"Saved {unique_gemm_sequences_data['summary']['total_kernels']} unique GEMM sequences to {unique_gemm_sequences_file}")
        print(f"\t* Uniqueness key: kernel_name + grid + block + shared_memory + input_dims")

    def replay_gemm_sequences(self) -> None:
        """
        Main pipeline: load -> setup -> replay all -> save.
        """
        # Clean previous kernel traces 
        kernel_traces_dir = utils.get_path("PYTORCH_KERNEL_TRACES_DIR", base_path=self.experiment_dir)
        utils.ensure_dir(kernel_traces_dir, cleanup=True)
        
        # Load unique GEMM sequences
        unique_gemm_sequences_file = utils.get_path("UNIQUE_GEMM_SEQUENCES", base_path=self.experiment_dir)
        unique_gemm_sequences_data = utils.load_json(unique_gemm_sequences_file)
        
        # Replay all event sequences from cpu ops
        replayed_sequences = self.replay_sequences_from_cpu_ops(
            unique_gemm_sequences_data["sequences"], 
            warmup=self.args.warmup, 
            runs=self.args.runs
        )
        
        # Some cpu ops can produce non GEMM kernels as side effects, filter them out
        replayed_gemm_sequences = utils.filter_gemm_sequences(replayed_sequences)

        replayed_gemm_sequences_file = utils.get_path("REPLAYED_GEMM_SEQUENCES", base_path=self.experiment_dir)
        replayed_gemm_sequences_data = {
            "summary": generate_summary(replayed_gemm_sequences),
            "sequences": replayed_gemm_sequences
        }
        utils.save_json(replayed_gemm_sequences_file, replayed_gemm_sequences_data)

    def verify_replayed_sequences(self) -> None:
        """
        Verify replayed sequences against original unique sequences.
        """
        from microbench.framework.pytorch.scripts.verify_replayed_kernels import run_verification_pipeline
        run_verification_pipeline(self.experiment_dir)

    def plot_kernel_tax(self) -> None:
        """
        Plot kernel tax graphs from replayed sequences.
        """
        from microbench.framework.pytorch.scripts.plot_kernel_tax import plot_kernel_tax_pipeline
        plot_kernel_tax_pipeline(self.experiment_dir)

    def run(self) -> None:
        """
        Run the complete microbenchmarking pipeline: extract -> replay -> verify -> plot.
        """
        self.extract_unique_gemm_sequences()
        self.replay_gemm_sequences()
        self.verify_replayed_sequences()
        self.plot_kernel_tax()
    
    def replay_sequences_from_cpu_ops(
        self, 
        event_sequences: List[Dict[str, Any]], 
        warmup: int,
        runs: int
    ) -> List[Dict[str, Any]]:
        """
        Replay all event sequences
        """
        # Save trace file in kernel_traces folder
        kernel_traces_dir = utils.get_path("PYTORCH_KERNEL_TRACES_DIR", base_path=self.experiment_dir)
        utils.ensure_dir(kernel_traces_dir)

        # Store sequences per replay index
        sequence_by_idx = {}
        
        print(f"Replaying {len(event_sequences)} kernels with {runs} run{'s' if runs > 1 else ''} each (warmup={warmup})")
        
        # Replay each event sequence
        for i, event_sequence in enumerate(event_sequences):
            cpu_op = event_sequence.get("cpu_op")
            kernel = event_sequence.get("kernel")
            
            assert kernel is not None, f"Kernel is None for event sequence {i}"
            assert cpu_op is not None, f"CPU operation is None for event sequence {i}"
            expected_kernel = utils.clean_kernel_name(event_sequence['kernel']['name'])
            seq_idx = i+1 
            
            print(f"* [{seq_idx}/{len(event_sequences)}] {cpu_op['name']} -> {expected_kernel}")
            
            # Generate trace filename based on operation name and expected kernel name
            op_name_short = cpu_op.get("name", "unknown").replace("::", "_")
            trace_file_name = f"{seq_idx:02d}_{op_name_short}_{expected_kernel}.json"
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

        return all_replayed_sequences

# Helper functions
def create_input_tensors(cpu_op: Dict[str, Any]) -> List[torch.Tensor]:
    """
    Create input tensors from CPU operation metadata.
    """
    input_dims = cpu_op["input_dims"]
    input_types = cpu_op.get("input_type", [])
    input_strides = cpu_op.get("input_strides", [])
    
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