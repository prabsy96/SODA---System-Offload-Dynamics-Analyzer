"""
SODA: System Offload Dynamics Analyzer
Analyze CPU–GPU dynamics of PyTorch models.
"""

import argparse
import json
import logging
import os
import sys
import torch
import traceback
import numpy as np
import transformers
from pathlib import Path
from collections import defaultdict, deque
from torch.profiler import ProfilerActivity, profile
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

# for fp8 e4m3 format support
try:
    from transformers.utils.quantization_config import FP8Config
    FP8_CONFIG_AVAILABLE = True
except ImportError:
    FP8Config = None
    FP8_CONFIG_AVAILABLE = False

def us_to_ms(microseconds: float) -> float:
    """Convert microseconds to milliseconds."""
    return microseconds / 1000.0

class SodaProfiler:
    """
    Handles model tracing, profile data parsing, and metric generation.
    """

    @staticmethod
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
            "-sl", "--seq-len", dest="seq_len", type=int, nargs="+", default=[512], 
            help="Sequence length(s) for synthetic input. Provide multiple values for sweep."
        )
        parser.add_argument(
            "-bs", "--batch-size", dest="batch_size", type=int, nargs="+", default=[1], 
            help="Batch size(s) for synthetic input. Provide multiple values for sweep."
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
            "--version", action="version", version="%(prog)s 0.1.0"
        )
        
        return parser

    @staticmethod
    def parse_and_validate_args(args=None) -> argparse.Namespace:
        """Parse and validate command-line arguments."""
        parser = SodaProfiler.get_args_parser()
        parsed_args = parser.parse_args(args)
        
        # Validate arguments
        if parsed_args.device == "cpu" and parsed_args.precision in ["float16", "float8_e4m3fn", "float8_e5m2", "bfloat16"]:
            print(f"Warning: {parsed_args.precision} is not supported on CPU. Forcing float32.")
            parsed_args.precision = "float32"

        if not torch.cuda.is_available() and parsed_args.device == "cuda":
            print("Error: CUDA is not available. Please select --device cpu.", file=sys.stderr)
            sys.exit(1)
        
        if parsed_args.precision in ["float8_e4m3fn"]:
            if not hasattr(torch, 'float8_e4m3fn'):
                print("Error: FP8 requires PyTorch 2.1+. Please upgrade PyTorch.", file=sys.stderr)
                sys.exit(1)

            if parsed_args.device == "cuda":
                capability = torch.cuda.get_device_capability()
                if capability[0] < 9 and not (capability[0] == 8 and capability[1] >= 9):
                    print(f"Warning: FP8 requires SM89+ (Ada/Hopper). Detected SM{capability[0]}{capability[1]}.", file=sys.stderr)
                    print("FP8 may not be hardware-accelerated on this device.", file=sys.stderr)

        return parsed_args

    @staticmethod
    def generate_experiment_name(args: argparse.Namespace) -> str:
        """
        Generates a unique experiment directory name from arguments.

        """
        # Batch size part
        if len(args.batch_size) == 1:
            bs_str = f"bs{args.batch_size[0]}"
        else:
            bs_str = "bs_sweep"
        
        # Sequence length part
        if len(args.seq_len) == 1:
            sl_str = f"sl{args.seq_len[0]}"
        else:
            sl_str = "sl_sweep"
            
        return f"{args.model.replace('/', '_')}_{args.compile_type}_{bs_str}_{sl_str}"

    def __init__(self, model_handler: 'ModelHandler', args: argparse.Namespace, log_console: bool = True, log_file: bool = True):
        """
        Initializes the profiler.

        Sets up the profiler and derives name, file, and path from parsed arguments.

        Args:
            model_handler: The ModelHandler class instance (contains pytorch_model, tokenizer, is_decoder).
            args: Parsed and validated command-line arguments.
            log_console: If True, write logs to console/stdout.
            log_file: If True, write logs to file.
        """
        from datetime import datetime

        self.args = args
        self.model_handler = model_handler

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Derive experiment_name, output_dir from args
        self.experiment_name = self.generate_experiment_name(args)
        self.output_dir = Path(args.output_dir) / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self._soda_logger = SodaLogger(self.output_dir, is_console=log_console, is_file=log_file)
        self.logger = self._soda_logger.logger
        
        # Set seed for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        self.trace_file_path = self.output_dir / "trace.json"
        self.report_file_path = self.output_dir / f"report_{self.run_id}.json"
        
        self.trace = None
        self.events = None
        self.results = None

        self._sweep_runs = []

    def trace_forward_pass_for_encoder(self, inputs: Dict[str, torch.Tensor]) -> str:
        """
        Profiles the forward pass of an encoder model.

        Args:
            inputs: A dictionary of tokenized inputs.
            tokenizer: The model's tokenizer.

        Returns:
            The path to the generated Chrome trace JSON file.
        """
        self.logger.info("=== Profiling Model Forward Pass ===")
        
        # Warm-up runs
        with torch.no_grad():
            for _ in range(5):
                self.model_handler.pytorch_model(**inputs)
        
        # Synchronize before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Profiled run with explicit timing
        with torch.no_grad():
            with profile(
                activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                with_modules=True,
                with_flops=True,
                profile_memory=True,
            ) as prof:
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                self.model_handler.pytorch_model(**inputs)
                end_time.record()
                
                torch.cuda.synchronize()
                self.measured_inference_time_ms = start_time.elapsed_time(end_time)

        prof.export_chrome_trace(str(self.trace_file_path))
        
        # Load trace data into memory immediately
        self.trace = SodaProfiler.load_json(self.trace_file_path)
        
        self.logger.info(f"* Chrome trace file generated at: {self.trace_file_path}")
        return str(self.trace_file_path)

    def profile_forward_pass(self, inputs: Dict[str, torch.Tensor], batch_size: int = None, seq_len: int = None) -> str:
        """
        Profiles the forward pass of the model (encoder or decoder).
        
        Args:
            inputs: A dictionary of tokenized inputs.
            batch_size: Optional batch size. Defaults to self.args.batch_size.
            seq_len: Optional sequence length. Defaults to self.args.seq_len.
            
        Returns:
            The path to the generated Chrome trace JSON file.
        """
        batch_size = self.args.batch_size if batch_size is None else batch_size
        seq_len = self.args.seq_len if seq_len is None else seq_len
            
        if self.model_handler.is_decoder:
            return self.trace_forward_pass_for_decoder(
                inputs, batch_size, seq_len
            )
        else:
            return self.trace_forward_pass_for_encoder(inputs)

    def trace_forward_pass_for_decoder(self, inputs: Dict[str, torch.Tensor], bs: int, sq: int) -> str:
        """
        Profiles the generate step of a decoder model.
        
        Args:
            inputs: A dictionary of tokenized inputs.
            bs: Batch size.
            sq: Sequence length.
            
        Returns:
            The path to the generated Chrome trace JSON file.
        """
        self.logger.info("=== Profiling Model Forward Pass ===")
        
        # Warm-up runs
        with torch.no_grad():
            for _ in range(5):
                self.model_handler.pytorch_model.generate(**inputs, max_new_tokens=1, do_sample=False, pad_token_id=self.model_handler.tokenizer.pad_token_id)

        # Synchronize before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Profiled run
        with torch.no_grad():
            with profile(
                activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                with_modules=True,
                with_flops=True,
                profile_memory=True,
            ) as prof:
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                self.model_handler.pytorch_model.generate(
                    **inputs, max_new_tokens=1, do_sample=False, 
                    pad_token_id=self.model_handler.tokenizer.pad_token_id
                )
                end_time.record()
                
                torch.cuda.synchronize()
                self.measured_inference_time_ms = start_time.elapsed_time(end_time)
        
        prof.export_chrome_trace(str(self.trace_file_path))
        self.trace = SodaProfiler.load_json(self.trace_file_path)
        
        self.logger.info(f"* Chrome trace file generated at: {self.trace_file_path}")
        return str(self.trace_file_path)
    
    def calculate_profiler_wall_clock_time(self) -> float:
        """
        Calculates wall-clock time from PyTorch profiler trace events.
        
        This measures the time span from the first to last trace event,
        which includes profiler overhead. Useful for comparison with
        CUDA event timing.
        
        Uses self.trace
            
        Returns:
            Wall-clock time in microseconds.
        """
        if isinstance(self.trace, list):
            trace_events = self.trace
        else:
            trace_events = self.trace.get("traceEvents", [])
        
        all_timestamps = []
        for event in trace_events:
            if event.get("ph") == "X":  # Complete events only
                start_time = float(event.get("ts", 0))
                duration = float(event.get("dur", 0))
                end_time = start_time + duration
                all_timestamps.append((start_time, end_time))
        
        if not all_timestamps:
            return 0.0
        
        min_start = min(start_time for start_time, _ in all_timestamps)
        max_end = max(end_time for _, end_time in all_timestamps)
        
        return max_end - min_start

    @staticmethod
    def get_path(env_var: str) -> Path:
        """Get path from environment variable."""
        return Path(os.environ[env_var])
    
    @staticmethod
    def ensure_dir(path) -> None:
        """Ensure directory exists, creating parent directories if needed. Accepts Path or str."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def load_json(file_path) -> Dict[str, Any]:
        """Load JSON file."""
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"File does not exist: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)


    def calculate_total_inference_time(self) -> float:
        """
        Calculates total wall-clock inference time from ALL trace events.
        
        Includes CPU ops, CUDA runtime calls, and GPU execution.
        Only considers complete events (ph="X") with timestamps and durations.
        Excludes flow markers, metadata, and instant events.
        
        Uses self.trace
            
        Returns:
            Total inference time in microseconds (max_end - min_start).
        """
        all_timestamps = []

        if isinstance(self.trace, list):
            trace_events = self.trace
        else:
            trace_events = self.trace.get("traceEvents", [])
        
        for event in trace_events:
            if event.get("ph") == "X":  # Only complete events (excludes flow markers)
                start_time = float(event.get("ts"))
                duration = float(event.get("dur", 0))
                end_time = start_time + duration
                all_timestamps.append((start_time, end_time))
        
        if not all_timestamps:
            return 0.0
        
        min_start = min(start_time for start_time, _ in all_timestamps)
        max_end = max(end_time for _, end_time in all_timestamps)
        
        return max_end - min_start

    def calculate_total_gpu_time_span(self) -> float:
        """
        Calculates the end-to-end GPU time span by finding min start and max end
        across GPU execution events (kernel, gpu_memcpy, gpu_memset).
        
        Measures the extreme time window of GPU execution (from first to last GPU event).
        Excludes cuda_runtime (CPU-side calls).
        
        Uses self.events.
            
        Returns:
            Time span in microseconds (max_end - min_start).
        """
        gpu_events = self.events["gpu"]["all"]
        
        if not gpu_events:
            return 0.0
        
        gpu_event_intervals = []
        for event in gpu_events:
            start_time = float(event.get("ts", 0))
            end_time = start_time + float(event.get("dur", 0))
            gpu_event_intervals.append((start_time, end_time))
        
        min_start = min(start_time for start_time, _ in gpu_event_intervals)
        max_end = max(end_time for _, end_time in gpu_event_intervals)
        
        return max_end - min_start

    def calculate_kernel_exec_time(self) -> Dict[str, float]:
        """
        Calculates total and average kernel execution time.
        
        Uses self.events.
            
        Returns:
            Dictionary with "total" and "avg" keys (in microseconds).
        """
        kernel_events = self.events["gpu"]["kernels"]
        num_kernels = len(kernel_events)
        
        total_kernel_exec_time = 0.0
        for kernel in kernel_events:
            total_kernel_exec_time += float(kernel.get("dur", 0))
        
        avg_kernel_exec_time = total_kernel_exec_time / num_kernels if num_kernels > 0 else 0.0
        
        return {
            "total": total_kernel_exec_time,
            "avg": avg_kernel_exec_time,
        }

    def calculate_true_gpu_busy_time(self) -> float:
        """
        Calculates GPU busy time by merging overlapping GPU event intervals.
        
        Accounts for concurrent GPU execution across all streams by merging
        overlapping time intervals. Includes kernel, gpu_memcpy, and gpu_memset
        events. If events run concurrently on different streams, their overlapping 
        time is counted once.
        
        Uses self.events.
            
        Returns:
            Merged GPU busy time in microseconds.
        """
        gpu_events = self.events["gpu"]["all"]
        
        # Edge case: no GPU events
        if not gpu_events:
            return 0.0
        
        # Extract GPU event intervals
        gpu_event_intervals = []
        for event in gpu_events:
            start_time = float(event.get("ts", 0))
            end_time = start_time + float(event.get("dur", 0))
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

    def calculate_gpu_utilization(self) -> float:
        """
        Calculates GPU utilization percentage.
        
        Uses self.events.
            
        Returns:
            GPU utilization as a percentage (0.0 to 100.0).
        """
        # Calculate denominator: time span of GPU execution events only 
        total_gpu_time_span = self.calculate_total_gpu_time_span()
        
        # Avoid division by zero
        if total_gpu_time_span == 0.0:
            return 0.0
        
        # Calculate numerator: non overlapping busy time of GPU execution events
        true_gpu_busy_time = self.calculate_true_gpu_busy_time()
        
        # Calculate GPU utilization percentage
        gpu_utilization = (true_gpu_busy_time / total_gpu_time_span)
        gpu_utilization = gpu_utilization * 100.0

        return gpu_utilization

    def analyze_per_stream(self) -> Dict:
        """
        Analyzes GPU events grouped by stream.
        
        Uses self.events.
        
        For each stream, calculates:
        - Total operations and kernel count
        - Total kernel execution time (sum of durations)
        - True GPU busy time (merged overlapping intervals)
            
        Returns:
            Dictionary mapping stream_id to stream metrics.
        """
        gpu_events = self.events["gpu"]["all"]
        
        stream_info = defaultdict(lambda: {
            "ops": [], "total_kernel_exec_time": 0.0, "true_gpu_busy_time": 0.0,
            "op_count": 0, "kernel_count": 0
        })
        
        for op in gpu_events:
            stream_id = op.get("stream", "unknown_stream")
            stream_info[stream_id]["ops"].append(op)
        
        for stream_id, data in stream_info.items():
            ops_on_stream = sorted(data["ops"], key=lambda x: float(x.get("ts", 0)))
            stream_info[stream_id]["ops"] = ops_on_stream
            stream_info[stream_id]["op_count"] = len(ops_on_stream)
            
            stream_kernels = [op for op in ops_on_stream if op.get("type") == "kernel"]
            stream_info[stream_id]["kernel_count"] = len(stream_kernels)
            stream_info[stream_id]["total_kernel_exec_time"] = sum(
                float(k.get("dur", 0)) for k in stream_kernels
            )
            
            if stream_kernels:
                stream_intervals = sorted([
                    (float(k["ts"]), float(k["ts"]) + float(k.get("dur", 0)))
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

    @staticmethod
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

    @staticmethod
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
        cuda_launch_events_by_ext_id = defaultdict(list) # New: fallback lookup
        kernel_events = []
        gpu_mem_events = []

        if isinstance(trace, list):
            trace_events = trace
        else:
            trace_events = trace.get("traceEvents", [])
        
        for event in trace_events:
            cat = event.get("cat")
            name = event.get("name", "")
            args = event.get("args", {})
            external_id = args.get("External id")
            correlation = args.get("correlation")
            
            if cat == "cpu_op" and external_id is not None:
                op_events_by_ext_id[external_id] = {
                    "type": "cpu_op",
                    "name": name,
                    "external_id": external_id,
                    "input_dims": args.get("Input Dims", []),
                    "input_strides": args.get("Input Strides", []),
                    "input_type": args.get("Input type", []),
                    "concrete_inputs": args.get("Concrete Inputs", []),
                    "ts": event.get("ts"),
                    "dur": event.get("dur")
                }
            elif (cat == "cuda_runtime" or cat == "Runtime") and ("LaunchKernel" in name or "Launch" in name):
                launch_event = {
                    "type": "cuda_launch",
                    "name": name,
                    "external_id": external_id,
                    "correlation": correlation,
                    "ts": event.get("ts"),
                    "dur": event.get("dur"),
                    "cbid": args.get("cbid")
                }
                if correlation is not None:
                    cuda_launch_events_by_corr[correlation] = launch_event
                
                # Index by external ID (fallback)
                if external_id is not None:
                    cuda_launch_events_by_ext_id[external_id].append(launch_event)

            elif cat == "kernel":
                if external_id is not None or correlation is not None:
                    kernel_events.append({
                        "type": "kernel",
                        "name": SodaProfiler.get_clean_kernel_name(name),
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
                    "name": name,
                    "correlation": correlation,
                    "stream": args.get("stream"),
                    "device": args.get("device"),
                    "context": args.get("context"),
                    "ts": event.get("ts"),
                    "dur": event.get("dur"),
                    "bytes": args.get("bytes"),
                    "memory_bandwidth_gbs": args.get("memory bandwidth (GB/s)") if cat == "gpu_memcpy" else None
                })
        
        return {
            "cpu": {
                "ops": op_events_by_ext_id,
                "launches": cuda_launch_events_by_corr,
                "launches_by_ext_id": cuda_launch_events_by_ext_id
            },
            "gpu": {
                "kernels": kernel_events,
                "memory": gpu_mem_events,
                "all": kernel_events + gpu_mem_events
            }
        }

    @staticmethod
    def get_linked_event_sequences(events: Dict[str, Any]) -> List[Dict]:
        """
        Get event sequences linking CPU operations, CUDA launches, and kernels.
        
        Args:
            events: Dictionary with hierarchical structure from collect_events_from_trace.

        Returns:
            List of event sequence dictionaries with keys: kernel, cuda_launch, cpu_op.
        """
        gpu_events = events["gpu"]
        cpu_ops = events["cpu"]["ops"]
        cuda_launches = events["cpu"]["launches"]
        cuda_launches_by_ext_id = events["cpu"].get("launches_by_ext_id", {})
        kernel_events = gpu_events["kernels"]

        event_sequences = []
        launch_usage_by_ext_id = defaultdict(int)

        for kernel in kernel_events:
            external_id = kernel.get("external_id")
            correlation = kernel.get("correlation")
            
            cpu_op = cpu_ops.get(external_id) if external_id is not None else None
            cuda_launch = cuda_launches.get(correlation) if correlation is not None else None

            if cuda_launch is None and external_id is not None:
                launches = cuda_launches_by_ext_id.get(external_id)
                if launches:
                    # Simple heuristic: map i-th kernel to i-th launch for this ID
                    # This assumes kernels appear in trace in same order as launches
                    idx = launch_usage_by_ext_id[external_id]
                    if idx < len(launches):
                        cuda_launch = launches[idx]
                        launch_usage_by_ext_id[external_id] += 1
                    else:
                        # If we ran out of launches, use the last one as fallback
                        cuda_launch = launches[-1]
            
            event_sequences.append({
                "kernel": kernel,
                "cuda_launch": cuda_launch,
                "cpu_op": cpu_op,
            })
        
        return event_sequences

    @staticmethod
    def calculate_per_seq_tklqt(event_sequences: List[Dict]) -> List[Dict]:
        """
        Calculates launch tax for each sequence and adds it to the sequence dict.

        Args:
            event_sequences: List of event sequence dictionaries.

        Returns:
            Modified event sequences with "kernel_tax" key added to each.
        """

        for seq in event_sequences:
            kernel = seq.get("kernel")
            cuda_launch = seq.get("cuda_launch")
            
            if kernel is None or cuda_launch is None:
                seq["tklqt_us"] = 0.0
                continue

            cuda_launch_start = float(cuda_launch.get("ts", 0))
            kernel_start = float(kernel.get("ts", 0))

            # TKLQT = Kernel Start (GPU) - Launch Start (CPU)
            val = kernel_start - cuda_launch_start

            if val < 0:
                # Kernel started before launch? Event correlation error
                seq["tklqt_us"] = 0.0
            elif val > 5000000:  # > 5 seconds is likely an artifact
                seq["tklqt_us"] = 0.0
            else:
                seq["tklqt_us"] = val
        
        return event_sequences


    @staticmethod
    def calculate_total_tklqt(event_sequences: List[Dict]) -> float:
        """
        Calculates total launch tax across all sequences.

        Args:
            event_sequences: List of event sequence dictionaries with "kernel_tax" key.

        Returns:
            Total launch tax in microseconds.
        """
        total = 0.0
        for seq in event_sequences:
            total += seq.get("launch_tax_us", 0.0)
        return total

    @staticmethod
    def calculate_avg_tklqt(event_sequences: List[Dict]) -> float:
        """
        Calculates average launch tax across all sequences.

        Args:
            event_sequences: List of event sequence dictionaries with "kernel_tax" key.

        Returns:
            Average launch tax in microseconds.
        """
        valid_sequences = [seq for seq in event_sequences if seq.get("launch_tax_us", 0.0) > 0]
        if not valid_sequences:
            return 0.0
        total = sum(seq.get("launch_tax_us", 0.0) for seq in valid_sequences)
        return total / len(valid_sequences)
    
    def get_average_kernel_duration(self) -> Dict[str, float]:
        """
        Calculates the average execution duration (aka operational intensity) for each unique kernel.
        
        Aggregates all instances of each kernel and computes the mean duration.
        
        Uses self.events.
            
        Returns:
            Dictionary mapping kernel name to average duration.
        """
        kernel_events = self.events["gpu"]["kernels"]
        
        kernel_stats = defaultdict(lambda: {"total_duration": 0.0, "count": 0})
    
        for kernel in kernel_events:
            kernel_name = kernel["name"]
            kernel_stats[kernel_name]["total_duration"] += kernel.get("dur", 0)
            kernel_stats[kernel_name]["count"] += 1
        
        avg_durations = {}
        for name, stat in kernel_stats.items():
            if stat["count"] > 0:
                avg_durations[name] = stat["total_duration"] / stat["count"]
            else:
                avg_durations[name] = 0.0
        
        return avg_durations
    
    def calculate_framework_overhead(self) -> Dict[str, float]:
        """
        Calculate framework overhead - the CPU-side time not spent on GPU computation.
        """
        if hasattr(self, 'measured_inference_time_ms') and self.measured_inference_time_ms is not None:
            total_inference_us = self.measured_inference_time_ms * 1000
        else:
            total_inference_us = self.calculate_profiler_wall_clock_time()
        
        gpu_active_time_us = self.calculate_true_gpu_busy_time()
        
        # Framework overhead = everything except GPU compute
        framework_tax_us = max(0.0, total_inference_us - gpu_active_time_us)

        is_framework_bound = framework_tax_us > gpu_active_time_us
        
        # Calculate percentages
        if total_inference_us > 0:
            tax_percent = (framework_tax_us / total_inference_us) * 100
            compute_percent = (gpu_active_time_us / total_inference_us) * 100
        else:
            tax_percent = 0.0
            compute_percent = 0.0
        
        return {
            "framework_tax_ms": us_to_ms(framework_tax_us),
            "framework_tax_percent": tax_percent,
            "gpu_active_time_ms": us_to_ms(gpu_active_time_us),
            "gpu_active_time_percent": compute_percent,
            "is_framework_bound": is_framework_bound,
            "bound_state": "Framework-Bound" if is_framework_bound else "Compute-Bound"
        }

    def get_top_k_kernels(self, k: int = 3) -> Dict[str, List[Tuple[str, Dict]]]:
        """
        Calculates the top-k most frequent and time-consuming kernels.
        
        Uses self.events.
        
        Args:
            k: Number of top kernels to return.
            
        Returns:
            Dictionary with keys:
            - "by_frequency": List of (kernel_name, stats_dict) tuples, sorted by frequency
            - "by_duration": List of (kernel_name, stats_dict) tuples, sorted by duration
            Each kernel_stats dict contains: frequency, duration
            Returns empty lists if no kernel events.
        """
        kernel_events = self.events["gpu"]["kernels"]
        
        if not kernel_events:
            return {"by_frequency": [], "by_duration": []}
        
        kernel_stats = defaultdict(lambda: {"frequency": 0, "duration": 0.0})
        
        for kernel in kernel_events:
            kernel_name = kernel["name"]
            kernel_stats[kernel_name]["frequency"] += 1
            kernel_stats[kernel_name]["duration"] += float(kernel.get("dur", 0))
        
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

    def kernelchains(self, event_sequences: List[Dict], exact_length: int, prox_score_threshold: float):
        """
        Analyzes kernel launch sequences to find opportunities for fusion.

        Args:
            event_sequences: List of event sequence dictionaries.
            exact_length: The exact length of sequences to analyze.
            prox_score_threshold: The proximity score required to recommend a fusion (e.g., 1.0 for deterministic).
        """
        if exact_length < 2:
            self.logger.warning("Sequence length must be at least 2.")
            return

        all_segments = []
        current_segment = []
        # Separate event sequences by synchronization points
        for seq in event_sequences:
            cuda_launch = seq.get("cuda_launch")
            if cuda_launch and 'cudaStreamSynchronize' in cuda_launch.get("name", ""):
                if current_segment:
                    all_segments.append(current_segment)
                current_segment = []
            current_segment.append(seq)
        if current_segment:
            all_segments.append(current_segment)

        total_kernel_launches = len(event_sequences)
        unique_fusion_candidates: Set[Tuple[str, ...]] = set()
        
        # Process each segment to find sequences
        for segment in all_segments:
            current_chain = deque(maxlen=exact_length)
            for seq in segment:
                kernel = seq.get("kernel")
                if kernel:
                    current_chain.append(kernel["name"])
                    if len(current_chain) == exact_length:
                        chain_tuple = tuple(current_chain)
                        unique_fusion_candidates.add(chain_tuple)

        # Calculate proximity scores
        fusion_recommendations = []
        kernel_freq: DefaultDict[str, int] = defaultdict(int)
        for seq in event_sequences:
            kernel = seq.get("kernel")
            if kernel:
                kernel_freq[kernel["name"]] += 1
            
        for chain in unique_fusion_candidates:
            # Count occurrences of this exact chain
            count = 0
            for segment in all_segments:
                segment_kernels = [seq["kernel"]["name"] for seq in segment if seq.get("kernel")]
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
        self.logger.info(f"=== Fusion Analysis (Length={exact_length}, Threshold={prox_score_threshold}) ===")
        if not fusion_recommendations:
            self.logger.info("\t* No kernel chains met the fusion criteria.")
            return None

        sorted_recommendations = sorted(fusion_recommendations, key=lambda x: x[1], reverse=True)
        self.logger.info(f"\t* Found {len(sorted_recommendations)} potential fusion candidates:")
        for idx, (chain, count, score) in enumerate(sorted_recommendations, 1):
            self.logger.info(f"\t* Chain {idx}\tFound {count} times\tProx. Score = {score:.2f}")
            for kernel in chain:
                self.logger.info(f"\t\t** {kernel}")
        
        self.logger.info("")
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

    
    def analyze(self) -> Dict[str, Any]:
        """
        Performs complete analysis of the trace data.
        
        Collects events, calculates metrics, and returns analysis results.
        This method mimics the analysis logic in main() function.
        
        Returns:
            Dictionary containing all analysis results including:
            - metrics: Performance metrics (inference time, GPU utilization, etc.)
            - stream_info: Per-stream analysis
            - top_k_kernels: Top-k kernels by frequency and duration
            - event_sequences: Event sequences
            - avg_kernel_dur: Average kernel duration results
        """
        self.logger.info("=== Analyzing Trace Data ===")
        # Collect events and build event sequences
        self.events = SodaProfiler.collect_events_from_trace(self.trace)
        self.logger.info(f"Analyzing {len(self.events['gpu']['kernels'])} kernel events from profiled run...")
        event_sequences = SodaProfiler.get_linked_event_sequences(self.events)
        # Calculate TKLQT (Launch + Queue Time)
        event_sequences = SodaProfiler.calculate_per_seq_tklqt(event_sequences)
        

        # Analyze per-stream metrics
        stream_info = self.analyze_per_stream()
        
        # Inference time metrics
        # Primary: CUDA event timing (most accurate)
        if hasattr(self, 'measured_inference_time_ms') and self.measured_inference_time_ms is not None:
            cuda_event_time_us = self.measured_inference_time_ms * 1000  # Convert ms to us
        else:
            cuda_event_time_us = None
        
        # Secondary: PyTorch profiler wall-clock (includes overhead)
        profiler_wall_clock_us = self.calculate_profiler_wall_clock_time()
        
        # Use CUDA event time if available, otherwise fall back to profiler wall-clock
        total_inference_time = cuda_event_time_us if cuda_event_time_us is not None else profiler_wall_clock_us
        
        # GPU metrics
        total_gpu_time_span = self.calculate_total_gpu_time_span()
        true_gpu_busy_time = self.calculate_true_gpu_busy_time()
        gpu_utilization = self.calculate_gpu_utilization()
        
        # Kernel metrics
        kernel_exec_time = self.calculate_kernel_exec_time()
        
        # TKLQT Metrics
        total_tklqt = SodaProfiler.calculate_total_tklqt(event_sequences)
        avg_tklqt = SodaProfiler.calculate_avg_tklqt(event_sequences)
        tklqt_metrics = {"total": total_tklqt, "avg": avg_tklqt}
        
        avg_kernel_dur = self.get_average_kernel_duration()
        top_k_kernels = self.get_top_k_kernels(k=3)
        
        # Fusion analysis
        fusion_results = None
        if self.args.fusion:
            self.logger.info("=== Kernel Fusion Analysis ===")
            fusion_results = {}
            for f in self.args.fusion:
                fusion_results[f] = self.kernelchains(event_sequences, f, self.args.prox_score)

        framework_overhead = self.calculate_framework_overhead()
        
        # Build metrics dictionary 
        metrics = {
            # Inference timing (primary metric)
            "inference_time_ms": us_to_ms(total_inference_time),
            
            # Timing breakdown
            "timing": {
                "cuda_event_ms": self.measured_inference_time_ms if hasattr(self, 'measured_inference_time_ms') and self.measured_inference_time_ms is not None else None,
                "profiler_wall_clock_ms": us_to_ms(profiler_wall_clock_us),
                "profiler_overhead_ms": us_to_ms(profiler_wall_clock_us - total_inference_time) if cuda_event_time_us is not None else None,
            },
            
            # Framework overhead (CPU-side latency)
            "framework_overhead": framework_overhead,
            
            # Stream info
            "active_streams": len(stream_info),

            # GPU metrics
            "total_gpu_time_span_ms": us_to_ms(total_gpu_time_span),
            "gpu_busy_time_ms": us_to_ms(true_gpu_busy_time),
            "gpu_idle_time_ms": us_to_ms(max(0.0, total_gpu_time_span - true_gpu_busy_time)),
            "gpu_utilization_percent": gpu_utilization,
 
            # Kernel metrics
            "total_kernel_exec_time_ms": us_to_ms(kernel_exec_time["total"]),
            "num_total_kernels": len(self.events["gpu"]["kernels"]),
            "avg_kernel_exec_time_ms": us_to_ms(kernel_exec_time["avg"]),
            
            # TKLQT Metrics
            "total_tklqt_ms": us_to_ms(tklqt_metrics["total"]),
            "avg_tklqt_ms": us_to_ms(tklqt_metrics["avg"]),
        }
        
        self.results = {
            "metrics": metrics,
            "stream_info": stream_info,
            "top_k_kernels": top_k_kernels,
            "event_sequences": event_sequences,
            "avg_kernel_dur": avg_kernel_dur,
            "fusion_results": fusion_results,
        }
        
        return self.results
    
    def report(self) -> None:
        
        metrics = self.results["metrics"]
        stream_info = self.results["stream_info"]
        top_k_kernels = self.results["top_k_kernels"]
        timing = metrics.get("timing", {})
        framework = metrics.get("framework_overhead", {})
        
        # --- Enhanced Reporting ---
        self.logger.info("")
        self.logger.info("=== Performance Metrics ===")
        self.logger.info(f"\t* Inference time (ms): {metrics['inference_time_ms']:.4f}")

        self.logger.info("")
        self.logger.info("=== Framework Tax Analysis ===")
        self.logger.info(f"\t* Workload State: {framework.get('bound_state', 'Unknown')}")
        self.logger.info(f"\t* Framework Tax (Exposed CPU Overhead): {framework.get('framework_tax_ms', 0):.4f} ms ({framework.get('framework_tax_percent', 0):.1f}%)")
        self.logger.info(f"\t* GPU Active Time (Compute): {framework.get('gpu_active_time_ms', 0):.4f} ms ({framework.get('gpu_active_time_percent', 0):.1f}%)")
        
        # Timing breakdown
        if timing.get("cuda_event_ms") is not None:
            self.logger.info(f"\t  - CUDA event timing (ms): {timing['cuda_event_ms']:.4f}")
            self.logger.info(f"\t  - Profiler wall-clock (ms): {timing['profiler_wall_clock_ms']:.4f}")
            if timing.get("profiler_overhead_ms") is not None:
                self.logger.info(f"\t  - Profiler overhead (ms): {timing['profiler_overhead_ms']:.4f}")
        
        # Framework overhead breakdown
        self.logger.info("")
        self.logger.info("=== GPU Metrics ===")
        self.logger.info(f"\t* Total kernel execution time (ms): {metrics['total_kernel_exec_time_ms']:.4f}")
        self.logger.info(f"\t* GPU busy time (concurrent-aware) (ms): {metrics['gpu_busy_time_ms']:.4f}")
        self.logger.info(f"\t* GPU idle time (ms): {metrics['gpu_idle_time_ms']:.4f}")
        self.logger.info(f"\t* GPU utilization: {metrics['gpu_utilization_percent']:.2f}%")
        self.logger.info(f"\t* Number of kernels: {metrics['num_total_kernels']}")
        self.logger.info(f"\t* Active streams: {metrics['active_streams']}")
        
        self.logger.info("")
        self.logger.info("=== Launch & Queue Latency (TKLQT) ===")
        self.logger.info(f"\t* Total TKLQT (ms): {metrics['total_tklqt_ms']:.4f}")
        if metrics['num_total_kernels'] > 0:
            self.logger.info(f"\t* Avg. TKLQT per kernel (ms): {metrics['avg_tklqt_ms']:.4f}")
            self.logger.info(f"\t* Avg. execution time per kernel (ms): {metrics['avg_kernel_exec_time_ms']:.4f}")
        
        self.logger.info("")
        # --- Per-Stream Breakdown ---
        self.logger.info("=== Per-Stream Analysis ===")
        for stream_id, data in stream_info.items():
            self.logger.info(
                f"\t* Stream {stream_id}: {data['op_count']} ops "
                f"({data['kernel_count']} kernels), "
                f"Busy Time: {us_to_ms(data['true_gpu_busy_time']):.4f} ms"
            )
        
        self.logger.info("")
        # Top-K kernels 
        if top_k_kernels["by_frequency"]:
            self.logger.info("=== Top-3 Kernels by Frequency ===")
            for i, (name, data) in enumerate(top_k_kernels["by_frequency"], 1):
                self.logger.info(
                    f"\t* #{i}: {name} "
                    f"(Frequency: {int(data['frequency'])}, "
                    f"Total Duration: {us_to_ms(data['duration']):.4f} ms)"
                )
            
            self.logger.info("")
            self.logger.info("=== Top-3 Kernels by Duration ===")
            for i, (name, data) in enumerate(top_k_kernels["by_duration"], 1):
                self.logger.info(
                    f"\t* #{i}: {name} "
                    f"(Total Duration: {us_to_ms(data['duration']):.4f} ms, "
                    f"Frequency: {int(data['frequency'])})"
                )
            
            self.logger.info("")
            self.logger.info("=== Top-3 Kernels by Duration ===")
            for i, (name, data) in enumerate(top_k_kernels["by_duration"], 1):
                self.logger.info(
                    f"\t* #{i}: {name} "
                    f"(Total Duration: {us_to_ms(data['duration']):.4f} ms, "
                    f"Frequency: {int(data['frequency'])})"
                )
    
    def save(self, batch_size: int = None, seq_len: int = None) -> str:
        """
        Saves analysis results to JSON file.
        Uses results stored in self.results from analyze().
        Generates model_name and config from self.args.
        
        Args:
            batch_size: The batch size used for this run.
            seq_len: The sequence length used for this run.
            
        Returns:
            Path to the saved report file.
        """
        if self.results is None:
            raise ValueError("No analysis results available. Call analyze() first.")
        
        from datetime import datetime
        
        metrics = self.results["metrics"]
        stream_info = self.results["stream_info"]
        top_k_kernels = self.results["top_k_kernels"]
        fusion_results = self.results.get("fusion_results")
        
        # Generate model_name and config from args
        model_name = self.args.model

        current_batch_size = batch_size if batch_size is not None else (
            self.args.batch_size[0] if isinstance(self.args.batch_size, list) else self.args.batch_size)
        
        current_seq_len = seq_len if seq_len is not None else (
            self.args.seq_len[0] if isinstance(self.args.seq_len, list) else self.args.seq_len)
        
        config = {
            "batch_size": current_batch_size,
            "seq_len": current_seq_len,
            "precision": self.args.precision,
            "compile_type": self.args.compile_type,
            "device": self.args.device,
        }
        
        # Build output structure
        run_result = {
            "metadata": {
                "model_name": model_name,
                "timestamp": datetime.now().isoformat(),
                "config": config
            },
            "performance_metrics": metrics, 
            "per_stream_analysis": {
                str(stream_id): {
                    "total_ops": data["op_count"],
                    "kernel_count": data["kernel_count"],
                    "busy_time_ms": us_to_ms(data["true_gpu_busy_time"]),
                    "total_kernel_exec_time_ms": us_to_ms(data["total_kernel_exec_time"]),
                }
                for stream_id, data in stream_info.items()
            },
            "top_kernels": {
                "by_frequency": [
                    {
                        "rank": i,
                        "name": name,
                        "frequency": data["frequency"],
                        "total_duration_ms": us_to_ms(data["duration"])
                    }
                    for i, (name, data) in enumerate(top_k_kernels["by_frequency"], 1)
                ],
                "by_duration": [
                    {
                        "rank": i,
                        "name": name,
                        "frequency": data["frequency"],
                        "total_duration_ms": us_to_ms(data["duration"])
                    }
                    for i, (name, data) in enumerate(top_k_kernels["by_duration"], 1)
                ]
            }
        }
        
        # Add fusion results if available
        if fusion_results is not None:
            run_result["fusion_analysis"] = fusion_results
        
        # Accumulate runs for sweep mode
        self._sweep_runs.append(run_result)
        
        batch_sizes = self.args.batch_size if isinstance(self.args.batch_size, list) else [self.args.batch_size]
        seq_lens = self.args.seq_len if isinstance(self.args.seq_len, list) else [self.args.seq_len]
        is_sweep = len(batch_sizes) > 1 or len(seq_lens) > 1
        
        if is_sweep:
            # Sweep mode: wrap all accumulated runs in sweep structure
            output = {
                "sweep_info": {
                    "run_id": self.run_id,
                    "model_name": model_name,
                    "batch_sizes": batch_sizes,
                    "seq_lens": seq_lens,
                    "precision": self.args.precision,
                    "compile_type": self.args.compile_type,
                },
                "runs": self._sweep_runs
            }
        else:
            # Single run: use simple format (no sweep wrapper)
            output = run_result
        
        # Save to file (overwrites each time with accumulated data)
        with open(self.report_file_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"* Metrics exported to: {self.report_file_path}")
        return str(self.report_file_path)
    
    def exit(self) -> None:
        """
        Cleanup function that prints log location and cleans up logger handlers.
        """
        print(f"\nLog output saved to {self._soda_logger.log_path}")
        self._soda_logger.cleanup()


class SodaLogger:
    """
    Logger class for SODA that supports both file and console output.
    """
    
    def __init__(self, output_dir: Path, is_console: bool = True, is_file: bool = True):
        """
        Initialize the logger.
        
        Args:
            output_dir: Directory where log file will be created.
            is_console: If True, write to console/stdout.
            is_file: If True, write to file.
        """
        self.log_path = output_dir / "soda.log"
        self.is_console = is_console

        self.is_file = is_file
        
        # Create logger
        self.logger = logging.getLogger("soda")
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Create formatter without timestamp
        formatter = logging.Formatter('%(message)s')
        
        # File handler - writes to file
        if self.is_file:
            file_handler = logging.FileHandler(self.log_path, mode='w')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Console handler - writes to stdout
        if self.is_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Log initial message
        self.logger.info(f"Results will be saved to: {output_dir.resolve()}")
    
    def cleanup(self):
        """Clean up logging handlers."""
        self.logger.handlers.clear()


class ModelHandler:
    """Handles loading of Hugging Face models with specific configurations."""

    def __init__(self, model_name: str, device: str, compile_type: str, precision: str):
        """
        Initializes the Model loader.

        Args:
            model_name: The name of the model from Hugging Face Hub.
            device: The device to load the model onto ('cpu' or 'cuda').
            compile_type: The compilation mode ('eager', 'torch.compile', 'flash-attention').
            precision: The desired data type ('float32', 'float16', 'bfloat16', 'float8_e4m3fn').
        """
        self.model_name = model_name
        self.device = torch.device(device)
        self.compile_type = compile_type
        self.precision_str = precision
        
        # FP8 requires special handling
        self.is_fp8 = precision == "float8_e4m3fn"
        
        # Map for loading precision (FP8 loads as bfloat16, then converts)
        self.load_precision_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float8_e4m3fn": torch.bfloat16,  # Load in bf16, convert later
        }
        
        self.precision_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        if hasattr(torch, 'float8_e4m3fn'):
            self.precision_map["float8_e4m3fn"] = torch.float8_e4m3fn

        self.precision = self.precision_map[precision]
        self.load_precision = self.load_precision_map[precision]
        
        # Determine if model is decoder or encoder
        self.is_decoder = not ("bert" in model_name.lower() or "roberta" in model_name.lower())
        
        # Load model; this will set self.pytorch_model and self.tokenizer
        self.pytorch_model = None
        self.tokenizer = None
        self.load()

    def get_kwargs(self) -> Dict[str, Any]:
        """Returns common kwargs for model loading."""
        kwargs = {
            "torch_dtype": self.load_precision,  # Use load_precision, not target precision
            "device_map": self.device if self.device.type == 'cuda' else 'cpu',
        }
        
        # FP8 Config Support
        if self.is_fp8 and FP8_CONFIG_AVAILABLE:
            # Use quantization_config if available for E4M3
            kwargs["quantization_config"] = FP8Config(fp8_format="e4m3")
            
        return kwargs
    
    def _convert_to_fp8(self, model: transformers.PreTrainedModel) -> transformers.PreTrainedModel:
        """
        Convert model linear layer weights to FP8 E4M3 format for inference.
        """
        print("Converting linear layer weights to float8_e4m3fn...")
        
        converted_count = 0
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                with torch.no_grad():
                    # FP8 E4M3 has range ~±448, clamp to avoid overflow
                    weight_clamped = module.weight.data.clamp(-448.0, 448.0)
                    module.weight.data = weight_clamped.to(torch.float8_e4m3fn)
                    converted_count += 1
        
        print(f"Converted {converted_count} linear layers to FP8 E4M3.")
        return model

    def load_encoder(self) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
        """Loads an encoder-only model (e.g., BERT)."""
        # Load tokenizer first to get eos_token_id
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        kwargs = self.get_kwargs()
        if self.compile_type == "torch.compile":
            kwargs.update({"attn_implementation": "sdpa"})
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                self.model_name, **kwargs
            ).eval()
            model.forward = torch.compile(model.forward, mode="reduce-overhead", backend="inductor")
        elif self.compile_type == "flash-attention":
            kwargs.update({"attn_implementation": "flash_attention_2"})
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                self.model_name, **kwargs
            ).eval()
        else:  # eager
            kwargs.update({"attn_implementation": "eager"})
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                self.model_name, **kwargs
            ).eval()
        
        # Convert to FP8 if requested AND NOT using quantization_config
        if self.is_fp8 and "quantization_config" not in kwargs:
            model = self._convert_to_fp8(model)
        
        if hasattr(model, 'generation_config') and model.generation_config.pad_token_id is None:
            model.generation_config.pad_token_id = tokenizer.eos_token_id
        
        # Store model and tokenizer
        self.pytorch_model = model
        self.tokenizer = tokenizer
        return model, tokenizer


    def load_decoder(self) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
        """Loads a decoder-only model (e.g., Llama)."""
        # Load tokenizer first to get eos_token_id
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load config and set pad_token_id before model initialization to prevent warning
        config = transformers.AutoConfig.from_pretrained(self.model_name)
        if hasattr(config, 'pad_token_id') and config.pad_token_id is None:
            config.pad_token_id = tokenizer.eos_token_id
        
        kwargs = self.get_kwargs()
        if self.compile_type == "torch.compile":
            kwargs.update({"attn_implementation": "sdpa"})
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name, config=config, **kwargs
            ).eval()
            model.generation_config.cache_implementation = "static"
            model.forward = torch.compile(model.forward, mode="reduce-overhead", backend="inductor")
        elif self.compile_type == "flash-attention":
            kwargs.update({"attn_implementation": "flash_attention_2"})
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name, config=config, **kwargs
            ).eval()
        else:  # eager
            kwargs.update({"attn_implementation": "eager"})
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name, config=config, **kwargs
            ).eval()
        

        # Convert to FP8 if requested AND NOT using quantization_config
        if self.is_fp8 and "quantization_config" not in kwargs:
            model = self._convert_to_fp8(model)
        
        # Store model and tokenizer
        self.pytorch_model = model
        self.tokenizer = tokenizer
        return model, tokenizer
    
    def load(self) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
        """
        Loads the model and tokenizer based on model type (encoder or decoder).
        
        Returns:
            Tuple of (model, tokenizer).
        """
        if self.is_decoder:
            return self.load_decoder()
        else:
            return self.load_encoder()
    
    def generate_synthetic_inputs(self, batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
        """
        Generates synthetic tokenized inputs for profiling.
        
        Args:
            batch_size: Batch size for the inputs.
            seq_len: Sequence length for the inputs.
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask' tensors.
        """
        return {
            "input_ids": torch.randint(
                1, self.tokenizer.vocab_size, size=(batch_size, seq_len), device=self.device
            ),
            "attention_mask": torch.ones(
                batch_size, seq_len, device=self.device
            ),
        }


def main() -> int:
    """Main entry point for the SODA CLI."""
    
    # Check if env.sh has been sourced and loaded
    if not os.environ.get("SODA_ENV_LOADED"):
        print("Error: SODA environment not loaded.", file=sys.stderr)
        print("Please run: source env.sh", file=sys.stderr)
        sys.exit(1)

    try:
        # Parse and validate arguments 
        args = SodaProfiler.parse_and_validate_args()

        # Prepare model handler
        print(f"Loading model: {args.model} with precision {args.precision}...")
        model_handler = ModelHandler(
            model_name=args.model,
            device=args.device,
            compile_type=args.compile_type,
            precision=args.precision,
        )

        # Get sweep parameters
        batch_sizes = args.batch_size if isinstance(args.batch_size, list) else [args.batch_size]
        seq_lens = args.seq_len if isinstance(args.seq_len, list) else [args.seq_len]
        is_sweep = len(batch_sizes) > 1 or len(seq_lens) > 1
        
        # Calculate total runs for sweep
        total_runs = len(batch_sizes) * len(seq_lens)
        
        if is_sweep:
            print(f"Running sweep: batch_sizes={batch_sizes}, seq_lens={seq_lens} ({total_runs} runs)")

        # Initialize profiler 
        profiler = SodaProfiler(model_handler=model_handler, args=args, log_console=True, log_file=True)

        print("Performing global warmup to stabilize profiler...")
        warmup_bs = batch_sizes[0]
        warmup_sl = seq_lens[0]
        warmup_inputs = model_handler.generate_synthetic_inputs(warmup_bs, warmup_sl)
        
        # We use a temporary trace file for warmup to avoid overwriting real results
        original_trace_path = profiler.trace_file_path
        profiler.trace_file_path = profiler.output_dir / "warmup_trace.json"
        
        # Run profile but don't analyze/save
        try:
            # Just run forward pass, no profiling overhead if possible, but we need to warm up profiler too
            # So we run the full profile method
            profiler.profile_forward_pass(warmup_inputs, batch_size=warmup_bs, seq_len=warmup_sl)
        except Exception as e:
            print(f"Warmup warning: {e}")
        
        # Restore path
        profiler.trace_file_path = original_trace_path
        print("Global warmup complete.")
        # --------------------------

        
        run_idx = 0
        for bs in batch_sizes:
            for sl in seq_lens:
                run_idx += 1
                
                if is_sweep:
                    profiler.logger.info(f"\n{'='*60}")
                    profiler.logger.info(f"=== Sweep Run {run_idx}/{total_runs}: bs={bs}, sl={sl} ===")
                    profiler.logger.info(f"{'='*60}")

                print(f"Generating synthetic input: batch_size={bs}, seq_len={sl}")
                model_inputs = model_handler.generate_synthetic_inputs(bs, sl)

                # Update trace file path for this run (separate traces per config)
                if is_sweep:
                    profiler.trace_file_path = profiler.output_dir / f"trace_bs{bs}_sl{sl}.json"

                # Profile forward pass and analyze 
                profiler.profile_forward_pass(model_inputs, batch_size=bs, seq_len=sl)
                profiler.analyze()

                # Report and save results
                profiler.report()
                profiler.save(batch_size=bs, seq_len=sl)
        
        # Cleanup and exit
        profiler.exit()
        
        return 0

    except FileNotFoundError as e:
        print(f"Error: File not found: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"Error: Runtime error during profiling: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: Unexpected error: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())