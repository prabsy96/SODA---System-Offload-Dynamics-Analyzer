"""
Utility classes and functions for SODA.

This module provides core functionalities including:
- Model: A wrapper for loading Hugging Face models with specific configurations.
- Benchmark: E2E latency benchmarking for TTFT and TPOT (currently unused by main but available).
- SodaProfiler: The main class for profiling a model's forward pass, parsing the
                PyTorch profiler's output, and calculating performance metrics.
- Graph: A utility for visualizing model layer to ATen op mappings (currently unused).
"""

import argparse
import json
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import transformers
from torch.profiler import ProfilerActivity, profile

# Configure logging
log = logging.getLogger(__name__)

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

class ModelHandler:
    """Handles loading of Hugging Face models with specific configurations."""

    def __init__(self, model_name: str, device: str, compile_type: str, precision: str):
        """
        Initializes the Model loader.

        Args:
            model_name: The name of the model from Hugging Face Hub.
            device: The device to load the model onto ('cpu' or 'cuda').
            compile_type: The compilation mode ('eager', 'torch.compile', 'flash-attention').
            precision: The desired data type ('float32', 'float16', 'bfloat16').
        """
        self.model_name = model_name
        self.device = torch.device(device)
        self.compile_type = compile_type
        self.precision_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.precision = self.precision_map[precision]
        
        # Determine if model is decoder or encoder
        self.is_decoder = not ("bert" in model_name.lower() or "roberta" in model_name.lower())
        
        # Load model; this will set self.pytorch_model and self.tokenizer
        self.pytorch_model = None
        self.tokenizer = None
        self.load()

    def get_kwargs(self) -> Dict[str, Any]:
        """Returns common kwargs for model loading."""
        return {
            "dtype": self.precision,
            "device_map": self.device if self.device.type == 'cuda' else 'cpu',
        }

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
    
    def generate_synthetic_inputs(self, batch_size: int, seq_len: int, device: str) -> Dict[str, torch.Tensor]:
        """
        Generates synthetic tokenized inputs for profiling.
        
        Args:
            batch_size: Batch size for the inputs.
            seq_len: Sequence length for the inputs.
            device: Device to create tensors on ('cpu' or 'cuda').
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask' tensors.
        """
        return {
            "input_ids": torch.randint(
                1, self.tokenizer.vocab_size, size=(batch_size, seq_len), device=device
            ),
            "attention_mask": torch.ones(
                batch_size, seq_len, device=device
            ),
        }


class SodaProfiler:
    """
    Handles model tracing, profile data parsing, and metric generation.
    """
    @staticmethod
    def generate_experiment_name(args: argparse.Namespace) -> str:
        """
        Generates a unique experiment directory name from arguments.
        
        Args:
            args: Parsed command-line arguments.
            
        Returns:
            Experiment directory name string.
        """
        return f"{args.model.replace('/', '_')}_{args.compile_type}_bs{args.batch_size}_sl{args.seq_len}"

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
        self.args = args
        self.model_handler = model_handler
        
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
        self.report_file_path = self.output_dir / "report.json"
        
        self.trace = None
        self.events = None
        self.results = None

    def trace_forward_pass_for_encoder(self, inputs: Dict[str, torch.Tensor]) -> str:
        """
        Profiles the forward pass of an encoder model.

        Args:
            inputs: A dictionary of tokenized inputs.
            tokenizer: The model's tokenizer.

        Returns:
            The path to the generated Chrome trace JSON file.
        """
        self.logger.info("Profiling model forward pass...")
        
        # Warm-up runs
        with torch.no_grad():
            for _ in range(5):
                self.model_handler.pytorch_model(**inputs)
        
        # Profiled run
        with torch.no_grad():
            with profile(
                activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                with_modules=True,
                with_flops=True,
                profile_memory=True,
            ) as prof:
                self.model_handler.pytorch_model(**inputs)

        prof.export_chrome_trace(str(self.trace_file_path))
        
        # Load trace data into memory immediately
        self.trace = self.load_trace_file(self.trace_file_path)
        
        self.logger.info(f"Chrome trace file generated at: {self.trace_file_path}")
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
        self.logger.info("Profiling model forward pass...")
        
        # Warm-up runs
        with torch.no_grad():
            for _ in range(5):
                self.model_handler.pytorch_model.generate(**inputs, max_new_tokens=1, do_sample=False, pad_token_id=self.model_handler.tokenizer.pad_token_id)

        # Profiled run
        with torch.no_grad():
            with profile(
                activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                with_modules=True,
                with_flops=True,
                profile_memory=True,
            ) as prof:
                self.model_handler.pytorch_model.generate(**inputs, max_new_tokens=1, do_sample=False, pad_token_id=self.model_handler.tokenizer.pad_token_id)
        
        prof.export_chrome_trace(str(self.trace_file_path))
        
        # Load trace data into memory immediately
        self.trace = self.load_trace_file(self.trace_file_path)
        
        self.logger.info(f"Chrome trace file generated at: {self.trace_file_path}")
        return str(self.trace_file_path)

    def load_trace_file(self, file_path: Path) -> Dict[str, Any]:
        """Loads and returns the content of a JSON trace file."""
        if not file_path.is_file():
            raise FileNotFoundError(f"Trace file does not exist: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def collect_events_from_trace(self) -> Dict[str, Any]:
        """
        Collects all events from trace organized by category.
        
        Uses self.trace
        
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
        
        for event in self.trace.get("traceEvents", []):
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
        self.events = events
        return events

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
        
        for event in self.trace.get("traceEvents", []):
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

    def generate_dependencies(self) -> List[Tuple]:
        """
        Analyzes dependencies between CUDA runtime calls and kernel launches.

        Uses self.events.

        Returns:
            List of (kernel, cuda_launch) dependency tuples.
        """
        gpu_events = self.events["gpu"]
        cuda_launches = self.events["cpu"]["launches"]
        kernel_events = gpu_events["kernels"]

        self.logger.info(f"Analyzing {len(kernel_events)} kernel events from profiled run.")

        dependencies = []
        for kernel in kernel_events:
            corr = kernel.get("correlation")
            if corr is not None and corr in cuda_launches:
                dependencies.append((kernel, cuda_launches[corr]))
        
        return dependencies

    def calculate_launch_tax(self, dependence: List[Tuple]) -> Dict[str, float]:
        """
        Calculates the total and average kernel launch tax.

        Args:
            dependence: List of (kernel, runtime) dependency tuples.

        Returns:
            Dictionary with "total" and "avg" keys (in microseconds).
        """
        if not dependence:
            return {"total": 0.0, "avg": 0.0}

        total_tax_us = 0.0
        for kernel, runtime in dependence:
            runtime_end = runtime["ts"] + runtime.get("dur", 0)
            kernel_start = kernel["ts"]

            gap_us = kernel_start - runtime_end

            if gap_us > 0:
                total_tax_us += gap_us

        num_kernels = len(dependence)
        avg_tax_us = total_tax_us / num_kernels if num_kernels > 0 else 0.0

        return {
            "total": total_tax_us,
            "avg": avg_tax_us,
        }
    
    def get_average_kernel_duration(self) -> Dict[str, float]:
        """
        Calculates the average execution duration (aka operational intensity) for each unique kernel.
        
        Aggregates all instances of each kernel and computes the mean duration.
        
        Uses self.events.
            
        Returns:
            Dictionary mapping kernel name to average duration in milliseconds.
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
                avg_duration_us = stat["total_duration"] / stat["count"]
                avg_durations[name] = float(avg_duration_us) / 1000.0
            else:
                avg_durations[name] = 0.0
        
        return avg_durations

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

    def kernelchains(self, dependence: List[Tuple], exact_length: int, prox_score_threshold: float):
        """
        Analyzes kernel launch sequences to find opportunities for fusion.

        Args:
            dependence: List of (kernel, runtime) dependency tuples.
            exact_length: The exact length of kernel chains to analyze.
            prox_score_threshold: The proximity score required to recommend a fusion (e.g., 1.0 for deterministic).
        """
        if exact_length < 2:
            log.warning("Kernel chain length must be at least 2.")
            return

        all_segments = []
        current_segment = []
        # Separate dependencies by synchronization points
        for kernel, runtime in dependence:
            if 'cudaStreamSynchronize' in runtime.get("name", ""):
                if current_segment:
                    all_segments.append(current_segment)
                current_segment = []
            current_segment.append((kernel, runtime))
        if current_segment:
            all_segments.append(current_segment)

        total_kernel_launches = len(dependence)
        unique_fusion_candidates: Set[Tuple[str, ...]] = set()
        
        # Process each segment to find chains
        for segment in all_segments:
            current_chain = deque(maxlen=exact_length)
            for kernel, _ in segment:
                current_chain.append(kernel["name"])
                if len(current_chain) == exact_length:
                    chain_tuple = tuple(current_chain)
                    unique_fusion_candidates.add(chain_tuple)

        # Calculate proximity scores
        fusion_recommendations = []
        kernel_freq: DefaultDict[str, int] = defaultdict(int)
        for kernel, _ in dependence:
            kernel_freq[kernel["name"]] += 1
            
        for chain in unique_fusion_candidates:
            # Count occurrences of this exact chain
            count = 0
            for segment in all_segments:
                segment_kernels = [k["name"] for k, r in segment]
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
        log.info(f"--- Fusion Analysis (Length={exact_length}, Threshold={prox_score_threshold}) ---")
        if not fusion_recommendations:
            log.info("No kernel chains met the fusion criteria.")
            return None

        sorted_recommendations = sorted(fusion_recommendations, key=lambda x: x[1], reverse=True)
        log.info(f"Found {len(sorted_recommendations)} potential fusion candidates:")
        for chain, count, score in sorted_recommendations:
            chain_str = ' -> '.join(chain)
            log.info(f"  - Chain: [{chain_str}] (Found {count} times, Proximity Score: {score:.2f})")
        
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

    
    def preprocess_trace(self) -> List[Tuple]:
        """
        Preprocesses trace data by collecting events and generating dependencies.
        
        Returns:
            Kernel dependency graph (list of tuples).
        """
        # Get all events organized by category 
        self.collect_events_from_trace()
        dependencies = self.generate_dependencies()
        return dependencies
    
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
            - dependencies: Kernel dependency graph
            - avg_kernel_dur: Average kernel duration results
        """
        self.logger.info("Analyzing trace data to generate reports...")
        # Preprocess trace data
        dependencies = self.preprocess_trace()
        
        # Analyze per-stream metrics
        stream_info = self.analyze_per_stream()
        
        # General metrics
        total_inference_time = self.calculate_total_inference_time()
        
        # GPU metrics
        total_gpu_time_span = self.calculate_total_gpu_time_span()
        true_gpu_busy_time = self.calculate_true_gpu_busy_time()
        gpu_utilization = self.calculate_gpu_utilization()
        
        # Kernel metrics
        kernel_exec_time = self.calculate_kernel_exec_time()
        launch_tax = self.calculate_launch_tax(dependencies)
        avg_kernel_dur = self.get_average_kernel_duration()
        top_k_kernels = self.get_top_k_kernels(k=3)
        
        # Fusion analysis
        fusion_results = None
        if self.args.fusion:
            self.logger.info("--- Kernel Fusion Analysis ---")
            fusion_results = {}
            for f in self.args.fusion:
                fusion_results[f] = self.kernelchains(dependencies, f, self.args.prox_score)
        
        # Build metrics dictionary 
        metrics = {
            # General metrics
            "inference_runtime_ms": total_inference_time / 1000,
            "active_streams": len(stream_info),

            # GPU metrics
            "total_gpu_time_span_ms": total_gpu_time_span / 1000,
            "gpu_busy_time_ms": true_gpu_busy_time / 1000,
            "gpu_idle_time_ms": max(0.0, (total_gpu_time_span - true_gpu_busy_time) / 1000),
            "gpu_utilization_percent": gpu_utilization,
 
            # Kernel metrics
            "total_kernel_exec_time_ms": kernel_exec_time["total"] / 1000,
            "num_total_kernels": len(self.events["gpu"]["kernels"]),
            "avg_kernel_exec_time_ms": kernel_exec_time["avg"] / 1000,
            "total_kernel_launch_tax_ms": launch_tax["total"] / 1000,
            "avg_kernel_launch_tax_ms": launch_tax["avg"] / 1000,
        }
        
        self.results = {
            "metrics": metrics,
            "stream_info": stream_info,
            "top_k_kernels": top_k_kernels,
            "dependencies": dependencies,
            "avg_kernel_dur": avg_kernel_dur,
            "fusion_results": fusion_results,
        }
        
        return self.results
    
    def report(self) -> None:
        """
        Prints performance metrics, stream analysis, and top-k kernels.
        Uses results stored in self.results from analyze().
        """
        if self.results is None:
            raise ValueError("No analysis results available. Call analyze() first.")
        
        metrics = self.results["metrics"]
        stream_info = self.results["stream_info"]
        top_k_kernels = self.results["top_k_kernels"]
        
        # --- Enhanced Reporting ---
        self.logger.info("--- Performance Metrics ---")
        self.logger.info(f"Inference runtime (ms): {metrics['inference_runtime_ms']:.4f}")
        self.logger.info(f"Total kernel execution time (ms): {metrics['total_kernel_exec_time_ms']:.4f}")
        self.logger.info(f"GPU busy time (concurrent-aware) (ms): {metrics['gpu_busy_time_ms']:.4f}")
        self.logger.info(f"GPU idle time (ms): {metrics['gpu_idle_time_ms']:.4f}")
        self.logger.info(f"GPU utilization: {metrics['gpu_utilization_percent']:.2f}%")
        self.logger.info(f"Total kernel launch tax (TKLQT) (ms): {metrics['total_kernel_launch_tax_ms']:.4f}")
        self.logger.info(f"Number of kernels: {metrics['num_total_kernels']}")
        self.logger.info(f"Active streams: {metrics['active_streams']}")
        
        if metrics['num_total_kernels'] > 0:
            self.logger.info(f"Avg. kernel launch tax per kernel (ms): {metrics['avg_kernel_launch_tax_ms']:.4f}")
            self.logger.info(f"Avg. execution time per kernel (ms): {metrics['avg_kernel_exec_time_ms']:.4f}")
        
        # --- Per-Stream Breakdown ---
        self.logger.info("--- Per-Stream Analysis ---")
        for stream_id, data in stream_info.items():
            self.logger.info(
                f"  Stream {stream_id}: {data['op_count']} ops "
                f"({data['kernel_count']} kernels), "
                f"Busy Time: {data['true_gpu_busy_time'] / 1000:.4f} ms"
            )
        
        # Top-K kernels 
        if top_k_kernels["by_frequency"]:
            self.logger.info("--- Top-3 Kernels by Frequency ---")
            for i, (name, data) in enumerate(top_k_kernels["by_frequency"], 1):
                self.logger.info(
                    f"#{i}: {name} "
                    f"(Frequency: {int(data['frequency'])}, "
                    f"Total Duration: {data['duration'] / 1000:.4f} ms)"
                )
            
            self.logger.info("--- Top-3 Kernels by Duration ---")
            for i, (name, data) in enumerate(top_k_kernels["by_duration"], 1):
                self.logger.info(
                    f"#{i}: {name} "
                    f"(Total Duration: {data['duration'] / 1000:.4f} ms, "
                    f"Frequency: {int(data['frequency'])})"
                )
    
    def save(self) -> str:
        """
        Saves analysis results to JSON file.
        Uses results stored in self.results from analyze().
        Generates model_name and config from self.args.
            
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
        config = {
            "batch_size": self.args.batch_size,
            "seq_len": self.args.seq_len,
            "precision": self.args.precision,
            "compile_type": self.args.compile_type,
            "device": self.args.device,
        }
        
        # Build output structure
        output = {
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
                    "busy_time_ms": data["true_gpu_busy_time"] / 1000,
                    "total_kernel_exec_time_ms": data["total_kernel_exec_time"] / 1000,
                }
                for stream_id, data in stream_info.items()
            },
            "top_kernels": {
                "by_frequency": [
                    {
                        "rank": i,
                        "name": name,
                        "frequency": data["frequency"],
                        "total_duration_ms": data["duration"] / 1000
                    }
                    for i, (name, data) in enumerate(top_k_kernels["by_frequency"], 1)
                ],
                "by_duration": [
                    {
                        "rank": i,
                        "name": name,
                        "frequency": data["frequency"],
                        "total_duration_ms": data["duration"] / 1000
                    }
                    for i, (name, data) in enumerate(top_k_kernels["by_duration"], 1)
                ]
            }
        }
        
        # Add fusion results if available
        if fusion_results is not None:
            output["fusion_analysis"] = fusion_results
        
        # Save to file
        with open(self.report_file_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        log.info(f"Metrics exported to: {self.report_file_path}")
        return str(self.report_file_path)
    
    def exit(self) -> None:
        """
        Cleanup function that prints log location and cleans up logger handlers.
        """
        print(f"\nLog output saved to {self._soda_logger.log_path}")
        self._soda_logger.cleanup()
