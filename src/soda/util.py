"""
Utility classes and functions for SODA.

This module provides core functionalities including:
- Model: A wrapper for loading Hugging Face models with specific configurations.
- Benchmark: E2E latency benchmarking for TTFT and TPOT (currently unused by main but available).
- TraceModel: The main class for profiling a model's forward pass, parsing the
              PyTorch profiler's output, and calculating performance metrics.
- Graph: A utility for visualizing model layer to ATen op mappings (currently unused).
"""

import heapq
import json
import logging
import os
import random
import re
import time
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

class Model:
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

    def _get_common_kwargs(self) -> Dict[str, Any]:
        """Returns common kwargs for model loading."""
        return {
            "torch_dtype": self.precision,
            "device_map": self.device if self.device.type == 'cuda' else 'cpu',
        }

    def load_encoder(self) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
        """Loads an encoder-only model (e.g., BERT)."""
        kwargs = self._get_common_kwargs()
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

        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        return model, tokenizer

    def load_decoder(self) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
        """Loads a decoder-only model (e.g., Llama)."""
        kwargs = self._get_common_kwargs()
        if self.compile_type == "torch.compile":
            kwargs.update({"attn_implementation": "sdpa"})
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name, **kwargs
            ).eval()
            model.generation_config.cache_implementation = "static"
            model.forward = torch.compile(model.forward, mode="reduce-overhead", backend="inductor")
        elif self.compile_type == "flash-attention":
            kwargs.update({"attn_implementation": "flash_attention_2"})
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name, **kwargs
            ).eval()
        else:  # eager
            kwargs.update({"attn_implementation": "eager"})
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name, **kwargs
            ).eval()

        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        return model, tokenizer


class TraceModel:
    """
    Handles model tracing, profile data parsing, and metric generation.
    """
    def __init__(self, name: str, file: str, path: str, model: nn.Module):
        """
        Initializes the tracer.

        Args:
            name: A unique name for this tracing run (e.g., model name + config).
            file: The filename for the output trace file (e.g., 'trace.json').
            path: The base directory to save all output artifacts.
            model: The PyTorch model instance to trace.
        """
        self.name = name
        self.file = file
        self.path = Path(path)
        self.model = model
        self.dump_dir = self.path / self.name
        self.dump_dir.mkdir(parents=True, exist_ok=True)

    def trace_forward_pass_for_encoder(self, inputs: Dict[str, torch.Tensor], tokenizer) -> str:
        """
        Profiles the forward pass of an encoder model.

        Args:
            inputs: A dictionary of tokenized inputs.
            tokenizer: The model's tokenizer.

        Returns:
            The path to the generated Chrome trace JSON file.
        """
        # Warm-up runs
        with torch.no_grad():
            for _ in range(5):
                self.model(**inputs)
        
        # Profiled run
        with torch.no_grad():
            with profile(
                activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                with_modules=True,
                with_flops=True,
                profile_memory=True,
            ) as prof:
                self.model(**inputs)

        json_file = self.dump_dir / self.file
        prof.export_chrome_trace(str(json_file))
        return str(json_file)

    def trace_forward_pass_for_decoder(self, inputs: Dict[str, torch.Tensor], tokenizer, bs: int, sq: int) -> str:
        """
        Profiles the generate step of a decoder model.

        Args:
            inputs: A dictionary of tokenized inputs.
            tokenizer: The model's tokenizer.
            bs: Batch size.
            sq: Sequence length.

        Returns:
            The path to the generated Chrome trace JSON file.
        """
        # Warm-up runs
        with torch.no_grad():
            for _ in range(5):
                self.model.generate(**inputs, max_new_tokens=1, do_sample=False, pad_token_id=tokenizer.pad_token_id)

        # Profiled run
        with torch.no_grad():
            with profile(
                activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                with_modules=True,
                with_flops=True,
                profile_memory=True,
            ) as prof:
                self.model.generate(**inputs, max_new_tokens=1, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        
        json_file = self.dump_dir / self.file
        prof.export_chrome_trace(str(json_file))
        return str(json_file)

    def _load_trace_file(self, file_path: Path) -> Dict[str, Any]:
        """Loads and returns the content of a JSON trace file."""
        if not file_path.is_file():
            raise FileNotFoundError(f"Trace file does not exist: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def collect_events_from_trace(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collects all events from trace organized by category.
        
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
        return {
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

    def calculate_total_gpu_time_span(self, events: Dict[str, Any]) -> float:
        """
        Calculates the end-to-end GPU time span by finding min start and max end
        across GPU execution events (kernel, gpu_memcpy, gpu_memset).
        
        Measures the extreme time window of GPU execution (from first to last GPU event).
        Excludes cuda_runtime (CPU-side calls).
        
        Args:
            events: Dictionary containing collected events from trace, organized hierarchically:
                - cpu: CPU-side events (ops, launches)
                - gpu: GPU-side events (kernels, memory, all)
            
        Returns:
            Time span in microseconds (max_end - min_start).
        """
        gpu_events = events["gpu"]["all"]
        
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

    def calculate_total_kernel_exec_time(self, events: Dict[str, Any]) -> float:
        """
        Calculates total kernel execution time by summing durations of all kernel events.
        
        Args:
            events: Dictionary containing collected events from trace, organized hierarchically:
                - cpu: CPU-side events (ops, launches)
                - gpu: GPU-side events (kernels, memory, all)
            
        Returns:
            Total kernel execution time in microseconds.
        """
        kernel_events = events["gpu"]["kernels"]
        
        total_kernel_exec_time = 0.0
        for kernel in kernel_events:
            total_kernel_exec_time += float(kernel.get("dur", 0))
        
        return total_kernel_exec_time

    def calculate_true_gpu_busy_time(self, events: Dict[str, Any]) -> float:
        """
        Calculates GPU busy time by merging overlapping GPU event intervals.
        
        Accounts for concurrent GPU execution across all streams by merging
        overlapping time intervals. Includes kernel, gpu_memcpy, and gpu_memset
        events. If events run concurrently on different streams, their overlapping 
        time is counted once.
        
        Args:
            events: Dictionary containing collected events from trace, organized hierarchically:
                - cpu: CPU-side events (ops, launches)
                - gpu: GPU-side events (kernels, memory, all)
            
        Returns:
            Merged GPU busy time in microseconds.
        """
        gpu_events = events["gpu"]["all"]
        
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

    def calculate_gpu_utilization(self, events: Dict[str, Any]) -> float:
        """
        Calculates GPU utilization percentage.
        
        Args:
            events: Dictionary containing collected events from trace, organized hierarchically:
                - cpu: CPU-side events (ops, launches)
                - gpu: GPU-side events (kernels, memory, all)
            
        Returns:
            GPU utilization as a percentage (0.0 to 100.0).
        """
        # Calculate denominator: time span of GPU execution events only 
        total_gpu_time_span = self.calculate_total_gpu_time_span(events)
        
        # Avoid division by zero
        if total_gpu_time_span == 0.0:
            return 0.0
        
        # Calculate numerator: non overlapping busy time of GPU execution events
        true_gpu_busy_time = self.calculate_true_gpu_busy_time(events)
        
        # Calculate GPU utilization percentage
        gpu_utilization = (true_gpu_busy_time / total_gpu_time_span)
        gpu_utilization = gpu_utilization * 100.0

        return gpu_utilization

    def analyze_per_stream(self, gpu_events: List[Dict]) -> Dict:
        """
        Analyzes GPU events grouped by stream.
        
        For each stream, calculates:
        - Total operations and kernel count
        - Total kernel execution time (sum of durations)
        - True GPU busy time (merged overlapping intervals)
        
        Args:
            gpu_events: List of GPU event dictionaries (kernel, gpu_memcpy, gpu_memset).
            
        Returns:
            Dictionary mapping stream_id to stream metrics.
        """
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

    def generate_dependencies(self, events: Dict[str, Any]) -> List[Tuple]:
        """
        Analyzes dependencies between CUDA runtime calls and kernel launches.

        Args:
            events: Dictionary containing collected events from trace, organized hierarchically:
                - cpu: CPU-side events (ops, launches)
                - gpu: GPU-side events (kernels, memory, all)

        Returns:
            List of (kernel, cuda_launch) dependency tuples.
        """
        gpu_events = events["gpu"]
        cuda_launches = events["cpu"]["launches"]
        kernel_events = gpu_events["kernels"]

        print(f"Analyzing {len(kernel_events)} kernel events from profiled run.")

        dependencies = []
        for kernel in kernel_events:
            corr = kernel.get("correlation")
            if corr is not None and corr in cuda_launches:
                dependencies.append((kernel, cuda_launches[corr]))
        
        return dependencies

    def calculate_launch_tax(self, dependence: List[Tuple]) -> float: 
        """
        Calculates the total and average launch overhead (tax).

        Args:
            dependence: List of (kernel, runtime) dependency tuples.
            kernel_count: Total number of kernels launched.

        Returns:
            The total launch overhead time.
        """
        if not dependence:
            return 0.0

        total_overhead_us = 0.0
        for kernel, runtime in dependence:
            runtime_end = runtime["ts"] + runtime.get("dur", 0)
            kernel_start = kernel["ts"]

            gap_us = kernel_start - runtime_end

            if gap_us > 0:
                total_overhead_us += gap_us

        return total_overhead_us
    
    def get_average_kernel_duration(self, kernel_events: List[Dict]) -> Dict[str, float]:
        """
        Calculates and saves the operational intensity (average kernel duration) for each unique kernel.
        
        Returns:
        Dict mapping kernel name -> average duration in milliseconds
        """
        kernel_stats = defaultdict(lambda: {"total_dur": 0.0, "count": 0})
    
        for k in kernel_events:
            name = k["name"]
            kernel_stats[name]["total_dur"] += k.get("dur", 0)
            kernel_stats[name]["count"] += 1
        
        akd_map = {}
        for name, data in kernel_stats.items():
            if data["count"] > 0:
                avg_dur_us = data["total_dur"] / data["count"]
                akd_map[name] = float(avg_dur_us)/1000.0
            else:
                akd_map[name] = 0.0
        
        return akd_map

    def get_top_k_kernels(self, kernel_events: List[Dict], k: int = 3):
        """
        Identifies the top-k most frequent and time-consuming kernels.
        
        Args:
            kernel_events: List of kernel event dictionaries.
            k: Number of top kernels to report.
        """
        if not kernel_events:
            print("Warning: No kernel events to analyze.")
            return
        
        kernel_data = defaultdict(lambda: {"frequency": 0, "duration": 0.0, "type": None, "streams": set()})
        
        for kernel in kernel_events:
            kernel_name = kernel["name"]
            kernel_data[kernel_name]["frequency"] += 1
            kernel_data[kernel_name]["duration"] += float(kernel.get("dur", 0))
            kernel_data[kernel_name]["type"] = kernel.get("type")
            if kernel.get("stream") is not None:
                kernel_data[kernel_name]["streams"].add(kernel["stream"])
        
        # Top k by frequency
        top_k_freq = heapq.nlargest(k, kernel_data.items(), key=lambda item: item[1]["frequency"])
        
        # Top k by duration
        top_k_dur = heapq.nlargest(k, kernel_data.items(), key=lambda item: item[1]["duration"])
        
        print(f"--- Top-{k} Kernels by Frequency ---")
        for i, (name, data) in enumerate(top_k_freq, 1):
            streams = list(data["streams"]) if data["streams"] else ["N/A"]
            print(
                f"#{i}: {name} "
                f"(Frequency: {int(data['frequency'])}, "
                f"Total Duration: {data['duration'] / 1000:.4f} ms, "
                f"Streams: {streams})"
            )
        
        print(f"--- Top-{k} Kernels by Duration ---")
        for i, (name, data) in enumerate(top_k_dur, 1):
            streams = list(data["streams"]) if data["streams"] else ["N/A"]
            print(
                f"#{i}: {name} "
                f"(Total Duration: {data['duration'] / 1000:.4f} ms, "
                f"Frequency: {int(data['frequency'])}, "
                f"Streams: {streams})"
            )

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
            return

        sorted_recommendations = sorted(fusion_recommendations, key=lambda x: x[1], reverse=True)
        log.info(f"Found {len(sorted_recommendations)} potential fusion candidates:")
        for chain, count, score in sorted_recommendations:
            chain_str = ' -> '.join(chain)
            log.info(f"  - Chain: [{chain_str}] (Found {count} times, Proximity Score: {score:.2f})")

    
    def export_metrics_to_json(
        self,
        model_name: str,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        stream_info: Dict[str, Dict],
        kernel_events: List[Dict],
    ) -> str:
       
        from datetime import datetime
        
        # Compute top kernels
        kernel_stats = self._get_kernel_stats(kernel_events)
        top_k_freq = heapq.nlargest(10, kernel_stats.items(), key=lambda x: x[1]["frequency"])
        top_k_dur = heapq.nlargest(10, kernel_stats.items(), key=lambda x: x[1]["duration"])
        
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
                        "total_duration_ms": data["duration"] / 1000,
                        "streams": sorted(list(data["streams"])) if data["streams"] else []
                    }
                    for i, (name, data) in enumerate(top_k_freq, 1)
                ],
                "by_duration": [
                    {
                        "rank": i,
                        "name": name,
                        "frequency": data["frequency"],
                        "total_duration_ms": data["duration"] / 1000,
                        "streams": sorted(list(data["streams"])) if data["streams"] else []
                    }
                    for i, (name, data) in enumerate(top_k_dur, 1)
                ]
            }
        }
        
        # Save to file
        output_path = self.dump_dir / "metrics_report.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        log.info(f"Metrics exported to: {output_path}")
        return str(output_path)
    
    def _get_kernel_stats(self, kernel_events: List[Dict]) -> Dict[str, Dict]:
        """
        Compute kernel statistics for export.
        
        Returns:
            Dict mapping kernel name to {frequency, duration, streams}
        """
        kernel_data = defaultdict(lambda: {"frequency": 0, "duration": 0.0, "streams": set()})
        
        for kernel in kernel_events:
            kernel_name = kernel["name"]
            kernel_data[kernel_name]["frequency"] += 1
            kernel_data[kernel_name]["duration"] += float(kernel.get("dur", 0))
            if kernel.get("stream") is not None:
                kernel_data[kernel_name]["streams"].add(kernel["stream"])
        
        return dict(kernel_data)
