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
import heapq

# Configure logging
log = logging.getLogger(__name__)


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

    def generate_cpu_specific_ops(self, file: str) -> float:
        """
        Parses the trace file to extract CPU operator metrics.

        Args:
            file: Path to the Chrome trace JSON file.

        Returns:
            The timestamp of the first CPU operation.
        """
        trace_obj = self._load_trace_file(Path(file))
        trace_events = trace_obj.get("traceEvents", [])

        cpu_ops = [
            {
                "name": v.get("name"),
                "begin": v.get("ts"),
                "dur": v.get("dur"),
            }
            for v in trace_events
            if v.get("cat") == "cpu_op"
        ]

        # Save all CPU ops
        with open(self.path / "cpuSpecificOPs.json", "w", encoding="utf-8") as f:
            json.dump({"cpu_ops": cpu_ops}, f, indent=4)

        # Save unique CPU ops
        unique_cpu_ops = sorted(list({op["name"] for op in cpu_ops}))
        with open(self.path / "uniqueCpuOPs.json", "w", encoding="utf-8") as f:
            f.write("\n".join(unique_cpu_ops))

        if not cpu_ops:
            return 0.0

        return float(cpu_ops[0]["begin"])

    def generate_gpu_specific_ops(self, file: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Extracts GPU kernel and CUDA runtime events from trace.
    
        Returns:
            (gpu_kernel_events, cuda_runtime_events)
        """
        trace_obj = self._load_trace_file(Path(file))
        trace_events = trace_obj.get("traceEvents", [])
        
        gpu_ops = []
        runtime_ops = []
        
        for v in trace_events:
            cat = v.get("cat", "")
            args = v.get("args", {})
            
            if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
                event_data = {
                    "Name": v.get("name"),
                    "Type": cat,
                    "Begin": float(v.get("ts", 0)),
                    "Dur": float(v.get("dur", 0)),
                    "Correlation": args.get("correlation"),
                    "Stream": args.get("stream"),
                    "Thread": v.get("tid"),
                }
                
                if cat == "kernel":
                    event_data.update({
                        "Registers per thread": args.get("registers per thread"),
                        "Shared memory": args.get("shared memory"),
                        "Grid": args.get("grid", []),
                        "Block": args.get("block", []),
                    })
                elif cat == "gpu_memcpy":
                    event_data.update({
                        "Bytes": args.get("bytes"),
                        "Mem Bandwidth": args.get("memory bandwidth (GB/s)"),
                    })
                elif cat == "gpu_memset":
                    event_data.update({
                        "Bytes": args.get("bytes"),
                    })
                
                gpu_ops.append(event_data)
            
            elif cat == "cuda_runtime":
                runtime_ops.append({
                    "Name": v.get("name"),
                    "Begin": float(v.get("ts", 0)),
                    "Dur": float(v.get("dur", 0)),
                    "Correlation": args.get("correlation"),
                })
        
        return gpu_ops, runtime_ops

    def generate_dependencies(self, all_gpu_events: Tuple[List[Dict], List[Dict]]) -> Tuple:
        """
        Analyzes dependencies between CUDA runtime calls and kernel launches.

        Args:
            file: Path to the JSON file with GPU metrics.

        Returns:
            A tuple containing:
            - A list of (kernel, cuda_runtime) dependencies.
            - Total kernel execution time.
            - Number of kernels.
            - End timestamp.
            - Total idle time between kernels.
            - Path to the kernels-only JSON file.
        """

        gpu_ops, runtime_ops = all_gpu_events

        # Global timing analysis

        min_start = min(float(evt["Begin"]) for evt in gpu_ops)
        max_end = max(float(evt["Begin"]) + float(evt.get("Dur", 0)) for evt in gpu_ops)
        end_to_end_gpu_time = max_end - min_start

        log.info(f"Analyzing {len(gpu_ops)} GPU events from profiled run.")

        kernel_events = [evt for evt in gpu_ops if evt.get("Type") == "kernel"]

        corr_runtime = {}

        # Dependency Linking
        for rt in runtime_ops:
            corr = rt.get("Correlation")
            if corr is not None:
                corr_runtime[corr] = rt

        dependencies = []
        for kernel in gpu_ops:
            corr = kernel.get("Correlation")
            if corr in corr_runtime:
                dependencies.append((kernel, corr_runtime[corr]))

        # Merge overlapping kernels
        if not kernel_events:
            merged_busy_time = 0.0
        else:
            intervals = sorted([
                (float(k["Begin"]), float(k["Begin"]) + float(k.get("Dur", 0)))
                for k in kernel_events
            ])
            merged = [intervals[0]]
            for current_start, current_end in intervals[1:]:
                last_start, last_end = merged[-1]
                if current_start < last_end:
                    merged[-1] = (last_start, max(last_end, current_end))
                else:
                    merged.append((current_start, current_end))
            merged_busy_time = sum(end - start for start, end in merged)
        
        gpu_idle_time = max(0.0, end_to_end_gpu_time - merged_busy_time)

        # Per-Stream Analysis
        stream_info = defaultdict(lambda: {
            "ops": [], "total_kernel_exec_time": 0.0, "merged_busy_time": 0.0,
            "op_count": 0, "kernel_count": 0
        })
        
        for op in gpu_ops:
            stream_id = op.get("Stream", "unknown_stream")
            stream_info[stream_id]["ops"].append(op)
        
        for stream_id, data in stream_info.items():
            ops_on_stream = sorted(data["ops"], key=lambda x: float(x.get("Begin", 0)))
            stream_info[stream_id]["ops"] = ops_on_stream
            stream_info[stream_id]["op_count"] = len(ops_on_stream)
            
            stream_kernels = [op for op in ops_on_stream if op.get("Type") == "kernel"]
            stream_info[stream_id]["kernel_count"] = len(stream_kernels)
            stream_info[stream_id]["total_kernel_exec_time"] = sum(
                float(k.get("Dur", 0)) for k in stream_kernels
            )
            
            if stream_kernels:
                stream_intervals = sorted([
                    (float(k["Begin"]), float(k["Begin"]) + float(k.get("Dur", 0)))
                    for k in stream_kernels
                ])
                s_merged = [stream_intervals[0]]
                for s_start, s_end in stream_intervals[1:]:
                    sl_start, sl_end = s_merged[-1]
                    if s_start < sl_end:
                        s_merged[-1] = (sl_start, max(sl_end, s_end))
                    else:
                        s_merged.append((s_start, s_end))
                stream_info[stream_id]["merged_busy_time"] = sum(
                    end - start for start, end in s_merged
                )
        
        # Final Aggregate Metrics
        total_kernel_exec_time = sum(float(k.get("Dur", 0)) for k in kernel_events)
        num_kernels = len(kernel_events)
        num_ops = len(gpu_ops)
        
        return (
            dependencies,
            total_kernel_exec_time,
            num_kernels,
            num_ops,
            end_to_end_gpu_time,
            gpu_idle_time,
            merged_busy_time,
            kernel_events,
            dict(stream_info)
        )



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
            runtime_end = runtime["Begin"] + runtime.get("Dur", 0)
            kernel_start = kernel["Begin"]

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
            name = k["Name"]
            kernel_stats[name]["total_dur"] += k.get("Dur", 0)
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
            logging.warning("No kernel events to analyze.")
            return
        
        kernel_data = defaultdict(lambda: {"frequency": 0, "duration": 0.0, "type": None, "streams": set()})
        
        for kernel in kernel_events:
            kernel_name = kernel["Name"]
            kernel_data[kernel_name]["frequency"] += 1
            kernel_data[kernel_name]["duration"] += float(kernel.get("Dur", 0))
            kernel_data[kernel_name]["type"] = kernel.get("Type")
            if kernel.get("Stream") is not None:
                kernel_data[kernel_name]["streams"].add(kernel["Stream"])
        
        # Top k by frequency
        top_k_freq = heapq.nlargest(k, kernel_data.items(), key=lambda item: item[1]["frequency"])
        
        # Top k by duration
        top_k_dur = heapq.nlargest(k, kernel_data.items(), key=lambda item: item[1]["duration"])
        
        logging.info(f"--- Top-{k} Kernels by Frequency ---")
        for i, (name, data) in enumerate(top_k_freq, 1):
            streams = list(data["streams"]) if data["streams"] else ["N/A"]
            logging.info(
                f"#{i}: {name} "
                f"(Frequency: {int(data['frequency'])}, "
                f"Total Duration: {data['duration'] / 1000:.4f} ms, "
                f"Streams: {streams})"
            )
        
        logging.info(f"--- Top-{k} Kernels by Duration ---")
        for i, (name, data) in enumerate(top_k_dur, 1):
            streams = list(data["streams"]) if data["streams"] else ["N/A"]
            logging.info(
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
            if 'cudaStreamSynchronize' in runtime["Name"]:
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
                current_chain.append(kernel["Name"])
                if len(current_chain) == exact_length:
                    chain_tuple = tuple(current_chain)
                    unique_fusion_candidates.add(chain_tuple)

        # Calculate proximity scores
        fusion_recommendations = []
        kernel_freq: DefaultDict[str, int] = defaultdict(int)
        for kernel, _ in dependence:
            kernel_freq[kernel["Name"]] += 1
            
        for chain in unique_fusion_candidates:
            # Count occurrences of this exact chain
            count = 0
            for segment in all_segments:
                segment_kernels = [k["Name"] for k, r in segment]
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
                    "busy_time_ms": data["merged_busy_time"] / 1000,
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
            kernel_name = kernel["Name"]
            kernel_data[kernel_name]["frequency"] += 1
            kernel_data[kernel_name]["duration"] += float(kernel.get("Dur", 0))
            if kernel.get("Stream") is not None:
                kernel_data[kernel_name]["streams"].add(kernel["Stream"])
        
        return dict(kernel_data)
