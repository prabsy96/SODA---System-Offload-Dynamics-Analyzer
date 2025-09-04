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

    def generateCPUSpecificOPs(self, file: str) -> float:
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

    def generateGPUSpecificOPs(self, file: str) -> str:
        """
        Parses the trace file to extract GPU runtime and kernel metrics.

        Args:
            file: Path to the Chrome trace JSON file.

        Returns:
            The path to the generated JSON file containing GPU metrics.
        """
        trace_obj = self._load_trace_file(Path(file))
        trace_events = trace_obj.get("traceEvents", [])
        
        gpu_events = []
        unique_gpu_kernels: Set[str] = set()
        
        for v in trace_events:
            if v.get("cat") in ('cuda_runtime', 'gpu_memcpy', 'gpu_memset', 'kernel'):
                unique_gpu_kernels.add(v["name"])
                event_data = {
                    "Name": v["name"],
                    "Type": v["cat"],
                    "Begin": v["ts"],
                    "Dur": v["dur"],
                    "Correlation": v["args"].get("correlation"),
                }
                # Add specific fields based on type
                if v["cat"] == "kernel":
                    event_data.update({
                        "Registers per thread": v["args"].get("registers per thread"),
                        "Shared memory": v["args"].get("shared memory"),
                        "Blocks per SM": v["args"].get("blocks per SM"),
                        "Grid": v["args"].get("grid"),
                        "Block": v["args"].get("block"),
                    })
                elif v["cat"] == "gpu_memcpy":
                    event_data.update({
                        "Bytes": v["args"].get("bytes"),
                        "Mem Bandwidth": v["args"].get("memory bandwidth (GB/s)"),
                    })
                gpu_events.append(event_data)

        output_file = self.path / "CudaRuntimeAndKernel.json"
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump({"kernels": gpu_events}, f, ensure_ascii=False, indent=4)

        with open(self.path / "uniqueGPUKernels.json", "w", encoding='utf-8') as f:
            f.write("\n".join(sorted(list(unique_gpu_kernels))))
            
        return str(output_file)

    def GenDependency(self, file: str) -> Tuple[List, float, int, float, float, str]:
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
        trace_obj = self._load_trace_file(Path(file))
        all_gpu_events = trace_obj.get("kernels", [])

        cuda_runtime_list = [evt for evt in all_gpu_events if evt["Type"] == "cuda_runtime"]
        kernel_list = [evt for evt in all_gpu_events if evt["Type"] != "cuda_runtime"]
        
        # Correlate kernels to their launching runtime calls
        correlation_map = {rt["Correlation"]: rt for rt in cuda_runtime_list}
        dependencies = []
        for kernel in kernel_list:
            if kernel["Correlation"] in correlation_map:
                dependencies.append((kernel, correlation_map[kernel["Correlation"]]))

        exec_time = sum(k["Dur"] for k in kernel_list)
        num_kernels = len(kernel_list)
        
        # Calculate idle time
        idle_time = 0.0
        if num_kernels > 1:
            sorted_kernels = sorted(kernel_list, key=lambda x: x["Begin"])
            for i in range(len(sorted_kernels) - 1):
                end_of_current = sorted_kernels[i]["Begin"] + sorted_kernels[i]["Dur"]
                start_of_next = sorted_kernels[i+1]["Begin"]
                idle = start_of_next - end_of_current
                if idle > 0:
                    idle_time += idle
        
        end_time = 0.0
        if cuda_runtime_list:
            last_runtime = max(cuda_runtime_list, key=lambda x: x["Begin"] + x["Dur"])
            end_time = last_runtime["Begin"] + last_runtime["Dur"]

        kernel_only_file = self.path / "OnlyGPUKernels.json"
        with open(kernel_only_file, "w") as f:
            json.dump(kernel_list, f, indent=4)

        return dependencies, exec_time, num_kernels, end_time, idle_time, str(kernel_only_file)

    def LaunchTax(self, dependence: List[Tuple], kernel_count: int) -> float:
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
            
        launch_overheads = [abs(kernel["Begin"] - runtime["Begin"]) for kernel, runtime in dependence]
        total_launch_overhead = sum(launch_overheads)

        return total_launch_overhead
    
    def OperationalIntensity(self, file: str):
        """
        Calculates and saves the operational intensity (average kernel duration) for each unique kernel.
        
        Args:
            file: Path to the kernels-only JSON file.
        """
        kernel_events = self._load_trace_file(Path(file))
        
        kernel_stats: DefaultDict[str, Dict[str, Any]] = defaultdict(lambda: {"dur": 0.0, "count": 0})
        
        for event in kernel_events:
            name = event["Name"]
            kernel_stats[name]["dur"] += event["Dur"]
            kernel_stats[name]["count"] += 1
            
        for name, data in kernel_stats.items():
            kernel_stats[name]["AKD"] = data["dur"] / data["count"] if data["count"] > 0 else 0.0
            
        # Sort by Average Kernel Duration (AKD)
        descending_stats = sorted(kernel_stats.items(), key=lambda item: item[1]["AKD"], reverse=True)
        
        output_file = self.path / "OperationalIntensity.json"
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(descending_stats, f, indent=4)

    def topKkernels(self, dependence: List[Tuple], k: int = 3):
        """
        Identifies the top-k most frequent and time-consuming kernels.

        Args:
            dependence: List of (kernel, runtime) dependency tuples.
            k: The number of top kernels to report.
        """
        kernel_data: DefaultDict[str, Dict[str, float]] = defaultdict(lambda: {"frequency": 0, "duration": 0.0})

        for kernel, _ in dependence:
            kernel_name = kernel["Name"]
            kernel_data[kernel_name]["frequency"] += 1
            kernel_data[kernel_name]["duration"] += float(kernel["Dur"])

        # Top k by frequency
        top_k_freq = heapq.nlargest(k, kernel_data.items(), key=lambda item: item[1]["frequency"])
        
        log.info(f"--- Top-{k} Kernels by Frequency ---")
        for i, (name, data) in enumerate(top_k_freq, 1):
            log.info(f"#{i}: {name} (Frequency: {int(data['frequency'])}, Total Duration: {data['duration']/1000:.4f} ms)")

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