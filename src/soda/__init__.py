"""
SODA: System Offload Dynamics Analyzer

"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
import traceback
import numpy as np
import torch
import transformers
from collections import defaultdict, deque
from pathlib import Path
from torch.profiler import ProfilerActivity, profile
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

# for fp8 e4m3 format support
try:
    from transformers.utils.quantization_config import FP8Config
    FP8_CONFIG_AVAILABLE = True
except ImportError:
    FP8Config = None
    FP8_CONFIG_AVAILABLE = False

# Import utilities and microbenchmark pipeline components.
from soda.common import utils
from soda.microbench.microbench import SodaMicrobench

# Global logger reference
LOGGER = logging.getLogger("soda")

# Public API
__all__ = ['ModelTracer', 'SodaAnalyzer', 'SodaLogger', 'LOGGER']


def _resolve_energy_measure_window(num_runs: int, requested_runs: int) -> Tuple[int, int]:
    """Return (start_index, window_size) for trailing-run energy measurement."""
    total_runs = max(1, int(num_runs))
    window_size = max(1, int(requested_runs))
    window_size = min(total_runs, window_size)
    return total_runs - window_size, window_size


def _build_inference_energy_measurement(
    delta_mj_total: float,
    duration_s_total: float,
    measurement_runs: int,
) -> Optional[Dict[str, Any]]:
    """Build per-inference energy measurement dict from a multi-run window.

    The energy counter and wall-clock duration can span multiple profiled runs
    for stability. This helper normalizes totals to per-inference values while
    preserving total-window metadata for diagnostics.
    """
    if duration_s_total <= 0.001 or delta_mj_total <= 0.0:
        return None

    runs = max(1, int(measurement_runs))
    per_inf_energy_mj = delta_mj_total / runs
    per_inf_duration_s = duration_s_total / runs
    power_w = (delta_mj_total * 1e-3) / duration_s_total

    return {
        "power_w": round(power_w, 3),
        "energy_mj": round(per_inf_energy_mj, 3),
        "duration_s": round(per_inf_duration_s, 6),
        "method": "energy_counter",
        "measurement_runs": runs,
        "total_energy_mj": round(delta_mj_total, 3),
        "total_duration_s": round(duration_s_total, 6),
    }

class SodaAnalyzer:
    """
    Handles model tracing, profile data parsing, and metric generation.
    """

    def __init__(self, tracer: 'ModelTracer', args: argparse.Namespace):
        """
        Initializes the profiler.

        Sets up the profiler and derives name, file, and path from parsed arguments.

        Args:
            tracer: The ModelTracer class instance (contains model, tokenizer, is_decoder).
            args: Parsed and validated command-line arguments.
            log_console: If True, write logs to console/stdout.
            log_file: If True, write logs to file.
        """
        self.args = args
        self.tracer = tracer
        
        # Use output paths from tracer
        self.experiment_name = tracer.experiment_name
        self.output_dir = tracer.output_dir
        
        self.trace_file = tracer.trace_file
        self.report_file = self.output_dir / "report.json"
        
        self.trace = tracer.trace_data
        self.events = tracer.events
        self.sequences = tracer.sequences
        self.results = None
    
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
            - sequences: Event sequences
            - avg_kernel_dur: Average kernel duration results
        """
        print("=== Analyzing Trace Data ===")
        print(f"Analyzing {len(self.sequences)} event sequences")
        sequences = utils.calculate_sequence_metrics(list(self.sequences), metrics=["T_launch", "T_dispatch", "T_Py"])
        
        # Analyze per-stream metrics
        stream_info = utils.analyze_per_stream(self.events)
        
        # General metrics
        trace_calculated_inference_time = utils.calculate_total_inference_time(self.trace)
        inference_time = self.tracer.torch_measured_inference_time_us
        
        # GPU metrics — single pass, per-device correct (Phase B + Fix B)
        total_gpu_time_span, true_gpu_busy_time, gpu_utilization, per_device_gpu = \
            utils.calculate_gpu_metrics(self.events)
        
        # Normalize GPU metrics by num_runs so they match per-inference time
        num_runs = getattr(self.tracer, "num_profiled_runs", 1) or 1
        total_gpu_time_span /= num_runs
        true_gpu_busy_time /= num_runs
        # gpu_utilization is a ratio, so it stays the same
        for dev_id in per_device_gpu:
            per_device_gpu[dev_id]["span_us"] /= num_runs
            per_device_gpu[dev_id]["busy_us"] /= num_runs
        
        # Kernel metrics (raw values span all num_runs in the trace)
        kernel_exec_time = utils.calculate_kernel_exec_time(self.events)
        total_T_launch = utils.calculate_total_tax(sequences, "T_launch")
        avg_T_launch = utils.calculate_avg_tax(sequences, "T_launch")
        total_T_dispatch = utils.calculate_total_tax(sequences, "T_dispatch")
        avg_T_dispatch = utils.calculate_avg_tax(sequences, "T_dispatch")
        total_T_Py = utils.calculate_total_tax(sequences, "T_Py")
        avg_T_Py = utils.calculate_avg_tax(sequences, "T_Py")
        avg_kernel_dur, top_k_kernels = utils.get_kernel_stats(self.events, k=3)  # Phase C: single pass

        # Normalize aggregate kernel metrics to per-inference values.
        # The trace contains all num_runs iterations; totals must be divided
        # so they represent a single inference pass (matching inference_time).
        # Per-kernel averages (avg_*_tax, avg_kernel_exec_time, TKLQT avg/min/max)
        # are already correct because both numerator and denominator scale equally.
        kernel_exec_time["total"] /= num_runs
        total_T_launch /= num_runs
        total_T_dispatch /= num_runs
        total_T_Py /= num_runs
        num_total_kernels = len(self.events["gpu"]["kernels"]) // num_runs

        # Warn about negative aggregate taxes (deep-queue GPU artifact, e.g. H100 bs=1).
        # Raw values are preserved as-is; use --taxbreak isolation replay for accurate KT.
        if total_T_launch < 0:
            print(
                f"Warning: total_T_launch={total_T_launch:.2f} µs is negative "
                f"(deep-queue GPU artifact). Values in report.json reflect raw "
                f"measurements; use --taxbreak for artifact-free per-kernel KT.",
                file=__import__('sys').stderr,
            )
        if total_T_dispatch < 0:
            print(
                f"Warning: total_T_dispatch={total_T_dispatch:.2f} µs is negative "
                f"(profiler ordering artifact). Use --taxbreak for accurate ΔCT.",
                file=__import__('sys').stderr,
            )

        # Normalize top-k frequency and total duration to per-inference
        for _bucket in ("by_frequency", "by_duration"):
            for _name, _data in top_k_kernels.get(_bucket, []):
                _data["frequency"] = int(round(_data["frequency"] / num_runs))
                _data["duration"] /= num_runs

        # Fusion analysis
        fusion_results = None
        if self.args.fusion:
            print("=== Kernel Fusion Analysis ===")
            fusion_results = {}
            for f in self.args.fusion:
                fusion_results[f] = utils.analyze_kernel_fusion_candidates(
                    sequences,
                    f,
                    self.args.prox_score,
                    logger=None
                )

        # TKLQT — normalize total and count to per-inference
        tklqt_metrics = utils.calculate_tklqt(sequences)
        if tklqt_metrics.get("count", 0) > 0:
            tklqt_metrics["total"] /= num_runs
            tklqt_metrics["count"] = int(round(tklqt_metrics["count"] / num_runs))

        # Inference throughput: TPOT vs TTFT labeling
        output_tokens = getattr(self.args, "max_new_tokens", 1) or 1
        is_ttft_run = (output_tokens == 1)
        inference_s = inference_time / 1_000_000.0
        tpot_ms = (inference_time / 1000.0) / output_tokens if output_tokens > 0 else None
        throughput_tok_s = (self.args.batch_size * output_tokens) / inference_s if inference_s > 0 else None
        interactivity_tok_s = output_tokens / inference_s if inference_s > 0 else None

        # Kernel fragmentation (MoE diagnostic) — normalize to per-inference
        fragmentation_metrics = utils.compute_kernel_fragmentation(self.events, output_tokens)
        fragmentation_metrics["total_kernel_launches"] = int(round(
            fragmentation_metrics["total_kernel_launches"] / num_runs
        ))
        if output_tokens > 0:
            fragmentation_metrics["kernels_per_output_token"] = round(
                fragmentation_metrics["total_kernel_launches"] / output_tokens, 2
            )
        # kernel_diversity_ratio uses unique count (unchanged) and normalized total
        n_frag = fragmentation_metrics["total_kernel_launches"]
        fragmentation_metrics["kernel_diversity_ratio"] = round(
            fragmentation_metrics["unique_kernel_count"] / n_frag, 4
        ) if n_frag > 0 else 0.0

        # Memory metrics from tracer + GPU memory transfer events
        # Normalize memcpy/memset counts and time to per-inference
        memory_metrics = dict(self.tracer.memory_metrics) if self.tracer.memory_metrics else {}
        gpu_mem_events = self.events["gpu"]["memory"]
        memory_metrics["num_memcpy_memset_ops"] = len(gpu_mem_events) // num_runs
        memory_metrics["total_memcpy_memset_time_ms"] = round(
            sum(e["dur"] for e in gpu_mem_events) / 1000.0 / num_runs, 4
        )

        # Carbon footprint estimation
        carbon_metrics = None
        try:
            from soda.carbon import compute_carbon_footprint, get_gpu_tdp
            device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
            gpu_tdp = get_gpu_tdp(device_name)
            if gpu_tdp is not None and inference_time > 0:
                carbon_metrics = compute_carbon_footprint(
                    inference_time_s=inference_time / 1_000_000.0,
                    gpu_tdp_w=gpu_tdp,
                    gpu_util_pct=gpu_utilization if gpu_utilization is not None else 50.0,
                    batch_size=getattr(self.args, "batch_size", 1),
                    num_tokens=output_tokens,
                    carbon_intensity_g_kwh=getattr(self.args, "carbon_intensity", 400.0),
                    pue=getattr(self.args, "pue", 1.1),
                )
        except Exception:
            pass

        # Power profile from NVML sampler (populated if --power-sample was set)
        power_profile = getattr(self.tracer, "_power_results", {})
        if power_profile.get("available"):
            n = power_profile.get("sample_count", 0)
            interval = power_profile.get("interval_ms", 50)
            backend = power_profile.get("backend", "unknown")
            if n == 0:
                print(
                    f"Warning: --power-sample: no power readings collected (backend={backend}, "
                    f"interval={interval} ms). The inference may be shorter than the sampling "
                    f"interval. Try --power-sample-interval {max(10, interval // 2)} to halve "
                    "the interval, or check that the backend has access to the GPU."
                )
            elif n < 3:
                print(
                    f"Warning: --power-sample: only {n} sample(s) collected (backend={backend}, "
                    f"interval={interval} ms). Power estimate may be inaccurate. "
                    f"Try --power-sample-interval {max(10, interval // 2)} for more samples."
                )
            if n > 0 and carbon_metrics is not None:
                # Augment carbon dict with measured (not TDP-estimated) power values
                carbon_metrics["measured_power_w"] = round(power_profile["mean_power_w"], 2)
                carbon_metrics["peak_power_w"] = round(power_profile["peak_power_w"], 2)
                carbon_metrics["power_sample_count"] = n
                carbon_metrics["power_backend"] = backend
                # Override the TDP-estimated value with the measured one for accuracy
                carbon_metrics["estimated_power_w"] = round(power_profile["mean_power_w"], 2)

        # Build metrics dictionary
        metrics = {
            # Inference time 
            "inference_time_ms": utils.us_to_ms(inference_time),
            # Inference time breakdown
            "inference_time_breakdown": {
                "torch_measured_inference_time_ms": utils.us_to_ms(self.tracer.torch_measured_inference_time_us),
                #"trace_calculated_inference_time_ms": utils.us_to_ms(trace_calculated_inference_time),
                "profiler_overhead_ms": utils.us_to_ms(trace_calculated_inference_time / num_runs - self.tracer.torch_measured_inference_time_us),
            },
            "active_streams": len(stream_info),

            # GPU metrics (avg per GPU for multi-GPU runs; Fix B)
            "total_gpu_time_span_ms": utils.us_to_ms(total_gpu_time_span),
            "gpu_busy_time_ms": utils.us_to_ms(true_gpu_busy_time),
            "true_gpu_busy_time_us": true_gpu_busy_time,  # µs version for summarizer
            "gpu_idle_time_ms": utils.us_to_ms(max(0.0, total_gpu_time_span - true_gpu_busy_time)),
            "gpu_utilization_percent": gpu_utilization,
            "per_device_gpu_metrics": per_device_gpu,   # {dev_id: {span_us, busy_us, utilization_pct}}

            # Kernel metrics
            "total_kernel_exec_time_ms": utils.us_to_ms(kernel_exec_time["total"]),
            "num_total_kernels": num_total_kernels,
            "avg_kernel_exec_time_ms": utils.us_to_ms(kernel_exec_time["avg"]),
            "total_T_dispatch_ms": utils.us_to_ms(total_T_dispatch),
            "avg_T_dispatch_ms": utils.us_to_ms(avg_T_dispatch),
            "total_T_launch_ms": utils.us_to_ms(total_T_launch),
            "avg_T_launch_ms": utils.us_to_ms(avg_T_launch),
            "total_T_Py_ms": utils.us_to_ms(total_T_Py),
            "avg_T_Py_ms": utils.us_to_ms(avg_T_Py),

            # TKLQT (HDBI requires accurate i_lib decomposition from --taxbreak)
            "tklqt": tklqt_metrics,

            # Inference throughput / latency
            "inference_throughput": {
                "tpot_ms": tpot_ms,
                "output_tokens": output_tokens,
                "is_ttft_run": is_ttft_run,
                "throughput_tok_s": throughput_tok_s,
                "interactivity_tok_s": interactivity_tok_s,
            },

            # Kernel fragmentation (MoE diagnostic)
            "kernel_fragmentation": fragmentation_metrics,

            # Memory profiling
            "memory_metrics": memory_metrics,

            # Carbon footprint
            "carbon_footprint": carbon_metrics,

            # Power profile from NVML sampler (empty dict if --power-sample not used)
            "power_profile": power_profile,

            # Ground truth inference power from energy counter (always attempted)
            "inference_energy": getattr(self.tracer, "_inference_energy_measurement", {}),
        }
        
        self.results = {
            "metrics": metrics,
            "stream_info": stream_info,
            "top_k_kernels": top_k_kernels,
            "sequences": sequences,
            "avg_kernel_dur": avg_kernel_dur,
            "fusion_results": fusion_results,
        }
        
        return self.results
    
    def report(self) -> None:
        """
        Prints performance metrics, stream analysis, and top-k kernels.
        Uses results stored in self.results from analyze().
        Default: compact layperson summary + summary.md written to output_dir.
        With --verbose: layperson summary followed by full expert tables.
        """
        if self.results is None:
            raise ValueError("No analysis results available. Call analyze() first.")

        # --- Layperson summary (always shown) ---
        from soda.common.summary_report import render_main_analysis
        render_main_analysis(self.results, self.args, self.output_dir)

        # --- Expert output (only with --verbose) ---
        verbose = getattr(self.args, "verbose", False)
        if not verbose:
            return

        metrics = self.results["metrics"]
        stream_info = self.results["stream_info"]
        top_k_kernels = self.results["top_k_kernels"]

        # --- Enhanced Reporting ---
        print("")
        print("=== Performance Metrics ===")
        print(f"\t* Inference runtime (ms): {metrics['inference_time_ms']:.4f}")
        
        # TKLQT Analysis (HDBI requires TaxBreak pipeline for dynamic T_sys)
        tklqt = metrics.get("tklqt", {})
        timing = metrics["inference_time_breakdown"]
        print("")
        print("=== TKLQT Analysis ===")
        if tklqt:
            for k, v in tklqt.items():
                print(f"\t* {k}: {v:.4f}" if isinstance(v, float) else f"\t* {k}: {v}")

        # Timing breakdown
        print(f"\t  - Torch measured inference time (ms): {timing['torch_measured_inference_time_ms']:.4f}")
        print(f"\t  - Profiler overhead (ms): {timing['profiler_overhead_ms']:.4f}")
        
        print("")
        print("=== GPU Metrics ===")
        print(f"\t* Total kernel execution time (ms): {metrics['total_kernel_exec_time_ms']:.4f}")
        print(f"\t* GPU busy time (concurrent-aware) (ms): {metrics['gpu_busy_time_ms']:.4f}")
        print(f"\t* GPU idle time (ms): {metrics['gpu_idle_time_ms']:.4f}")
        print(f"\t* GPU utilization: {metrics['gpu_utilization_percent']:.2f}%")
        print(f"\t* Number of kernels: {metrics['num_total_kernels']}")
        print(f"\t* Active streams: {metrics['active_streams']}")
        
        print("")
        print("=== Taxes ===")
        print(f"\t* ΔFT_py  (T_Py, Python layer) (ms): {metrics['total_T_Py_ms']:.4f}")
        print(f"\t* ΔFT_disp + δCT  (T_dispatch, undifferentiated in standard mode) (ms): {metrics['total_T_dispatch_ms']:.4f}")
        print(f"\t*   (use --taxbreak for per-kernel FT_dispatch / δCT decomposition)")
        print(f"\t* ΔKT  (T_launch + queue latency) (ms): {metrics['total_T_launch_ms']:.4f}")
        if metrics['num_total_kernels'] > 0:
            print(f"\t* Avg. T_Py per kernel (ms): {metrics['avg_T_Py_ms']:.4f}")
            print(f"\t* Avg. T_dispatch per kernel (ms): {metrics['avg_T_dispatch_ms']:.4f}")
            print(f"\t* Avg. T_launch (+ queue latency) per kernel (ms): {metrics['avg_T_launch_ms']:.4f}")
            print(f"\t* Avg. execution time per kernel (ms): {metrics['avg_kernel_exec_time_ms']:.4f}")
        
        # Memory profiling
        mem = metrics.get("memory_metrics", {})
        if mem:
            print("")
            print("=== Memory Profiling ===")
            if "model_memory_mb" in mem:
                print(f"\t* Model memory (allocated): {mem['model_memory_mb']:.2f} MB")
            if "pre_inference_memory_mb" in mem:
                print(f"\t* Pre-inference memory: {mem['pre_inference_memory_mb']:.2f} MB")
            if "peak_memory_allocated_mb" in mem:
                print(f"\t* Peak memory (allocated): {mem['peak_memory_allocated_mb']:.2f} MB")
            if "peak_memory_reserved_mb" in mem:
                print(f"\t* Peak memory (reserved): {mem['peak_memory_reserved_mb']:.2f} MB")
            if "memory_delta_mb" in mem:
                print(f"\t* Inference memory delta: {mem['memory_delta_mb']:.2f} MB")
            if "num_memcpy_memset_ops" in mem:
                print(f"\t* GPU memcpy/memset ops: {mem['num_memcpy_memset_ops']}")
            if "total_memcpy_memset_time_ms" in mem:
                print(f"\t* Total memcpy/memset time (ms): {mem['total_memcpy_memset_time_ms']:.4f}")

        print("")
        # --- Per-Stream Breakdown ---
        print("=== Per-Stream Analysis ===")
        for stream_id, data in stream_info.items():
            print(
                f"\t* Stream {stream_id}: {data['op_count']} ops "
                f"({data['kernel_count']} kernels), "
                f"Busy Time: {utils.us_to_ms(data['true_gpu_busy_time']):.4f} ms"
            )
        
        print("")
        # Top-K kernels 
        if top_k_kernels["by_frequency"]:
            print("=== Top-3 Kernels by Frequency ===")
            for i, (name, data) in enumerate(top_k_kernels["by_frequency"], 1):
                print(
                    f"\t* #{i}: {name} "
                    f"(Frequency: {int(data['frequency'])}, "
                    f"Total Duration: {utils.us_to_ms(data['duration']):.4f} ms)"
                )
            
            print("")
            print("=== Top-3 Kernels by Duration ===")
            for i, (name, data) in enumerate(top_k_kernels["by_duration"], 1):
                print(
                    f"\t* #{i}: {name} "
                    f"(Total Duration: {utils.us_to_ms(data['duration']):.4f} ms, "
                    f"Frequency: {int(data['frequency'])})"
                )
    
    def save(self) -> str:
        """
        Saves analysis results to JSON file.
        """
        if self.results is None:
            raise ValueError("No analysis results available. Call analyze() first.")
        
        metrics = self.results["metrics"]
        stream_info = self.results["stream_info"]
        top_k_kernels = self.results["top_k_kernels"]
        
        # Generate model_name and config from args
        model_name = self.args.model
        
        # Add GPU name(s) to config
        if torch.cuda.is_available():
            n = getattr(self.tracer, "num_gpus", 1)
            if n > 1:
                gpu_name = " + ".join(
                    torch.cuda.get_device_name(i) for i in range(n)
                )
            else:
                gpu_name = torch.cuda.get_device_name(0)
        else:
            gpu_name = "cpu"

        config = {
            "batch_size": self.args.batch_size,
            "seq_len": self.args.seq_len,
            "max_new_tokens": getattr(self.args, "max_new_tokens", None),
            "precision": self.args.precision,
            "compile_type": self.args.compile_type,
            "device": self.args.device,
            "gpu_name": gpu_name,
            "num_gpus": getattr(self.tracer, "num_gpus", 1),
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
                    "busy_time_ms": utils.us_to_ms(data["true_gpu_busy_time"]),
                    "total_kernel_exec_time_ms": utils.us_to_ms(data["total_kernel_exec_time"]),
                }
                for stream_id, data in stream_info.items()
            },
            "top_kernels": {
                "by_frequency": [
                    {
                        "rank": i,
                        "name": name,
                        "frequency": data["frequency"],
                        "total_duration_ms": utils.us_to_ms(data["duration"])
                    }
                    for i, (name, data) in enumerate(top_k_kernels["by_frequency"], 1)
                ],
                "by_duration": [
                    {
                        "rank": i,
                        "name": name,
                        "frequency": data["frequency"],
                        "total_duration_ms": utils.us_to_ms(data["duration"])
                    }
                    for i, (name, data) in enumerate(top_k_kernels["by_duration"], 1)
                ]
            }
        }
        
        # Add fusion results if available
        if "fusion_results" in self.results:
            output["fusion_analysis"] = self.results["fusion_results"]
        
        # Save to file
        utils.save_json(self.report_file, output)
        
        print(f"Metrics exported to: {self.report_file}")
        return str(self.report_file)
    
    def run(self) -> str:
        """
        Runs the complete analysis pipeline: analyze -> report -> save.
        
        Returns:
            Path to the saved report file.
        """
        self.analyze()
        self.report()
        return self.save()

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
        global LOGGER
        self.logger = LOGGER
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


def _compute_kv_cache_bytes(past_key_values) -> int:
    """Sum actual bytes of all K and V tensors in a HuggingFace KV cache.
    Handles DynamicCache/StaticCache (HF>=4.36) and legacy tuple-of-tuples.
    Returns 0 for None or unrecognised types.
    """
    if past_key_values is None:
        return 0
    # DynamicCache / StaticCache (transformers >= 4.36)
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return sum(
            k.element_size() * k.nelement() + v.element_size() * v.nelement()
            for k, v in zip(past_key_values.key_cache, past_key_values.value_cache)
        )
    # Legacy: tuple/list of (key_tensor, value_tensor) per layer
    if isinstance(past_key_values, (tuple, list)):
        total = 0
        for layer in past_key_values:
            if isinstance(layer, (tuple, list)) and len(layer) >= 2:
                k, v = layer[0], layer[1]
                if hasattr(k, "element_size"):
                    total += k.element_size() * k.nelement() + v.element_size() * v.nelement()
        return total
    return 0


class ModelTracer:
    """Handles loading of Hugging Face models with specific configurations."""

    def __init__(self, args: argparse.Namespace):
        """
        Initializes the Model loader.

        Args:
            args: Parsed CLI arguments containing model/configuration settings.
        """
        self.args = args
        self.model_name = args.model
        
        # Detect CUDA availability once and store for use throughout
        self._has_cuda = torch.cuda.is_available()
        
        # If device is explicitly set to cuda but CUDA is not available, fall back to CPU
        requested_device = args.device
        if 'cuda' in requested_device and not self._has_cuda:
            print(f"Warning: CUDA requested ('{requested_device}') but not available. Falling back to CPU.")
            requested_device = 'cpu'
        
        self.device = torch.device(requested_device)

        requested_gpus = max(1, getattr(args, "num_gpus", 1))
        available_gpus = torch.cuda.device_count() if self._has_cuda else 0
        self.num_gpus = min(requested_gpus, max(1, available_gpus))
        if self.num_gpus < requested_gpus:
            print(
                f"Warning: requested {requested_gpus} GPUs but only "
                f"{available_gpus} available. Using {self.num_gpus}."
            )
        if self.num_gpus > 1:
            print(f"Multi-GPU mode: distributing model across {self.num_gpus} GPUs "
                  f"(device_map=\"balanced\").")

        self.compile_type = args.compile_type
        self.is_fp8 = args.precision == "float8_e4m3fn"
        
        self.precision = utils.parse_dtype_to_torch(args.precision)
        self.load_precision = torch.bfloat16 if self.is_fp8 else self.precision
        # Set random seeds
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if self._has_cuda:
            torch.cuda.manual_seed_all(args.seed)

        # Setup deterministic mode for microbench
        if bool(getattr(args, "microbench", False)):
            print("Setting up deterministic mode for microbench")
            utils.setup_deterministic_mode()
        
        # Store run parameters
        self.batch_size = args.batch_size
        self.seq_len = args.seq_len
        self.max_new_tokens = args.max_new_tokens
        
        # Determine if model is decoder or encoder
        encoder_models = ["bert", "roberta"]
        whisper_models = ["whisper"]

        self.is_whisper = any(model.lower() in self.model_name.lower() for model in whisper_models)
        self.is_encoder = any(model.lower() in self.model_name.lower() for model in encoder_models)
        self.is_decoder = not (self.is_encoder or self.is_whisper)

        # Derive experiment/output paths
        self.experiment_name = utils.generate_experiment_name(
            self.model_name,
            self.compile_type,
            args.precision,
            self.batch_size,
            self.seq_len,
            self.max_new_tokens,
            num_gpus=self.num_gpus,
        )

        # Output directory for trace: <output_dir>/<experiment_name>
        self.output_dir = args.output_dir / self.experiment_name
        utils.ensure_dir(self.output_dir)
        
        # Set EXPERIMENT_DIR environment variable for microbench scripts
        os.environ["EXPERIMENT_DIR"] = str(self.output_dir)

        # Trace file: <output_dir>/<experiment_name>/trace.json
        self.trace_file = self.output_dir / "trace.json"
        utils.ensure_dir(self.trace_file.parent)

        # Collect and save env_metadata in experiment directory
        env_metadata = utils.collect_env_metadata()
        env_metadata_file = utils.get_path("ENV_METADATA")
        utils.save_json(env_metadata_file, env_metadata)

        # Objects related to model loading and tracing
        self.model = None
        self.tokenizer = None
        self.model_inputs = None

        # Objects related to trace data collection and processing
        self.trace_data = None
        self.events = None
        self.sequences = None
        self.torch_measured_inference_time_us = None

        # Memory profiling (populated during setup/trace)
        self.memory_metrics = None
        self._model_memory_bytes = 0
        self._model_memory_reserved_bytes = 0
        self._pre_inference_bytes = 0
        self._peak_allocated_bytes = 0
        self._peak_reserved_bytes = 0
        self._kv_cache_bytes = 0

        # Power sampling results (populated during trace if --power-sample is set)
        self._power_results: dict = {}

        # Ground truth inference power: measured via NVML energy counter during
        # the last profiled inference pass.  Always attempted (no CLI flag needed).
        # Dict with keys: power_w, energy_mj, duration_s, method.  Empty if unavailable.
        self._inference_energy_measurement: dict = {}

    def _read_energy_counter_mj(self) -> Optional[float]:
        """Read cumulative GPU energy counter (mJ) via pynvml, or None."""
        try:
            import pynvml  # type: ignore[import]
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            # nvmlDeviceGetTotalEnergyConsumption returns millijoules
            val = float(pynvml.nvmlDeviceGetTotalEnergyConsumption(h))
            return val
        except Exception as e:
            print(f"Warning: energy counter read failed: {e}", file=sys.stderr)
            return None

    def _make_power_sampler(self):
        """Return a power sampler (NVMLPowerSampler or no-op) based on CLI args."""
        from soda.power_sampler import make_power_sampler
        enabled = getattr(self.args, "power_sample", False)
        interval_ms = getattr(self.args, "power_sample_interval", 50)
        gpu_ids = list(range(self.num_gpus))
        return make_power_sampler(gpu_ids=gpu_ids, interval_ms=interval_ms, enabled=enabled)

    def _capture_power_results(
        self,
        sampler,
        profile_start_s: Optional[float],
        profile_end_s: Optional[float],
        sampling_start_s: Optional[float],
        sampling_end_s: Optional[float],
    ) -> None:
        """Capture aggregated and raw power sampling data for Stage-2 reuse."""
        results = sampler.get_results()

        raw_samples = getattr(sampler, "_samples", None)
        if isinstance(raw_samples, dict):
            serialized = {}
            for gpu_id, readings in raw_samples.items():
                gpu_key = str(gpu_id)
                gpu_readings = []
                if isinstance(readings, list):
                    for pair in readings:
                        if not isinstance(pair, tuple) or len(pair) != 2:
                            continue
                        try:
                            t_ms = float(pair[0])
                            watts = float(pair[1])
                        except Exception:
                            continue
                        gpu_readings.append([t_ms, watts])
                serialized[gpu_key] = gpu_readings
            results["samples_ms"] = serialized

        if profile_start_s is not None and profile_end_s is not None and profile_end_s > profile_start_s:
            results["profile_window_s"] = {
                "start": float(profile_start_s),
                "end": float(profile_end_s),
            }
        if sampling_start_s is not None and sampling_end_s is not None and sampling_end_s > sampling_start_s:
            results["sampling_window_s"] = {
                "start": float(sampling_start_s),
                "end": float(sampling_end_s),
            }

        self._power_results = results

    def _get_profiler_activities(self) -> List:
        """Returns the appropriate profiler activities based on device availability."""
        activities = [ProfilerActivity.CPU]
        if self._has_cuda:
            activities.append(ProfilerActivity.CUDA)
            print("Profiling: CPU + CUDA")
        else:
            print("Profiling: CPU-only (no CUDA device detected)")
        return activities
    
    def _sync_device(self) -> None:
        """Synchronize all active CUDA devices."""
        if self._has_cuda:
            for i in range(self.num_gpus):
                torch.cuda.synchronize(i)

    def _reset_memory_stats(self) -> None:
        """Reset peak memory stats on all active GPUs."""
        if self._has_cuda:
            for i in range(self.num_gpus):
                torch.cuda.reset_peak_memory_stats(i)

    def _capture_pre_inference_memory(self) -> None:
        """Capture memory baseline after warmup, before profiled inference."""
        if self._has_cuda:
            for i in range(self.num_gpus):
                torch.cuda.synchronize(i)
            self._pre_inference_bytes = sum(
                torch.cuda.memory_allocated(i) for i in range(self.num_gpus)
            )

    def _capture_peak_memory(self) -> None:
        """Capture peak memory after profiled inference completes."""
        if self._has_cuda:
            for i in range(self.num_gpus):
                torch.cuda.synchronize(i)
            self._peak_allocated_bytes = sum(
                torch.cuda.max_memory_allocated(i) for i in range(self.num_gpus)
            )
            self._peak_reserved_bytes = sum(
                torch.cuda.max_memory_reserved(i) for i in range(self.num_gpus)
            )

    def setup(self) -> None:
        """
        Loads the model/tokenizer and prepares synthetic inputs.
        """

        # Load model and tokenizer
        if self.is_whisper:
            self.model, self.tokenizer = self.load_whisper()
            # FIX: Generate audio inputs for Whisper, not text inputs
            print(f"Generating synthetic audio input: batch_size={self.batch_size}, seq_len={self.seq_len}")
            self.model_inputs = self.generate_audio_inputs()
        elif self.is_decoder:
            self.model, self.tokenizer = self.load_decoder()
            # print(f"Generating synthetic input: batch_size={self.batch_size}, seq_len={self.seq_len}")
            self.model_inputs = utils.generate_synthetic_inputs(
                self.tokenizer, self.device, self.batch_size, self.seq_len, model_config=self.model.config
            )
        else:
            self.model, self.tokenizer = self.load_encoder()
            # print(f"Generating synthetic input: batch_size={self.batch_size}, seq_len={self.seq_len}")
            self.model_inputs = utils.generate_synthetic_inputs(
                self.tokenizer, self.device, self.batch_size, self.seq_len, model_config=self.model.config
            )

        # Capture model memory footprint after loading model + inputs
        if self._has_cuda:
            for i in range(self.num_gpus):
                torch.cuda.synchronize(i)
            self._model_memory_bytes = sum(
                torch.cuda.memory_allocated(i) for i in range(self.num_gpus)
            )
            self._model_memory_reserved_bytes = sum(
                torch.cuda.memory_reserved(i) for i in range(self.num_gpus)
            )

    def get_kwargs(self) -> Dict[str, Any]:
        """Returns common kwargs for model loading."""
        if self.num_gpus > 1:
            device_map = "balanced"
        elif self._has_cuda:
            device_map = self.device
        else:
            device_map = "cpu"
        return {
            "dtype": self.load_precision,
            "device_map": device_map,
            "trust_remote_code": True,
        }

    def _convert_to_fp8(self, model: transformers.PreTrainedModel) -> transformers.PreTrainedModel:
        """
        Convert Linear layer weights to FP8 E4M3 for inference if quantization config is unavailable.
        """
        print("Converting linear layer weights to float8_e4m3fn...")

        converted_count = 0
        for _, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                with torch.no_grad():
                    # Clamp to avoid overflow in FP8 E4M3 range before conversion
                    clamped = module.weight.data.clamp(-448.0, 448.0)
                    module.weight.data = clamped.to(torch.float8_e4m3fn)
                    converted_count += 1

        print(f"Converted {converted_count} linear layers to FP8 E4M3.")
        return model

    def _replace_linear_with_te(self, module: torch.nn.Module) -> None:
        """
        Recursively replace nn.Linear with transformer_engine.pytorch.Linear.
        """
        import transformer_engine.pytorch as te
        import gc
        
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Linear):
                has_bias = child.bias is not None
                
                # Create TE layer
                # We keep params in BF16 (master weights) and let fp8_autocast handle the cast
                te_linear = te.Linear(
                    child.in_features,
                    child.out_features,
                    bias=has_bias,
                    params_dtype=child.weight.dtype
                )
                
                # Copy weights
                with torch.no_grad():
                    te_linear.weight.copy_(child.weight)
                    if has_bias:
                        te_linear.bias.copy_(child.bias)
                
                # Move to correct device
                te_linear.to(child.weight.device)
                
                # Replace the layer
                setattr(module, name, te_linear)
                
                # OPTIONAL: Explicitly delete original child to free memory immediately
                del child
            else:
                self._replace_linear_with_te(child)
        
        # Force GC to reclaim memory after replacements
        gc.collect()
        if self._has_cuda:
            torch.cuda.empty_cache()

            
    def _load_model_fp8(self, model_name: str) -> Any:
        """Load model with FP8 quantization using transformer-engine or fallback."""
        print(f"DEBUG: Entering _load_model_fp8 for {model_name}")
        try:
            # Option 1: Use Transformer Engine (recommended for H100)
            import transformer_engine.pytorch as te
            from transformer_engine.common.recipe import DelayedScaling, Format
            from transformers import AutoModelForCausalLM, AutoConfig
            
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            print("DEBUG: Loading model in BFloat16 before TE replacement...")
            
            # Load in bfloat16 first
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            
            # Replace nn.Linear with TE Linear layers to enable FP8 autocast
            LOGGER.info("Replacing nn.Linear with TransformerEngine Linear layers...")
            self._replace_linear_with_te(model)
            
            # Apply FP8 recipe using correct API import
            fp8_recipe = DelayedScaling(
                margin=0,
                fp8_format=Format.E4M3,
                amax_history_len=16,
                amax_compute_algo="max",
            )
            
            LOGGER.info(f"Loaded {model_name} with Transformer Engine FP8")
            # NOTE: Do NOT manually convert to FP8 here. TE handles it via autocast.
            return model, fp8_recipe
            
        except ImportError:
            # Option 2: Fallback to manual FP8 casting (limited support)
            LOGGER.warning("Transformer Engine not available. Using manual FP8 casting.")
            from transformers import AutoModelForCausalLM
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,  # Load as bf16, cast later
                device_map="auto",
                trust_remote_code=True,
            )
            # Convert weights manually
            model = self._convert_to_fp8(model)
            return model, None
        except AttributeError:
             # Option 3: Fallback if TE API is different/older
            LOGGER.warning("Transformer Engine API mismatch. Using manual FP8 casting.")
            from transformers import AutoModelForCausalLM
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            model = self._convert_to_fp8(model)
            return model, None

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
            if self.is_fp8 and "quantization_config" not in kwargs:
                model = self._convert_to_fp8(model)
            model.forward = torch.compile(model.forward, mode="reduce-overhead", backend="inductor")
        elif self.compile_type == "flash-attention":
            kwargs.update({"attn_implementation": "flash_attention_2"})
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                self.model_name, **kwargs
            ).eval()
            if self.is_fp8 and "quantization_config" not in kwargs:
                model = self._convert_to_fp8(model)
        else:  # eager
            kwargs.update({"attn_implementation": "eager"})
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                self.model_name, **kwargs
            ).eval()
            if self.is_fp8 and "quantization_config" not in kwargs:
                model = self._convert_to_fp8(model)
        
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None: 
            pad_token_id = getattr(generation_config, "pad_token_id", None)
            if pad_token_id is None:
                generation_config.pad_token_id = tokenizer.eos_token_id
        
        return model, tokenizer

    def load_decoder(self) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
        """Loads a decoder-only model (e.g., Llama)."""
        # Load tokenizer first to get eos_token_id
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load config and set pad_token_id before model initialization
        config = transformers.AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        if hasattr(config, 'pad_token_id') and config.pad_token_id is None:
            config.pad_token_id = tokenizer.eos_token_id
        
        # FP8 path: Use Transformer Engine if available
        if self.is_fp8:
            model, self.fp8_recipe = self._load_model_fp8(self.model_name)
            # FIX: Ensure model config has pad_token_id after FP8 loading
            if hasattr(model, 'config') and model.config.pad_token_id is None:
                model.config.pad_token_id = tokenizer.pad_token_id
            return model, tokenizer
        
        # Non-FP8 paths (existing code)
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
        
        return model, tokenizer
    
    def load_whisper(self) -> Tuple[transformers.PreTrainedModel, transformers.AutoProcessor]:
        """Loads Whisper encoder-decoder model."""
        processor = transformers.AutoProcessor.from_pretrained(self.model_name)
        
        kwargs = self.get_kwargs()
        if self.compile_type == "torch.compile":
            kwargs.update({"attn_implementation": "sdpa"})
            model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name, **kwargs
            ).eval()
            if self.is_fp8 and "quantization_config" not in kwargs:
                model = self._convert_to_fp8(model)
            model.generate = torch.compile(model.generate, mode="reduce-overhead", backend="inductor")
        elif self.compile_type == "flash-attention":
            kwargs.update({"attn_implementation": "flash_attention_2"})
            model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name, **kwargs
            ).eval()
            if self.is_fp8 and "quantization_config" not in kwargs:
                model = self._convert_to_fp8(model)
        else:  # eager
            kwargs.update({"attn_implementation": "eager"})
            model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name, **kwargs
            ).eval()
            if self.is_fp8 and "quantization_config" not in kwargs:
                model = self._convert_to_fp8(model)
        
        return model, processor

    def generate_audio_inputs(self) -> Dict[str, torch.Tensor]:
        """
        Generates synthetic audio features for Whisper.
        Handles both raw samples (large seq_len) and frames (small seq_len).
        """
        # Determine correct mel bins from model config (v3=128, v1/v2=80)
        num_mel_bins = getattr(self.model.config, "num_mel_bins", 80)
        WHISPER_EXPECTED_FRAMES = 3000

        # Case 1: seq_len looks like sample count (e.g. 480000 for 30s)
        if self.seq_len > 10000:
            # Generate raw audio waveform
            audio = torch.randn(self.seq_len, dtype=torch.float32)
            
            # Use processor to extract features (handles mel conversion & padding)
            inputs = self.tokenizer(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            )
            features = inputs.input_features # [1, bins, frames]
            
            # Expand to batch size
            if self.batch_size > 1:
                features = features.repeat(self.batch_size, 1, 1)
                
        # Case 2: seq_len looks like frame count (e.g. 3000)
        else:
            features = torch.randn(
                self.batch_size, 
                num_mel_bins, 
                self.seq_len,
                dtype=self.precision,
                device=self.device
            )
            
            # Pad or truncate to expected 3000 frames
            if self.seq_len < WHISPER_EXPECTED_FRAMES:
                features = torch.nn.functional.pad(
                    features, (0, WHISPER_EXPECTED_FRAMES - self.seq_len)
                )
            elif self.seq_len > WHISPER_EXPECTED_FRAMES:
                features = features[:, :, :WHISPER_EXPECTED_FRAMES]

        # Create attention mask (all 1s since we have valid audio)
        # Shape matches the frames dimension of features: [batch_size, frames]
        # Note: Whisper attention mask is usually for the encoder inputs
        attention_mask = torch.ones(
            features.shape[0], 
            features.shape[2], 
            dtype=torch.long, 
            device=self.device
        )

        return {
            "input_features": features.to(self.device).to(self.precision),
            "attention_mask": attention_mask
        }

    def run(self) -> None:
        """
        Complete tracing pipeline
        """
        self.setup()
        self.trace()
        self.process()
    
    def trace(self) -> None:
        """
        Profiles the forward pass of the model (encoder or decoder).
        """
        if self.is_whisper:
            self.trace_forward_pass_for_whisper()
        elif self.is_decoder:
            self.trace_forward_pass_for_decoder()
        else:
            self.trace_forward_pass_for_encoder()
        
        # Load trace data into memory immediately
        self.trace_data = utils.load_json(self.trace_file)
        print(f"Chrome trace file generated at: {self.trace_file}")

        # Build memory metrics from snapshots captured during setup/trace
        if self._has_cuda:
            self.memory_metrics = {
                "model_memory_mb": round(self._model_memory_bytes / (1024**2), 2),
                "model_memory_reserved_mb": round(self._model_memory_reserved_bytes / (1024**2), 2),
                "pre_inference_memory_mb": round(self._pre_inference_bytes / (1024**2), 2),
                "peak_memory_allocated_mb": round(self._peak_allocated_bytes / (1024**2), 2),
                "peak_memory_reserved_mb": round(self._peak_reserved_bytes / (1024**2), 2),
                "memory_delta_mb": round((self._peak_allocated_bytes - self._pre_inference_bytes) / (1024**2), 2),
                "kv_cache_mb": round(self._kv_cache_bytes / (1024**2), 2),
            }

    def process(self) -> None:
        """
        Parses the trace to collect events and build linked event sequences.
        """
        self.events = utils.collect_events(self.trace_data)
        self.sequences = utils.link_sequences(self.events)
        print(f"Collected {len(self.sequences)} event sequences.")

    def trace_forward_pass_for_whisper(self) -> None:
        """
        Profiles the generate step of Whisper (encoder-decoder).

        Runs multiple profiled inferences (controlled by --runs) for statistical robustness.
        """
        num_runs = getattr(self.args, 'runs', 1)
        LOGGER.info(f"=== Profiling Whisper Model Forward Pass ({num_runs} runs) ===")

        # Reset peak memory stats before warmup
        self._reset_memory_stats()

        # Warm-up runs
        with torch.no_grad():
            for _ in range(max(0, self.args.warmup)):
                self.model.generate(
                    **self.model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

        # Synchronize before timing
        self._sync_device()

        # Capture pre-inference memory baseline (after warmup, before profiling)
        self._reset_memory_stats()
        self._capture_pre_inference_memory()

        # Report GPU clocks for reproducibility
        if self._has_cuda:
            utils.report_gpu_clocks(context="after warmup, before profiling")

        # Profiled runs
        _profile_wall_start = time.perf_counter()
        with torch.no_grad():
            with profile(
                activities=self._get_profiler_activities(),
                with_modules=True,
                with_flops=True,
                profile_memory=True,
                record_shapes=True,
            ) as prof:
                _sampler = self._make_power_sampler()
                if self._has_cuda and self.num_gpus == 1:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                else:
                    self._sync_device()
                    wall_start = time.perf_counter()

                _energy_start_mj: Optional[float] = None
                _energy_wall_start: Optional[float] = None
                _sampling_wall_start: Optional[float] = None
                _sampling_wall_end: Optional[float] = None
                _energy_start_idx, _energy_measure_runs = _resolve_energy_measure_window(
                    num_runs,
                    getattr(self.args, "energy_measure_runs", 5),
                )
                for run_idx in range(num_runs):
                    _is_last = (run_idx == num_runs - 1)
                    _is_energy_start = (run_idx == _energy_start_idx)
                    if _is_energy_start:
                        _sampler.start()
                        self._sync_device()
                        _energy_start_mj = self._read_energy_counter_mj()
                        _energy_wall_start = time.perf_counter()
                        _sampling_wall_start = _energy_wall_start
                    self.model.generate(
                        **self.model_inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                    )
                    self._sync_device()
                    if _is_last:
                        _sampler.stop()
                        _sampling_wall_end = time.perf_counter()
                        if _energy_start_mj is not None and _energy_wall_start is not None:
                            _energy_end_mj = self._read_energy_counter_mj()
                            _energy_wall_end = time.perf_counter()
                            if _energy_end_mj is not None:
                                _dur_s_total = _energy_wall_end - _energy_wall_start
                                _delta_mj_total = _energy_end_mj - _energy_start_mj
                                _measurement = _build_inference_energy_measurement(
                                    _delta_mj_total,
                                    _dur_s_total,
                                    _energy_measure_runs,
                                )
                                if _measurement is not None:
                                    self._inference_energy_measurement = _measurement
                                else:
                                    print(
                                        "Warning: NVML energy counter returned zero/invalid delta"
                                        " — counter may not be supported on this GPU or"
                                        " the measurement window was too short."
                                        " energy_balance will be null in power_report.json.",
                                        file=sys.stderr,
                                    )

                if self._has_cuda and self.num_gpus == 1:
                    end_event.record()
                    torch.cuda.synchronize()
                    total_time_us = utils.ms_to_us(start_event.elapsed_time(end_event))
                else:
                    self._sync_device()
                    wall_end = time.perf_counter()
                    total_time_us = (wall_end - wall_start) * 1e6  # seconds -> µs

                self.torch_measured_inference_time_us = total_time_us / num_runs

        _profile_wall_end = time.perf_counter()
        self._capture_power_results(
            sampler=_sampler,
            profile_start_s=_profile_wall_start,
            profile_end_s=_profile_wall_end,
            sampling_start_s=_sampling_wall_start,
            sampling_end_s=_sampling_wall_end,
        )

        self.num_profiled_runs = num_runs
        print(f"Mean time per inference: {utils.us_to_ms(self.torch_measured_inference_time_us):.2f} ms")

        # Capture peak memory after profiled inference
        self._capture_peak_memory()

        prof.export_chrome_trace(str(self.trace_file))

    def trace_forward_pass_for_decoder(self) -> None:
        """
        Profiles the generate step of a decoder model.

        Runs multiple profiled inferences (controlled by --runs) to compute
        mean metrics for statistical robustness (addresses Reviewer B Comment 1).
        """
        num_runs = getattr(self.args, 'runs', 1)
        print(f"=== Profiling Model Forward Pass ({num_runs} runs) ===")

        # Reset peak memory stats before warmup
        self._reset_memory_stats()

        # Warm-up runs
        with torch.no_grad():
            for _ in range(max(0, self.args.warmup)):
                self.model.generate(
                    **self.model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        # Synchronize before timing
        self._sync_device()

        # Capture pre-inference memory baseline (after warmup, before profiling)
        self._reset_memory_stats()
        self._capture_pre_inference_memory()

        # Report GPU clocks for reproducibility
        if self._has_cuda:
            utils.report_gpu_clocks(context="after warmup, before profiling")

        # Profiled runs - run num_runs inferences within profiler
        # All runs are captured in a single trace; metrics are averaged by frequency
        _profile_wall_start = time.perf_counter()
        with torch.no_grad():
            with profile(
                activities=self._get_profiler_activities(),
                with_modules=True,
                with_flops=True,
                profile_memory=True,
                record_shapes=True,
            ) as prof:
                _sampler = self._make_power_sampler()
                if self._has_cuda and self.num_gpus == 1:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                else:
                    self._sync_device()
                    wall_start = time.perf_counter()

                _last_kv_output = None
                _energy_start_mj: Optional[float] = None
                _energy_wall_start: Optional[float] = None
                _sampling_wall_start: Optional[float] = None
                _sampling_wall_end: Optional[float] = None
                _energy_start_idx, _energy_measure_runs = _resolve_energy_measure_window(
                    num_runs,
                    getattr(self.args, "energy_measure_runs", 5),
                )
                for run_idx in range(num_runs):
                    _is_last = (run_idx == num_runs - 1)
                    _is_energy_start = (run_idx == _energy_start_idx)
                    if _is_energy_start:
                        _sampler.start()
                        # Ground truth: read energy counter at start of trailing window
                        self._sync_device()
                        _energy_start_mj = self._read_energy_counter_mj()
                        _energy_wall_start = time.perf_counter()
                        _sampling_wall_start = _energy_wall_start
                    # FIX: Use TE FP8 autocast if recipe is available
                    if self.is_fp8 and getattr(self, 'fp8_recipe', None):
                        import transformer_engine.pytorch as te
                        with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
                            _out = self.model.generate(
                                **self.model_inputs,
                                max_new_tokens=self.max_new_tokens,
                                do_sample=False,
                                pad_token_id=self.tokenizer.pad_token_id,
                                return_dict_in_generate=_is_last,
                            )
                    else:
                        _out = self.model.generate(
                            **self.model_inputs,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                            return_dict_in_generate=_is_last,
                        )
                    if _is_last:
                        _last_kv_output = _out
                    del _out

                    # Sync between runs to ensure clean measurements
                    self._sync_device()
                    if _is_last:
                        _sampler.stop()
                        _sampling_wall_end = time.perf_counter()
                        # Ground truth: read energy counter at end of trailing window
                        if _energy_start_mj is not None and _energy_wall_start is not None:
                            _energy_end_mj = self._read_energy_counter_mj()
                            _energy_wall_end = time.perf_counter()
                            if _energy_end_mj is not None:
                                _dur_s_total = _energy_wall_end - _energy_wall_start
                                _delta_mj_total = _energy_end_mj - _energy_start_mj
                                _measurement = _build_inference_energy_measurement(
                                    _delta_mj_total,
                                    _dur_s_total,
                                    _energy_measure_runs,
                                )
                                if _measurement is not None:
                                    self._inference_energy_measurement = _measurement
                                else:
                                    print(
                                        "Warning: NVML energy counter returned zero/invalid delta"
                                        " — counter may not be supported on this GPU or"
                                        " the measurement window was too short."
                                        " energy_balance will be null in power_report.json.",
                                        file=sys.stderr,
                                    )

                if self._has_cuda and self.num_gpus == 1:
                    end_event.record()
                    torch.cuda.synchronize()
                    total_time_us = utils.ms_to_us(start_event.elapsed_time(end_event))
                else:
                    self._sync_device()
                    wall_end = time.perf_counter()
                    total_time_us = (wall_end - wall_start) * 1e6  # seconds -> µs

                self.torch_measured_inference_time_us = total_time_us / num_runs

        _profile_wall_end = time.perf_counter()
        self._capture_power_results(
            sampler=_sampler,
            profile_start_s=_profile_wall_start,
            profile_end_s=_profile_wall_end,
            sampling_start_s=_sampling_wall_start,
            sampling_end_s=_sampling_wall_end,
        )

        # Store num_runs for downstream analysis
        self.num_profiled_runs = num_runs
        print(f"Total time for {num_runs} runs: {utils.us_to_ms(total_time_us):.2f} ms")
        print(f"Mean time per inference: {utils.us_to_ms(self.torch_measured_inference_time_us):.2f} ms")

        # Capture peak memory after profiled inference
        self._capture_peak_memory()

        prof.export_chrome_trace(str(self.trace_file))

        # Measure KV cache from last profiled run
        if _last_kv_output is not None:
            try:
                self._kv_cache_bytes = _compute_kv_cache_bytes(
                    getattr(_last_kv_output, "past_key_values", None)
                )
            except Exception:
                self._kv_cache_bytes = 0
            finally:
                del _last_kv_output

    def trace_forward_pass_for_encoder(self) -> None:
        """
        Profiles the forward pass of an encoder model.

        Runs multiple profiled inferences (controlled by --runs) for statistical robustness.
        """
        num_runs = getattr(self.args, 'runs', 1)
        print(f"=== Profiling Model Forward Pass ({num_runs} runs) ===")

        # Reset peak memory stats before warmup
        self._reset_memory_stats()

        # Warm-up runs
        with torch.no_grad():
            for _ in range(max(0, self.args.warmup)):
                self.model(**self.model_inputs)

        # Synchronize before timing
        self._sync_device()

        # Capture pre-inference memory baseline (after warmup, before profiling)
        self._reset_memory_stats()
        self._capture_pre_inference_memory()

        # Report GPU clocks for reproducibility
        if self._has_cuda:
            utils.report_gpu_clocks(context="after warmup, before profiling")

        # Profiled runs
        _profile_wall_start = time.perf_counter()
        with torch.no_grad():
            with profile(
                activities=self._get_profiler_activities(),
                with_modules=True,
                with_flops=True,
                profile_memory=True,
                record_shapes=True,
            ) as prof:
                _sampler = self._make_power_sampler()
                if self._has_cuda and self.num_gpus == 1:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                else:
                    self._sync_device()
                    wall_start = time.perf_counter()

                _energy_start_mj: Optional[float] = None
                _energy_wall_start: Optional[float] = None
                _sampling_wall_start: Optional[float] = None
                _sampling_wall_end: Optional[float] = None
                _energy_start_idx, _energy_measure_runs = _resolve_energy_measure_window(
                    num_runs,
                    getattr(self.args, "energy_measure_runs", 5),
                )
                for run_idx in range(num_runs):
                    _is_last = (run_idx == num_runs - 1)
                    _is_energy_start = (run_idx == _energy_start_idx)
                    if _is_energy_start:
                        _sampler.start()
                        self._sync_device()
                        _energy_start_mj = self._read_energy_counter_mj()
                        _energy_wall_start = time.perf_counter()
                        _sampling_wall_start = _energy_wall_start
                    self.model(**self.model_inputs)
                    self._sync_device()
                    if _is_last:
                        _sampler.stop()
                        _sampling_wall_end = time.perf_counter()
                        if _energy_start_mj is not None and _energy_wall_start is not None:
                            _energy_end_mj = self._read_energy_counter_mj()
                            _energy_wall_end = time.perf_counter()
                            if _energy_end_mj is not None:
                                _dur_s_total = _energy_wall_end - _energy_wall_start
                                _delta_mj_total = _energy_end_mj - _energy_start_mj
                                _measurement = _build_inference_energy_measurement(
                                    _delta_mj_total,
                                    _dur_s_total,
                                    _energy_measure_runs,
                                )
                                if _measurement is not None:
                                    self._inference_energy_measurement = _measurement
                                else:
                                    print(
                                        "Warning: NVML energy counter returned zero/invalid delta"
                                        " — counter may not be supported on this GPU or"
                                        " the measurement window was too short."
                                        " energy_balance will be null in power_report.json.",
                                        file=sys.stderr,
                                    )

                if self._has_cuda and self.num_gpus == 1:
                    end_event.record()
                    torch.cuda.synchronize()
                    total_time_us = utils.ms_to_us(start_event.elapsed_time(end_event))
                else:
                    self._sync_device()
                    wall_end = time.perf_counter()
                    total_time_us = (wall_end - wall_start) * 1e6  # seconds -> µs

                self.torch_measured_inference_time_us = total_time_us / num_runs

        _profile_wall_end = time.perf_counter()
        self._capture_power_results(
            sampler=_sampler,
            profile_start_s=_profile_wall_start,
            profile_end_s=_profile_wall_end,
            sampling_start_s=_sampling_wall_start,
            sampling_end_s=_sampling_wall_end,
        )

        self.num_profiled_runs = num_runs
        print(f"Mean time per inference: {utils.us_to_ms(self.torch_measured_inference_time_us):.2f} ms")

        # Capture peak memory after profiled inference
        self._capture_peak_memory()

        prof.export_chrome_trace(str(self.trace_file))

def main() -> int:
    """Main entry point for the SODA CLI."""
    
    # Check if env.sh has been sourced and loaded
    if "SODA_ENV_LOADED" not in os.environ:
        # Use stderr for early errors before logger is set up
        print("Error: SODA environment not loaded.", file=sys.stderr)
        print("Please run: source env.sh", file=sys.stderr)
        sys.exit(1)

    try:
        # Parse and validate arguments
        args = utils.parse_and_validate_args()

        # --- Enhanced TaxBreak mode (Stage 2: no model loading) ---
        if getattr(args, "taxbreak", False):
            from soda.taxbreak.pipeline import TaxBreakPipeline

            db_path = getattr(args, "kernel_db_path", None)
            if not db_path:
                print("Error: --taxbreak requires --kernel-db-path", file=sys.stderr)
                return 1
            db_path = Path(db_path)
            if not db_path.exists():
                print(f"Error: kernel DB not found: {db_path}", file=sys.stderr)
                return 1

            # Set EXPERIMENT_DIR so relative path resolution works
            os.environ["EXPERIMENT_DIR"] = str(db_path.parent.resolve())

            pipeline = TaxBreakPipeline(kernel_db_path=db_path, args=args)
            pipeline.run()
            return 0

        # --- MoE per-operator memory profiling via CUPTI ---
        if getattr(args, "moe_profile", False):
            from soda.moe.pipeline import MoEProfilePipeline

            pipeline = MoEProfilePipeline(model_name=args.model, args=args)
            pipeline.run()
            return 0

        # Create tracer (derives experiment/output paths internally)
        print(f"Loading model: {args.model} with precision {args.precision}")
        tracer = ModelTracer(args=args)
        print(f"Results will be saved to: {tracer.output_dir.resolve()}")

        # Run the tracing pipeline
        tracer.run()

        if args.microbench:
            # Microbench mode: extract -> replay -> verify -> plot
            microbench = SodaMicrobench(tracer=tracer, args=args)
            microbench.run()
            return 0

        # Standard analysis
        analyzer = SodaAnalyzer(tracer=tracer, args=args)
        analyzer.run()

        # Optional: generate kernel database after analysis
        if getattr(args, "kernel_db", False):
            from soda.kerneldb import generate_kernel_database
            db_path = tracer.output_dir / "kernel_database.json"
            generate_kernel_database(tracer=tracer, args=args, output_path=db_path)

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
