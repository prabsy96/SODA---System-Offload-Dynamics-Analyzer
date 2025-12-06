"""
SODA: System Offload Dynamics Analyzer
Analyze CPU–GPU dynamics of PyTorch models.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
import traceback
import numpy as np
import torch
import transformers
from collections import defaultdict, deque
from pathlib import Path
from torch.profiler import ProfilerActivity, profile
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

# Treat this module as the root of the soda package so submodules like
# soda.common.* resolve to the sibling directories under src/.
_PACKAGE_ROOT = Path(__file__).resolve().parent
__path__ = [str(_PACKAGE_ROOT)]
if __spec__ is not None:
    __spec__.submodule_search_locations = __path__

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
        LOGGER.info("=== Analyzing Trace Data ===")
        LOGGER.info(f"Analyzing {len(self.sequences)} event sequences")
        sequences = utils.calculate_per_seq_launch_tax(list(self.sequences))
        
        # Analyze per-stream metrics
        stream_info = utils.analyze_per_stream(self.events)
        
        # General metrics
        trace_calculated_inference_time = utils.calculate_total_inference_time(self.trace)
        inference_time = self.tracer.torch_measured_inference_time_us
        
        # GPU metrics
        total_gpu_time_span = utils.calculate_total_gpu_time_span(self.events)
        true_gpu_busy_time = utils.calculate_true_gpu_busy_time(self.events)
        gpu_utilization = utils.calculate_gpu_utilization(self.events)
        
        # Kernel metrics
        kernel_exec_time = utils.calculate_kernel_exec_time(self.events)
        total_launch_tax = utils.calculate_total_launch_tax(sequences)
        avg_launch_tax = utils.calculate_avg_launch_tax(sequences)
        avg_kernel_dur = utils.get_average_kernel_duration(self.events)
        top_k_kernels = utils.get_top_k_kernels(self.events, k=3)
        
        # Fusion analysis
        fusion_results = None
        if self.args.fusion:
            LOGGER.info("=== Kernel Fusion Analysis ===")
            fusion_results = {}
            for f in self.args.fusion:
                fusion_results[f] = utils.analyze_kernel_fusion_candidates(sequences, f, self.args.prox_score, logger=LOGGER)

        # Framework overhead (CPU-side latency)
        framework_overhead = utils.calculate_framework_tax(
            inference_time,
            true_gpu_busy_time
        )

        # Build metrics dictionary 
        metrics = {
            # Inference time 
            "inference_time_ms": utils.us_to_ms(inference_time),
            # Inference time breakdown
            "inference_time_breakdown": {
                "torch_measured_inference_time_ms": utils.us_to_ms(self.tracer.torch_measured_inference_time_us),
                "trace_calculated_inference_time_ms": utils.us_to_ms(trace_calculated_inference_time),
                "profiler_overhead_ms": utils.us_to_ms(trace_calculated_inference_time - self.tracer.torch_measured_inference_time_us),
            },
            "framework_overhead": framework_overhead,
            "active_streams": len(stream_info),

            # GPU metrics
            "total_gpu_time_span_ms": utils.us_to_ms(total_gpu_time_span),
            "gpu_busy_time_ms": utils.us_to_ms(true_gpu_busy_time),
            "gpu_idle_time_ms": utils.us_to_ms(max(0.0, total_gpu_time_span - true_gpu_busy_time)),
            "gpu_utilization_percent": gpu_utilization,
 
            # Kernel metrics
            "total_kernel_exec_time_ms": utils.us_to_ms(kernel_exec_time["total"]),
            "num_total_kernels": len(self.events["gpu"]["kernels"]),
            "avg_kernel_exec_time_ms": utils.us_to_ms(kernel_exec_time["avg"]),
            "total_kernel_launch_tax_ms": utils.us_to_ms(total_launch_tax),
            "avg_kernel_launch_tax_ms": utils.us_to_ms(avg_launch_tax),
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
        """
        if self.results is None:
            raise ValueError("No analysis results available. Call analyze() first.")
        
        metrics = self.results["metrics"]
        stream_info = self.results["stream_info"]
        top_k_kernels = self.results["top_k_kernels"]
        
        # --- Enhanced Reporting ---
        LOGGER.info("")
        LOGGER.info("=== Performance Metrics ===")
        LOGGER.info(f"\t* Inference runtime (ms): {metrics['inference_time_ms']:.4f}")
        
        # Framework Tax Analysis
        framework = metrics["framework_overhead"]
        timing = metrics["inference_time_breakdown"]
        LOGGER.info("")
        LOGGER.info("=== Framework Tax Analysis ===")
        LOGGER.info(f"\t* Framework Tax (Exposed CPU Overhead): {framework['framework_tax_ms']:.4f} ms ({framework['framework_tax_percent']:.1f}%)")
        LOGGER.info(f"\t* GPU Active Time (Compute): {metrics['gpu_busy_time_ms']:.4f} ms ({framework['gpu_busy_time_percent']:.1f}%)")
        
        # Timing breakdown
        LOGGER.info(f"\t  - Torch measured inference time (ms): {timing['torch_measured_inference_time_ms']:.4f}")
        LOGGER.info(f"\t  - Trace calculated inference time (ms): {timing['trace_calculated_inference_time_ms']:.4f}")
        LOGGER.info(f"\t  - Profiler overhead (ms): {timing['profiler_overhead_ms']:.4f}")
        
        LOGGER.info("")
        LOGGER.info("=== GPU Metrics ===")
        LOGGER.info(f"\t* Total kernel execution time (ms): {metrics['total_kernel_exec_time_ms']:.4f}")
        LOGGER.info(f"\t* GPU busy time (concurrent-aware) (ms): {metrics['gpu_busy_time_ms']:.4f}")
        LOGGER.info(f"\t* GPU idle time (ms): {metrics['gpu_idle_time_ms']:.4f}")
        LOGGER.info(f"\t* GPU utilization: {metrics['gpu_utilization_percent']:.2f}%")
        LOGGER.info(f"\t* Number of kernels: {metrics['num_total_kernels']}")
        LOGGER.info(f"\t* Active streams: {metrics['active_streams']}")
        
        LOGGER.info("")
        LOGGER.info("=== Launch & Queue Latency (TKLQT) ===")
        LOGGER.info(f"\t* Total TKLQT (ms): {metrics['total_kernel_launch_tax_ms']:.4f}")
        if metrics['num_total_kernels'] > 0:
            LOGGER.info(f"\t* Avg. TKLQT per kernel (ms): {metrics['avg_kernel_launch_tax_ms']:.4f}")
            LOGGER.info(f"\t* Avg. execution time per kernel (ms): {metrics['avg_kernel_exec_time_ms']:.4f}")
        
        LOGGER.info("")
        # --- Per-Stream Breakdown ---
        LOGGER.info("=== Per-Stream Analysis ===")
        for stream_id, data in stream_info.items():
            LOGGER.info(
                f"\t* Stream {stream_id}: {data['op_count']} ops "
                f"({data['kernel_count']} kernels), "
                f"Busy Time: {utils.us_to_ms(data['true_gpu_busy_time']):.4f} ms"
            )
        
        LOGGER.info("")
        # Top-K kernels 
        if top_k_kernels["by_frequency"]:
            LOGGER.info("=== Top-3 Kernels by Frequency ===")
            for i, (name, data) in enumerate(top_k_kernels["by_frequency"], 1):
                LOGGER.info(
                    f"\t* #{i}: {name} "
                    f"(Frequency: {int(data['frequency'])}, "
                    f"Total Duration: {utils.us_to_ms(data['duration']):.4f} ms)"
                )
            
            LOGGER.info("")
            LOGGER.info("=== Top-3 Kernels by Duration ===")
            for i, (name, data) in enumerate(top_k_kernels["by_duration"], 1):
                LOGGER.info(
                    f"\t* #{i}: {name} "
                    f"(Total Duration: {utils.us_to_ms(data['duration']):.4f} ms, "
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
        
        metrics = self.results["metrics"]
        stream_info = self.results["stream_info"]
        top_k_kernels = self.results["top_k_kernels"]
        
        # Generate model_name and config from args
        model_name = self.args.model
        config = {
            "batch_size": self.args.batch_size,
            "seq_len": self.args.seq_len,
            "max_new_tokens": getattr(self.args, "max_new_tokens", None),
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
        
        LOGGER.info(f"Metrics exported to: {self.report_file}")
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
        self.device = torch.device(args.device)
        self.compile_type = args.compile_type
        self.is_fp8 = args.precision == "float8_e4m3fn"
        self.precision = utils.parse_dtype_to_torch(args.precision)
        self.load_precision = torch.bfloat16 if self.is_fp8 else self.precision

        # Set random seeds
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
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
    
    def setup(self) -> None:
        """
        Loads the model/tokenizer and prepares synthetic inputs.
        """

        # Load model and tokenizer
        if self.is_whisper:
            self.model, self.tokenizer = self.load_whisper()
        elif self.is_decoder:
            self.model, self.tokenizer = self.load_decoder()
        else:
            self.model, self.tokenizer = self.load_encoder()

        print(f"Generating synthetic input: batch_size={self.batch_size}, seq_len={self.seq_len}")
        if self.is_whisper:
            self.model_inputs = self.generate_audio_inputs()
        else:
            self.model_inputs = utils.generate_synthetic_inputs(
                self.tokenizer, self.device, self.batch_size, self.seq_len
            )


    def get_kwargs(self) -> Dict[str, Any]:
        """Returns common kwargs for model loading."""
        kwargs = {
            "dtype": self.load_precision,
            "device_map": self.device if self.device.type == 'cuda' else 'cpu',
            "trust_remote_code": True,
        }
        
        if self.is_fp8 and FP8_CONFIG_AVAILABLE:
            kwargs["quantization_config"] = FP8Config(fp8_format="e4m3")

        return kwargs

    def _convert_to_fp8(self, model: transformers.PreTrainedModel) -> transformers.PreTrainedModel:
        """
        Convert Linear layer weights to FP8 E4M3 for inference if quantization config is unavailable.
        """
        LOGGER.info("Converting linear layer weights to float8_e4m3fn...")

        converted_count = 0
        for _, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                with torch.no_grad():
                    # Clamp to avoid overflow in FP8 E4M3 range before conversion
                    clamped = module.weight.data.clamp(-448.0, 448.0)
                    module.weight.data = clamped.to(torch.float8_e4m3fn)
                    converted_count += 1

        LOGGER.info(f"Converted {converted_count} linear layers to FP8 E4M3.")
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
            if self.is_fp8 and "quantization_config" not in kwargs:
                model = self._convert_to_fp8(model)
            model.forward = torch.compile(model.forward, mode="reduce-overhead", backend="inductor")
        elif self.compile_type == "flash-attention":
            kwargs.update({"attn_implementation": "flash_attention_2"})
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name, config=config, **kwargs
            ).eval()
            if self.is_fp8 and "quantization_config" not in kwargs:
                model = self._convert_to_fp8(model)
        else:  # eager
            kwargs.update({"attn_implementation": "eager"})
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name, config=config, **kwargs
            ).eval()
            if self.is_fp8 and "quantization_config" not in kwargs:
                model = self._convert_to_fp8(model)
        
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
        LOGGER.info(f"Chrome trace file generated at: {self.trace_file}")

    def process(self) -> None:
        """
        Parses the trace to collect events and build linked event sequences.
        """
        self.events = utils.collect_events(self.trace_data)
        self.sequences = utils.link_sequences(self.events)
        LOGGER.info(f"Collected {len(self.sequences)} event sequences.")

    def trace_forward_pass_for_whisper(self) -> None:
        """
        Profiles the generate step of Whisper (encoder-decoder).
        """
        LOGGER.info("=== Profiling Whisper Model Forward Pass ===")
        
        # Warm-up runs
        with torch.no_grad():
            for _ in range(max(0, self.args.warmup)):
                self.model.generate(
                    **self.model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

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
                record_shapes=True,
            ) as prof:
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                self.model.generate(
                    **self.model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
                end_time.record()
                
                torch.cuda.synchronize()
                self.torch_measured_inference_time_us = utils.ms_to_us(start_time.elapsed_time(end_time))
        
        prof.export_chrome_trace(str(self.trace_file))

    def trace_forward_pass_for_decoder(self) -> None:
        """
        Profiles the generate step of a decoder model.
        
        Args:
            inputs: A dictionary of tokenized inputs.
        """
        LOGGER.info("=== Profiling Model Forward Pass ===")
        
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
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Profiled run 
        with torch.no_grad():
            with profile(
                activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                with_modules=True,
                with_flops=True,
                profile_memory=True,
                record_shapes=True,
            ) as prof:
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                self.model.generate(
                    **self.model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                end_time.record()
                
                torch.cuda.synchronize()
                self.torch_measured_inference_time_us = utils.ms_to_us(start_time.elapsed_time(end_time))
        
        prof.export_chrome_trace(str(self.trace_file))

    def trace_forward_pass_for_encoder(self) -> None:
        """
        Profiles the forward pass of an encoder model.
        
        Args:
            inputs: A dictionary of tokenized inputs.
        """
        LOGGER.info("=== Profiling Model Forward Pass ===")
        
        # Warm-up runs
        with torch.no_grad():
            for _ in range(max(0, self.args.warmup)):
                self.model(**self.model_inputs)
        
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
                record_shapes=True,
            ) as prof:
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                self.model(**self.model_inputs)
                end_time.record()
                
                torch.cuda.synchronize()
                self.torch_measured_inference_time_us = utils.ms_to_us(start_time.elapsed_time(end_time))

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

        # Create tracer (derives experiment/output paths internally)
        print(f"Loading model: {args.model} with precision {args.precision}")
        tracer = ModelTracer(args=args)
        
        # Setup logger for tracer
        SodaLogger(tracer.output_dir, is_console=True, is_file=True)
        
        # Run the tracing pipeline
        tracer.run()

        if args.microbench:
            # Microbench mode: extract -> replay -> verify -> plot
            microbench = SodaMicrobench(tracer=tracer, args=args)
            microbench.run()
            return 0
        else:
            # Create analyzer and run
            analyzer = SodaAnalyzer(tracer=tracer, args=args)
            analyzer.run()
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
