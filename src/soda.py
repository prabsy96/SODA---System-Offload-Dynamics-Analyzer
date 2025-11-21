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

# Import and expose utils module for 'from soda import utils'
import utils

# Make utils accessible as soda.utils
__all__ = ['ModelHandler', 'SodaProfiler', 'utils']

class SodaProfiler:
    """
    Handles model tracing, profile data parsing, and metric generation.
    """
    
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
        self.experiment_name = utils.generate_experiment_name(args.model, args.compile_type, args.batch_size, args.seq_len)
        self.output_dir = Path(args.output_dir) / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self._soda_logger = SodaLogger(self.output_dir, is_console=log_console, is_file=log_file)
        self.logger = self._soda_logger.logger
        
        # Setup deterministic mode for reproducibility
        utils.setup_deterministic_mode(seed=args.seed)
        
        self.trace_file_path = self.output_dir / "trace.json"
        self.report_file_path = self.output_dir / "report.json"
        
        self.trace = None
        self.events = None
        self.results = None

    def trace_forward_pass_for_encoder(self, inputs: Dict[str, torch.Tensor]) -> None:
        """
        Profiles the forward pass of an encoder model.

        Args:
            inputs: A dictionary of tokenized inputs.
        """
        self.logger.info("=== Profiling Model Forward Pass ===")
        
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
                record_shapes=True,
            ) as prof:
                self.model_handler.pytorch_model(**inputs)

        prof.export_chrome_trace(str(self.trace_file_path))

    def profile_forward_pass(self, inputs: Dict[str, torch.Tensor], batch_size: int = None, seq_len: int = None) -> None:
        """
        Profiles the forward pass of the model (encoder or decoder).
        
        Args:
            inputs: A dictionary of tokenized inputs.
            batch_size: Optional batch size. Defaults to self.args.batch_size.
            seq_len: Optional sequence length. Defaults to self.args.seq_len.
        """
        batch_size = self.args.batch_size if batch_size is None else batch_size
        seq_len = self.args.seq_len if seq_len is None else seq_len
        
        self.trace_forward_pass_for_encoder(inputs)
        # Load trace data into memory immediately
        self.trace = utils.load_json(self.trace_file_path)
        self.logger.info(f"* Chrome trace file generated at: {self.trace_file_path}")

        # FIXME
        # if self.model_handler.is_decoder:
        #     self.trace_forward_pass_for_decoder(inputs, batch_size, seq_len)
        # else:
        #     self.trace_forward_pass_for_encoder(inputs)

    def trace_forward_pass_for_decoder(self, inputs: Dict[str, torch.Tensor], bs: int, sq: int) -> None:
        """
        Profiles the generate step of a decoder model.
        
        Args:
            inputs: A dictionary of tokenized inputs.
            bs: Batch size.
            sq: Sequence length.
        """
        self.logger.info("=== Profiling Model Forward Pass ===")
        
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
                record_shapes=True,
            ) as prof:
                self.model_handler.pytorch_model.generate(**inputs, max_new_tokens=1, do_sample=False, pad_token_id=self.model_handler.tokenizer.pad_token_id)
        
        prof.export_chrome_trace(str(self.trace_file_path))
    
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
        self.events = utils.collect_events(self.trace)
        self.logger.info(f"Analyzing {len(self.events['gpu']['kernels'])} kernel events from profiled run...")
        event_sequences = utils.get_linked_event_sequences(self.events)
        event_sequences = utils.calculate_per_seq_launch_tax(event_sequences)
        
        # Analyze per-stream metrics
        stream_info = utils.analyze_per_stream(self.events)
        
        # General metrics
        total_inference_time = utils.calculate_total_inference_time(self.trace)
        
        # GPU metrics
        total_gpu_time_span = utils.calculate_total_gpu_time_span(self.events)
        true_gpu_busy_time = utils.calculate_true_gpu_busy_time(self.events)
        gpu_utilization = utils.calculate_gpu_utilization(self.events)
        
        # Kernel metrics
        kernel_exec_time = utils.calculate_kernel_exec_time(self.events)
        total_launch_tax = utils.calculate_total_launch_tax(event_sequences)
        avg_launch_tax = utils.calculate_avg_launch_tax(event_sequences)
        avg_kernel_dur = utils.get_average_kernel_duration(self.events)
        top_k_kernels = utils.get_top_k_kernels(self.events, k=3)
        
        # Fusion analysis
        fusion_results = None
        if self.args.fusion:
            self.logger.info("=== Kernel Fusion Analysis ===")
            fusion_results = {}
            for f in self.args.fusion:
                fusion_results[f] = utils.analyze_kernel_fusion_candidates(event_sequences, f, self.args.prox_score, logger=self.logger)
        
        utils.run_extraction_pipeline(self.trace_file_path, event_sequences)
        
        # Build metrics dictionary 
        metrics = {
            # General metrics
            "inference_runtime_ms": utils.us_to_ms(total_inference_time),
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
            "event_sequences": event_sequences,
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
        self.logger.info("")
        self.logger.info("=== Performance Metrics ===")
        self.logger.info(f"\t* Inference runtime (ms): {metrics['inference_runtime_ms']:.4f}")
        self.logger.info(f"\t* Total kernel execution time (ms): {metrics['total_kernel_exec_time_ms']:.4f}")
        self.logger.info(f"\t* GPU busy time (concurrent-aware) (ms): {metrics['gpu_busy_time_ms']:.4f}")
        self.logger.info(f"\t* GPU idle time (ms): {metrics['gpu_idle_time_ms']:.4f}")
        self.logger.info(f"\t* GPU utilization: {metrics['gpu_utilization_percent']:.2f}%")
        self.logger.info(f"\t* Total kernel launch tax (TKLQT) (ms): {metrics['total_kernel_launch_tax_ms']:.4f}")
        self.logger.info(f"\t* Number of kernels: {metrics['num_total_kernels']}")
        self.logger.info(f"\t* Active streams: {metrics['active_streams']}")
        
        if metrics['num_total_kernels'] > 0:
            self.logger.info(f"\t* Avg. kernel launch tax per kernel (ms): {metrics['avg_kernel_launch_tax_ms']:.4f}")
            self.logger.info(f"\t* Avg. execution time per kernel (ms): {metrics['avg_kernel_exec_time_ms']:.4f}")
        
        self.logger.info("")
        # --- Per-Stream Breakdown ---
        self.logger.info("=== Per-Stream Analysis ===")
        for stream_id, data in stream_info.items():
            self.logger.info(
                f"\t* Stream {stream_id}: {data['op_count']} ops "
                f"({data['kernel_count']} kernels), "
                f"Busy Time: {utils.us_to_ms(data['true_gpu_busy_time']):.4f} ms"
            )
        
        self.logger.info("")
        # Top-K kernels 
        if top_k_kernels["by_frequency"]:
            self.logger.info("=== Top-3 Kernels by Frequency ===")
            for i, (name, data) in enumerate(top_k_kernels["by_frequency"], 1):
                self.logger.info(
                    f"\t* #{i}: {name} "
                    f"(Frequency: {int(data['frequency'])}, "
                    f"Total Duration: {utils.us_to_ms(data['duration']):.4f} ms)"
                )
            
            self.logger.info("")
            self.logger.info("=== Top-3 Kernels by Duration ===")
            for i, (name, data) in enumerate(top_k_kernels["by_duration"], 1):
                self.logger.info(
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
        if fusion_results is not None:
            output["fusion_analysis"] = fusion_results
        
        # Save to file
        utils.save_json(self.report_file_path, output)
        
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
            "trust_remote_code": True,
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
        
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None: 
            pad_token_id = getattr(generation_config, "pad_token_id", None)
            if pad_token_id is None:
                generation_config.pad_token_id = tokenizer.eos_token_id
        
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
        # Use stderr for early errors before logger is set up
        print("Error: SODA environment not loaded.", file=sys.stderr)
        print("Please run: source env.sh", file=sys.stderr)
        sys.exit(1)

    try:
        # Parse and validate arguments 
        args = utils.parse_and_validate_args()

        # Prepare model handler
        print(f"Loading model: {args.model} with precision {args.precision}...")
        model_handler = ModelHandler(
            model_name=args.model,
            device=args.device,
            compile_type=args.compile_type,
            precision=args.precision,
        )
        
        # Generate synthetic inputs
        print(f"Generating synthetic input: batch_size={args.batch_size}, seq_len={args.seq_len}")
        model_inputs = model_handler.generate_synthetic_inputs(
            args.batch_size, args.seq_len
        )

        # Initialize profiler 
        profiler = SodaProfiler(model_handler=model_handler, args=args, log_console=True, log_file=True)

        # Profile forward pass and analyze 
        profiler.profile_forward_pass(model_inputs)
        profiler.analyze()

        # Report and save results
        profiler.report()
        profiler.save()

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
