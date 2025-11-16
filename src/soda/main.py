"""
SODA: System-Offload-Dynamics-Analyzer

A lightweight profiler for analyzing the performance dynamics of PyTorch models, focusing on CPU/GPU interactions, kernel launches, and execution dependencies.
This tool is inspired by the principles of detailed system analysis seen in tools like SKIP.
"""

import argparse
import builtins
import os
import sys
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from . import util

def set_seed(seed: int) -> None:
    """
    Set seed for reproducibility.
    
    Args:
        seed: Seed value to ensure deterministic behavior.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_args_parser() -> argparse.ArgumentParser:
    """Create and return argument parser."""
    parser = argparse.ArgumentParser(
        description="SODA: System Offload Dynamics Analyzer.",
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
        default=Path(os.environ["SODA_RESULTS"]),
        help="Output directory for analysis artifacts (traces, reports, etc.)",
    )
    parser.add_argument(
        "-c",
        "--compile_type",
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
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Precision for model weights and operations",
    )
    parser.add_argument(
        "-sl", "--seq_len", type=int, default=512, 
        help="Sequence length for synthetic input."
    )
    parser.add_argument(
        "-bs", "--batch_size", type=int, default=1, 
        help="Batch size for synthetic input."
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
        "--prox_score",
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

def validate_args(args: argparse.Namespace) -> None:
    """Validate parsed arguments."""
    if args.device == "cpu" and args.precision == "float16":
        print("Warning: float16 is not supported on CPU. Forcing float32.")
        args.precision = "float32"

    if not torch.cuda.is_available() and args.device == "cuda":
        print("Error: CUDA is not available. Please select --device cpu.", file=sys.stderr)
        sys.exit(1)

def setup_logging(output_dir: Path) -> tuple:
    """
    Setup logging: redirect print to both stdout and log file.
    
    Args:
        output_dir: Directory where log file will be created.
    
    Returns:
        Tuple of (log_path, output_file, original_print) for cleanup.
    """
    log_path = output_dir / "soda.log"
    output_file = open(log_path, "w")
    
    original_print = builtins.print
    
    def print_and_write(*args_print, **kwargs):
        """Print to stdout and write to file."""
        original_print(*args_print, **kwargs)
        line = ' '.join(str(arg) for arg in args_print)
        output_file.write(line + '\n')
        output_file.flush()
    
    builtins.print = print_and_write
    
    # Print initial message 
    print(f"Results will be saved to: {output_dir.resolve()}")
    
    return log_path, output_file, original_print

def main() -> int:
    """Main entry point for the SODA CLI."""

    # Check if env.sh has been sourced and loaded
    if not os.environ.get("SODA_ENV_LOADED"):
        print("Error: SODA environment not loaded.", file=sys.stderr)
        print("Please run: source env.sh", file=sys.stderr)
        sys.exit(1)
    
    # Parse and validate arguments
    parser = get_args_parser()
    args = parser.parse_args()
    validate_args(args)

    # Set seed for reproducibility
    set_seed(args.seed)

    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging: redirect print to both stdout and log file
    log_path, output_file, original_print = setup_logging(args.output_dir)

    try:
        # --- Model Loading ---
        print(f"Loading model: {args.model} with precision {args.precision}...")
        model_obj = util.Model(
            model_name=args.model,
            device=args.device,
            compile_type=args.compile_type,
            precision=args.precision,
        )

        if "bert" in args.model or "roberta" in args.model:
            model, tokenizer = model_obj.load_encoder()
            is_decoder = False
        else:
            model, tokenizer = model_obj.load_decoder()
            is_decoder = True
        print("Model loaded successfully.")

        # --- Synthetic Data Generation ---
        print(f"Generating synthetic input: batch_size={args.batch_size}, seq_len={args.seq_len}")
        token_ids = {
            "input_ids": torch.randint(
                1, tokenizer.vocab_size, size=(args.batch_size, args.seq_len), device=args.device
            ),
            "attention_mask": torch.ones(
                args.batch_size, args.seq_len, device=args.device
            ),
        }

        # --- Tracing and Analysis ---
        trace_dir_name = f"{args.model.replace('/', '_')}_{args.compile_type}_bs{args.batch_size}_sl{args.seq_len}"
        trace_obj = util.TraceModel(
            name=trace_dir_name,
            file="trace.json",
            path=str(args.output_dir),
            model=model,
        )

        #  Single warm-up and trace run is performed.
        print("Starting model profiling run...")
        if is_decoder:
            json_file = trace_obj.trace_forward_pass_for_decoder(
                token_ids, tokenizer, args.batch_size, args.seq_len
            )
        else:
            json_file = trace_obj.trace_forward_pass_for_encoder(token_ids, tokenizer)
        print(f"Chrome trace file generated at: {json_file}")

        # Data Processing and Reporting
        print("Analyzing trace data to generate reports...")
        
        # Get all events organized by category
        trace_data = trace_obj.load_trace_file(Path(json_file))
        events = trace_obj.collect_events_from_trace(trace_data)
        
        dependencies = trace_obj.generate_dependencies(events)
        
        # Analyze per-stream metrics
        stream_info = trace_obj.analyze_per_stream(events["gpu"]["all"])
        
        # Calculate metrics 
        total_inference_time = trace_obj.calculate_total_inference_time(trace_data)
        total_gpu_time_span = trace_obj.calculate_total_gpu_time_span(events)
        true_gpu_busy_time = trace_obj.calculate_true_gpu_busy_time(events)
        gpu_utilization = trace_obj.calculate_gpu_utilization(events)
        gpu_idle_time = max(0.0, total_gpu_time_span - true_gpu_busy_time)
        total_kernel_exec_time = trace_obj.calculate_total_kernel_exec_time(events)
        
        num_kernels = len(events["gpu"]["kernels"])
        num_gpu_events = len(events["gpu"]["all"])
        
        # Calculate launch overhead
        launch_overhead = trace_obj.calculate_launch_tax(dependencies)
        
        # Calculate average kernel duration
        akd_results = trace_obj.get_average_kernel_duration(events["gpu"]["kernels"])
        
        
        # --- Enhanced Reporting ---
        print("--- Performance Metrics ---")
        print(f"Inference runtime (ms): {total_inference_time / 1000:.4f}")
        print(f"Total kernel execution time (ms): {total_kernel_exec_time / 1000:.4f}")
        print(f"GPU busy time (concurrent-aware) (ms): {true_gpu_busy_time / 1000:.4f}")
        print(f"GPU idle time (ms): {gpu_idle_time / 1000:.4f}")
        print(f"GPU utilization: {gpu_utilization:.2f}%")
        print(f"Total launch overhead (TKLQT) (ms): {launch_overhead / 1000:.4f}")
        print(f"Number of kernels: {num_kernels}")
        print(f"Total GPU operations: {num_gpu_events}")
        print(f"Active streams: {len(stream_info)}")
        
        if num_kernels > 0:
            print(f"Avg. launch overhead per kernel (ms): {(launch_overhead / num_kernels) / 1000:.4f}")
            print(f"Avg. execution time per kernel (ms): {(total_kernel_exec_time / num_kernels) / 1000:.4f}")
        
        # --- Per-Stream Breakdown ---
        print("--- Per-Stream Analysis ---")
        for stream_id, data in stream_info.items():
            print(
                f"  Stream {stream_id}: {data['op_count']} ops "
                f"({data['kernel_count']} kernels), "
                f"Busy Time: {data['true_gpu_busy_time'] / 1000:.4f} ms"
            )
        
        # Top-K kernels 
        top_k_kernels = trace_obj.get_top_k_kernels(events["gpu"]["kernels"], k=3)
        
        if top_k_kernels["by_frequency"]:
            print("--- Top-3 Kernels by Frequency ---")
            for i, (name, data) in enumerate(top_k_kernels["by_frequency"], 1):
                print(
                    f"#{i}: {name} "
                    f"(Frequency: {int(data['frequency'])}, "
                    f"Total Duration: {data['duration'] / 1000:.4f} ms)"
                )
            
            print("--- Top-3 Kernels by Duration ---")
            for i, (name, data) in enumerate(top_k_kernels["by_duration"], 1):
                print(
                    f"#{i}: {name} "
                    f"(Total Duration: {data['duration'] / 1000:.4f} ms, "
                    f"Frequency: {int(data['frequency'])})"
                )
        
        # Fusion analysis
        if args.fusion:
            print("--- Kernel Fusion Analysis ---")
            for f in args.fusion:
                trace_obj.kernelchains(dependencies, f, args.prox_score)
        
        # --- Export Metrics to JSON ---
        metrics_dict = {
            "inference_runtime_ms": total_inference_time / 1000,
            "total_kernel_execution_ms": total_kernel_exec_time / 1000,
            "gpu_busy_time_ms": true_gpu_busy_time / 1000,
            "gpu_idle_time_ms": gpu_idle_time / 1000,
            "gpu_utilization_percent": gpu_utilization,
            "total_launch_overhead_ms": launch_overhead / 1000,
            "num_kernels": num_kernels,
            "total_gpu_operations": num_gpu_events,
            "active_streams": len(stream_info),
            "avg_launch_overhead_per_kernel_ms": (launch_overhead / num_kernels / 1000) if num_kernels > 0 else 0.0,
            "avg_kernel_execution_ms": (total_kernel_exec_time / num_kernels / 1000) if num_kernels > 0 else 0.0,
        }
        
        config_dict = {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "precision": args.precision,
            "compile_type": args.compile_type,
            "device": args.device,
        }
        
        trace_obj.export_metrics_to_json(
            model_name=args.model,
            config=config_dict,
            metrics=metrics_dict,
            stream_info=stream_info,
            top_k_kernels=top_k_kernels,
        )
        
        print("Analysis complete.")
        print(f"\nLog output saved to {log_path}")
        
        # Restore original print and close log file
        builtins.print = original_print
        output_file.close()
        
        return 0

    except FileNotFoundError as e:
        builtins.print = original_print
        output_file.close()
        print(f"Error: File not found: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        builtins.print = original_print
        output_file.close()
        print(f"Error: Runtime error during profiling: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        builtins.print = original_print
        output_file.close()
        print(f"Error: Unexpected error: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
