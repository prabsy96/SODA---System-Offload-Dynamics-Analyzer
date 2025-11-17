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
from .util import SodaProfiler, ModelHandler

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
    
    # Setup logging: redirect print to both stdout and log file
    log_path, output_file, original_print = setup_logging(args.output_dir)

    try:
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
            args.batch_size, args.seq_len, args.device
        )

        # Tracing and Analysis
        profiler = SodaProfiler(model_handler=model_handler, args=args)
        print("Profiling model forward pass...")
        trace_file_path = profiler.profile_forward_pass(model_inputs)
        print(f"Chrome trace file generated at: {trace_file_path}")

        # Data Processing and Reporting
        print("Analyzing trace data to generate reports...")
        
        # Run analysis
        profiler.analyze()
        
        # Report results
        profiler.report()
        
        # Save results to JSON
        profiler.save()
        
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
