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
        import logging
        
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

def main() -> int:
    """Main entry point for the SODA CLI."""

    # Check if env.sh has been sourced and loaded
    if not os.environ.get("SODA_ENV_LOADED"):
        # Use stderr for early errors before logger is set up
        print("Error: SODA environment not loaded.", file=sys.stderr)
        print("Please run: source env.sh", file=sys.stderr)
        sys.exit(1)
    
    # Parse and validate arguments
    parser = get_args_parser()
    args = parser.parse_args()
    validate_args(args)

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
