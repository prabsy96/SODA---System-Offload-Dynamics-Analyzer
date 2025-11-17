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

def main() -> int:
    """Main entry point for the SODA CLI."""

    # Check if env.sh has been sourced and loaded
    if not os.environ.get("SODA_ENV_LOADED"):
        # Use stderr for early errors before logger is set up
        print("Error: SODA environment not loaded.", file=sys.stderr)
        print("Please run: source env.sh", file=sys.stderr)
        sys.exit(1)
    
    # Parse and validate arguments 
    args = SodaProfiler.parse_and_validate_args()

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
