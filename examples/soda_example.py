#!/usr/bin/env python3
"""
Example script showing how to use SODA.

This demonstrates how to import and use SodaProfiler and ModelHandler
directly in Python code instead of using the CLI.
"""

import os
import sys

from soda import SodaProfiler, ModelHandler
import utils

def main():
    """Example usage of SODA programmatically."""
    
    # Check if SODA environment is loaded
    if not os.environ.get("SODA_ENV_LOADED"):
        print("Error: SODA environment not loaded.", file=sys.stderr)
        print("Please run: source env.sh", file=sys.stderr)
        sys.exit(1)
    
    # Parse arguments (or create a simple namespace object)
    # Option 1: Use the built-in argument parser
    args = utils.parse_and_validate_args([
        "--model", "gpt2",
        "--output-dir", os.environ.get("SODA_OUTPUT", "."),
        "--compile-type", "eager",
        "--device", "cuda",
        "--precision", "float16",
        "--batch-size", "1",
        "--seq-len", "128",
        "--fusion", "2", "3",
        "--prox-score", "1.0",
    ])
    
    # Option 2: Create a simple namespace object manually
    # from types import SodaNamespace
    # args = SodaNamespace(
    #     model="gpt2",
    #     output_dir=Path(os.environ.get("SODA_OUTPUT", ".")),
    #     compile_type="eager",
    #     device="cuda",
    #     precision="float16",
    #     batch_size=1,
    #     seq_len=128,
    #     fusion=[2, 3],
    #     prox_score=1.0,
    #     seed=42
    # )
    
    # Prepare model handler
    model_handler = ModelHandler(
        model_name=args.model,
        device=args.device,
        compile_type=args.compile_type,
        precision=args.precision,
    )
    
    # Generate synthetic inputs
    model_inputs = model_handler.generate_synthetic_inputs(
        args.batch_size, args.seq_len
    )
    
    # Initialize profiler
    profiler = SodaProfiler(
        model_handler=model_handler, 
        args=args, 
        log_console=True, 
        log_file=True
    )
    
    # Profile forward pass and analyze
    profiler.profile_forward_pass(model_inputs)
    profiler.analyze()
    
    # Report and save results
    profiler.report()
    profiler.save()
    
    # Cleanup and exit
    profiler.exit()

if __name__ == "__main__":
    main()

