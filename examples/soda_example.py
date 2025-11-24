#!/usr/bin/env python3
"""
Example script showing how to use SODA.

This demonstrates how to import and use SodaAnalyzer and ModelTracer
directly in Python code instead of using the CLI.
"""

import os
import sys

from soda import SodaAnalyzer, ModelTracer, SodaLogger
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
        "--precision", "float32",
        "--batch-size", "1",
        "--seq-len", "16",
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
    #     precision="float32",
    #     batch_size=1,
    #     seq_len=16,
    #     fusion=[2, 3],
    #     prox_score=1.0,
    #     seed=42
    # )

    # Create tracer (derives experiment/output paths internally)
    tracer = ModelTracer(args=args)
    
    # Setup logger for tracer
    SodaLogger(tracer.output_dir, is_console=True, is_file=True)
    
    tracer.run()

    # Create analyzer and analyze (all analyzer operations)
    profiler = SodaAnalyzer(tracer=tracer, args=args)
    profiler.analyze()
    
    # Report and save results
    profiler.report()
    profiler.save()

if __name__ == "__main__":
    main()

