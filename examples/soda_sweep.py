#!/usr/bin/env python3
"""
Quick sweep helper for SODA.

Runs a small grid over batch sizes and sequence lengths using the
SodaAnalyzer + ModelTracer pipeline with an HF model. Swap the lists
below to try different shapes or precisions.
"""

import os
import sys
from argparse import Namespace
from itertools import product
from pathlib import Path

from soda import ModelTracer, SodaAnalyzer, SodaLogger
import utils


def ensure_env_loaded() -> None:
    """Exit early if env.sh was not sourced."""
    if not os.environ.get("SODA_ENV_LOADED"):
        print("Error: SODA environment not loaded.", file=sys.stderr)
        print("Please run: source env.sh", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    ensure_env_loaded()

    model = "meta-llama/Llama-3.2-3B"
    batch_sizes = [1, 2, 4, 8, 16, 32]
    seq_lens = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    for bs, sl in product(batch_sizes, seq_lens):
        print(f"\n=== Running sweep point: batch_size={bs}, seq_len={sl} ===")
        # Edit this list directly to tweak model/device/precision per experiment.
        cli_args = [
            "--model", model,
            "--batch-size", str(bs),
            "--seq-len", str(sl),
            "--precision", "bfloat16",
            "--compile-type", "eager",
            "--device", "cuda",
            "--warmup", "1000",
            "--runs", "5000",
        ]
        args = utils.parse_and_validate_args(cli_args)

        tracer = ModelTracer(args=args)
        SodaLogger(tracer.output_dir, is_console=True, is_file=True)
        tracer.run()

        analyzer = SodaAnalyzer(tracer=tracer, args=args)
        report_path = analyzer.run()
        print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
