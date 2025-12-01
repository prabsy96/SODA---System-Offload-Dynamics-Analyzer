#!/usr/bin/env python3
"""
Microbench sweep helper for SODA.

Profiles a small grid of batch sizes and sequence lengths, then runs the
microbenchmark pipeline (PyTorch + baremetal GEMM checks) for each
point. Start with small grids; the baremetal steps can be long-running.
"""

import os
import sys
from argparse import Namespace
from itertools import product
from pathlib import Path

from soda import ModelTracer, SodaLogger
from soda_microbench import SodaMicrobench
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
    batch_sizes = [1]
    seq_lens = [128, 256]
    warmup = 1000
    runs = 5000

    for bs, sl in product(batch_sizes, seq_lens):
        print(f"\n=== Microbench sweep point: batch_size={bs}, seq_len={sl} ===")

        # Edit this list directly to tweak experiment parameters
        cli_args = [
            "--model", model,
            "--batch-size", str(bs),
            "--seq-len", str(sl),
            "--precision", "bfloat16",
            "--compile-type", "eager",
            "--device", "cuda",
            "--output-dir", os.environ.get("SODA_OUTPUT", "."),
            "--microbench",
            "--warmup", str(warmup),
            "--runs", str(runs),
        ]
        args = utils.parse_and_validate_args(cli_args)

        # If you want to bypass CLI parsing, uncomment and edit the namespace
        # below. All supported args are included for convenience.
        #
        # args = Namespace(
        #     model=model,
        #     output_dir=Path(os.environ.get("SODA_OUTPUT", ".")),
        #     compile_type="eager",
        #     device="cuda",
        #     precision="bfloat16",
        #     batch_size=bs,
        #     seq_len=sl,
        #     fusion=None,
        #     prox_score=1.0,
        #     seed=42,
        #     microbench=True,
        #     runs=runs,
        #     warmup=warmup,
        # )

        tracer = ModelTracer(args=args)
        SodaLogger(tracer.output_dir, is_console=True, is_file=True)
        tracer.run()

        microbench = SodaMicrobench(tracer=tracer, args=args)
        microbench.run()


if __name__ == "__main__":
    main()
