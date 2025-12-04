#!/usr/bin/env python3
"""
SodaAnalyzer sweep helper

Runs a small grid over batch sizes, sequence lengths, and maximum new tokens
using the SodaAnalyzer + ModelTracer pipeline with an HF model. Swap the lists
below to try different shapes or precisions.
"""

import os
import sys
from argparse import Namespace
from itertools import product
from pathlib import Path

import torch

from soda import ModelTracer, SodaAnalyzer, SodaLogger
from soda.common import utils

# Each config declares the model to test and the BS/SL sweeps to run.
CONFIGS = {
    "gpt2_short_ctx": {
        "model_name": "gpt2",
        # "batch_sizes": sorted([1, 2, 4, 8], reverse=True),
        # "seq_lens": sorted([128, 256, 512, 1024], reverse=True),
        "batch_sizes": sorted([1, 2], reverse=True),
        "seq_lens": sorted([128, 256], reverse=True),
        "max_new_toks": [1],
    },
    "llama_3.2_1b_short_ctx": {
        "model_name": "meta-llama/Llama-3.2-1B",
        "batch_sizes": sorted([1, 2, 4, 8, 16], reverse=True),
        "seq_lens": sorted([128, 256, 512, 1024, 2048], reverse=True),
        "max_new_toks": [1],
    },
    # "tinyllama_1.1b": {
    #     "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    #     "batch_sizes": sorted([1, 2, 4, 8, 16], reverse=True),
    #     "seq_lens": sorted([128, 256, 512, 1024, 2048], reverse=True),
    #     "max_new_toks": [1],
    # },
}


def ensure_env_loaded() -> None:
    """Exit early if env.sh was not sourced."""
    if not os.environ.get("SODA_ENV_LOADED"):
        print("Error: SODA environment not loaded.", file=sys.stderr)
        print("Please run: source env.sh", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    ensure_env_loaded()

    compile_type = "eager"
    precision = "bfloat16"

    for config_name, cfg in CONFIGS.items():
        model = cfg["model_name"]
        batch_sizes = cfg["batch_sizes"]
        seq_lens = cfg["seq_lens"]
        max_new_toks = cfg["max_new_toks"]

        # Group sweep outputs under a common prefix (model + compile_type + precision)
        sweep_root = Path(os.environ.get("SODA_OUTPUT", "output")) / f"{model.replace('/', '_')}_{compile_type}_{precision}"
        print(f"\n=== Running config: {config_name} ({model}) ===")

        for bs, sl, max_new_tokens in product(batch_sizes, seq_lens, max_new_toks):
            print(f"\n=== Running sweep point: batch_size={bs}, seq_len={sl}, max_new_tokens={max_new_tokens} ===")
            exp_name = utils.generate_experiment_name(model, compile_type, precision, bs, sl, max_new_tokens)
            cli_args = [
                "--model", model,
                "--output-dir", str(sweep_root),
                "--batch-size", str(bs),
                "--seq-len", str(sl),
                "--max-new-tokens", str(max_new_tokens),
                "--precision", precision,
                "--compile-type", compile_type,
                "--device", "cuda",
                "--warmup", "5",
                # "--runs", "5000",
            ]
            args = utils.parse_and_validate_args(cli_args)

            try:
                tracer = ModelTracer(args=args)
                SodaLogger(tracer.output_dir, is_console=True, is_file=True)
                tracer.run()

                analyzer = SodaAnalyzer(tracer=tracer, args=args)
                report_path = analyzer.run()
                print(f"Report saved to: {report_path}")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"Skipping bs={bs}, sl={sl}, max_new_tokens={max_new_tokens} due to OOM: {e}", file=sys.stderr)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise


if __name__ == "__main__":
    main()
