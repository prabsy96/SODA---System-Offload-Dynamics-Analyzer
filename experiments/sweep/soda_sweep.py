#!/usr/bin/env python3
"""
SodaAnalyzer sweep helper

Runs a small grid over batch sizes, sequence lengths, and maximum new tokens
using the SodaAnalyzer + ModelTracer pipeline with an HF model. Swap the lists
below to try different shapes or precisions.
"""

import os
import sys
from itertools import product
from pathlib import Path

import torch

from soda import ModelTracer, SodaAnalyzer
from soda.common import utils
from experiments.sweep.summarize_soda_sweep import summarize as summarize_soda_sweep
from experiments.sweep.config import PARAMS, SWEEP_CONFIGS

def ensure_env_loaded() -> None:
    """Exit early if env.sh was not sourced."""
    if not os.environ.get("SODA_ENV_LOADED"):
        print("Error: SODA environment not loaded.", file=sys.stderr)
        print("Please run: source env.sh", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    ensure_env_loaded()

    sweep_roots = set()

    compile_type = PARAMS["compile_type"]
    precision = PARAMS["precision"]
    device = PARAMS["device"]
    warmup = PARAMS["inference_warmup"]

    for config_name, cfg in SWEEP_CONFIGS.items():
        model = cfg["model_name"]
        batch_sizes = cfg["batch_sizes"]
        seq_lens = cfg["seq_lens"]
        max_new_toks = cfg["max_new_toks"]

        # Group sweep outputs under a common prefix (model + compile_type + precision)
        sweep_root = Path(os.environ.get("SODA_OUTPUT", "output")) / f"{model.replace('/', '_')}_{compile_type}_{precision}"
        sweep_roots.add(sweep_root)
        print(f"\n=== Running config: {config_name} ({model}) ===")

        for bs, sl, max_new_tokens in product(batch_sizes, seq_lens, max_new_toks):
            print(f"\n\n\n=== Running sweep point: batch_size={bs}, seq_len={sl}, max_new_tokens={max_new_tokens} ===")
            exp_name = utils.generate_experiment_name(model, compile_type, precision, bs, sl, max_new_tokens)
            cli_args = [
                "--model", model,
                "--output-dir", str(sweep_root),
                "--batch-size", str(bs),
                "--seq-len", str(sl),
                "--max-new-tokens", str(max_new_tokens),
                "--precision", precision,
                "--compile-type", compile_type,
                "--device", device,
                "--warmup", warmup,
                # Extra parser knobs (fusion + microbench) left at defaults:
                # "--fusion", "2",
                # "--prox-score", "1.0",
                # "--seed", "42",
                # "--version",

                # NOTE: for SodaAnalyzer
                # "--microbench", # DONOT use microbench 
                # "--warmup", "10", # 10 is ok 
                # "--runs", "5", # This doesn't matter 
            ]
            args = utils.parse_and_validate_args(cli_args)

            try:
                tracer = ModelTracer(args=args)
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

    if sweep_roots:
        print("\n=== Summarizing completed sweep directories ===")
        for root in sorted(sweep_roots, key=lambda p: str(p)):
            if not root.exists():
                print(f"Skipping summary for {root}: path does not exist", file=sys.stderr)
                continue
            try:
                summarize_soda_sweep(root)
            except RuntimeError as exc:
                print(f"Failed to summarize {root}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
