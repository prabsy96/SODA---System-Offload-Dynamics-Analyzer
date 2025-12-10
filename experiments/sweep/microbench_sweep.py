#!/usr/bin/env python3
"""
SodaMicrobench sweep helper 

Runs a small grid over batch sizes, sequence lengths, and maximum new tokens
using the SodaAnalyzer + ModelTracer pipeline with an HF model. Swap the lists
below to try different shapes or precisions.
"""

import os
import sys
from itertools import product
from pathlib import Path
import torch
from soda import ModelTracer
from soda.microbench.microbench import SodaMicrobench
from soda.common import utils
from experiments.sweep.config import PARAMS, PREF_SWEEP_CONFIGS, DEC_SWEEP_CONFIGS, DEBUG_SWEEP_CONFIGS

def ensure_env_loaded() -> None:
    """Exit early if env.sh was not sourced."""
    if not os.environ.get("SODA_ENV_LOADED"):
        print("Error: SODA environment not loaded.", file=sys.stderr)
        print("Please run: source env.sh", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    ensure_env_loaded()

    compile_type = PARAMS["compile_type"]
    precision = PARAMS["precision"]
    device = PARAMS["device"]
    warmup = PARAMS["microbench_warmup"]
    runs = PARAMS["microbench_runs"]

    # Select the sweep configs to use
    SWEEP_CONFIGS = PREF_SWEEP_CONFIGS
    # SWEEP_CONFIGS = DEC_SWEEP_CONFIGS
    # SWEEP_CONFIGS = DEBUG_SWEEP_CONFIGS

    for config_name, cfg in SWEEP_CONFIGS.items():
        model = cfg["model_name"]
        batch_sizes = cfg["batch_sizes"]
        seq_lens = cfg["seq_lens"]
        max_new_toks = cfg["max_new_toks"]

        sweep_root = Path(os.environ.get("SODA_OUTPUT", "output")) / f"{model.replace('/', '_')}_{compile_type}_{precision}"

        for bs, sl, max_new_tokens in product(batch_sizes, seq_lens, max_new_toks):
            print(f"\n=== Microbench sweep {config_name}: batch_size={bs}, seq_len={sl}, max_new_tokens={max_new_tokens} ===")

            exp_name = utils.generate_experiment_name(model, compile_type, precision, bs, sl, max_new_tokens)
            run_dir = sweep_root / exp_name
            cli_args = [
                "--model", model,
                "--output-dir", str(sweep_root),
                "--batch-size", str(bs),
                "--seq-len", str(sl),
                "--max-new-tokens", str(max_new_tokens),
                "--precision", precision,
                "--compile-type", compile_type,
                "--device", device,
                "--microbench",
                "--warmup", warmup,
                "--runs", runs,
            ]
            args = utils.parse_and_validate_args(cli_args)

            try:
                tracer = ModelTracer(args=args)
                tracer.run()

                microbench = SodaMicrobench(tracer=tracer, args=args)
                microbench.run()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"Skipping {config_name} bs={bs}, sl={sl}, max_new_tokens={max_new_tokens} due to OOM: {e}", file=sys.stderr)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise


if __name__ == "__main__":
    main()
