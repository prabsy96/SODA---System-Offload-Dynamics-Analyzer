#!/usr/bin/env python3
"""
SodaMicrobench sweep helper 

Runs a small grid over batch sizes, sequence lengths, and maximum new tokens
using the SodaAnalyzer + ModelTracer pipeline with an HF model.
"""

import os
import sys
from itertools import product
from pathlib import Path
import torch
from soda import ModelTracer
from soda.microbench.microbench import SodaMicrobench
from soda.common import utils
from experiments.sweep.config import PARAMS, PREF_SWEEP_CONFIG, DEC_SWEEP_CONFIG, DEBUG_SWEEP_CONFIG  # Fixed import names


def get_gpu_name() -> str:
    """Get sanitized GPU name for output directory."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_name = gpu_name.replace("NVIDIA ", "").replace(" ", "_")
        for key in ["H100", "H200", "A100", "V100", "A10", "L40", "RTX"]:
            if key in gpu_name.upper():
                return key
        return gpu_name.split()[0]
    return "unknown_gpu"


def ensure_env_loaded() -> None:
    """Exit early if env.sh was not sourced."""
    if not os.environ.get("SODA_ENV_LOADED"):
        print("Error: SODA environment not loaded.", file=sys.stderr)
        print("Please run: source env.sh", file=sys.stderr)
        sys.exit(1)


def get_sweep_configs():
    """
    Select sweep configs based on environment variables.
    
    SODA_SWEEP_CONFIG: "prefill", "decode", "debug", or "all" (default: "prefill")
    SODA_SWEEP_MODEL: Specific model key to run, or "all" (default: "all")
    """
    config_type = os.environ.get("SODA_SWEEP_CONFIG", "prefill").lower()
    model_filter = os.environ.get("SODA_SWEEP_MODEL", "all")
    
    # Select config set based on type
    if config_type == "decode":
        base_configs = DEC_SWEEP_CONFIG
    elif config_type == "debug":
        base_configs = DEBUG_SWEEP_CONFIG
    elif config_type == "all":
        # Merge all configs
        base_configs = {}
        base_configs.update(PREF_SWEEP_CONFIG)
        base_configs.update({f"dec_{k}": v for k, v in DEC_SWEEP_CONFIG.items()})
    else:  # default to prefill
        base_configs = PREF_SWEEP_CONFIG
    
    # Filter by model if specified
    if model_filter != "all":
        if model_filter in base_configs:
            return {model_filter: base_configs[model_filter]}
        else:
            print(f"Warning: Model '{model_filter}' not found in {config_type} configs.", file=sys.stderr)
            print(f"Available models: {list(base_configs.keys())}", file=sys.stderr)
            sys.exit(1)
    
    return base_configs


def main() -> None:
    ensure_env_loaded()

    compile_type = PARAMS["compile_type"]
    precision = PARAMS["precision"]
    device = PARAMS["device"]
    warmup = PARAMS["microbench_warmup"]
    runs = PARAMS["microbench_runs"]

    gpu_name = get_gpu_name()
    print(f"Detected GPU: {gpu_name}")

    # Get configs from environment variables
    SWEEP_CONFIGS = get_sweep_configs()
    
    config_type = os.environ.get("SODA_SWEEP_CONFIG", "prefill")
    model_filter = os.environ.get("SODA_SWEEP_MODEL", "all")
    print(f"Sweep config: {config_type}")
    print(f"Model filter: {model_filter}")
    print(f"Running {len(SWEEP_CONFIGS)} model configurations: {list(SWEEP_CONFIGS.keys())}")

    for config_name, cfg in SWEEP_CONFIGS.items():
        model = cfg["model_name"]
        batch_sizes = cfg["batch_sizes"]
        seq_lens = cfg["seq_lens"]
        max_new_toks = cfg["max_new_toks"]
        
        # Allow per-model precision override (e.g., for FP8)
        model_precision = cfg.get("precision", precision)

        sweep_root = (
            Path(os.environ.get("SODA_OUTPUT", "output")) 
            / f"{gpu_name}_microbench"
            / f"{model.replace('/', '_')}_{compile_type}_{model_precision}"
        )

        total_runs = len(batch_sizes) * len(seq_lens) * len(max_new_toks)
        print(f"\n{'='*60}")
        print(f"Config: {config_name} ({total_runs} sweep points)")
        print(f"Model: {model}")
        print(f"Batch sizes: {batch_sizes}")
        print(f"Sequence lengths: {seq_lens}")
        print(f"Max new tokens: {max_new_toks}")
        print(f"{'='*60}")

        run_count = 0
        for bs, sl, max_new_tokens in product(batch_sizes, seq_lens, max_new_toks):
            run_count += 1
            print(f"\n[{run_count}/{total_runs}] {config_name}: bs={bs}, sl={sl}, max_new_tokens={max_new_tokens}")

            exp_name = utils.generate_experiment_name(model, compile_type, model_precision, bs, sl, max_new_tokens)
            run_dir = sweep_root / exp_name
            
            # Skip if already completed
            taxbreak_json = run_dir / "microbench" / "taxbreak.json"
            if taxbreak_json.exists():
                print(f"  Skipping (already completed): {run_dir}")
                continue

            cli_args = [
                "--model", model,
                "--output-dir", str(sweep_root),
                "--batch-size", str(bs),
                "--seq-len", str(sl),
                "--max-new-tokens", str(max_new_tokens),
                "--precision", model_precision,
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
                
                print(f"  Completed: {run_dir}")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM Error - Skipping: {e}", file=sys.stderr)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)
                continue


if __name__ == "__main__":
    main()