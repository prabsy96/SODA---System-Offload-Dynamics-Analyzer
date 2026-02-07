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
import json 
from datetime import datetime 
from transformers import AutoConfig
from soda import ModelTracer, SodaAnalyzer
from soda.common import utils
from experiments.sweep.summarize_soda_sweep import summarize as summarize_soda_sweep
from experiments.sweep.config import PARAMS, PREF_SWEEP_CONFIG, DEC_SWEEP_CONFIG, DEBUG_SWEEP_CONFIG, FP8_SWEEP_CONFIG

def ensure_env_loaded() -> None:
    """Exit early if env.sh was not sourced."""
    if not os.environ.get("SODA_ENV_LOADED"):
        print("Error: SODA environment not loaded.", file=sys.stderr)
        print("Please run: source env.sh", file=sys.stderr)
        sys.exit(1)


def get_gpu_suffix() -> str:
    """Returns a short GPU suffix (e.g., H100, A100) or 'cpu'."""
    if not torch.cuda.is_available():
        return "cpu"
    name = torch.cuda.get_device_name(0)
    if "H100" in name: return "H100"
    if "H200" in name: return "H200"
    if "A100" in name: return "A100"
    if "V100" in name: return "V100"
    if "T4" in name: return "T4"     # ADD THIS LINE
    if "L4" in name: return "L4"     # ADD THIS LINE (for the L4 node)
    if "4090" in name: return "4090"
    return "gpu"

def main() -> None:
    ensure_env_loaded()

    sweep_roots = set()
    gpu_suffix = get_gpu_suffix()

    compile_type = PARAMS["compile_type"]
    precision = PARAMS["precision"]
    device = PARAMS["device"]
    warmup = PARAMS["inference_warmup"]

    # select config from env variable 
    config_type = os.environ.get("SODA_SWEEP_CONFIG", "prefill").lower()
    if config_type == "decode":
        SWEEP_CONFIG = DEC_SWEEP_CONFIG
    elif config_type == "debug":
        SWEEP_CONFIG = DEBUG_SWEEP_CONFIG
    elif config_type == "fp8":
        # Verify H100/H200 for FP8
        if gpu_suffix not in ("H100", "H200"):
            print(f"Error: FP8 requires H100/H200 GPU. Detected: {gpu_suffix}", file=sys.stderr)
            sys.exit(1)
        SWEEP_CONFIG = FP8_SWEEP_CONFIG
        # FIX: Update precision variable so the sweep root directory is named correctly
        precision = "float8_e4m3fn"
    elif config_type == "all":
        SWEEP_CONFIG = dict(PREF_SWEEP_CONFIG)
        for k, v in DEC_SWEEP_CONFIG.items():
            key = f"{k}_dec" if k in SWEEP_CONFIG else k
            SWEEP_CONFIG[key] = v
    else:
        SWEEP_CONFIG = PREF_SWEEP_CONFIG

    model_filter = os.environ.get("SODA_SWEEP_MODEL")
    if model_filter:
        models = [m.strip() for m in model_filter.split(",")]
        filtered = {}
        for m in models:
            if m in SWEEP_CONFIG:
                filtered[m] = SWEEP_CONFIG[m]
            else:
                print(f"Warning: Model '{m}' not found in {config_type} config. Available: {list(SWEEP_CONFIG.keys())}", file=sys.stderr)
        if not filtered:
            print(f"Error: No valid models found. Exiting.", file=sys.stderr)
            sys.exit(1)
        SWEEP_CONFIG = filtered
    
    print(f"Running sweep config: {config_type}")
    print(f"Models: {list(SWEEP_CONFIG.keys())}")


    for config_name, cfg in SWEEP_CONFIG.items():
        model = cfg["model_name"]
        batch_sizes = cfg["batch_sizes"]
        seq_lens = cfg["seq_lens"]
        max_new_toks = cfg["max_new_toks"]

        run_precision = cfg.get("precision", precision)

        try:
            model_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
            max_pos = getattr(model_config, 'max_position_embeddings',
                    getattr(model_config, 'n_positions',
                    getattr(model_config, 'n_ctx', float('inf'))))
        except Exception as e:
            print(f"Warning: Could not load config for {model}: {e}. Skipping seq_len validation.")
            max_pos = float('inf')

        # Group sweep outputs under a common prefix (model + compile_type + precision)
        max_tok_str = f"mt{max_new_toks[0]}"
        sweep_root = Path(os.environ.get("SODA_OUTPUT", "output")) / f"{model.replace('/', '_')}_{compile_type}_{run_precision}_{max_tok_str}_{gpu_suffix}"
        sweep_roots.add(sweep_root)
        print(f"\n=== Running config: {config_name} ({model}) with precision={run_precision}, max_pos={max_pos} ===")

        for bs, sl, max_new_tokens in product(batch_sizes, seq_lens, max_new_toks):
            # Before running each sweep point, check if seq_len is valid for this model
            if sl > max_pos:
                print(f"Skipping bs={bs}, sl={sl}: exceeds model max_position_embeddings ({max_pos})")
                continue
            
            

            print(f"\n\n\n=== Running sweep point: batch_size={bs}, seq_len={sl}, max_new_tokens={max_new_tokens} ===")
            exp_name = utils.generate_experiment_name(model, compile_type, run_precision, bs, sl, max_new_tokens)
            cli_args = [
                "--model", model,
                "--output-dir", str(sweep_root),
                "--batch-size", str(bs),
                "--seq-len", str(sl),
                "--max-new-tokens", str(max_new_tokens),
                "--precision", run_precision,
                "--compile-type", compile_type,
                "--device", device,
                "--warmup", warmup,
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
                    
                    # Generate a report.json that MATCHES soda.py structure
                    run_output_dir = sweep_root / exp_name
                    run_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    oom_report = {
                        "metadata": {
                            "model_name": model,
                            "timestamp": datetime.now().isoformat(),
                            "config": {
                                "batch_size": bs,
                                "seq_len": sl,
                                "max_new_tokens": max_new_tokens,
                                "precision": run_precision,
                                "compile_type": compile_type,
                                "device": device,
                                "gpu_name": gpu_suffix
                            }
                        },
                        "performance_metrics": {
                            "inference_time_ms": "OOM",
                            "error": str(e)
                        }
                    }
                    
                    with open(run_output_dir / "report.json", "w") as f:
                        json.dump(oom_report, f, indent=4)

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise

        # === FIX START: Generate summary immediately after this model finishes ===
        if sweep_root.exists():
            print(f"\n=== Generating summary for {sweep_root} ===")
            try:
                summarize_soda_sweep(sweep_root, gpu_name_override=gpu_suffix)
                print(f"Summary generated in {sweep_root}/summary")
            except Exception as e:
                print(f"Warning: Could not generate summary for {sweep_root}: {e}")
                import traceback
                traceback.print_exc()
        # === FIX END ===

    # Remove the redundant summary blocks at the end of main()
    print("\n=== Sweep Completed ===")


if __name__ == "__main__":
    main()
