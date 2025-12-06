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

from soda import ModelTracer, SodaAnalyzer, SodaLogger
from soda.common import utils
from experiments.sweep.summarize_soda_sweep import summarize as summarize_soda_sweep

# Each config declares the model to test and the BS/SL sweeps to run.

## PREFILL

# CONFIGS = {
    
#     # "gpt2_short_ctx": {
#     #     "model_name": "gpt2",
#     #     # "batch_sizes": sorted([1, 2, 4, 8], reverse=True),
#     #     # "seq_lens": sorted([128, 256, 512, 1024], reverse=True),
#     #     # FIXME: DEBUG ONLY
#     #     "batch_sizes": sorted([1, 2], reverse=True),
#     #     "seq_lens": sorted([128, 256], reverse=True),
#     #     "max_new_toks": [1],
#     # },
#     # "llama_3.2_1b_short_ctx": {
#     #     "model_name": "meta-llama/Llama-3.2-1B",
#     #     # FIXME: DEBUG ONLY
#     #     # "batch_sizes": sorted([1, 2, 4, 8, 16], reverse=True),
#     #     # "seq_lens": sorted([128, 256, 512, 1024, 2048], reverse=True),
#     #     "batch_sizes": sorted([1, 2, 4, 8, 16], reverse=True),
#     #     "seq_lens": sorted([512, 1024, 2048, 4096, 8192], reverse=True),
#     #     "max_new_toks": [1],
#     # },
#     # "tinyllama_1.1b": {
#     #     "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     #     "batch_sizes": sorted([1, 2, 4, 8, 16], reverse=True),
#     #     "seq_lens": sorted([128, 256, 512, 1024, 2048], reverse=True),
#     #     "max_new_toks": [1],
#     # },
#     # "deepseek_moe_16b_prefill": {
#     #     "model_name": "deepseek-ai/deepseek-moe-16b-base",
#     #     "batch_sizes": sorted([1, 2, 4, 8], reverse=True),  # Lower due to MoE memory
#     #     "seq_lens": sorted([512, 1024, 2048, 4096], reverse=True),  # Reduced max seq_len
#     #     "max_new_toks": [1],
#     # },
#     "whisper_large_v3_prefill": {
#         "model_name": "openai/whisper-large-v3",
#         "batch_sizes": sorted([1, 2, 4, 8], reverse=True),
#         # Sweep Audio Durations: 30s (Standard), 10s, 2s
#         # Note: Whisper pads to 30s internally, so these might have similar runtime
#         "seq_lens": [480000, 160000, 32000], 
#         "max_new_toks": [1], 
#     },
# }

# DECODE

CONFIGS = {

#     # "gpt2_short_ctx": {
#     #     "model_name": "gpt2",
#     #     # "batch_sizes": sorted([1, 2, 4, 8], reverse=True),
#     #     # "seq_lens": sorted([128, 256, 512, 1024], reverse=True),
#     #     # FIXME: DEBUG ONLY
#     #     "batch_sizes": sorted([1, 2], reverse=True),
#     #     "seq_lens": sorted([128, 256], reverse=True),
#     #     "max_new_toks": [1],
#     # },
#     # "llama_3.2_1b_decode_ispass": {
#     #     "model_name": "meta-llama/Llama-3.2-1B",
#     #     # ISPASS "Tax" Zone: Batch 1-4 is where Framework Tax kills performance.
#     #     # We sweep 1, 2, 4, 8, 16, 32 to show the curve.
#     #     "batch_sizes": [1, 2, 4, 8, 16, 32], 
        
#     #     # Simulating different KV Cache fill levels
#     #     "seq_lens": [128, 1024, 2048, 4096], 

#     #     "max_new_toks": [10], 
#     # },
#     "deepseek_moe_16b_decode": {
#         "model_name": "deepseek-ai/deepseek-moe-16b-base",  # or "deepseek-ai/DeepSeek-V2-Lite"
#         "batch_sizes": [1, 2, 4, 8, 16],  # Smaller batch sizes due to MoE memory overhead
#         "seq_lens": [128, 1024, 2048, 4096],
#         "max_new_toks": [10],
#     },
    "whisper_large_v3_prefill": {
        "model_name": "openai/whisper-large-v3",
        "batch_sizes": sorted([1, 2, 4, 8], reverse=True),
        # Sweep Audio Durations: 30s (Standard), 10s, 2s
        # Note: Whisper pads to 30s internally, so these might have similar runtime
        "seq_lens": [480000, 160000, 32000], 
        "max_new_toks": [50], 
    },
}


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

    compile_type = "eager"
    precision = "bfloat16"
    sweep_roots = set()
    gpu_suffix = get_gpu_suffix()

    for config_name, cfg in CONFIGS.items():
        model = cfg["model_name"]
        batch_sizes = cfg["batch_sizes"]
        seq_lens = cfg["seq_lens"]
        max_new_toks = cfg["max_new_toks"]

        # Group sweep outputs under a common prefix (model + compile_type + precision)
        max_tok_str = f"mt{max_new_toks[0]}"  # Use the first value since it's typically a single-element list
        sweep_root = Path(os.environ.get("SODA_OUTPUT", "output")) / f"{model.replace('/', '_')}_{compile_type}_{precision}_{max_tok_str}_{gpu_suffix}"
        sweep_roots.add(sweep_root)
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
                # Extra parser knobs (fusion + microbench) left at defaults:
                # "--fusion", "2",
                # "--prox-score", "1.0",
                # "--seed", "42",
                # "--microbench",
                # "--warmup", "10",
                # "--runs", "5",
                # "--version",
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

    if sweep_roots:
        print("\n=== Summarizing completed sweep directories ===")
        for root in sorted(sweep_roots, key=lambda p: str(p)):
            if not root.exists():
                print(f"Skipping summary for {root}: path does not exist", file=sys.stderr)
                continue
            
            # Extract max_tok_str from the folder name
            # e.g., "meta-llama_Llama-3.2-1B_eager_bfloat16_mt10_T4" -> "mt10"
            folder_name = root.name
            max_tok_str = None
            for part in folder_name.split("_"):
                if part.startswith("mt"):
                    max_tok_str = part
                    break
            
            try:
                # PASS both gpu_suffix and max_tok_str
                summarize_soda_sweep(root, gpu_name_override=gpu_suffix, max_tok_override=max_tok_str)
            except RuntimeError as exc:
                print(f"Failed to summarize {root}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
