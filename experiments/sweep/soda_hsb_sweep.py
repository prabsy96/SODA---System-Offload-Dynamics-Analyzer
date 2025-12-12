"""
SODA HSB (Hardware-Software Balance) Sweep

HSB = 1 - (T_Exposed / T_Structural)

Where:
- T_Exposed: Total GPU idle time during inference (gaps in GPU execution)
- T_Structural: Sum of per-kernel framework overheads from taxbreak LUT

    
Environment variables:
    SODA_SWEEP_CONFIG: "prefill", "decode", "debug", "all" (default: "prefill")
    SODA_SWEEP_MODEL: Comma-separated list of model configs to run
    SODA_OUTPUT: Output directory (default: "output")
"""

import os
import sys
import json
from itertools import product
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import torch

from soda import ModelTracer, SodaAnalyzer
from soda.common import utils
from soda.microbench.microbench import SodaMicrobench
from experiments.sweep.config import PARAMS, PREF_SWEEP_CONFIG, DEC_SWEEP_CONFIG, DEBUG_SWEEP_CONFIG
from experiments.sweep.hsb import (
    load_taxbreak_lut,
    calculate_t_exposed,
    calculate_t_structural,
    calculate_hsb,
    HSBResult,
)
from experiments.sweep.summarize_hsb_sweep import summarize_hsb_sweep

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
    for key in ["H200", "H100", "A100", "V100", "T4", "L4", "4090", "3090"]:
        if key in name:
            return key
    return "gpu"

def run_single_experiment( args, taxbreak_lut: Dict[str, float],) -> Tuple[Optional[HSBResult], Optional[Dict[str, Any]]]:
    """
    Run a single experiment and compute HSB metrics.
    
    Args:
        args: Parsed command line arguments
        taxbreak_lut: Pre-loaded taxbreak lookup table
    
    Returns:
        Tuple of (HSBResult, full_report_dict) or (None, None) on failure
    """

    try:
        # Run tracing
        tracer = ModelTracer(args=args)
        tracer.run()
        
        # Run analysis
        analyzer = SodaAnalyzer(tracer=tracer, args=args)
        results = analyzer.analyze()
        
        # Extract metrics for HSB calculation
        metrics = results.get("metrics", {})
        sequences = results.get("sequences", [])
        
        # Calculate T_Exposed (GPU idle time)
        inference_time_us = utils.ms_to_us(metrics.get("inference_time_ms", 0))
        gpu_busy_time_us = utils.ms_to_us(metrics.get("gpu_busy_time_ms", 0))
        t_exposed = calculate_t_exposed(inference_time_us, gpu_busy_time_us)
        
        # Calculate T_Structural (sum of framework overheads)
        t_structural = calculate_t_structural(sequences, taxbreak_lut)
        
        # Calculate HSB
        hsb = calculate_hsb(t_exposed, t_structural)
        
        # Build HSB result
        hsb_result = HSBResult(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            max_new_tokens=args.max_new_tokens,
            inference_time_ms=metrics.get("inference_time_ms", 0),
            gpu_busy_time_ms=metrics.get("gpu_busy_time_ms", 0),
            gpu_idle_time_ms=metrics.get("gpu_idle_time_ms", 0),
            t_exposed_us=t_exposed,
            t_structural_us=t_structural,
            hsb=hsb,
            num_kernels=len(sequences),
            num_kernels_in_lut=sum(1 for s in sequences if _get_kernel_name(s) in taxbreak_lut),
        )
        
        # Save analyzer report
        analyzer.save()
        
        # Build full report
        full_report = {
            "metadata": {
                "model_name": args.model,
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "batch_size": args.batch_size,
                    "seq_len": args.seq_len,
                    "max_new_tokens": args.max_new_tokens,
                    "precision": args.precision,
                    "compile_type": args.compile_type,
                    "device": args.device,
                    "gpu_name": get_gpu_suffix(),
                }
            },
            "performance_metrics": metrics,
            "hsb_metrics": hsb_result.to_dict(),
        }
        
        return hsb_result, full_report
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"OOM: {e}", file=sys.stderr)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, None
        raise

def _get_kernel_name(sequence: Dict[str, Any]) -> str:
    """Extract kernel name from sequence dict."""
    kernel = sequence.get("kernel", {})
    if isinstance(kernel, dict):
        return kernel.get("name", "")
    return ""


def load_or_generate_taxbreak_lut(
    model_name: str,
    compile_type: str,
    precision: str,
    gpu_suffix: str,
    base_output_dir: Path,
    args_template: Any,
) -> Dict[str, float]:
    """
    Load existing taxbreak LUT or generate one if not found.
    
    The LUT is generated from a reference configuration (smallest BS/SL)
    and cached for reuse across the sweep.
    """
    # Check for existing LUT
    lut_pattern = f"{model_name.replace('/', '_')}_{compile_type}_{precision}_*_{gpu_suffix}"
    
    for existing_dir in base_output_dir.glob(lut_pattern):
        taxbreak_file = existing_dir / "microbench" / "taxbreak.json"
        if taxbreak_file.exists():
            print(f"Loading existing taxbreak LUT from: {taxbreak_file}")
            return load_taxbreak_lut(taxbreak_file)
    
    # Generate new LUT using reference configuration
    print(f"No existing taxbreak LUT found. Generating from reference configuration...")
    
    # Use smallest configuration for LUT generation
    ref_args = utils.parse_and_validate_args([
        "--model", model_name,
        "--output-dir", str(base_output_dir),
        "--batch-size", "1",
        "--seq-len", "128",
        "--max-new-tokens", "1",
        "--precision", precision,
        "--compile-type", compile_type,
        "--device", "cuda",
        "--microbench",
        "--warmup", str(PARAMS.get("microbench_warmup", "100")),
        "--runs", str(PARAMS.get("microbench_runs", "1000")),
    ])
    
    try:
        # Run full microbench to generate taxbreak LUT
        tracer = ModelTracer(args=ref_args)
        tracer.run()
        
        microbench = SodaMicrobench(tracer=tracer, args=ref_args)
        microbench.run()
        
        # Load the generated LUT
        taxbreak_file = Path(tracer.output_dir) / "microbench" / "taxbreak.json"
        if taxbreak_file.exists():
            return load_taxbreak_lut(taxbreak_file)
        else:
            print(f"Warning: taxbreak.json not generated at {taxbreak_file}", file=sys.stderr)
            return {}
            
    except Exception as e:
        print(f"Error generating taxbreak LUT: {e}", file=sys.stderr)
        return {}


def main() -> None:
    ensure_env_loaded()
    
    gpu_suffix = get_gpu_suffix()
    compile_type = PARAMS["compile_type"]
    precision = PARAMS["precision"]
    device = PARAMS["device"]
    warmup = PARAMS["inference_warmup"]
    
    # Select sweep config
    config_type = os.environ.get("SODA_SWEEP_CONFIG", "prefill").lower()
    if config_type == "decode":
        SWEEP_CONFIG = DEC_SWEEP_CONFIG
    elif config_type == "debug":
        SWEEP_CONFIG = DEBUG_SWEEP_CONFIG
    elif config_type == "all":
        SWEEP_CONFIG = {**PREF_SWEEP_CONFIG, **DEC_SWEEP_CONFIG}
    else:
        SWEEP_CONFIG = PREF_SWEEP_CONFIG
    
    # Filter models if specified
    model_filter = os.environ.get("SODA_SWEEP_MODEL")
    if model_filter:
        models = [m.strip() for m in model_filter.split(",")]
        SWEEP_CONFIG = {k: v for k, v in SWEEP_CONFIG.items() if k in models}
        if not SWEEP_CONFIG:
            print("Error: No valid models found.", file=sys.stderr)
            sys.exit(1)
    
    print(f"Running HSB sweep config: {config_type}")
    print(f"Models: {list(SWEEP_CONFIG.keys())}")
    
    base_output_dir = Path(os.environ.get("SODA_OUTPUT", "output"))
    sweep_roots = set()
    
    for config_name, cfg in SWEEP_CONFIG.items():
        model = cfg["model_name"]
        batch_sizes = cfg["batch_sizes"]
        seq_lens = cfg["seq_lens"]
        max_new_toks = cfg["max_new_toks"]
        run_precision = cfg.get("precision", precision)
        
        max_tok_str = f"mt{max_new_toks[0]}"
        sweep_root = base_output_dir / f"{model.replace('/', '_')}_{compile_type}_{run_precision}_{max_tok_str}_{gpu_suffix}_hsb"
        sweep_roots.add(sweep_root)
        sweep_root.mkdir(parents=True, exist_ok=True)
        
        print(f"\n=== Running HSB sweep: {config_name} ({model}) ===")
        
        # Load or generate taxbreak LUT
        taxbreak_lut = load_or_generate_taxbreak_lut(
            model_name=model,
            compile_type=compile_type,
            precision=run_precision,
            gpu_suffix=gpu_suffix,
            base_output_dir=base_output_dir,
            args_template=None,
        )
        
        if not taxbreak_lut:
            print(f"Warning: Empty taxbreak LUT for {model}. HSB calculations may be inaccurate.")
        else:
            print(f"Loaded taxbreak LUT with {len(taxbreak_lut)} kernels")
        
        # Store HSB results for this model
        hsb_results: List[HSBResult] = []
        
        for bs, sl, max_new_tokens in product(batch_sizes, seq_lens, max_new_toks):
            print(f"\n--- HSB sweep: bs={bs}, sl={sl}, mt={max_new_tokens} ---")
            
            exp_name = utils.generate_experiment_name(
                model, compile_type, run_precision, bs, sl, max_new_tokens
            )
            
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
            
            hsb_result, full_report = run_single_experiment(args, taxbreak_lut)
            
            if hsb_result is not None:
                hsb_results.append(hsb_result)
                
                # Save individual HSB report
                exp_dir = sweep_root / exp_name
                exp_dir.mkdir(parents=True, exist_ok=True)
                
                hsb_report_file = exp_dir / "hsb_report.json"
                with open(hsb_report_file, "w") as f:
                    json.dump(full_report, f, indent=2)
                
                print(f"HSB = {hsb_result.hsb:.4f} | T_exposed = {hsb_result.t_exposed_us:.2f} us | T_structural = {hsb_result.t_structural_us:.2f} us")
            else:
                # OOM case
                hsb_results.append(HSBResult(
                    batch_size=bs,
                    seq_len=sl,
                    max_new_tokens=max_new_tokens,
                    inference_time_ms=None,
                    gpu_busy_time_ms=None,
                    gpu_idle_time_ms=None,
                    t_exposed_us=None,
                    t_structural_us=None,
                    hsb=None,
                    num_kernels=0,
                    num_kernels_in_lut=0,
                    status="oom",
                ))
        
        # Save sweep-level HSB summary
        sweep_summary = {
            "model_name": model,
            "compile_type": compile_type,
            "precision": run_precision,
            "gpu_name": gpu_suffix,
            "taxbreak_lut_size": len(taxbreak_lut),
            "results": [r.to_dict() for r in hsb_results],
        }
        
        summary_file = sweep_root / "hsb_sweep_summary.json"
        with open(summary_file, "w") as f:
            json.dump(sweep_summary, f, indent=2)
        print(f"Saved HSB sweep summary to: {summary_file}")
    
    # Generate visualizations for all sweep roots
    print("\n=== Generating HSB visualizations ===")
    for root in sorted(sweep_roots):
        if root.exists():
            try:
                summarize_hsb_sweep(root, gpu_suffix)
            except Exception as e:
                print(f"Error summarizing {root}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()