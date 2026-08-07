"""
Experiment 1 Runner with Live Progress Tracking & Transparent Error Logging.
Executes batch size scaling sweeps for Llama3 and GPT-2 models.
"""
import os
import sys
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = REPO_ROOT / "configs" / "exp1_batch_scaling"
OUTPUT_BASE_DIR = REPO_ROOT / "model_dir" / "soda_output"
LOG_FILE = REPO_ROOT / "exp1_sweep_status.log"
CEREBRAS_VENV = Path(
    os.environ.get(
        "SODA_CEREBRAS_VENV", "/cra-1684/tutorial/pvellaisamy/venvs/r2.10"
    )
)
BENCHMARK_MODE = os.environ.get("SODA_BENCHMARK_MODE", "decode").lower()
if BENCHMARK_MODE not in {"decode", "prefill"}:
    raise ValueError("SODA_BENCHMARK_MODE must be decode or prefill")

MODELS = ["llama3", "gpt2", "llama3_3b"]
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

def log_message(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(formatted + "\n")

def run_single_experiment(model: str, bs: int, idx: int, total: int):
    suffix = "" if BENCHMARK_MODE == "decode" else "_prefill"
    cfg_path = CONFIG_DIR / f"{model}_bs_{bs}{suffix}.yaml"
    if not cfg_path.exists():
        log_message(f"ERROR: Configuration {cfg_path} not found. Skipping.")
        return False

    log_message("=" * 70)
    log_message(f"PROGRESS [{idx}/{total}]: Launching {model.upper()} with Batch Size = {bs}")
    log_message(f"Config File: {cfg_path}")
    log_message("=" * 70)

    # Prepare specific output directory
    run_out_dir = OUTPUT_BASE_DIR / f"exp1_{model}_{BENCHMARK_MODE}_bs_{bs}"
    run_out_dir.mkdir(parents=True, exist_ok=True)

    # A valid completed benchmark must include verified fixed-length generation.
    model_prefix = "olmoe" if "olmoe" in model else ("llama" if "llama" in model else "gpt")
    trace_file = run_out_dir / "soda_output" / f"{model_prefix}_cerebras_trace.json"
    performance_file = run_out_dir / "soda_output" / "performance.json"
    if trace_file.exists() and performance_file.exists():
        try:
            import json
            performance = json.loads(performance_file.read_text())
            if (
                performance.get("benchmark_valid")
                and performance.get("workload", {}).get("mode") == BENCHMARK_MODE
            ):
                log_message(f"SKIP: {model.upper()} {BENCHMARK_MODE} BS={bs} already has a valid result.")
                return True
        except Exception:
            pass

    cmd = [
        str(CEREBRAS_VENV / "bin" / "python"),
        str(REPO_ROOT / "src" / "soda" / "cerebras" / "soda_inference.py"),
        "CSX",
        "--params",
        str(cfg_path),
        "--disable_version_check",
        "-o",
        str(run_out_dir),
    ]

    log_message(f"Executing command:\n  {' '.join(cmd)}")

    start_time = time.time()
    try:
        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env={
                **os.environ,
                "PATH": f"{CEREBRAS_VENV / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
                "PYTHONPATH": f"{REPO_ROOT / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            line_str = line.strip()
            # Print key progress indicators live
            if any(k in line_str for k in ["INFO", "WARNING", "ERROR", "Step", "Rate", "Compile", "Ingress", "Saved", "Exported"]):
                log_message(f"  [{model}-BS{bs}] {line_str}")

        process.wait()
        elapsed = time.time() - start_time

        if process.returncode == 0:
            log_message(f"SUCCESS: Completed {model}-BS{bs} in {elapsed:.1f} seconds.")
            return True
        else:
            log_message(f"FAILURE: {model}-BS{bs} failed with exit code {process.returncode} after {elapsed:.1f} seconds.")
            return False

    except Exception as e:
        log_message(f"EXCEPTION during {model}-BS{bs}: {str(e)}")
        return False

def main():
    log_message("Resuming Experiment 1 Phase 1: Dense Models Batch Scaling Sweep")
    
    total_runs = len(MODELS) * len(BATCH_SIZES)
    current_idx = 1
    
    success_count = 0
    failure_count = 0
    
    for model in MODELS:
        skip_model_larger_bs = False
        for bs in BATCH_SIZES:
            if skip_model_larger_bs:
                log_message(f"PRUNING: Skipping {model.upper()} BS={bs} because a smaller batch size failed or overflowed memory.")
                current_idx += 1
                continue

            success = run_single_experiment(model, bs, current_idx, total_runs)
            if success:
                success_count += 1
            else:
                failure_count += 1
                log_message(f"OVERFLOW PRUNING TRIGGERED: {model.upper()} failed at BS={bs}. Pruning remaining batch sizes for {model.upper()}.")
                skip_model_larger_bs = True
            current_idx += 1

    log_message("=" * 70)
    log_message(f"EXPERIMENT 1 SWEEP COMPLETE: {success_count} Succeeded, {failure_count} Failed.")
    log_message("=" * 70)

if __name__ == "__main__":
    main()
