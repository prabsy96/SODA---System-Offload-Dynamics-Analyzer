"""
Experiment 1 Runner with Live Progress Tracking & Transparent Error Logging.
Executes batch size scaling sweeps for Llama3 and GPT-2 models.
"""
import os
import sys
import subprocess
import time
from pathlib import Path

CONFIG_DIR = Path("/cra-1684/igormolybog/SODA---System-Offload-Dynamics-Analyzer/configs/exp1_batch_scaling")
OUTPUT_BASE_DIR = Path("/cra-1684/igormolybog/SODA---System-Offload-Dynamics-Analyzer/model_dir/soda_output")
LOG_FILE = Path("/cra-1684/igormolybog/SODA---System-Offload-Dynamics-Analyzer/exp1_sweep_status.log")
VENV_ACTIVATE = "/cra-1684/tutorial/igormolybog/venvs/r2.10/bin/activate"

# Queue up Llama3.2-1B batch sizes 128 through 1024
MODELS = ["llama3"]
BATCH_SIZES = [128, 256, 512, 1024]

def log_message(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(formatted + "\n")

def run_single_experiment(model: str, bs: int, idx: int, total: int):
    cfg_path = CONFIG_DIR / f"{model}_bs_{bs}.yaml"
    if not cfg_path.exists():
        log_message(f"ERROR: Configuration {cfg_path} not found. Skipping.")
        return False

    log_message("=" * 70)
    log_message(f"PROGRESS [{idx}/{total}]: Launching {model.upper()} with Batch Size = {bs}")
    log_message(f"Config File: {cfg_path}")
    log_message("=" * 70)

    # Prepare specific output directory
    run_out_dir = OUTPUT_BASE_DIR / f"exp1_{model}_bs_{bs}"
    run_out_dir.mkdir(parents=True, exist_ok=True)

    # Check if run already has a complete 10-step trace file
    trace_file = run_out_dir / "soda_output" / f"{'llama' if 'llama' in model else 'gpt'}_cerebras_trace.json"
    if trace_file.exists():
        try:
            import json
            data = json.loads(trace_file.read_text())
            if len(data.get("metrics", [])) >= 10:
                log_message(f"SKIP: {model.upper()} BS={bs} already has valid 10-step trace ({len(data['metrics'])} steps). Skipping.")
                return True
        except Exception:
            pass

    cmd = (
        f"source {VENV_ACTIVATE} && "
        f"export PYTHONPATH=/cra-1684/igormolybog/SODA---System-Offload-Dynamics-Analyzer/src:$PYTHONPATH && "
        f"python3 src/soda/cerebras/soda_inference.py CSX --params {cfg_path} --disable_version_check -o {run_out_dir}"
    )

    log_message(f"Executing command:\n  {cmd}")

    start_time = time.time()
    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            executable="/bin/bash",
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
    log_message("Resuming Experiment 1: Multi-Model Batch Scaling Sweep")
    
    total_runs = len(MODELS) * len(BATCH_SIZES)
    current_idx = 1
    
    success_count = 0
    failure_count = 0
    
    for model in MODELS:
        for bs in BATCH_SIZES:
            success = run_single_experiment(model, bs, current_idx, total_runs)
            if success:
                success_count += 1
            else:
                failure_count += 1
            current_idx += 1

    log_message("=" * 70)
    log_message(f"EXPERIMENT 1 SWEEP COMPLETE: {success_count} Succeeded, {failure_count} Failed.")
    log_message("=" * 70)

if __name__ == "__main__":
    main()
