"""
Script to extract and plot Figure 11(a) Inference Time and Figure 11(c) CPU Idle Time
comparing the three GPU systems from the SODA paper (AMD+A100, Intel+H100, GH200)
with our measured Cerebras CS-3 Wafer-Scale Engine results up to BS=64,
including non-trivial 95% Confidence Interval error bars for Cerebras CS-3 measurements.
"""
import os
import json
import math
import statistics
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("/cra-1684/igormolybog/SODA---System-Offload-Dynamics-Analyzer/model_dir/soda_output")

# GPU baseline batch sizes (from paper Figure 11)
GPU_BS = [1, 2, 4, 8, 16]
# Full Cerebras batch sizes up to BS=64
CEREBRAS_BS = [1, 2, 4, 8, 16, 32, 64]

# Digitized baseline values from Figure 11 of SODA paper (arXiv:2504.11750v1)
# --- Llama-3 (1.2B) ---
llama_inference_time_gpu = {
    "AMD + A100": [45.0, 55.0, 110.0, 210.0, 400.0],
    "Intel + H100": [45.0, 55.0, 90.0, 180.0, 380.0],
    "GH200": [45.0, 50.0, 70.0, 140.0, 275.0],
}

llama_cpu_idle_gpu = {
    "AMD + A100": [0.0, 5.0, 80.0, 190.0, 380.0],
    "Intel + H100": [0.0, 2.0, 60.0, 170.0, 360.0],
    "GH200": [0.0, 0.0, 25.0, 50.0, 140.0],
}

# --- GPT-2 (137M) ---
gpt2_inference_time_gpu = {
    "AMD + A100": [10.0, 15.0, 25.0, 50.0, 100.0],
    "Intel + H100": [10.0, 15.0, 22.0, 45.0, 90.0],
    "GH200": [20.0, 22.0, 25.0, 35.0, 60.0],
}

gpt2_cpu_idle_gpu = {
    "AMD + A100": [0.0, 0.0, 5.0, 35.0, 80.0],
    "Intel + H100": [0.0, 0.0, 3.0, 30.0, 75.0],
    "GH200": [0.0, 0.0, 0.0, 12.0, 40.0],
}

def parse_cerebras_metrics(model: str):
    inference_times = []
    inference_time_moes = []
    cpu_idles = []
    cpu_idle_moes = []

    for bs in CEREBRAS_BS:
        run_dir = OUTPUT_DIR / f"exp1_{model}_bs_{bs}"
        trace_files = list(run_dir.rglob("*_trace.json"))
        if trace_files and trace_files[0].exists():
            try:
                data = json.loads(trace_files[0].read_text())
                metrics = data.get("metrics", [])
                steps = metrics[1:] if len(metrics) > 1 else metrics
                if steps:
                    per_sample_ms = [s["step_latency_s"] * 1000.0 for s in steps]
                    batch_times = [l * bs for l in per_sample_ms]
                    host_idles = [s["host_sync_idle_s"] * 1000.0 for s in steps]
                    n = len(steps)

                    mean_bt = statistics.mean(batch_times)
                    moe_bt = 1.96 * (statistics.stdev(batch_times) / math.sqrt(n)) if n > 1 else 0.0

                    mean_hi = statistics.mean(host_idles)
                    moe_hi = 1.96 * (statistics.stdev(host_idles) / math.sqrt(n)) if n > 1 else 0.0

                    inference_times.append(mean_bt)
                    inference_time_moes.append(moe_bt)
                    cpu_idles.append(mean_hi)
                    cpu_idle_moes.append(moe_hi)
                    continue
            except Exception as e:
                print(f"Error reading trace for {model} BS={bs}: {e}")

        # Fallback dummy
        inference_times.append(np.nan)
        inference_time_moes.append(0.0)
        cpu_idles.append(np.nan)
        cpu_idle_moes.append(0.0)

    return (
        np.array(inference_times),
        np.array(inference_time_moes),
        np.array(cpu_idles),
        np.array(cpu_idle_moes),
    )

def plot_comparison():
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    colors = {
        "AMD + A100": "#e74c3c",      # Red
        "Intel + H100": "#3498db",    # Blue
        "GH200": "#2ecc71",           # Green
        "Cerebras CS-3": "#9b59b6"    # Purple
    }

    markers = {
        "AMD + A100": "o",
        "Intel + H100": "s",
        "GH200": "^",
        "Cerebras CS-3": "D"
    }

    # Parse Cerebras metrics with error bars
    llama_inf_t, llama_inf_moe, llama_cpu_t, llama_cpu_moe = parse_cerebras_metrics("llama3")
    gpt2_inf_t, gpt2_inf_moe, gpt2_cpu_t, gpt2_cpu_moe = parse_cerebras_metrics("gpt2")

    all_bs = [1, 2, 4, 8, 16, 32, 64]

    # 1. Llama-3 Inference Time
    ax = axes[0, 0]
    for sys_name, vals in llama_inference_time_gpu.items():
        ax.plot(GPU_BS, vals, label=sys_name, color=colors[sys_name], marker=markers[sys_name], linewidth=2, markersize=7)
    
    ax.errorbar(
        CEREBRAS_BS, llama_inf_t, yerr=llama_inf_moe,
        label="Cerebras CS-3 (95% CI)", color=colors["Cerebras CS-3"], marker=markers["Cerebras CS-3"],
        linewidth=2.5, markersize=8, capsize=5, capthick=1.5, elinewidth=1.5
    )
    ax.set_title("Llama-3.2-1B: Fig. 11(a) Inference Time (ms)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Batch Size", fontsize=11)
    ax.set_ylabel("Inference Time (ms)", fontsize=11)
    ax.set_xticks(all_bs)
    ax.set_xticklabels([str(b) for b in all_bs])
    ax.legend(fontsize=10)

    # 2. Llama-3 CPU Idle Time
    ax = axes[0, 1]
    for sys_name, vals in llama_cpu_idle_gpu.items():
        ax.plot(GPU_BS, vals, label=sys_name, color=colors[sys_name], marker=markers[sys_name], linewidth=2, markersize=7)
    
    ax.errorbar(
        CEREBRAS_BS, llama_cpu_t, yerr=llama_cpu_moe,
        label="Cerebras CS-3 (95% CI)", color=colors["Cerebras CS-3"], marker=markers["Cerebras CS-3"],
        linewidth=2.5, markersize=8, capsize=5, capthick=1.5, elinewidth=1.5
    )
    ax.set_title("Llama-3.2-1B: Fig. 11(c) CPU / Host Idle Time (ms)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Batch Size", fontsize=11)
    ax.set_ylabel("CPU Idle Time (ms)", fontsize=11)
    ax.set_xticks(all_bs)
    ax.set_xticklabels([str(b) for b in all_bs])
    ax.legend(fontsize=10)

    # 3. GPT-2 Inference Time
    ax = axes[1, 0]
    for sys_name, vals in gpt2_inference_time_gpu.items():
        ax.plot(GPU_BS, vals, label=sys_name, color=colors[sys_name], marker=markers[sys_name], linewidth=2, markersize=7)
    
    ax.errorbar(
        CEREBRAS_BS, gpt2_inf_t, yerr=gpt2_inf_moe,
        label="Cerebras CS-3 (95% CI)", color=colors["Cerebras CS-3"], marker=markers["Cerebras CS-3"],
        linewidth=2.5, markersize=8, capsize=5, capthick=1.5, elinewidth=1.5
    )
    ax.set_title("GPT-2: Fig. 11(a) Inference Time (ms)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Batch Size", fontsize=11)
    ax.set_ylabel("Inference Time (ms)", fontsize=11)
    ax.set_xticks(all_bs)
    ax.set_xticklabels([str(b) for b in all_bs])
    ax.legend(fontsize=10)

    # 4. GPT-2 CPU Idle Time
    ax = axes[1, 1]
    for sys_name, vals in gpt2_cpu_idle_gpu.items():
        ax.plot(GPU_BS, vals, label=sys_name, color=colors[sys_name], marker=markers[sys_name], linewidth=2, markersize=7)
    
    ax.errorbar(
        CEREBRAS_BS, gpt2_cpu_t, yerr=gpt2_cpu_moe,
        label="Cerebras CS-3 (95% CI)", color=colors["Cerebras CS-3"], marker=markers["Cerebras CS-3"],
        linewidth=2.5, markersize=8, capsize=5, capthick=1.5, elinewidth=1.5
    )
    ax.set_title("GPT-2: Fig. 11(c) CPU / Host Idle Time (ms)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Batch Size", fontsize=11)
    ax.set_ylabel("CPU Idle Time (ms)", fontsize=11)
    ax.set_xticks(all_bs)
    ax.set_xticklabels([str(b) for b in all_bs])
    ax.legend(fontsize=10)

    plt.tight_layout()
    chart_path = OUTPUT_DIR / "fig11_4system_comparison.png"
    plt.savefig(chart_path, dpi=300)
    print(f"Saved augmented 4-system comparison plot with error bars up to BS=64 to: {chart_path}")

if __name__ == "__main__":
    plot_comparison()
