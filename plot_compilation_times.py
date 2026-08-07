"""
Script to extract AOT Compilation Times vs Queue Wait Times from run.log files
and plot a stacked bar chart of Pure AOT Compilation Time vs Cluster Queue Wait Time.
"""
import os
import re
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("/cra-1684/igormolybog/SODA---System-Offload-Dynamics-Analyzer/model_dir/soda_output")
RESULTS_DIR = Path("/cra-1684/igormolybog/SODA---System-Offload-Dynamics-Analyzer/experiments/fig11/results")
ARTIFACT_DIR = Path("/cloud_psc_homedir/igorm-cloud/.gemini/antigravity-ide/brain/161834b6-0ff6-44c2-ac39-adff7941022b")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]

def extract_detailed_times():
    data = {}
    for model in ["llama3", "gpt2"]:
        data[model] = {}
        for bs in BATCH_SIZES:
            run_dir = OUTPUT_DIR / f"exp1_{model}_bs_{bs}"
            log_files = sorted(run_dir.glob("cerebras_logs/inference/*/run.log"))
            if not log_files:
                continue
            
            latest_log = log_files[-1]
            text = latest_log.read_text()
            
            t_submit = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*Initiating a new compile wsjob", text)
            t_scheduled = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*Recording the timestamp when jobs is scheduled", text)
            t_success = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*Compile was successful", text)
            
            if t_submit and t_scheduled and t_success:
                t0 = datetime.strptime(t_submit.group(1), "%Y-%m-%d %H:%M:%S,%f")
                t1 = datetime.strptime(t_scheduled.group(1), "%Y-%m-%d %H:%M:%S,%f")
                t2 = datetime.strptime(t_success.group(1), "%Y-%m-%d %H:%M:%S,%f")
                
                q_wait = (t1 - t0).total_seconds() / 60.0
                pure_compile = (t2 - t1).total_seconds() / 60.0
                data[model][bs] = {
                    "pure_compile": pure_compile,
                    "queue_wait": q_wait,
                    "total": (t2 - t0).total_seconds() / 60.0
                }
    return data

def plot_compilation_times():
    timing_data = extract_detailed_times()
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    bs_keys = [1, 2, 4, 8, 16, 32, 64, 128]

    # 1. Llama-3 Stacked Bar Chart
    ax = axes[0]
    llama_bs = [b for b in bs_keys if b in timing_data.get("llama3", {})]
    x = np.arange(len(llama_bs))
    
    pure_c = [timing_data["llama3"][b]["pure_compile"] for b in llama_bs]
    q_wait = [timing_data["llama3"][b]["queue_wait"] for b in llama_bs]

    p1 = ax.bar(x, pure_c, label='Pure AOT Compilation Time', color='#9b59b6', width=0.55)
    p2 = ax.bar(x, q_wait, bottom=pure_c, label='Cluster Queue Wait Time', color='#e74c3c', alpha=0.7, width=0.55)

    ax.set_ylabel('Duration (Minutes)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_title('Llama-3.2-1B: Pure Compile vs. Queue Wait Time', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in llama_bs], fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    for i in range(len(llama_bs)):
        tot = pure_c[i] + q_wait[i]
        ax.annotate(f'{pure_c[i]:.1f}m pure\n({tot:.1f}m tot)',
                    xy=(x[i], tot), xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    # 2. GPT-2 Stacked Bar Chart
    ax = axes[1]
    gpt2_bs = [b for b in bs_keys if b in timing_data.get("gpt2", {})]
    x_gpt = np.arange(len(gpt2_bs))
    
    pure_c_gpt = [timing_data["gpt2"][b]["pure_compile"] for b in gpt2_bs]
    q_wait_gpt = [timing_data["gpt2"][b]["queue_wait"] for b in gpt2_bs]

    p1_g = ax.bar(x_gpt, pure_c_gpt, label='Pure AOT Compilation Time', color='#3498db', width=0.55)
    p2_g = ax.bar(x_gpt, q_wait_gpt, bottom=pure_c_gpt, label='Cluster Queue Wait Time', color='#e67e22', alpha=0.7, width=0.55)

    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_title('GPT-2: Pure Compile vs. Queue Wait Time', fontsize=13, fontweight='bold')
    ax.set_xticks(x_gpt)
    ax.set_xticklabels([str(b) for b in gpt2_bs], fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    for i in range(len(gpt2_bs)):
        tot = pure_c_gpt[i] + q_wait_gpt[i]
        ax.annotate(f'{pure_c_gpt[i]:.1f}m pure\n({tot:.1f}m tot)',
                    xy=(x_gpt[i], tot), xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    chart_path = OUTPUT_DIR / "compilation_times_comparison.png"
    plt.savefig(chart_path, dpi=300)
    
    res_path = RESULTS_DIR / "compilation_times_comparison.png"
    plt.savefig(res_path, dpi=300)
    
    artifact_chart_path = ARTIFACT_DIR / "compilation_times_comparison.png"
    plt.savefig(artifact_chart_path, dpi=300)
    print(f"Saved updated timing breakdown plot to: {res_path}")

if __name__ == "__main__":
    plot_compilation_times()
