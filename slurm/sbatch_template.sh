#!/bin/bash
# =============================================================================
# SODA SBATCH Template
# =============================================================================
# Configurable SLURM batch script for running SODA on GPU clusters.
# Copy this file, fill in the USER CONFIGURATION section, and submit:
#
#   cp slurm/sbatch_template.sh slurm/my_job.sh
#   # Edit my_job.sh with your settings
#   sbatch slurm/my_job.sh
#
# =============================================================================



#SBATCH --job-name=soda_run
#SBATCH --output=soda_%j.out
#SBATCH --error=soda_%j.err
#SBATCH -t 0-06:00:00

# SBATCH -p partition_name_here   # e.g., gpu, H100
#SBATCH --nodes=1   # Number of nodes (adjust if your job requires multiple nodes)
#SBATCH --ntasks=1  # Number of tasks (usually 1 for single-process jobs)
#SBATCH --gres=gpu:1               # Number of GPUs (e.g., gpu:2 for multi-GPU)
#SBATCH --cpus-per-gpu=6     # CPU cores per GPU (adjust based on your workload)
#SBATCH --mem-per-gpu=64G          # Memory per GPU (adjust based on your model size and batch size)

# =============================================================================
# USER CONFIGURATION — Edit these values for your setup
# =============================================================================

# Path to your SODA repository clone
SODA_PROJECT_ROOT="$HOME/SODA---System-Offload-Dynamics-Analyzer"

# Conda environment name (set to "" to skip conda activation)
CONDA_ENV=""

# CUDA toolkit module to load (set to "" to skip module load)
CUDA_MODULE="cuda12.6/toolkit"

# HuggingFace token for gated models (e.g., Llama, Gemma)
# Option 1: Set directly here (not recommended for shared scripts)
# HF_TOKEN="hf_your_token_here"
# Option 2: Export in your ~/.bashrc or pass via --export when submitting
# Option 3: Use huggingface-cli login before submitting
HF_TOKEN="${HF_TOKEN:-}"

# HuggingFace cache directory (default: /scratch/$USER/hf_cache)
HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"

# Optional packages (space-separated, set to "" to skip)
# flash-attn:         Flash Attention kernels (requires CUDA toolkit + ninja)
# transformer-engine: FP8 support on H100/H200 (requires CUDA 12+)
EXTRA_PIP_PACKAGES="flash-attn transformer-engine"

# =============================================================================
# ENVIRONMENT SETUP — Typically no changes needed below this line
# =============================================================================

echo "============================================================================"
echo "SODA Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "============================================================================"

# Load CUDA module
if [ -n "$CUDA_MODULE" ]; then
    module load "$CUDA_MODULE"
fi

# Activate conda environment
if [ -n "$CONDA_ENV" ]; then
    export PATH=~/miniconda3/bin:$PATH
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV"
fi

# Source SODA environment
if [ -f "$SODA_PROJECT_ROOT/env.sh" ]; then
    source "$SODA_PROJECT_ROOT/env.sh"
else
    echo "Error: env.sh not found at $SODA_PROJECT_ROOT/env.sh"
    exit 1
fi

cd "$SODA_ROOT"

# Set HuggingFace config
export HF_HOME="$HF_HOME"
if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN="$HF_TOKEN"
    echo "HF_TOKEN: set"
else
    echo "HF_TOKEN: not set (gated models will fail)"
fi

# Install SODA package
pip install -e "$SODA_PROJECT_ROOT" --quiet 2>/dev/null

# Install extra packages (flash-attn, transformer-engine, etc.)
if [ -n "$EXTRA_PIP_PACKAGES" ]; then
    echo "Installing extra packages: $EXTRA_PIP_PACKAGES"
    for pkg in $EXTRA_PIP_PACKAGES; do
        pip install "$pkg" --quiet --no-build-isolation 2>/dev/null || \
            echo "Warning: Failed to install $pkg (may need CUDA toolkit or ninja)"
    done
fi

# Verify environment
echo ""
echo "Environment:"
echo "  SODA_ROOT: $SODA_ROOT"
echo "  HF_HOME:   $HF_HOME"
python -c "
import torch
print(f'  PyTorch:    {torch.__version__}')
print(f'  CUDA:       {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:        {torch.cuda.get_device_name(0)}')
    print(f'  GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
echo ""

# =============================================================================
# JOB COMMANDS — Uncomment ONE section below (or write your own)
# =============================================================================

# -----------------------------------------------------------------------------
# 1) Single model analysis via soda-cli
# -----------------------------------------------------------------------------
# soda-cli \
#     --model gpt2 \
#     --output-dir output/ \
#     --seq-len 512 \
#     --batch-size 1 \
#     --precision float16 \
#     2>&1 | tee "$SODA_OUTPUT/console.log"



# -----------------------------------------------------------------------------
# 2) Prefill sweep — batch size x sequence length grid (max_new_tokens=1)
#    SODA_SWEEP_CONFIG: prefill | decode | fp8 | debug | all
#    SODA_SWEEP_MODEL:  leave empty for all, or comma-separated config keys:
#      Prefill: gpt2_short_ctx, llama_3.2_1b_short_ctx, tinyllama_1.1b,
#               olmoe_1b_7b, qwen1.5_moe_a2.7b, gpt_oss_20b,
#               whisper_large_v3, gemma_2b
# -----------------------------------------------------------------------------
# export SODA_SWEEP_CONFIG="prefill"
# export SODA_SWEEP_MODEL=""
# python -u "$SODA_ROOT/experiments/sweep/soda_sweep.py" \
#     2>&1 | tee "$SODA_OUTPUT/prefill_sweep.log"

# -----------------------------------------------------------------------------
# 3) Decode sweep — autoregressive generation (max_new_tokens=10+)
#    SODA_SWEEP_MODEL keys for decode:
#      gpt_oss_20b, llama_3.2_1b_short_ctx, tinyllama_1.1b,
#      qwen1.5_moe_a2.7b, olmoe_1b_7b, whisper_large_v3
# -----------------------------------------------------------------------------
# export SODA_SWEEP_CONFIG="decode"
# export SODA_SWEEP_MODEL=""
# python -u "$SODA_ROOT/experiments/sweep/soda_sweep.py" \
#     2>&1 | tee "$SODA_OUTPUT/decode_sweep.log"

# -----------------------------------------------------------------------------
# 4) FP8 sweep (H100/H200 only, requires transformer-engine)
#    Model: gpt_oss_20b_fp8 with float8_e4m3fn precision
# -----------------------------------------------------------------------------
# export SODA_SWEEP_CONFIG="fp8"
# python -u "$SODA_ROOT/experiments/sweep/soda_sweep.py" \
#     2>&1 | tee "$SODA_OUTPUT/fp8_sweep.log"

# -----------------------------------------------------------------------------
# 5) Debug sweep — quick sanity check (gpt2, bs=[1,2], sl=[128,256])
# -----------------------------------------------------------------------------
# export SODA_SWEEP_CONFIG="debug"
# python -u "$SODA_ROOT/experiments/sweep/soda_sweep.py" \
#     2>&1 | tee "$SODA_OUTPUT/debug_sweep.log"

# -----------------------------------------------------------------------------
# 6) Microbenchmark sweep — GEMM kernel analysis (baremetal + PyTorch)
#    Requires baremetal binary. Uses same SODA_SWEEP_CONFIG / SODA_SWEEP_MODEL
#    / SODA_CUSTOM_MODEL env vars as soda_sweep.py.
# -----------------------------------------------------------------------------
# build
# export SODA_SWEEP_CONFIG="prefill"
# export SODA_SWEEP_MODEL=""
# python -u "$SODA_ROOT/experiments/sweep/microbench_sweep.py" \
#     2>&1 | tee "$SODA_OUTPUT/microbench_sweep.log"

# -----------------------------------------------------------------------------
# 7) Custom model sweep — any HuggingFace model, no config.py needed
#    Works with both soda_sweep.py and microbench_sweep.py.
#    Env vars:
#      SODA_CUSTOM_MODEL   (required) Any HF model ID
#      SODA_BATCH_SIZES    (default: "1,2,4,8,16")
#      SODA_SEQ_LENS       (default: "128,256,512,1024")
#      SODA_MAX_NEW_TOKENS (default: "1")
#      SODA_PRECISION      (default: "bfloat16")
# -----------------------------------------------------------------------------
# export SODA_CUSTOM_MODEL="meta-llama/Llama-3.2-1B"
# export SODA_BATCH_SIZES="1,2,4,8"
# export SODA_SEQ_LENS="512,1024,2048,4096"
# export SODA_MAX_NEW_TOKENS="1"
# export SODA_PRECISION="bfloat16"
# python -u "$SODA_ROOT/experiments/sweep/soda_sweep.py" \
#     2>&1 | tee "$SODA_OUTPUT/custom_sweep.log"

# =============================================================================

echo ""
echo "============================================================================"
echo "Job Complete"
echo "Finished: $(date)"
echo "============================================================================"
