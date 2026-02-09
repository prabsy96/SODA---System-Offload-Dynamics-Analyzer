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

# -----------------------------------------------------------------------------
# SLURM Directives
# -----------------------------------------------------------------------------
# Cluster partitions and defaults:
#   Partition   DefaultTime  MaxTime   CPUs/GPU  Mem/GPU  Max GPUs/user
#   defq        12h          3d 23h    1         64G      H100:8
#   HGPU        6h           2d        1         32G      H200:4
#   deadline    4h           24h       1         32G      H100:4
#
# Node hardware:
#   DGX H100 (node-gpu01/02): 8x H100 80GB, 2x Xeon 8480C (56c), 2TB RAM, 28TB /scratch
#   H200 (node-gpu03):        10x H200 NVL 141GB, 2x Xeon 6538Y (32c), 2TB RAM, 5TB /scratch
#   cyh-c0-gpu00x:            2x Tesla T4 (or 1x L4), 2x Xeon 6126, 256GB RAM, 768GB /scratch
# -----------------------------------------------------------------------------

#SBATCH --job-name=soda_run
#SBATCH --output=soda_%j.out
#SBATCH --error=soda_%j.err
#SBATCH -t 0-06:00:00

#SBATCH -p defq                    # Partition: defq | HGPU | deadline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1               # Number of GPUs (e.g., gpu:2 for multi-GPU)
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=64G          # defq: up to 64G, HGPU: up to 180G

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

# Additional pip packages to install at job start (space-separated)
# Examples: "transformer-engine" for FP8, "flash-attn" for Flash Attention
EXTRA_PIP_PACKAGES=""

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

# Install extra packages
if [ -n "$EXTRA_PIP_PACKAGES" ]; then
    echo "Installing extra packages: $EXTRA_PIP_PACKAGES"
    pip install $EXTRA_PIP_PACKAGES --quiet 2>/dev/null
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
# JOB COMMANDS — Replace with your SODA commands
# =============================================================================

# Example 1: Run soda-cli
# soda-cli \
#     --model gpt2 \
#     --output-dir output/ \
#     --seq-len 512 \
#     --batch-size 1 \
#     --precision float16 \
#     2>&1 | tee "$SODA_OUTPUT/console.log"

# Example 2: Programmatic API
# python -u -c "
# from soda import ModelTracer, SodaAnalyzer
# from soda.common import utils
#
# args = utils.parse_and_validate_args([
#     '--model', 'meta-llama/Llama-3.2-1B',
#     '--output-dir', 'output/',
#     '--batch-size', '1',
#     '--seq-len', '512',
#     '--precision', 'bfloat16',
#     '--compile-type', 'eager',
#     '--device', 'cuda',
#     '--warmup', '3',
# ])
#
# tracer = ModelTracer(args=args)
# tracer.run()
# analyzer = SodaAnalyzer(tracer=tracer, args=args)
# analyzer.run()
# "

echo "YOUR COMMAND HERE"

# =============================================================================

echo ""
echo "============================================================================"
echo "Job Complete"
echo "Finished: $(date)"
echo "============================================================================"
