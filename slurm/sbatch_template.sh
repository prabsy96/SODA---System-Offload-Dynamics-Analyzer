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
#SBATCH --cpus-per-gpu=6     # REQUIRED: 6 CPU cores per GPU (paper Table II §4.1)
#SBATCH --mem-per-gpu=32G          # REQUIRED: 32 GB per GPU (paper Table II §4.1)

# =============================================================================
# USER CONFIGURATION — Edit these values for your setup
# =============================================================================

# Path to your SODA repository clone
SODA_PROJECT_ROOT="$HOME/SODA---System-Offload-Dynamics-Analyzer"

# Conda environment name (set to "" to skip conda activation)
CONDA_ENV=""

# CUDA toolkit module to load (set to "" to skip module load)
CUDA_MODULE="cuda12.6/toolkit"

# Nsight module for nsys/ncu (required for --taxbreak and --ncu)
# Set to "" to skip if nsys/ncu are already in PATH
NSIGHT_MODULE="cuda12.8/nsight/12.8.1"

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

# Load Nsight module (provides nsys + ncu for profiling)
if [ -n "$NSIGHT_MODULE" ]; then
    module load "$NSIGHT_MODULE"
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
source "$SODA_ROOT/slurm/output_paths.sh"

# Default to a per-submission output root so repeated submissions never share
# caches or overwrite one another. Override SODA_OUTPUT before submission if needed.
export SODA_OUTPUT="$(soda_make_output_root "slurm-template")"
mkdir -p "$SODA_OUTPUT"

# Set HuggingFace config
export HF_HOME="$HF_HOME"
if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN="$HF_TOKEN"
    echo "HF_TOKEN: set"
else
    echo "HF_TOKEN: not set (gated models will fail)"
fi

# Install SODA package
pip install -e "$SODA_PROJECT_ROOT" --quiet 2>/dev/null || true

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
    print(f'  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
echo ""

# =============================================================================
# JOB COMMANDS — Uncomment ONE section below (or write your own)
# =============================================================================
#
# OUTPUT FLAGS (applicable to all soda-cli commands):
#   --verbose            Print full expert tables (per-kernel, HDBI breakdown)
#                        Default: compact layperson summary only
#   --carbon-intensity   Grid carbon intensity in gCO2eq/kWh (default: 400.0)
#                        Presets: 58=France, 295=EU avg, 386=US avg, 581=China
#   --pue                Power Usage Effectiveness multiplier (default: 1.1)
#
# AUTOMATIC OUTPUTS (always generated, no flag needed):
#   summary.md           Layperson-friendly Markdown report
#   pareto.png           Throughput–interactivity Pareto plot
#   report.json          Full machine-readable metrics
# =============================================================================

# -----------------------------------------------------------------------------
# 1) Standard analysis — compact summary (default output mode)
#    Writes: report.json, summary.md, pareto.png
# -----------------------------------------------------------------------------
# soda-cli \
#     --model gpt2 \
#     --output-dir output/ \
#     --seq-len 512 \
#     --batch-size 1 \
#     --precision float16 \
#     2>&1 | tee "$SODA_OUTPUT/console.log"

# -----------------------------------------------------------------------------
# 1b) Standard analysis — verbose expert output
#     Adds full per-kernel tables, HDBI derivation, and tax breakdown detail.
# -----------------------------------------------------------------------------
# soda-cli \
#     --model gpt2 \
#     --output-dir output/ \
#     --seq-len 512 \
#     --batch-size 1 \
#     --precision float16 \
#     --verbose \
#     2>&1 | tee "$SODA_OUTPUT/console.log"

# -----------------------------------------------------------------------------
# 1c) Standard analysis — custom carbon footprint parameters
#     Useful for matching the reported CO2 to your actual grid.
#     --carbon-intensity: regional grid intensity in gCO2eq/kWh
#     --pue:              facility power overhead (1.0 = GPU only, 1.5 = avg DC)
# -----------------------------------------------------------------------------
# soda-cli \
#     --model gpt2 \
#     --output-dir output/ \
#     --seq-len 512 \
#     --batch-size 1 \
#     --precision float16 \
#     --carbon-intensity 386 \
#     --pue 1.2 \
#     2>&1 | tee "$SODA_OUTPUT/console.log"

# -----------------------------------------------------------------------------
# 1d) Standard analysis — all options combined
# -----------------------------------------------------------------------------
# soda-cli \
#     --model meta-llama/Llama-3.2-1B \
#     --output-dir output/ \
#     --seq-len 1024 \
#     --batch-size 4 \
#     --precision bfloat16 \
#     --verbose \
#     --carbon-intensity 400 \
#     --pue 1.1 \
#     2>&1 | tee "$SODA_OUTPUT/console.log"

# -----------------------------------------------------------------------------
# 2) Stage 1: Generate kernel database (required before --taxbreak)
#    Output dir: output/<model>_<compile>_<precision>_bs<B>_sl<S>_mt<T>/
#    Produces: report.json, summary.md, pareto.png, kernel_database.json
# -----------------------------------------------------------------------------
# soda-cli \
#     --model gpt2 \
#     --output-dir output/ \
#     --seq-len 512 \
#     --batch-size 1 \
#     --precision float16 \
#     --kernel-db \
#     2>&1 | tee "$SODA_OUTPUT/stage1.log"

# -----------------------------------------------------------------------------
# 2b) Stage 1 — with verbose output and carbon tracking
# -----------------------------------------------------------------------------
# soda-cli \
#     --model meta-llama/Llama-3.2-1B \
#     --output-dir output/ \
#     --seq-len 1024 \
#     --batch-size 1 \
#     --precision bfloat16 \
#     --kernel-db \
#     --verbose \
#     --carbon-intensity 386 \
#     --pue 1.1 \
#     2>&1 | tee "$SODA_OUTPUT/stage1.log"

# -----------------------------------------------------------------------------
# 3) Stage 2: Enhanced TaxBreak — per-kernel isolation replay
#    Replays each kernel from kernel_database.json in isolation under nsys.
#    Writes: taxbreak/enhanced_taxbreak.json, taxbreak/roofline.png, summary.md
#    NOTE: --kernel-db-path must match the output dir from Stage 1.
# -----------------------------------------------------------------------------
# soda-cli \
#     --taxbreak \
#     --kernel-db-path output/gpt2_eager_float16_bs1_sl512_mt1/kernel_database.json \
#     2>&1 | tee "$SODA_OUTPUT/taxbreak.log"

# -----------------------------------------------------------------------------
# 3b) Stage 2 — with ncu hardware counter profiling
#     Adds L1/L2 cache hit rates, DRAM throughput, compute utilization.
#     --ncu-top-n: number of kernels to profile (sorted by total GPU time)
#     Requires: nsys + ncu in PATH (load NSIGHT_MODULE above)
# -----------------------------------------------------------------------------
# soda-cli \
#     --taxbreak \
#     --kernel-db-path output/gpt2_eager_float16_bs1_sl512_mt1/kernel_database.json \
#     --ncu \
#     --ncu-top-n 10 \
#     2>&1 | tee "$SODA_OUTPUT/taxbreak_ncu.log"

# -----------------------------------------------------------------------------
# 3c) Stage 2 — verbose expert output (full per-kernel table + roofline)
# -----------------------------------------------------------------------------
# soda-cli \
#     --taxbreak \
#     --kernel-db-path output/gpt2_eager_float16_bs1_sl512_mt1/kernel_database.json \
#     --ncu \
#     --ncu-top-n 10 \
#     --verbose \
#     2>&1 | tee "$SODA_OUTPUT/taxbreak_ncu_verbose.log"

# -----------------------------------------------------------------------------
# 4) Full two-stage pipeline (Stage 1 then Stage 2 in one job)
#    Generates kernel DB then immediately runs enhanced TaxBreak analysis.
# -----------------------------------------------------------------------------
# MODEL="gpt2"
# OUTPUT_DIR="output/"
# SEQ_LEN=512
# BATCH=1
# PRECISION="float16"
#
# # Stage 1: profiled inference + kernel database
# soda-cli \
#     --model "$MODEL" \
#     --output-dir "$OUTPUT_DIR" \
#     --seq-len "$SEQ_LEN" \
#     --batch-size "$BATCH" \
#     --precision "$PRECISION" \
#     --kernel-db \
#     --carbon-intensity 400 \
#     2>&1 | tee "$SODA_OUTPUT/stage1.log"
#
# # Stage 2: per-kernel isolation replay + ncu
# EXP_DIR=$(ls -td "$OUTPUT_DIR"/${MODEL}_*_bs${BATCH}_sl${SEQ_LEN}_* 2>/dev/null | head -1)
# if [ -n "$EXP_DIR" ]; then
#     soda-cli \
#         --taxbreak \
#         --kernel-db-path "$EXP_DIR/kernel_database.json" \
#         --ncu \
#         --ncu-top-n 10 \
#         --verbose \
#         2>&1 | tee "$SODA_OUTPUT/stage2.log"
# else
#     echo "Error: could not locate experiment directory under $OUTPUT_DIR"
#     exit 1
# fi

# -----------------------------------------------------------------------------
# 5) Prefill sweep — batch size × sequence length grid (max_new_tokens=1)
#    SODA_SWEEP_CONFIG: prefill | decode | fp8 | debug | all
#    Pareto plot is generated per-run and automatically picks up sibling runs
#    from the same output directory.
# -----------------------------------------------------------------------------
# export SODA_SWEEP_CONFIG="prefill"
# export SODA_SWEEP_MODEL=""
# python -u "$SODA_ROOT/experiments/sweep/soda_sweep.py" \
#     2>&1 | tee "$SODA_OUTPUT/prefill_sweep.log"

# -----------------------------------------------------------------------------
# 6) Decode sweep — autoregressive generation (max_new_tokens=10+)
# -----------------------------------------------------------------------------
# export SODA_SWEEP_CONFIG="decode"
# export SODA_SWEEP_MODEL=""
# python -u "$SODA_ROOT/experiments/sweep/soda_sweep.py" \
#     2>&1 | tee "$SODA_OUTPUT/decode_sweep.log"

# -----------------------------------------------------------------------------
# 7) FP8 sweep (H100/H200 only, requires transformer-engine)
#    Model: gpt_oss_20b_fp8 with float8_e4m3fn precision
# -----------------------------------------------------------------------------
# export SODA_SWEEP_CONFIG="fp8"
# python -u "$SODA_ROOT/experiments/sweep/soda_sweep.py" \
#     2>&1 | tee "$SODA_OUTPUT/fp8_sweep.log"

# -----------------------------------------------------------------------------
# 8) Debug sweep — quick sanity check (gpt2, bs=[1,2], sl=[128,256])
# -----------------------------------------------------------------------------
# export SODA_SWEEP_CONFIG="debug"
# python -u "$SODA_ROOT/experiments/sweep/soda_sweep.py" \
#     2>&1 | tee "$SODA_OUTPUT/debug_sweep.log"

# -----------------------------------------------------------------------------
# 9) Microbenchmark sweep — GEMM kernel analysis (baremetal + PyTorch)
#    Requires baremetal binary. Uses same SODA_SWEEP_CONFIG / SODA_SWEEP_MODEL
#    / SODA_CUSTOM_MODEL env vars as soda_sweep.py.
# -----------------------------------------------------------------------------
# export SODA_SWEEP_CONFIG="prefill"
# export SODA_SWEEP_MODEL=""
# python -u "$SODA_ROOT/experiments/sweep/microbench_sweep.py" \
#     2>&1 | tee "$SODA_OUTPUT/microbench_sweep.log"

# -----------------------------------------------------------------------------
# 10) Custom model sweep — any HuggingFace model, no config.py needed
#     Works with both soda_sweep.py and microbench_sweep.py.
#     Env vars:
#       SODA_CUSTOM_MODEL   (required) Any HF model ID
#       SODA_BATCH_SIZES    (default: "1,2,4,8,16")
#       SODA_SEQ_LENS       (default: "128,256,512,1024")
#       SODA_MAX_NEW_TOKENS (default: "1")
#       SODA_PRECISION      (default: "bfloat16")
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
