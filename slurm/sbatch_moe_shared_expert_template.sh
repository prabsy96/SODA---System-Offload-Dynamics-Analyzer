#!/bin/bash
# =============================================================================
# SODA SBATCH Template: MoE Shared-Expert Pipeline (Stage 1 + Stage 3)
# =============================================================================
# Commit-safe, reusable SLURM script for MoE profiling with optional NVBit pass.
#
# This template intentionally does NOT embed HF tokens or read token files.
# For gated models, authenticate before submit:
#   huggingface-cli login
#
# Usage:
#   cp slurm/sbatch_moe_shared_expert_template.sh slurm/my_moe_job.sh
#   # Edit USER CONFIGURATION section
#   sbatch slurm/my_moe_job.sh
# =============================================================================

#SBATCH --job-name=soda_moe_profile
#SBATCH --output=soda_moe_%j.out
#SBATCH --error=soda_moe_%j.err
#SBATCH -t 0-06:00:00

# SBATCH -p partition_name_here
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=32G

set -euo pipefail

# =============================================================================
# USER CONFIGURATION — Edit these values for your setup
# =============================================================================

# Path to your SODA repository clone
SODA_PROJECT_ROOT="$HOME/SODA---System-Offload-Dynamics-Analyzer"

# Conda environment name (set to "" to skip conda activation)
CONDA_ENV=""

# CUDA toolkit module to load (set to "" to skip)
CUDA_MODULE="cuda12.6/toolkit"

# Nsight module for ncu (set to "" if ncu is already in PATH)
NSIGHT_MODULE="cuda12.8/nsight/12.8.1"

# Model/profile config
MODEL="Qwen/Qwen1.5-MoE-A2.7B"
BATCH=1
SEQ_LEN=1024
MAX_NEW_TOKENS=1
PRECISION="bfloat16"

# Output tag used by soda_make_output_root
OUTPUT_TAG="moe-shared-expert"

# Optional HuggingFace cache location
HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"

# NVBit settings
# Existing .so path for Stage 3 --nvbit-lib
NVBIT_LIB_PATH="${NVBIT_LIB_PATH:-$SODA_PROJECT_ROOT/src/soda/moe/nvbit_tool/mem_reuse_tracker.so}"

# Set to 1 to auto-build mem_reuse_tracker.so if missing
AUTO_BUILD_NVBIT=0

# Optional external NVBit SDK root (the extracted nvbit_release_x86_64 dir)
# If empty, build uses bundled SDK in src/soda/moe/nvbit_tool/nvbit_release_x86_64
NVBIT_SDK_PATH=""

# =============================================================================
# ENVIRONMENT SETUP — Usually no changes needed below
# =============================================================================

echo "============================================================================"
echo "SODA MoE Shared-Expert Pipeline"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Config: model=$MODEL bs=$BATCH sl=$SEQ_LEN mt=$MAX_NEW_TOKENS p=$PRECISION"
echo "============================================================================"

if [ -n "$CUDA_MODULE" ]; then
	module load "$CUDA_MODULE" 2>/dev/null || true
fi

if [ -n "$NSIGHT_MODULE" ]; then
	module load "$NSIGHT_MODULE" 2>/dev/null || true
fi

if [ -n "$CONDA_ENV" ]; then
	export PATH=~/miniconda3/bin:$PATH
	source ~/miniconda3/etc/profile.d/conda.sh
	conda activate "$CONDA_ENV"
fi

if [ -f "$SODA_PROJECT_ROOT/env.sh" ]; then
	source "$SODA_PROJECT_ROOT/env.sh"
else
	echo "Error: env.sh not found at $SODA_PROJECT_ROOT/env.sh"
	exit 1
fi

cd "$SODA_ROOT"
source "$SODA_ROOT/slurm/output_paths.sh"

export HF_HOME="$HF_HOME"

if ! command -v ncu >/dev/null 2>&1; then
	echo "Error: ncu not found in PATH. Load NSIGHT_MODULE or install Nsight Compute."
	exit 1
fi

pip install -e "$SODA_PROJECT_ROOT" --quiet 2>/dev/null || true

echo ""
echo "Environment:"
echo "  SODA_ROOT: $SODA_ROOT"
echo "  HF_HOME:   $HF_HOME"
python -c "
import torch
print(f'  PyTorch:   {torch.__version__}')
print(f'  CUDA:      {torch.cuda.is_available()}')
if torch.cuda.is_available():
	print(f'  GPU:       {torch.cuda.get_device_name(0)}')
"
echo ""

OUTPUT_ROOT="$(soda_make_output_root "$OUTPUT_TAG")"
mkdir -p "$OUTPUT_ROOT"
echo "Output root: $OUTPUT_ROOT"

if [ ! -f "$NVBIT_LIB_PATH" ] && [ "$AUTO_BUILD_NVBIT" -eq 1 ]; then
	echo "NVBit library not found at $NVBIT_LIB_PATH; attempting build..."
	pushd "$SODA_ROOT/src/soda/moe/nvbit_tool" >/dev/null
	if [ -n "$NVBIT_SDK_PATH" ]; then
		make clean
		make ARCH=all NVBIT_SDK_PATH="$NVBIT_SDK_PATH"
	else
		make clean
		make ARCH=all
	fi
	popd >/dev/null
fi

if [ ! -f "$NVBIT_LIB_PATH" ]; then
	echo "Error: NVBit library not found at $NVBIT_LIB_PATH"
	echo "Build manually: cd src/soda/moe/nvbit_tool && make ARCH=all"
	echo "Or set AUTO_BUILD_NVBIT=1 and optionally NVBIT_SDK_PATH."
	exit 1
fi

# =============================================================================
# Stage 1: Profiled inference + kernel DB
# =============================================================================

echo ""
echo "============================================"
echo "Stage 1: Profiled Inference + Kernel DB"
echo "============================================"

soda-cli \
	--model "$MODEL" \
	--output-dir "$OUTPUT_ROOT" \
	--batch-size "$BATCH" \
	--seq-len "$SEQ_LEN" \
	--max-new-tokens "$MAX_NEW_TOKENS" \
	--precision "$PRECISION" \
	--kernel-db \
	2>&1 | tee "$OUTPUT_ROOT/stage1.log"

echo "Stage 1 complete: $(date)"

EXP_DIR=$(ls -td "${OUTPUT_ROOT}"/*_bs${BATCH}_sl${SEQ_LEN}_mt${MAX_NEW_TOKENS}* 2>/dev/null | head -1)
if [ -z "${EXP_DIR:-}" ] || [ ! -f "$EXP_DIR/kernel_database.json" ]; then
	echo "Error: kernel_database.json not found under $OUTPUT_ROOT"
	exit 1
fi
echo "Experiment dir: $EXP_DIR"

# =============================================================================
# Stage 3: MoE profiling + NVBit in-context pass
# =============================================================================

echo ""
echo "============================================"
echo "Stage 3: MoE Op Profile + NVBit"
echo "============================================"

soda-cli \
	--moe-profile \
	--kernel-db-path "$EXP_DIR/kernel_database.json" \
	--nvbit-lib "$NVBIT_LIB_PATH" \
	2>&1 | tee "$OUTPUT_ROOT/stage3.log"

echo ""
echo "============================================"
echo "Done"
echo "Finished: $(date)"
echo "============================================"
echo ""
echo "Outputs:"
echo "  report.json      $EXP_DIR/report.json"
echo "  summary.md       $EXP_DIR/summary.md"
echo "  kernel_database  $EXP_DIR/kernel_database.json"
echo "  op_profile.json  $EXP_DIR/moe_profile/op_profile.json"
echo "  moe_profile.json $EXP_DIR/moe_profile/moe_profile.json"

if [ -f "$EXP_DIR/moe_profile/op_profile.json" ]; then
	total=$(python -c "import json; d=json.load(open('$EXP_DIR/moe_profile/op_profile.json')); print(len(d))")
	shared=$(python -c "import json; d=json.load(open('$EXP_DIR/moe_profile/op_profile.json')); print(sum(1 for r in d if r.get('is_shared_expert')))" )
	echo ""
	echo "  op_profile records:    $total"
	echo "  shared expert records: $shared"
	echo "  non-shared records:    $((total - shared))"
fi

