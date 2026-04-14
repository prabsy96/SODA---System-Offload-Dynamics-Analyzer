#!/bin/bash
#SBATCH --job-name=ct_standalone
#SBATCH --output=ct_standalone.out
#SBATCH --error=ct_standalone.err
#SBATCH -t 02:00:00

#SBATCH -p HGPU
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=180G

echo "============================================================================"
echo "Standalone ΔCT Validation Microbenchmark - Reviewer D"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "============================================================================"

# Load modules
module load cuda12.6/toolkit

# Activate environment
export PATH=~/miniconda3/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

SODA_PROJECT_ROOT="/home/pvellais/SODA---System-Offload-Dynamics-Analyzer"

if [ -f "$SODA_PROJECT_ROOT/env.sh" ]; then
    source "$SODA_PROJECT_ROOT/env.sh"
else
    echo "Error: env.sh not found"
    exit 1
fi

cd "$SODA_ROOT"

# Verify environment
echo "SODA_ENV_LOADED: $SODA_ENV_LOADED"
echo "SODA_ROOT: $SODA_ROOT"
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Build baremetal binary (always rebuild to pick up latest C++ fixes)
echo ""
echo "Building baremetal binary..."
build
echo "Baremetal binary: $BAREMETAL_BINARY"

GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0).replace(' ', '_'))" 2>/dev/null || echo "unknown")
OUTPUT_DIR="$SODA_OUTPUT/ct_standalone_${GPU_NAME}"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "============================================================================"
echo "Running Standalone ΔCT Validation (15 shapes, FP32)"
echo "Output: $OUTPUT_DIR"
echo "============================================================================"
echo ""

python -u "$SODA_ROOT/experiments/rebuttal/ct_standalone_microbench.py" \
    --output-dir "$OUTPUT_DIR" \
    --warmup 50 \
    --runs 200 \
    2>&1 | tee "$OUTPUT_DIR/console.log"

echo ""
echo "============================================================================"
echo "Validation Complete"
echo "Finished: $(date)"
echo "Results: $OUTPUT_DIR"
echo "============================================================================"
