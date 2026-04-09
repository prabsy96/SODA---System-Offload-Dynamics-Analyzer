#!/bin/bash
#SBATCH --job-name=ct_val_model
#SBATCH --output=ct_validation_model.out
#SBATCH --error=ct_validation_model.err
#SBATCH -t 1-00:00:00

# Specify the HGPU partition for H200 nodes
#SBATCH -p HGPU

#SBATCH -p HGPU                   
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1              
#SBATCH --cpus-per-gpu=6          # Native ratio is 6.4. 
#SBATCH --mem-per-gpu=180G     

echo "============================================================================"
echo "Model-Driven ΔCT Validation - Reviewer D (FP32)"
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

# Build baremetal binary (always rebuild to pick up latest changes)
BAREMETAL_BIN="src/soda/microbench/baremetal/build/main_gemm_bm"
echo "Building baremetal binary..."
build
echo "Baremetal binary: $BAREMETAL_BIN"

OUTPUT_DIR="$SODA_OUTPUT/ct_validation_model_$(hostname | cut -d. -f1)"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "============================================================================"
echo "Running Model-Driven ΔCT Validation (FP32 for Replay Compatibility)"
echo "============================================================================"
echo ""

# Run model-driven validation with FP32
python -u "$SODA_ROOT/experiments/rebuttal/ct_validation_microbench.py" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 1 \
    --seq-len 512 \
    --num-layers 12 \
    --warmup 10 \
    --runs 5 \
    --dtype float32 \
    2>&1 | tee "$OUTPUT_DIR/console.log"

echo ""
echo "============================================================================"
echo "Validation Complete"
echo "Finished: $(date)"
echo "Results: $OUTPUT_DIR"
echo "============================================================================"