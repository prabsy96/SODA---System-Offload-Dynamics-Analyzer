#!/bin/bash
# Jobname
#SBATCH -J run_tool # Updated job name for clarity
# Output and error files
#SBATCH -o run_tool.out
#SBATCH -e run_tool.err

# Job time limit (1 day 4 hours). Max for HGPU partition is 2 days.
#SBATCH -t 1-04:00:00

# Specify the HGPU partition for H200 nodes
#SBATCH -p HGPU

# Node configuration
#SBATCH --nodes 1                 # Request 1 full node

# Task configuration
#SBATCH --ntasks 1                # Run a single task

# --- MODIFIED RESOURCE ALLOCATION ---
# GPU allocation: Request 1 H200 GPU
#SBATCH --gres=gpu:H200:1

# CPU allocation: Request 1 CPU core for the GPU, per HGPU partition policy
#SBATCH --cpus-per-gpu=1

# Memory allocation: Request 32GB of RAM for the GPU, per HGPU partition policy
#SBATCH --mem-per-gpu=32G
# --- END MODIFICATIONS ---

# --- Execution Block ---

# Load necessary modules
 echo "Loading modules..."
 module load nvhpc/24.9 cuda12.2/nsight/12.2.2 cuda12.2/blas/12.2.2 cuda12.2/fft/12.2.2 cuda12.2/toolkit/12.2.2 cuda12.2/profiler/12.2.2 gcc11/11.3.0
 module list # Optional: lists loaded modules for debugging

# Activate your environment
echo "Activating environment..."
export PATH=~/miniconda3/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate base

# Verify activation
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Install dependencies if not present
if ! python -c "import torch, torchvision, transformers, accelerate, matplotlib, numpy"; then
    echo "Installing dependencies..."
    conda install -c conda-forge accelerate matplotlib numpy transformers -y
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
fi

# Run your script
echo "Running script..."

# Check if SODA environment is loaded
if [ -z "$SODA_ENV_LOADED" ]; then
    echo "Error: SODA environment not loaded."
    echo "Please run: source env.sh"
    exit 1
fi

# Change to SODA root
cd "$SODA_ROOT"

python -m soda \
  --model gpt2 \
  --output-dir "$SODA_RESULTS" \
  --batch-size 1 \
  --seq-len 128 \
  --fusion 2 3 \
  --prox-score 1.0
echo "Job finished."
