# Dates 
# #!/bin/bash
# # Jobname
# #SBATCH -J run_tool # Updated job name for clarity
# # Output and error files
# #SBATCH -o run_tool.out
# #SBATCH -e run_tool.err

# # Job time limit (1 day 4 hours). Max for HGPU partition is 2 days.
# #SBATCH -t 1-04:00:00

# # Specify the HGPU partition for H200 nodes
# #SBATCH -p HGPU

# # Node configuration
# #SBATCH --nodes 1                 # Request 1 full node

# # Task configuration
# #SBATCH --ntasks 1                # Run a single task

# # --- MODIFIED RESOURCE ALLOCATION ---
# # GPU allocation: Request 1 H200 GPU
# #SBATCH --gres=gpu:H200:1

# # CPU allocation: Request 1 CPU core for the GPU, per HGPU partition policy
# #SBATCH --cpus-per-gpu=1

# # Memory allocation: Request 32GB of RAM for the GPU, per HGPU partition policy
# #SBATCH --mem-per-gpu=32G
# # --- END MODIFICATIONS ---

# # --- Execution Block ---

# # Load necessary modules
#  echo "Loading modules"
#  module load nvhpc/24.9 cuda12.2/nsight/12.2.2 cuda12.2/blas/12.2.2 cuda12.2/fft/12.2.2 cuda12.2/toolkit/12.2.2 cuda12.2/profiler/12.2.2 gcc11/11.3.0
#  module list # Optional: lists loaded modules for debugging

# # Activate your environment
# echo "Activating environment"
# export PATH=~/miniconda3/bin:$PATH
# source ~/miniconda3/etc/profile.d/conda.sh
# conda init bash
# conda activate base

# # Verify activation
# echo "Conda environment: $CONDA_DEFAULT_ENV"
# echo "Python path: $(which python)"
# echo "Python version: $(python --version)"

# # Install dependencies if not present
# if ! python -c "import torch, torchvision, transformers, accelerate, matplotlib, numpy"; then
#     echo "Installing dependencies"
#     conda install -c conda-forge accelerate matplotlib numpy transformers -y
#     conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
# fi

# # Run your script
# echo "Running script"

# # Check if SODA environment is loaded
# if [ -z "$SODA_ENV_LOADED" ]; then
#     echo "Error: SODA environment not loaded."
#     echo "Please run: source env.sh"
#     exit 1
# fi

# # Change to SODA root
# cd "$SODA_ROOT"

# python -m soda \
#   --model gpt2 \
#   --output-dir "$SODA_OUTPUT" \
#   --batch-size 1 \
#   --seq-len 128 \
#   --fusion 2 3 \
#   --prox-score 1.0
# echo "Job finished."

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
module load cuda12.2/toolkit

# Activate your environment
echo "Activating environment..."
export PATH=~/miniconda3/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# Verify activation
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Check PyTorch version and install/upgrade if needed
echo "Checking PyTorch version..."
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)

if [ -z "$PYTORCH_VERSION" ]; then
    echo "PyTorch not installed. Installing PyTorch 2.x..."
    pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu121 --quiet
elif [[ ! "$PYTORCH_VERSION" =~ ^2\. ]]; then
    echo "PyTorch version $PYTORCH_VERSION detected. Upgrading to PyTorch 2.x..."
    pip install --upgrade torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu121 --quiet
else
    echo "PyTorch $PYTORCH_VERSION already installed (2.x confirmed)"
fi

# Install other dependencies if not present
if ! python -c "import transformers, accelerate, matplotlib, numpy" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install transformers accelerate matplotlib numpy --quiet
fi

# Verify PyTorch 2.x and CUDA
echo "Verifying PyTorch installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Compute Capability: {torch.cuda.get_device_capability(0)}')
# Verify PyTorch 2.x features
assert torch.__version__.startswith('2.'), f'Expected PyTorch 2.x, got {torch.__version__}'
print('PyTorch 2.x verification: PASSED')
"

# Set SODA project root (absolute path - most reliable for SLURM)
SODA_PROJECT_ROOT="/home/pvellais/SODA---System-Offload-Dynamics-Analyzer"

# Source SODA environment
source "$SODA_PROJECT_ROOT/env.sh"

# Debug: verify env.sh was sourced
echo "SODA_ENV_LOADED: $SODA_ENV_LOADED"
echo "SODA_ROOT: $SODA_ROOT"

# Check if SODA environment is loaded
if [ -z "$SODA_ENV_LOADED" ]; then
    echo "Error: SODA environment not loaded"
    exit 1
fi

# Install soda package if soda-cli not available
if ! command -v soda-cli &> /dev/null; then
    echo "Installing soda package..."
    pip install -e "$SODA_PROJECT_ROOT" --quiet
fi

# Change to SODA root
cd "$SODA_ROOT"

# Run your script
echo "Running SODA analysis..."
# @prabhu did this 
# soda-cli --model gpt2 -bs 1 2 4 8 16 -sl 1024
# @shreesh did this 
python examples/soda_sweep.py
