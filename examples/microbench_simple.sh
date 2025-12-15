#!/bin/bash
# Microbenchmark example script
# Runs microbenchmark sweep for multiple batch sizes and sequence lengths

set -e

# Check running from root directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the root directory of the SODA repository."
    exit 1
fi

# Check if SODA environment is loaded
if [ -z "$SODA_ENV_LOADED" ]; then
    echo "Error: SODA environment not loaded."
    echo "Please run: source env.sh"
    exit 1
fi

# Activate Python environment (supports conda or venv)
if [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "base" ]; then
    echo "Using conda environment: $CONDA_DEFAULT_ENV"
elif [ -d "$PYTHON_VENV" ]; then
    echo "Activating virtual environment at $PYTHON_VENV"
    source "$PYTHON_VENV/bin/activate"
elif [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Using conda base environment"
elif command -v python &> /dev/null; then
    echo "Using system Python"
else
    echo "Error: No Python environment found."
    exit 1
fi

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

GPU_NAME=$(python -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0).replace('NVIDIA ', '').replace(' ', '_')
    for key in ['H100', 'H200', 'A100', 'V100', 'A10', 'L40', 'RTX']:
        if key in name.upper():
            print(key)
            exit()
    print(name.split()[0])
else:
    print('unknown_gpu')
" 2>/dev/null || echo "unknown_gpu")

echo "Detected GPU: $GPU_NAME"
echo ""

# Configuration 
MODEL="gpt2"
# MODEL="meta-llama/Meta-Llama-3-8B"
# MODEL="meta-llama/Llama-3.2-3B"
# MODEL="meta-llama/Llama-3.2-1B"

PRECISION="bfloat16"
COMPILE_TYPE="eager"
WARMUP="1000"
RUNS="5000"
MAX_NEW_TOKENS="1"

# Sweep Configuration
BATCH_SIZES=(1 2 4 8 16)
SEQ_LENS=(128 256 512 1024)

echo "Starting sweep for model: $MODEL"
echo "Batch Sizes: ${BATCH_SIZES[*]}"
echo "Sequence Lengths: ${SEQ_LENS[*]}"
echo "---------------------------------------------------"

for BS in "${BATCH_SIZES[@]}"; do
    for SL in "${SEQ_LENS[@]}"; do
        echo "Running microbenchmark: BS=$BS, SL=$SL"
        
        python -m soda \
          --model "$MODEL" \
          --batch-size "$BS" \
          --seq-len "$SL" \
          --max-new-tokens "$MAX_NEW_TOKENS" \
          --precision "$PRECISION" \
          --compile-type "$COMPILE_TYPE" \
          --warmup "$WARMUP" \
          --runs "$RUNS" --microbench
          
        echo "Completed: BS=$BS, SL=$SL"
        echo "---------------------------------------------------"
    done
done

echo "Sweep completed successfully."