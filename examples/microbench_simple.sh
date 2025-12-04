#!/bin/bash
# Microbenchmark example script
# Runs microbenchmark for a single point (bs=1, sl=128, max_new_tokens=1)

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

# Configuration 
# MODEL="gpt2"
# MODEL="meta-llama/Meta-Llama-3-8B"
MODEL="meta-llama/Llama-3.2-3B"
BATCH_SIZE="1"
SEQ_LEN="128"
PRECISION="bfloat16"
COMPILE_TYPE="eager"
WARMUP="1"
RUNS="1"
MAX_NEW_TOKENS="1"

python -m soda \
  --model "$MODEL" \
  --batch-size "$BATCH_SIZE" \
  --seq-len "$SEQ_LEN" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --precision "$PRECISION" \
  --compile-type "$COMPILE_TYPE" \
  --warmup "$WARMUP" \
  --runs "$RUNS" --microbench
