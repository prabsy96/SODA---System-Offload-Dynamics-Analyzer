#!/bin/bash
# Microbenchmark example script
# Runs the complete microbenchmark suite: framework + baremetal

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

# Activate virtual environment
if [ ! -d "$PYTHON_VENV" ]; then
    echo "Error: Virtual environment not found at $PYTHON_VENV"
    echo "Create it with: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

source "$PYTHON_VENV/bin/activate"

echo "=============================================="
echo "Kernel Launch Tax Microbenchmark Suite"
echo "=============================================="
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# HuggingFace cache is already set in env.sh
mkdir -p "$HF_HOME"

# Configuration 
# MODEL="gpt2"
# MODEL="meta-llama/Meta-Llama-3-8B"
# MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"
MODEL="meta-llama/Llama-3.2-3B"
BATCH_SIZE="1"
SEQ_LEN="16"
# PRECISION="float32"
PRECISION="bfloat16"
# PRECISION="float16"
COMPILE_TYPE="eager"
WARMUP="1000"
RUNS="5000"
# WARMUP="1"
# RUNS="1"

echo "=== Microbenchmarking GEMM Kernels via SodaMicrobench ==="
python -m soda \
  --model "$MODEL" \
  --batch-size "$BATCH_SIZE" \
  --seq-len "$SEQ_LEN" \
  --precision "$PRECISION" \
  --compile-type "$COMPILE_TYPE" \
  --microbench \
  --warmup "$WARMUP" \
  --runs "$RUNS"

echo ""
echo "=============================================="
echo "Done! Check results in the experiment output directory."
echo "=============================================="
