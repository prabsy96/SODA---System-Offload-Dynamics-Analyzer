#!/bin/bash
# Run complete microbenchmark suite: framework + baremetal
# This orchestrates both pipelines end to end 

set -e

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

# Run PyTorch profiling
cd "$PYTORCH_MICROBENCH_DIR"
./run_pytorch.sh
echo ""

# Run baremetal profiling
cd "$BAREMETAL_MICROBENCH_DIR"
./run_baremetal.sh
