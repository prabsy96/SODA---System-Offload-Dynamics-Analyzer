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

# Activate Python environment (supports both conda and venv)
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