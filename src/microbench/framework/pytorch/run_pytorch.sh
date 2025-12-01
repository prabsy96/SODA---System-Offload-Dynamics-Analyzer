#!/bin/bash
# Run the PyTorch profiling and analysis pipeline

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

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# HuggingFace cache is already set in env.sh
mkdir -p "$HF_HOME"

# Profiling parameters (shared with baremetal)
WARMUP_RUNS=1000
MEASUREMENT_RUNS=2000

# Export for use in other scripts
export WARMUP_RUNS
export MEASUREMENT_RUNS

echo "=== Phase 1: Extract Event Sequences ==="
python scripts/extract_kernel_sequences.py \
  --model gpt2 \
  --batch-size 1 \
  --seq-len 16 \
  --precision float32 \
  --compile-type eager
echo ""

echo "=== Phase 2: Replay Event Sequences ==="
python scripts/replay_kernel_sequences.py --runs "$MEASUREMENT_RUNS" --warmup "$WARMUP_RUNS"
echo ""

echo "=== Phase 3: Verify Replayed Kernels ==="
python scripts/verify_replayed_kernels.py
echo ""

echo "=== Phase 4: Plot Kernel Tax Graphs ==="
python scripts/plot_kernel_tax.py

echo ""
echo "=============================================="
echo "Done! Check results in:"
echo "  $PYTORCH_OUTPUT"
echo "=============================================="