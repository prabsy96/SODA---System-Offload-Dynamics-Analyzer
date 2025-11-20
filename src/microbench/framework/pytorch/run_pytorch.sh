#!/bin/bash
# Framework (PyTorch) GEMM profiling microbenchmark
# Profiles PyTorch model, extracts GEMM kernels, replays, verifies, and plots

set -e

# Check if SODA environment is loaded
if [ -z "$SODA_ENV_LOADED" ]; then
    echo "Error: SODA environment not loaded."
    echo "Please run: source env.sh"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment
if [ ! -d "$PYTHON_VENV" ]; then
    echo "Error: Virtual environment not found at $PYTHON_VENV"
    echo "Create it with: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

source "$PYTHON_VENV/bin/activate"

echo "=============================================="
echo "PyTorch GEMM Profiling"
echo "=============================================="
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

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

