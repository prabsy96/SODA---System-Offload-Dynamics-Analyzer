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
WARMUP_RUNS=100
MEASUREMENT_RUNS=20

# Export for use in other scripts
export WARMUP_RUNS
export MEASUREMENT_RUNS
export MICROBENCH_WARMUP_RUNS="$WARMUP_RUNS"
export MICROBENCH_REPLAY_RUNS="$MEASUREMENT_RUNS"

# MODEL="${PYTORCH_MODEL:-meta-llama/Llama-3.2-3B}"
# MODEL="${PYTORCH_MODEL:-cerebras/btlm-3b-8k-base}"
MODEL="${PYTORCH_MODEL:-gpt2}"
BATCH_SIZE="${PYTORCH_BATCH_SIZE:-1}"
SEQ_LEN="${PYTORCH_SEQ_LEN:-16}"
PRECISION="${PYTORCH_PRECISION:-float32}"
COMPILE_TYPE="${PYTORCH_COMPILE_TYPE:-eager}"

echo "=== Phase 1: Extract & Replay via SODA (microbench mode) ==="
echo "Model: $MODEL (batch=$BATCH_SIZE, seq_len=$SEQ_LEN, precision=$PRECISION, compile=$COMPILE_TYPE)"
python -m soda \
  --model "$MODEL" \
  --batch-size "$BATCH_SIZE" \
  --seq-len "$SEQ_LEN" \
  --precision "$PRECISION" \
  --compile-type "$COMPILE_TYPE" \
  --microbench
echo ""

echo "=== Phase 2: Verify Replayed Kernels ==="
python scripts/verify_replayed_kernels.py
echo ""

echo "=== Phase 3: Plot Kernel Tax Graphs ==="
python scripts/plot_kernel_tax.py

echo ""
echo "=============================================="
echo "Done! Check results in:"
echo "  $PYTORCH_OUTPUT"
echo "=============================================="

