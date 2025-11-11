#!/bin/bash
# Framework (PyTorch) GEMM profiling microbenchmark
# Profiles PyTorch model, extracts GEMM kernels, replays, verifies, and plots

set -e

# Get script directory and activate venv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Go up 4 levels: pytorch -> framework -> microbench -> src -> soda
SODA_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
VENV_PATH="$SODA_ROOT/.venv"

# Activate virtual environment
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "Using Python: $(which python)"
    echo "Python version: $(python --version)"
    echo ""
else
    echo "Warning: Virtual environment not found at $VENV_PATH"
    echo "Using system Python: $(which python3)"
    echo ""
fi

cd "$SCRIPT_DIR"

# Use local /tmp for HuggingFace cache 
export HF_HOME=/tmp/hf_cache_$USER
mkdir -p "$HF_HOME"

echo "=== Phase 1: Extract Kernel Chains ==="
python scripts/extract_kernel_chains.py
echo ""

echo "=== Phase 2: Replay Kernel Chains ==="
python scripts/replay_kernel_chains.py output/unique_gemm_kernel_chains.json --runs 300 --warmup 500
echo ""

echo "=== Phase 3: Verify Replayed Kernels ==="
python scripts/verify_replayed_kernels.py output/unique_gemm_kernel_chains.json output/replayed_gemm_kernel_chains.json
echo ""

echo "=== Phase 4: Plot Kernel Tax Graphs ==="
python scripts/plot_kernel_tax.py output/replayed_gemm_kernel_chains.json --out output/graphs/

echo ""
echo "=============================================="
echo "Done! Check results in:"
echo "  $SCRIPT_DIR/output/"
echo "=============================================="

