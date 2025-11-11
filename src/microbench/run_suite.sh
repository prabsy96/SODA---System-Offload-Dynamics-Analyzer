#!/bin/bash
# Run complete microbenchmark suite: framework + bare-metal
# This orchestrates both pipelines end-to-end

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SODA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PATH="$SODA_ROOT/.venv"

# Activate virtual environment
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

source "$VENV_PATH/bin/activate"

echo "=============================================="
echo "Kernel Launch Tax Microbenchmark Suite"
echo "=============================================="
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Phase 1: Run PyTorch profiling
echo "=== Phase 1: PyTorch Profiling ==="
cd "$SCRIPT_DIR/framework/pytorch"
./run_pytorch.sh
echo ""

# Phase 2: Run bare-metal profiling
echo "=== Phase 2: Bare-Metal Profiling ==="
cd "$SCRIPT_DIR/baremetal"
./run_baremetal.sh

echo ""
echo "=============================================="
echo "Done! Check results in:"
echo "  PyTorch:    framework/pytorch/output/"
echo "  Bare-metal: baremetal/output/"
echo "  Comparison: baremetal/output/bm_vs_framework_report.json"
echo "=============================================="

