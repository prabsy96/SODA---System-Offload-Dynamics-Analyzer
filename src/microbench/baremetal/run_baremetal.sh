#!/bin/bash
# Run bare-metal GEMM launch tax microbenchmark
# Compares framework (PyTorch) vs bare-metal kernel launch overhead
# Uses the .venv from scratchpad/pytorch

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MICROBENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# Go up 3 levels: baremetal -> microbench -> src -> soda
SODA_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV_PATH="$SODA_ROOT/.venv"

# Activate virtual environment
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

source "$VENV_PATH/bin/activate"

echo "=============================================="
echo "Bare-Metal Kernel Launch Tax Suite"
echo "=============================================="
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Check if PyTorch results exist
PYTORCH_RESULTS="$MICROBENCH_DIR/framework/pytorch/output/unique_gemm_kernel_chains.json"
if [ ! -f "$PYTORCH_RESULTS" ]; then
    echo "WARNING: PyTorch results not found at:"
    echo "  framework/pytorch/output/unique_gemm_kernel_chains.json"
    echo ""
    echo "You need to run the PyTorch profiling pipeline first:"
    echo "  cd framework/pytorch && ./run_pytorch.sh"
    echo ""
    read -p "Run PyTorch pipeline now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "=== Running PyTorch Profiling Pipeline ==="
        cd "$MICROBENCH_DIR/framework/pytorch"
        ./run_pytorch.sh
        cd "$SCRIPT_DIR"
        echo ""
    else
        echo "Exiting. Run PyTorch pipeline first."
        exit 1
    fi
fi

cd "$SCRIPT_DIR"

# Phase 1: Generate jobs
echo "=== Phase 1: Generate Bare-Metal Jobs ==="
python scripts/gen_bm_jobs.py
echo ""

# Phase 2: Run bare-metal suite (NOTE: This runs nsys, may take a while)
echo "=== Phase 2: Run Bare-Metal Suite (under nsys profiling) ==="
echo "WARNING: This will run nsys profiling for 7 jobs, may take several minutes..."
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python scripts/run_bm_suite.py
    echo ""
    
    # Phase 3: Compare results
    echo "=== Phase 3: Compare PyTorch vs Bare-Metal ==="
    python scripts/compare_kernel_tax.py
else
    echo "Skipped bare-metal suite execution"
fi

echo ""
echo "=============================================="
echo "Done! Check results in:"
echo "  - baremetal/output/jobs.json"
echo "  - baremetal/output/baremetal_gemm_runs.json"
echo "  - baremetal/output/bm_vs_framework_report.json"
echo "=============================================="

