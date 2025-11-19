#!/bin/bash
# Run baremetal GEMM launch tax microbenchmark
# Compares framework (PyTorch) vs baremetal kernel launch overhead

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
echo "Baremetal GEMM Profiling"
echo "=============================================="
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Check if PyTorch results exist
if [ ! -f "$PYTORCH_UNIQUE_KERNELS" ]; then
    echo "WARNING: PyTorch results not found at:"
    echo "  $PYTORCH_UNIQUE_KERNELS"
    echo ""
    echo "You need to run the PyTorch profiling pipeline first:"
    echo "  cd $PYTORCH_MICROBENCH_DIR && ./run_pytorch.sh"
    echo ""
    read -p "Run PyTorch pipeline now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "=== Running PyTorch Profiling Pipeline ==="
        cd "$PYTORCH_MICROBENCH_DIR"
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
echo "=== Phase 1: Generate Baremetal Jobs ==="
python scripts/gen_bm_jobs.py
echo ""

# Phase 2: Search for algorithm indices (offline, can run anytime)
echo "=== Phase 2: Offline Search for Algorithm Indices ==="
echo "Searching for cuBLASLt algorithm indices for each job..."
python scripts/search_algorithm_indices.py
echo ""

# Phase 3: Profile baremetal (uses matched algorithms)
echo "=== Phase 3: Profile Baremetal (under nsys profiling) ==="
echo "This will run nsys profiling for multiple jobs, may take several minutes..."
python scripts/profile_baremetal.py
echo ""

# Phase 4: Compare results
echo "=== Phase 4: Compare PyTorch vs Baremetal ==="
python scripts/compare_kernel_tax.py

echo ""
echo "=============================================="
echo "Done! Check results in:"
echo "  - $BAREMETAL_JOBS"
echo "  - $BAREMETAL_RUNS"
echo "  - $BAREMETAL_REPORT"
echo "=============================================="

