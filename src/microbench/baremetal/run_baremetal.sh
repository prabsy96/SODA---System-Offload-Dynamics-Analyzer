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

