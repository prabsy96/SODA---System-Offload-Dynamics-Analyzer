#!/bin/bash
# SODA Environment Configuration
# Source this file to set up all path variables for the SODA repository
#
# Usage: source env.sh (or . env.sh)
#

# Determine SODA_ROOT dynamically (location of this script)
export SODA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Flag to indicate env.sh has been sourced
export SODA_ENV_LOADED=1

# Core directory structure
export SODA_SRC="$SODA_ROOT/src"
export SODA_EXAMPLES="$SODA_ROOT/examples"
export SODA_RESULTS="$SODA_ROOT/soda_results"

# SODA Python package
export SODA_PACKAGE="$SODA_SRC/soda"

# Microbenchmark directories
export MICROBENCH_DIR="$SODA_SRC/microbench"
export BAREMETAL_MICROBENCH_DIR="$MICROBENCH_DIR/baremetal"
export FRAMEWORK_DIR="$MICROBENCH_DIR/framework"
export PYTORCH_MICROBENCH_DIR="$FRAMEWORK_DIR/pytorch"

# Microbenchmark output directories
export BAREMETAL_OUTPUT="$BAREMETAL_MICROBENCH_DIR/output"
export PYTORCH_OUTPUT="$PYTORCH_MICROBENCH_DIR/output"

# Microbenchmark script directories
export BAREMETAL_SCRIPTS="$BAREMETAL_MICROBENCH_DIR/scripts"
export PYTORCH_SCRIPTS="$PYTORCH_MICROBENCH_DIR/scripts"

# Build directories
export BAREMETAL_BUILD="$BAREMETAL_MICROBENCH_DIR/build"

# Virtual environment
export PYTHON_VENV="$SODA_ROOT/.venv"

# Common data files referenced across scripts
export PYTORCH_UNIQUE_KERNELS="$PYTORCH_OUTPUT/unique_gemm_kernel_chains.json"
export PYTORCH_ALL_KERNELS="$PYTORCH_OUTPUT/all_kernel_chains.json"
export PYTORCH_GEMM_KERNELS="$PYTORCH_OUTPUT/gemm_kernel_chains.json"
export PYTORCH_REPLAYED_KERNELS="$PYTORCH_OUTPUT/replayed_gemm_kernel_chains.json"

export BAREMETAL_JOBS="$BAREMETAL_OUTPUT/jobs.json"
export BAREMETAL_RUNS="$BAREMETAL_OUTPUT/baremetal_gemm_runs.json"
export BAREMETAL_REPORT="$BAREMETAL_OUTPUT/bm_vs_framework_report.json"

# Trace directories
export BAREMETAL_TRACES="$BAREMETAL_OUTPUT/traces"
export PYTORCH_TRACES="$PYTORCH_OUTPUT/traces"

# Graphs output
export PYTORCH_GRAPHS="$PYTORCH_OUTPUT/graphs"

# HuggingFace cache (set default if not already set)
export HF_HOME="${HF_HOME:-/tmp/hf_cache_$USER}"

# Python path setup for imports
export PYTHONPATH="$SODA_SRC:$PYTHONPATH"

# Helper function to activate virtual environment
activate_venv() {
    if [ -d "$PYTHON_VENV" ]; then
        source "$PYTHON_VENV/bin/activate"
        echo "Virtual environment activated: $PYTHON_VENV"
    else
        echo "Warning: Virtual environment not found at $PYTHON_VENV"
        return 1
    fi
}

# Helper function to print all paths (for debugging)
print_soda_env() {
    echo "=== SODA Environment Variables ==="
    echo "SODA_ROOT: $SODA_ROOT"
    echo "SODA_SRC: $SODA_SRC"
    echo "SODA_PACKAGE: $SODA_PACKAGE"
    echo "PYTHON_VENV: $PYTHON_VENV"
    echo ""
    echo "=== Microbenchmark Directories ==="
    echo "MICROBENCH_DIR: $MICROBENCH_DIR"
    echo "BAREMETAL_MICROBENCH_DIR: $BAREMETAL_MICROBENCH_DIR"
    echo "PYTORCH_MICROBENCH_DIR: $PYTORCH_MICROBENCH_DIR"
    echo ""
    echo "=== Output Directories ==="
    echo "BAREMETAL_OUTPUT: $BAREMETAL_OUTPUT"
    echo "PYTORCH_OUTPUT: $PYTORCH_OUTPUT"
    echo ""
    echo "=== Key Files ==="
    echo "PYTORCH_UNIQUE_KERNELS: $PYTORCH_UNIQUE_KERNELS"
    echo "BAREMETAL_JOBS: $BAREMETAL_JOBS"
    echo "BAREMETAL_RUNS: $BAREMETAL_RUNS"
    echo "BAREMETAL_REPORT: $BAREMETAL_REPORT"
    echo "=================================="
}

# Optional: Print confirmation when sourced
if [ "${SODA_ENV_VERBOSE:-0}" = "1" ]; then
    echo "SODA environment loaded from: $SODA_ROOT"
fi

