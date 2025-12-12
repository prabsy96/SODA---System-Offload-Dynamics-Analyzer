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
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# Core directory structure
export SODA_SRC="$SODA_ROOT/src"
export SODA_EXAMPLES="$SODA_ROOT/examples"
export SODA_OUTPUT="$SODA_ROOT/output"

# Microbenchmark directories
export MICROBENCH_DIR="$SODA_SRC/microbench"
export BAREMETAL_MICROBENCH_DIR="$MICROBENCH_DIR/baremetal"
export FRAMEWORK_DIR="$MICROBENCH_DIR/framework"
export PYTORCH_MICROBENCH_DIR="$FRAMEWORK_DIR/pytorch"

# Microbenchmark output directories
export BAREMETAL_OUTPUT_DIR="microbench/baremetal"
export PYTORCH_OUTPUT_DIR="microbench/framework/pytorch"

# Microbenchmark script directories
export BAREMETAL_SCRIPTS="$BAREMETAL_MICROBENCH_DIR/scripts"
export PYTORCH_SCRIPTS="$PYTORCH_MICROBENCH_DIR/scripts"

# Build directories
export BAREMETAL_BUILD="$BAREMETAL_MICROBENCH_DIR/build"
export BAREMETAL_BINARY="$BAREMETAL_BUILD/main_gemm_bm"

# Virtual environment
export PYTHON_VENV="$SODA_ROOT/.venv"

# Environment metadata file
export ENV_METADATA="env_metadata.json"

# Experiment directory (set by tracer, DO NOT set manually)
export EXPERIMENT_DIR=""

# Common data files referenced across scripts (relative to experiment directory)
export ALL_SEQUENCES="sequences/all_sequences.json"
export ALL_GEMM_SEQUENCES="sequences/all_gemm_sequences.json"
export UNIQUE_GEMM_SEQUENCES="sequences/unique_gemm_sequences.json"

# Framework/pytorch
export PYTORCH_GEMM_SEQUENCES="$PYTORCH_OUTPUT_DIR/pytorch_gemm_profile.json"
export PYTORCH_TRACES="$PYTORCH_OUTPUT_DIR/traces"
export PYTORCH_LAUNCH_TAX_GRAPHS="$PYTORCH_OUTPUT_DIR/graphs/launch_tax"

# Baremetal
export BAREMETAL_JOBS="$BAREMETAL_OUTPUT_DIR/jobs.json"
export BAREMETAL_GEMM_KERNELS="$BAREMETAL_OUTPUT_DIR/baremetal_gemm_profile.json"
export BAREMETAL_TRACES="$BAREMETAL_OUTPUT_DIR/traces"

# Tax break files (relative to experiment directory)
export TAX_BREAK_SUMMARY="microbench/taxbreak.json"
export TAX_BREAK_PLOT="microbench/taxbreak.png"

# Log files (relative to experiment directory)
export PYTORCH_VERIFY_LOG="$PYTORCH_OUTPUT_DIR/microbench.log"
export ASSERT_LOG="assertions.log"

# HuggingFace cache (set default if not already set)
# export HF_HOME="${HF_HOME:-/tmp/hf_cache_$USER}"
export HF_HOME="/scratch/$USER/hf_cache"

# Python path setup for imports
export PYTHONPATH="$SODA_SRC:$PYTHONPATH"

# ============================================================
# Microbench paths
# ============================================================

# Raw sequences from trace
export ALL_SEQUENCES="microbench/all_sequences.json"

# All kernel sequences (GEMM + non-GEMM)
export ALL_KERNEL_SEQUENCES="microbench/all_kernel_sequences.json"
export UNIQUE_ALL_SEQUENCES="microbench/unique_all_sequences.json"

# GEMM-only sequences (for baremetal comparison)
export ALL_GEMM_SEQUENCES="microbench/all_gemm_sequences.json"
export UNIQUE_GEMM_SEQUENCES="microbench/unique_gemm_sequences.json"

# PyTorch profiling outputs
export PYTORCH_TRACES="microbench/framework/pytorch/traces"
export PYTORCH_GEMM_SEQUENCES="microbench/framework/pytorch/output/pytorch_gemm_sequences.json"
export PYTORCH_ALL_SEQUENCES="microbench/framework/pytorch/output/pytorch_all_sequences.json"

# Baremetal outputs (GEMM only - cuBLAS comparison)
export BAREMETAL_JOBS="microbench/baremetal/output/jobs.json"
export BAREMETAL_GEMM_KERNELS="microbench/baremetal/output/baremetal_gemm_kernels.json"

# TaxBreak report outputs
export TAX_BREAK_SUMMARY="microbench/taxbreak.json"
export TAX_BREAK_PLOT="microbench/taxbreak_plot.png"

# Logs
export ASSERT_LOG="microbench/assert.log"

# Helper function to activate Python environment (supports conda or venv)
activate_env() {
    if [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "base" ]; then
        echo "Using conda environment: $CONDA_DEFAULT_ENV"
    elif [ -d "$PYTHON_VENV" ]; then
        echo "Activating venv at $PYTHON_VENV"
        source "$PYTHON_VENV/bin/activate"
    elif [ -n "$CONDA_DEFAULT_ENV" ]; then
        echo "Using conda base environment"
    else
        echo "Warning: No virtual environment found. Using system Python."
    fi
}

# Helper function to cleanup output directory
cleanup() {
    if [ -d "$SODA_OUTPUT" ]; then
        echo "Deleting output directory: $SODA_OUTPUT"
        rm -rf "$SODA_OUTPUT"
        echo "Cleanup complete"
    else
        echo "Output directory does not exist: $SODA_OUTPUT"
    fi
}

# Helper function to reinstall the soda package
reinstall() {
    echo "Reinstalling soda package"
    pip install --ignore-installed --force-reinstall --no-deps -e "$SODA_ROOT"
    echo "Soda package reinstalled"
}

# Build the baremetal binary using env-provided paths
build() {
    local build_dir="${BAREMETAL_BUILD:-$SODA_ROOT/build}"
    local build_type="${CMAKE_BUILD_TYPE:-Release}"
    local jobs="${NUM_JOBS:-$(nproc)}"

    mkdir -p "$build_dir"
    cmake -S "$BAREMETAL_MICROBENCH_DIR" -B "$build_dir" -DCMAKE_BUILD_TYPE="$build_type"
    cmake --build "$build_dir" -- -j"$jobs"
    echo "Built binary at ${BAREMETAL_BINARY:-$build_dir/main_gemm_bm}"
}

# Print SODA banner when sourced
print_soda_banner() {
    # ANSI color codes for pastel colors
    PASTEL_RED='\033[38;5;217m'      # Pastel pink/red
    PASTEL_ORANGE='\033[38;5;223m'   # Pastel peach/orange
    PASTEL_YELLOW='\033[38;5;229m'   # Pastel yellow
    PASTEL_GREEN='\033[38;5;157m'    # Pastel mint/green
    BLUE='\033[38;5;33m'              # Previous nice blue
    RESET='\033[0m'                   # Reset color
    WHITE='\033[1;37m'                # Bright white
    
    # CMU brand color (official red) - hex #C41230, RGB: 196, 18, 48
    CMU_RED='\033[38;2;196;18;48m'
    
    # Banner width (63 characters between borders)
    BANNER_WIDTH=63
    
    # Helper function to center content in banner line
    format_banner_line() {
        local content="$1"
        # Strip ANSI escape sequences to count visible characters
        local visible=$(echo -e "$content" | sed 's/\x1b\[[0-9;]*m//g')
        local visible_len=${#visible}
        local total_padding=$((BANNER_WIDTH - visible_len))
        local left_padding=$((total_padding / 2))
        local right_padding=$((total_padding - left_padding))
        local left_spaces=$(printf "%*s" $left_padding "")
        local right_spaces=$(printf "%*s" $right_padding "")
        echo -e "${WHITE}║${RESET}${left_spaces}${content}${right_spaces}${WHITE}║${RESET}"
    }
    
    echo -e "${WHITE}╔═══════════════════════════════════════════════════════════════╗${RESET}"
    format_banner_line ""
    format_banner_line "${PASTEL_RED}███████╗${RESET} ${PASTEL_ORANGE}██████╗${RESET} ${PASTEL_YELLOW}██████╗${RESET}  ${PASTEL_GREEN}█████╗${RESET}"
    format_banner_line "${PASTEL_RED}██╔════╝${RESET}${PASTEL_ORANGE}██╔═══██╗${RESET}${PASTEL_YELLOW}██╔══██╗${RESET}${PASTEL_GREEN}██╔══██╗${RESET}"
    format_banner_line "${PASTEL_RED}███████╗${RESET}${PASTEL_ORANGE}██║   ██║${RESET}${PASTEL_YELLOW}██║  ██║${RESET}${PASTEL_GREEN}███████║${RESET}"
    format_banner_line "${PASTEL_RED}╚════██║${RESET}${PASTEL_ORANGE}██║   ██║${RESET}${PASTEL_YELLOW}██║  ██║${RESET}${PASTEL_GREEN}██╔══██║${RESET}"
    format_banner_line "${PASTEL_RED}███████║${RESET}${PASTEL_ORANGE}╚██████╔╝${RESET}${PASTEL_YELLOW}██████╔╝${RESET}${PASTEL_GREEN}██║  ██║${RESET}"
    format_banner_line "${PASTEL_RED}╚══════╝${RESET} ${PASTEL_ORANGE}╚═════╝${RESET} ${PASTEL_YELLOW}╚═════╝${RESET} ${PASTEL_GREEN}╚═╝  ╚═╝${RESET}"
    format_banner_line ""
    format_banner_line "${BLUE}System Offload Dynamics Analyzer${RESET}"
    format_banner_line "${CMU_RED}Carnegie Mellon University${RESET}"
    echo -e "${WHITE}╚═══════════════════════════════════════════════════════════════╝${RESET}"
}

# Print banner (unless SODA_ENV_QUIET is set)
if [ -z "${SODA_ENV_QUIET:-}" ]; then
    print_soda_banner
    echo ""
    echo "Get started:"
    echo "  * activate_env     - Activate Python virtual environment"
    echo "  * cleanup           - Delete output directory ($SODA_OUTPUT)"
    echo "  * build            - Build the baremetal binary"
    echo "  * reinstall    - Reinstall the soda package (use after making changes)"
    echo ""
fi
