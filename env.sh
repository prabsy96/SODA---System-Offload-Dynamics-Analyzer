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
export SODA_OUTPUT="$SODA_ROOT/output"

# Microbenchmark directories
export MICROBENCH_DIR="$SODA_SRC/microbench"
export BAREMETAL_MICROBENCH_DIR="$MICROBENCH_DIR/baremetal"
export FRAMEWORK_DIR="$MICROBENCH_DIR/framework"
export PYTORCH_MICROBENCH_DIR="$FRAMEWORK_DIR/pytorch"

# Microbenchmark output directories
export BAREMETAL_OUTPUT="$SODA_OUTPUT/microbench/baremetal"
export PYTORCH_OUTPUT="$SODA_OUTPUT/microbench/framework/pytorch"

# Microbenchmark script directories
export BAREMETAL_SCRIPTS="$BAREMETAL_MICROBENCH_DIR/scripts"
export PYTORCH_SCRIPTS="$PYTORCH_MICROBENCH_DIR/scripts"

# Build directories
export BAREMETAL_BUILD="$BAREMETAL_MICROBENCH_DIR/build"

# Virtual environment
export PYTHON_VENV="$SODA_ROOT/.venv"

# Common data files referenced across scripts
export PYTORCH_UNIQUE_KERNELS="$PYTORCH_OUTPUT/unique_gemm_kernel_sequences.json"
export PYTORCH_ALL_KERNELS="$PYTORCH_OUTPUT/all_kernel_sequences.json"
export PYTORCH_GEMM_KERNELS="$PYTORCH_OUTPUT/gemm_kernel_sequences.json"
export PYTORCH_REPLAYED_KERNELS="$PYTORCH_OUTPUT/replayed_gemm_kernel_sequences.json"

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
    echo "SODA_EXAMPLES: $SODA_EXAMPLES"
    echo "SODA_OUTPUT: $SODA_OUTPUT"
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
    echo "  * print_soda_env    - Show all environment variables and paths"
    echo "  * activate_venv     - Activate Python virtual environment"
    echo ""
fi
