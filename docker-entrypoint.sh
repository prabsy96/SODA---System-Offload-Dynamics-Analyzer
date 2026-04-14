#!/bin/bash
set -e

# Verify NVIDIA GPU access
if ! command -v nvidia-smi &>/dev/null; then
    echo "WARNING: nvidia-smi not found. Ensure --gpus all is passed to docker run."
fi

# Ensure output directory exists and is writable
mkdir -p "${SODA_OUTPUT}"

# Ensure HuggingFace cache directory exists
mkdir -p "${HF_HOME}"

# Verify baremetal binary exists
if [ ! -f "${BAREMETAL_BINARY}" ]; then
    echo "WARNING: Baremetal GEMM binary not found at ${BAREMETAL_BINARY}"
    echo "Microbenchmark mode (--microbench) will not work."
fi

# Print environment summary (suppress with SODA_QUIET=1)
if [ -z "${SODA_QUIET:-}" ]; then
    echo "============================================"
    echo "  SODA - System Offload Dynamics Analyzer"
    echo "============================================"
    echo "  SODA_ROOT:     ${SODA_ROOT}"
    echo "  SODA_OUTPUT:   ${SODA_OUTPUT}"
    echo "  HF_HOME:       ${HF_HOME}"
    echo "  HF_TOKEN:      ${HF_TOKEN:+set (hidden)}${HF_TOKEN:-not set}"
    echo "  BAREMETAL:     ${BAREMETAL_BINARY}"
    if command -v nvidia-smi &>/dev/null; then
        echo "  GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
    fi
    echo "============================================"
    echo ""
fi

# Execute the provided command
exec "$@"
