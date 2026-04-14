#!/bin/bash
# ============================================================================
# ΔCT Validation Experiment for Reviewer D
# ============================================================================
#
# Purpose: Validate ΔCT (CUDA Library Translation overhead) calculation
# by comparing direct trace analysis vs replay-based measurement.
#
# Methodology:
#   1. Direct mode: ΔCT = max(0, T_aten_xlat - T_aten_base)
#      - T_aten_base = median of non-GEMM kernel T_aten_xlat values
#      - This is a "subtractive lower bound" estimate
#
#   2. Replay mode: Isolates GEMM ops and measures:
#      - PyTorch: T_py + T_aten_xlat + T_sys (full framework overhead)
#      - Baremetal: T_culib_xlat + T_shim + T_sys (direct cuBLAS)
#      - The difference isolates the CUDA library front-end overhead (ΔCT)
#
# Output: Both direct and replay results are saved for comparison analysis.
# ============================================================================

set -e

# Check running from root directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the root directory of the SODA repository."
    exit 1
fi

# Check if SODA environment is loaded
if [ -z "$SODA_ENV_LOADED" ]; then
    echo "Error: SODA environment not loaded."
    echo "Please run: source env.sh"
    exit 1
fi

# Activate Python environment
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Using conda environment: $CONDA_DEFAULT_ENV"
elif [ -d "$PYTHON_VENV" ]; then
    echo "Activating virtual environment at $PYTHON_VENV"
    source "$PYTHON_VENV/bin/activate"
else
    echo "Using system Python"
fi

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Detect GPU
GPU_NAME=$(python -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0).replace('NVIDIA ', '').replace(' ', '_')
    for key in ['H100', 'H200', 'A100', 'V100', 'A10', 'L40', 'RTX']:
        if key in name.upper():
            print(key)
            exit()
    print(name.split()[0])
else:
    print('unknown_gpu')
" 2>/dev/null || echo "unknown_gpu")

echo "Detected GPU: $GPU_NAME"
echo ""

# ============================================================================
# Experiment Configuration
# ============================================================================
# Use GPT-2 for controlled comparison (matches paper Section V baseline)
MODEL="gpt2"
PRECISION="bfloat16"
COMPILE_TYPE="eager"

# Single configuration for detailed analysis
# BS=1 forces host-bound regime where ΔCT is most significant
BATCH_SIZE=1
SEQ_LEN=512
MAX_NEW_TOKENS=1  # Prefill only

# Microbench parameters
# Use modest runs for faster turnaround while maintaining statistical validity
WARMUP=50
RUNS=100

# IMPORTANT: Both modes (direct and replay) use the SAME trace data source.
# They differ only in HOW the analysis is performed:
#   - Direct: Analyzes timing from the original trace directly
#   - Replay: Re-profiles each unique kernel in isolation for comparison
#
# The kernel identity is preserved via:
#   - ATen op name + input_dims + input_type (for matching)
#   - kernel name + grid/block/shared_mem (for grouping)

echo "============================================================================"
echo "ΔCT Validation Experiment (Reviewer D)"
echo "============================================================================"
echo "Model: $MODEL"
echo "Configuration: BS=$BATCH_SIZE, SL=$SEQ_LEN, MaxNewTokens=$MAX_NEW_TOKENS"
echo "Precision: $PRECISION"
echo "Warmup: $WARMUP, Runs: $RUNS"
echo "============================================================================"
echo ""

# Build baremetal binary (required for replay mode)
echo "Building baremetal binary..."
build
echo ""

# Create output directory for validation results
VALIDATION_DIR="$SODA_OUTPUT/delta_ct_validation_${GPU_NAME}"
mkdir -p "$VALIDATION_DIR"

# ============================================================================
# Step 1: Run Direct Trace Analysis (Subtractive Method)
# ============================================================================
echo ""
echo "============================================================================"
echo "STEP 1: Direct Trace Analysis (Subtractive ΔCT)"
echo "============================================================================"
echo "Running: python -m soda --microbench --direct-trace"
echo "Output: $VALIDATION_DIR/direct/"
echo ""

python -m soda \
  --model "$MODEL" \
  --batch-size "$BATCH_SIZE" \
  --seq-len "$SEQ_LEN" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --precision "$PRECISION" \
  --compile-type "$COMPILE_TYPE" \
  --warmup "$WARMUP" \
  --runs "$RUNS" \
  --microbench \
  --direct-trace \
  --output-dir "$VALIDATION_DIR/direct/"

echo ""
echo "Direct trace analysis complete."
echo "Results saved to: $VALIDATION_DIR/direct/"

# ============================================================================
# Step 2: Run Replay Analysis (PyTorch + Baremetal cuBLAS)
# ============================================================================
echo ""
echo "============================================================================"
echo "STEP 2: Replay Analysis (PyTorch vs Baremetal cuBLAS)"
echo "============================================================================"
echo "Running: python -m soda --microbench --replay"
echo "Output: $VALIDATION_DIR/replay/"
echo ""
echo "This step will:"
echo "  1. Extract unique GEMM sequences from trace"
echo "  2. Profile each GEMM in isolation via PyTorch"
echo "  3. Profile each GEMM via baremetal cuBLAS (C++ binary)"
echo "  4. Compare overhead components"
echo ""

python -m soda \
  --model "$MODEL" \
  --batch-size "$BATCH_SIZE" \
  --seq-len "$SEQ_LEN" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --precision "$PRECISION" \
  --compile-type "$COMPILE_TYPE" \
  --warmup "$WARMUP" \
  --runs "$RUNS" \
  --microbench \
  --replay \
  --output-dir "$VALIDATION_DIR/replay/"

echo ""
echo "Replay analysis complete."
echo "Results saved to: $VALIDATION_DIR/replay/"

# ============================================================================
# Step 3: Generate Comparison Summary
# ============================================================================
echo ""
echo "============================================================================"
echo "STEP 3: Comparison Summary"
echo "============================================================================"
echo ""

# Export GPU_NAME for Python script
export GPU_NAME

# Run the dedicated validation analysis script
python "$SODA_ROOT/experiments/rebuttal/validate_ct.py" \
  --direct-dir "$VALIDATION_DIR/direct/" \
  --replay-dir "$VALIDATION_DIR/replay/" \
  --output "$VALIDATION_DIR/validation_report.json"

echo ""
echo "============================================================================"
echo "Experiment Complete"
echo "============================================================================"
echo "Results saved to: $VALIDATION_DIR"
echo ""
echo "Key files:"
echo "  - Direct trace: $VALIDATION_DIR/direct/microbench/taxbreak_summary.json"
echo "  - Replay:       $VALIDATION_DIR/replay/microbench/taxbreak_summary.json"
echo "  - Baremetal:    $VALIDATION_DIR/replay/microbench/baremetal/output/"
echo ""
