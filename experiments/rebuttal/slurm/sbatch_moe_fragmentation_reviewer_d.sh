#!/bin/bash
#SBATCH -J moe_fragmentation
#SBATCH -o moe_fragmentation.out
#SBATCH -e moe_fragmentation.err
#SBATCH -t 0-06:00:00

#SBATCH -p HGPU
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=180G

# ============================================
# MoE Fragmentation Analysis for Reviewer D Q2
# ============================================
# Disambiguates host orchestration overhead vs device-side fragmentation
# in MoE models using TaxBreak decomposition and HDBI.
#
# This script:
# 1. Generates SODA traces for OLMoE-1B-7B (MoE) and Llama-3.2-1B (Dense)
#    ONLY if report.json doesn't already exist
# 2. Runs report-driven fragmentation analysis with HDBI comparison
# 3. Optional sampled gap analysis from trace files
# ============================================

echo "============================================"
echo "MoE Fragmentation Analysis - Reviewer D Q2"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "============================================"

# Load modules
module load cuda12.6/toolkit

# Activate environment
export PATH=~/miniconda3/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

SODA_PROJECT_ROOT="/home/pvellais/SODA---System-Offload-Dynamics-Analyzer"

if [ -f "$SODA_PROJECT_ROOT/env.sh" ]; then
    source "$SODA_PROJECT_ROOT/env.sh"
else
    echo "Error: env.sh not found"
    exit 1
fi

cd "$SODA_ROOT"

# Verify environment
echo "SODA_ENV_LOADED: $SODA_ENV_LOADED"
echo "SODA_ROOT: $SODA_ROOT"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

pip install -e "$SODA_PROJECT_ROOT" --quiet 2>/dev/null

OUTPUT_BASE="output/moe_fragmentation_analysis"

# ============================================
# Step 1: Generate traces if needed
# ============================================

OLMOE_DIR="${OUTPUT_BASE}/allenai_OLMoE-1B-7B-0924_eager_bfloat16_bs1_sl512_mt10"
LLAMA_DIR="${OUTPUT_BASE}/meta-llama_Llama-3.2-1B_eager_bfloat16_bs1_sl512_mt10"

OLMOE_REPORT="${OLMOE_DIR}/report.json"
LLAMA_REPORT="${LLAMA_DIR}/report.json"

# OLMoE trace generation (only if report.json missing)
if [ -f "$OLMOE_REPORT" ]; then
    echo ""
    echo "OLMoE report already exists: $OLMOE_REPORT"
    echo "Skipping trace generation."
else
    echo ""
    echo "============================================"
    echo "Generating OLMoE-1B-7B trace (decode mode)"
    echo "============================================"
    python -u -c "
from soda import ModelTracer, SodaAnalyzer
from soda.common import utils

args = utils.parse_and_validate_args([
    '--model', 'allenai/OLMoE-1B-7B-0924',
    '--output-dir', '${OUTPUT_BASE}',
    '--batch-size', '1',
    '--seq-len', '512',
    '--max-new-tokens', '10',
    '--precision', 'bfloat16',
    '--compile-type', 'eager',
    '--device', 'cuda',
    '--warmup', '3',
])

tracer = ModelTracer(args=args)
tracer.run()
analyzer = SodaAnalyzer(tracer=tracer, args=args)
analyzer.run()
print(f'OLMoE trace saved to: {tracer.trace_file}')
"
fi

# Llama trace generation (only if report.json missing)
if [ -f "$LLAMA_REPORT" ]; then
    echo ""
    echo "Llama report already exists: $LLAMA_REPORT"
    echo "Skipping trace generation."
else
    echo ""
    echo "============================================"
    echo "Generating Llama-3.2-1B trace (decode mode)"
    echo "============================================"
    python -u -c "
from soda import ModelTracer, SodaAnalyzer
from soda.common import utils

args = utils.parse_and_validate_args([
    '--model', 'meta-llama/Llama-3.2-1B',
    '--output-dir', '${OUTPUT_BASE}',
    '--batch-size', '1',
    '--seq-len', '512',
    '--max-new-tokens', '10',
    '--precision', 'bfloat16',
    '--compile-type', 'eager',
    '--device', 'cuda',
    '--warmup', '3',
])

tracer = ModelTracer(args=args)
tracer.run()
analyzer = SodaAnalyzer(tracer=tracer, args=args)
analyzer.run()
print(f'Llama trace saved to: {tracer.trace_file}')
"
fi

# ============================================
# Step 2: Run Fragmentation Analysis
# ============================================
echo ""
echo "============================================"
echo "Step 2: MoE vs Dense Fragmentation Analysis"
echo "============================================"

if [ ! -f "$OLMOE_REPORT" ] || [ ! -f "$LLAMA_REPORT" ]; then
    echo "Error: Missing report files"
    echo "  OLMoE: $OLMOE_REPORT (exists: $([ -f "$OLMOE_REPORT" ] && echo yes || echo no))"
    echo "  Llama: $LLAMA_REPORT (exists: $([ -f "$LLAMA_REPORT" ] && echo yes || echo no))"
    exit 1
fi

# Find trace files for optional gap analysis
OLMOE_TRACE="${OLMOE_DIR}/trace.json"
LLAMA_TRACE="${LLAMA_DIR}/trace.json"

# Build the analysis command
ANALYSIS_CMD="python -u $SODA_ROOT/experiments/rebuttal/moe_fragmentation_analysis.py \
    --moe-report $OLMOE_REPORT \
    --dense-report $LLAMA_REPORT \
    --output-dir ${OUTPUT_BASE} \
    --profiling-runs 150"

# Only add trace paths for gap analysis if traces are < 10GB
if [ -f "$LLAMA_TRACE" ]; then
    LLAMA_SIZE=$(stat --printf="%s" "$LLAMA_TRACE" 2>/dev/null || echo 0)
    if [ "$LLAMA_SIZE" -lt 10737418240 ]; then
        ANALYSIS_CMD="$ANALYSIS_CMD --dense-trace $LLAMA_TRACE"
        echo "Including Llama trace for gap analysis ($(numfmt --to=iec $LLAMA_SIZE))"
    else
        echo "Skipping Llama trace gap analysis (too large: $(numfmt --to=iec $LLAMA_SIZE))"
    fi
fi

# OLMoE trace is typically 69GB+ — skip gap analysis for it
if [ -f "$OLMOE_TRACE" ]; then
    OLMOE_SIZE=$(stat --printf="%s" "$OLMOE_TRACE" 2>/dev/null || echo 0)
    if [ "$OLMOE_SIZE" -lt 10737418240 ]; then
        ANALYSIS_CMD="$ANALYSIS_CMD --moe-trace $OLMOE_TRACE"
        echo "Including OLMoE trace for gap analysis ($(numfmt --to=iec $OLMOE_SIZE))"
    else
        echo "Skipping OLMoE trace gap analysis (too large: $(numfmt --to=iec $OLMOE_SIZE))"
    fi
fi

echo ""
eval $ANALYSIS_CMD

echo ""
echo "============================================"
echo "MoE Fragmentation Analysis Complete"
echo "Finished: $(date)"
echo "Results: ${OUTPUT_BASE}/"
echo "============================================"
