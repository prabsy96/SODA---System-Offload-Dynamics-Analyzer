#!/bin/bash
# =============================================================================
# SODA Unit-Test Runner — SLURM batch script
# =============================================================================
# Runs the full pytest suite (225 tests, no GPU required).
#
# Usage:
#   sbatch slurm/run_tests.sh
#
# Output files (written to the repo root):
#   tests_<JOBID>.out   — combined stdout/stderr
#   tests_<JOBID>.err   — separate stderr (usually empty)
# =============================================================================

#SBATCH --job-name=soda_tests
#SBATCH --output=tests_%j.out
#SBATCH --error=tests_%j.err
#SBATCH -t 0-00:15:00              # 15 minutes is far more than needed (~5 s)

# Resource request — tests are CPU-only.
# If your cluster has a dedicated CPU partition, prefer that:
#   #SBATCH -p cpu
# Otherwise, request a single GPU node with minimal GPU allocation:
#SBATCH -p HGPU
#SBATCH --nodes=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# GPU not required; comment the next line in if your partition mandates it:
# #SBATCH --gres=gpu:1

# =============================================================================
# CONFIGURATION — adjust paths if your layout differs
# =============================================================================

SODA_PROJECT_ROOT="$HOME/SODA/SODA---System-Offload-Dynamics-Analyzer"
PYTHON="/home/pvellais/miniconda3/bin/python3.13"   # must be >= 3.11

# =============================================================================
# SETUP
# =============================================================================

echo "============================================================================"
echo "SODA Unit Tests"
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $(hostname)"
echo "Started : $(date)"
echo "Python  : $($PYTHON --version 2>&1)"
echo "Repo    : $SODA_PROJECT_ROOT"
echo "============================================================================"
echo ""

cd "$SODA_PROJECT_ROOT" || { echo "ERROR: repo root not found: $SODA_PROJECT_ROOT"; exit 1; }

# Ensure the package is importable (editable install) — needed if PYTHONPATH
# is not set.  The -q flag suppresses pip noise; failures are non-fatal because
# PYTHONPATH=src is set explicitly below.
"$PYTHON" -m pip install -e . --quiet 2>/dev/null || true

# =============================================================================
# RUN TESTS
# =============================================================================

echo "Running full pytest suite ..."
echo ""

PYTHONPATH=src "$PYTHON" -m pytest tests/ -v --tb=short 2>&1
EXIT_CODE=$?

echo ""
echo "============================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Result  : ALL TESTS PASSED"
else
    echo "Result  : SOME TESTS FAILED (exit code $EXIT_CODE)"
fi
echo "Finished: $(date)"
echo "============================================================================"

exit $EXIT_CODE
