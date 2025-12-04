#!/bin/bash
# SODA example run script

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

# Activate Python environment (supports conda or venv)
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

# Verify soda-cli is available, install if not
if ! command -v soda-cli &> /dev/null; then
    echo "Installing soda package..."
    pip install -e "$SODA_ROOT" --quiet
fi

# Run SODA using CLI command
soda-cli --model "meta-llama/Llama-3.2-3B" \
  --output-dir "$SODA_OUTPUT" \
  --batch-size 1 \
  --seq-len 512 \
  --max-new-tokens 1 \
  --fusion 2 3 \
  --prox-score 1.0

# # Alternative: Run SODA using Python directly
# python src/soda.py \
#   --model gpt2 \
#   --output-dir "$SODA_OUTPUT" \
#   --batch-size 1 \
#   --seq-len 128 \
#   --max-new-tokens 1 \
#   --fusion 2 3 \
#   --prox-score 1.0

# # Alternative: Run SODA using Python module format
# python -m soda \
#   --model gpt2 \
#   --output-dir "$SODA_OUTPUT" \
#   --batch-size 1 \
#   --seq-len 128 \
#   --max-new-tokens 1 \
#   --fusion 2 3 \
#   --prox-score 1.0
