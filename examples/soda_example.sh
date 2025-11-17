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

# Activate virtual environment
if [ ! -d "$PYTHON_VENV" ]; then
    echo "Error: Virtual environment not found at $PYTHON_VENV"
    echo "Create it with: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

source "$PYTHON_VENV/bin/activate"

# Run SODA using CLI command
soda-cli \
  --model gpt2 \
  --output-dir "$SODA_OUTPUT" \
  --batch-size 1 \
  --seq-len 128 \
  --fusion 2 3 \
  --prox-score 1.0

# # Alternative: Run SODA using Python directly
# python src/soda.py \
#   --model gpt2 \
#   --output-dir "$SODA_OUTPUT" \
#   --batch-size 1 \
#   --seq-len 128 \
#   --fusion 2 3 \
#   --prox-score 1.0

# # Alternative: Run SODA using Python module format
# python -m soda \
#   --model gpt2 \
#   --output-dir "$SODA_OUTPUT" \
#   --batch-size 1 \
#   --seq-len 128 \
#   --fusion 2 3 \
#   --prox-score 1.0