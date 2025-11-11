#!/bin/bash
# Quickstart example for SODA


# Check if SODA environment is loaded
if [ -z "$SODA_ENV_LOADED" ]; then
    echo "Error: SODA environment not loaded."
    echo "Please run: source env.sh"
    exit 1
fi

# Check running from root directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the root directory of the SODA repository."
    exit 1
fi

# Run SODA
python -m soda.main \
  --model gpt2 \
  --output-dir "$SODA_RESULTS" \
  --batch_size 1 \
  --seq_len 128 \
  --fusion 2 3 \
  --prox_score 1.0