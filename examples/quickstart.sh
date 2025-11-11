if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the root directory of the SODA repository."
    exit 1
fi
export HF_HOME=${HF_HOME:-/tmp/hf}
python -m soda.main \
  --model gpt2 \
  --output-dir ./soda_results \
  --batch_size 1 \
  --seq_len 128 \
  --fusion 2 3 \
  --prox_score 1.0