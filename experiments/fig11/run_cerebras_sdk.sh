#!/usr/bin/env bash
# Launch the Cerebras SDK batch-prefill sweep and plot mean ± 95% CI.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/results/fig11/cerebras-sdk}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-gpt-oss-120b}"

: "${CEREBRAS_API_KEY:?Set CEREBRAS_API_KEY before launching the SDK experiment.}"
for batch_size in 1 2 4 8 16; do
  "${PYTHON_BIN}" "${ROOT_DIR}/experiments/fig11/cerebras_sdk_sweep.py" \
    --model "${MODEL}" \
    --batch-size "${batch_size}" \
    --warmup "${WARMUP:-5}" \
    --trials "${TRIALS:-30}" \
    --output-dir "${RESULTS_DIR}/bs${batch_size}"
done

"${PYTHON_BIN}" "${ROOT_DIR}/experiments/fig11/plot_cerebras_sdk.py" \
  --results-dir "${RESULTS_DIR}" \
  --output "${RESULTS_DIR}/figure_11_cerebras_sdk.png"
