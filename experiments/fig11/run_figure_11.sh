#!/usr/bin/env bash
# Launch Figure 11 prefill sweeps on one local CUDA platform.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/results/fig11/${PLATFORM_LABEL:?Set PLATFORM_LABEL (for example intel-h100)}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

for model in gpt2 llama-3.2-1b; do
  for batch_size in 1 2 4 8 16; do
    run_dir="${RESULTS_DIR}/${model}/bs${batch_size}"
    "${PYTHON_BIN}" "${ROOT_DIR}/experiments/fig11/benchmark_prefill.py" \
      --model "${model}" \
      --batch-size "${batch_size}" \
      --platform-label "${PLATFORM_LABEL}" \
      --seq-len 512 \
      --warmup "${WARMUP:-20}" \
      --runs "${RUNS:-50}" \
      --output-dir "${run_dir}"
  done
done

"${PYTHON_BIN}" "${ROOT_DIR}/experiments/fig11/plot_figure_11.py" \
  --results-dir "${RESULTS_DIR}" \
  --output "${RESULTS_DIR}/figure_11_${PLATFORM_LABEL}.png"
