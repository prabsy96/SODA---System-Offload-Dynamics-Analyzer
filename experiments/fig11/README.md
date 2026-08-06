# Figure 11 reproduction

This experiment reproduces the paper's decoder-only prefill protocol:

- models: GPT-2 and Llama-3.2-1B;
- eager PyTorch FP16 forward pass with KV-cache creation;
- fixed 512-token synthetic input;
- batches `1, 2, 4, 8, 16`;
- latency from CUDA events, plus Chrome traces for idle metrics.

It must run on each physical CUDA host independently. The original Figure 11
compares AMD+A100, Intel+H100, and GH200. Do not use `cszoo validate` or the
public Cerebras Inference SDK for this protocol: neither gives a local eager
prefill trace or exposes the CPU/GPU idle metrics.

## Prerequisites

Use a CUDA-enabled environment with PyTorch, `transformers`, and `matplotlib`.
For the Llama model, accept the model license and authenticate before launch:

```bash
huggingface-cli login
```

Record the host CPU, GPU, driver, CUDA, PyTorch, and Transformers versions in
the experiment log. Pin the same software stack across all hosts wherever
possible. The script records GPU, CUDA runtime, and PyTorch versions in each
`result.json`.

## Launch

From the repository root, on each CUDA platform:

```bash
export PLATFORM_LABEL=intel-h100
export RESULTS_DIR="$PWD/results/fig11/$PLATFORM_LABEL"
experiments/fig11/run_figure_11.sh
```

For a quick smoke test before reserving hardware:

```bash
python3 experiments/fig11/benchmark_prefill.py \
  --model gpt2 --batch-size 1 --warmup 2 --runs 3 \
  --output-dir results/fig11/smoke/gpt2/bs1
```

Copy every platform directory below one common `results/fig11/` directory, then
make the combined six-panel plot:

```bash
python3 experiments/fig11/plot_figure_11.py \
  --results-dir results/fig11 \
  --output results/fig11/figure_11_reproduction.png
```

## Metric scope

`gpu_idle_ms` is trace span minus union-of-CUDA-kernel time. It is comparable
across the local CUDA hosts when the same tracing setup is used.

`cpu_idle_proxy_ms` is trace span minus union-of-PyTorch CPU-op time. It is a
useful host-wait proxy, but is not physical per-core CPU idleness; label it as a
proxy in any result that cannot obtain the paper's original SKIP measurement.

The Cerebras Inference SDK belongs in a separate service-level extension that
reports TTFT and request concurrency—not in this Figure 11 reproduction.

## Cerebras CS-3-compatible experiment

For the CS-3 result, use the SDK batch endpoint rather than ModelZoo. The
completion API accepts an array of prompts, so each measurement is one request
containing a true API batch of `1, 2, 4, 8, 16` prompts. It reports server-side
`prompt_time`, which is the prefill-latency analogue plotted against batch size.

This creates a Figure-11-compatible latency curve, but it cannot reproduce the
paper's GPU-idle and CPU-idle panels: those hardware counters are not exposed by
the public SDK. The substitute second panel is SDK-reported service queue time.
Every point is the mean of 30 measured requests after five warmups, with a
non-parametric bootstrap 95% confidence interval. Actual prompt-token counts
are retained from API usage in each `result.json`.

```bash
export CEREBRAS_API_KEY=...
export MODEL=gpt-oss-120b  # or an approved dedicated endpoint model ID
experiments/fig11/run_cerebras_sdk.sh
```
