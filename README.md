# SODA — System Offload Dynamics Analyzer

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-enabled-green?logo=nvidia&logoColor=white)
![Profiler](https://img.shields.io/badge/PyTorch%20Profiler-supported-blueviolet)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![System Analysis](https://img.shields.io/badge/System%20Profiler-Enabled-blue)

SODA implements the **TaxBreak methodology** for decomposing host-side overhead in LLM inference. It parses PyTorch profiler traces to quantify where inference time is spent across the CPU-GPU execution stack.

## What SODA Measures

SODA profiles a model for 150 inference runs and produces a detailed breakdown of where time is spent:

| Metric | Description |
|--------|-------------|
| **GPU Utilization** | Percentage of GPU time span that is actively executing kernels (concurrent-aware). |
| **Inference Time** | Wall-clock inference time (torch-measured) with profiler overhead breakdown. |
| **TKLQT** | Top-K Launch Queue Tax — total observable CPU→GPU launch overhead for the most frequent kernels. Reported in standard mode without requiring a kernel database. |
| **Inference Throughput** | TTFT (prefill) or TPOT (decode) latency, interactivity (tokens/sec/user), and system throughput (tokens/sec). Automatically labeled based on `--max-new-tokens`. |
| **Memory Profiling** | Model memory, peak allocated/reserved memory, inference delta, empirical KV cache size (GQA/MLA-aware), and GPU memcpy/memset operations. |
| **Kernel Fragmentation** | Total kernel launches, unique kernel count, diversity ratio, and per-token dispatch rate — useful for MoE and Triton-heavy models. |
| **Carbon Footprint** | Estimated gCO₂eq per inference run from GPU TDP × utilization × grid carbon intensity × PUE. |
| **Top-K Kernels** | Highest-frequency and longest-duration kernels across the trace. |
| **Per-Stream Analysis** | Kernel count, op count, and busy time broken down by CUDA stream. |
| **Kernel Fusion** | (Optional) Identifies adjacent kernel chains that are candidates for fusion. |
| **HDBI** | *(TaxBreak pipeline only)* Host-Device Balance Index — requires per-kernel `i_lib` labels, isolation-replay taxes, and a dynamically measured system floor. Run `--taxbreak` to compute it. |
| **Kernel Launch Tax** | *(TaxBreak pipeline only)* Per-kernel isolation-replay launch overhead with T_sys floor subtracted. |
| **ATen Translation Tax** | *(TaxBreak pipeline only)* Per-kernel overhead between ATen dispatch and CUDA runtime entry, with CudaT separated from CT using `i_lib` labels. |
| **GPU Roofline** | *(TaxBreak + `--ncu`)* Arithmetic intensity, achieved GFLOP/s, compute/memory bound classification, and roofline plot. |

## Output Files

Each profiling run produces an experiment directory under `<output-dir>/<model>_<precision>/`:

| File | Description |
|------|-------------|
| `trace.json` | Raw PyTorch profiler trace (Chrome trace format). Load in `chrome://tracing` or Perfetto. |
| `report.json` | Full analysis report: metadata, performance metrics, per-stream analysis, top-K kernels. |
| `env_metadata.json` | Snapshot of the runtime environment (GPU, PyTorch version, CUDA version, env vars). |
| `soda.log` | Console output log. |
| `kernel_database.json` | (With `--kernel-db`) Op-kernel mapping from the last profiled run, used by Stage 2. |
| `taxbreak/enhanced_taxbreak.json` | (Stage 2) Per-kernel isolation-replay overhead breakdown with optional ncu metrics and roofline data. |
| `taxbreak/roofline.png` | (Stage 2 + ncu) GPU roofline plot showing kernel arithmetic intensity vs achieved performance. |
| `summary/comparative_report.html` | (Sweep) Self-contained dark-mode HTML report comparing all batch × seq-len configurations side-by-side with sortable table and heatmap links. |

## Installation

```bash
conda create -y -n soda-311 python=3.11
conda activate soda-311

# Install CUDA-enabled PyTorch
conda install -y -c pytorch -c nvidia pytorch pytorch-cuda=12.1

# Clone and install
git clone https://github.com/prabsy96/soda.git
cd SODA---System-Offload-Dynamics-Analyzer
pip install -e .
```

## Quick Start

```bash
# Load environment
source env.sh

# Run on a model
soda-cli --model gpt2 --output-dir output/ --seq-len 512 --batch-size 1 --precision float16

# Or use python -m
python -m soda --model gpt2 --output-dir output/ --seq-len 512 --batch-size 1 --precision float16
```

## CLI Reference

### Core Profiling

| Argument | Default | Description |
|----------|---------|-------------|
| `-m`, `--model` | *(required)* | HuggingFace model name or local path |
| `--output-dir` | `$SODA_OUTPUT` | Output directory for traces and reports |
| `-p`, `--precision` | `bfloat16` | Weight precision: `float32`, `float16`, `bfloat16`, `float8_e4m3fn` |
| `-sl`, `--seq-len` | `128` | Sequence length for synthetic input |
| `-bs`, `--batch-size` | `1` | Batch size |
| `--max-new-tokens` | `1` | Tokens to generate per decoder run |
| `-c`, `--compile-type` | `eager` | Execution mode: `eager`, `torch.compile`, `flash-attention` |
| `-d`, `--device` | `cuda` | Device: `cuda` or `cpu` |
| `--seed` | `42` | Random seed for reproducibility |
| `--runs` | `150` | Number of profiled inference iterations |
| `--warmup` | `50` | Warmup iterations before profiling |

### Analysis Options

| Argument | Description |
|----------|-------------|
| `-f`, `--fusion` | Kernel chain lengths to analyze for fusion (e.g., `-f 2 3`) |
| `-ps`, `--prox-score` | Proximity score threshold (0.0-1.0) for fusion recommendations |
| `--kernel-db` | Generate kernel database after profiling (Stage 1 of enhanced pipeline) |
| `--verbose` | Print full expert-level output (per-kernel tables, derivation details). Default shows compact summary. |

### Carbon & Environment

| Argument | Default | Description |
|----------|---------|-------------|
| `--carbon-intensity` | `400` | Grid carbon intensity in gCO₂eq/kWh. Presets: `FR=58`, `EU=295`, `US=386`, `CN=581` |
| `--pue` | `1.1` | Power Usage Effectiveness of the data centre (1.05–1.6 typical range) |

### Enhanced TaxBreak (Stage 2)

| Argument | Default | Description |
|----------|---------|-------------|
| `--taxbreak` | — | Run enhanced TaxBreak pipeline (no model loading required) |
| `--kernel-db-path` | — | Path to `kernel_database.json` from Stage 1 |
| `--ncu` | — | Enable ncu profiling on top-N kernels |
| `--ncu-top-n` | `10` | Number of top kernels (by duration) to profile with ncu |

### Legacy Microbench

| Argument | Description |
|----------|-------------|
| `--microbench` | Run full microbench pipeline (baremetal + PyTorch replay) |
| `--skip-offline-cublas-algo-search` | Skip cuBLASLt algorithm search (use heuristic) |
| `--skip-pytorch-profile` | Skip PyTorch GEMM kernel profiling |
| `--skip-baremetal-profile` | Skip baremetal GEMM kernel profiling |

## Running on SLURM Clusters

A template SLURM sbatch script is provided at [`slurm/run_job.sh`](slurm/sbatch_template.sh). Edit the configuration variables at the top and the job commands at the bottom, then submit:

```bash
sbatch slurm/sbatch_template.sh
```

## SODA Pipeline

The pipeline measures per-kernel overhead in isolation, replacing hardcoded baselines with dynamic measurements. It runs in two decoupled stages:

```bash
# Stage 1: Profile model and generate kernel database
soda-cli -m gpt2 --kernel-db

# Stage 2: Run enhanced TaxBreak (no model loading required)
soda-cli --taxbreak --kernel-db-path output/gpt2_eager_bfloat16_bs1_sl128_mt1/kernel_database.json

# With optional ncu profiling on the top-3 kernels
soda-cli --taxbreak --kernel-db-path output/gpt2_eager_bfloat16_bs1_sl128_mt1/kernel_database.json --ncu --ncu-top-n 3
```

Stage 2 performs:
1. **Dynamic system floor** — measures null-kernel launch tax via nsys (replaces hardcoded baselines)
2. **Isolation replay** — replays each kernel individually under nsys for clean launch-tax measurement
3. **ncu profiling** (optional) — collects L1/L2 hit rates, DRAM throughput, and compute utilization
4. **GPU roofline analysis** (with ncu) — computes arithmetic intensity, classifies kernels as compute/memory bound, generates `roofline.png`
5. **Enhanced report** — writes `taxbreak/enhanced_taxbreak.json` with per-kernel breakdown and roofline data

## Sweep Summary

After running a batch × sequence-length sweep with `experiments/sweep/soda_sweep.py`, generate a summary report:

```bash
python experiments/sweep/summarize_soda_sweep.py output/<sweep_root>/
```

This produces `<sweep_root>/summary/` containing:

| File | Description |
|------|-------------|
| `comparative_report.html` | Self-contained dark-mode HTML report. Per model-group: heatmaps (embedded as base64) and a sortable run comparison table with inference time, throughput, GPU utilization, TKLQT, and peak memory. Opens correctly from any path or viewer. |
| `<slug>_heatmap.png` | Inference time heatmap (batch size × sequence length) |
| `<slug>_gpu_active_heatmap.png` | GPU active time heatmap |
| `<slug>_t_exposed_heatmap.png` | T_exposed (GPU idle) heatmap |
| `<slug>_tklqt_heatmap.png` | TKLQT overhead heatmap |
| `<slug>_peak_memory_heatmap.png` | Peak memory heatmap |
| `<slug>_*_pivot.csv` | Raw pivot tables for each metric |

OOM cells are shown in red. GPU utilization is color-coded (green ≥60%, yellow 30–60%, red <30%). All table columns are client-side sortable with no external dependencies.

## Project Structure

```
SODA---System-Offload-Dynamics-Analyzer/
├── src/soda/                      # Python package
│   ├── __init__.py                # SodaAnalyzer, ModelTracer, main()
│   ├── __main__.py                # python -m soda support
│   ├── common/                    # Utilities, data structures, trace parsing
│   ├── kerneldb.py                # Kernel database generator (Stage 1)
│   ├── ncu.py                     # NVIDIA Compute Profiler integration
│   ├── roofline.py                # GPU roofline analysis and plot generation
│   ├── taxbreak/                  # Enhanced TaxBreak pipeline (Stage 2)
│   │   ├── null_kernel.py         # Dynamic system floor measurement
│   │   ├── nsys_replay.py         # nsys isolation replay for any kernel
│   │   ├── pipeline.py            # Pipeline orchestrator
│   │   └── report.py              # Enhanced report generation
│   └── microbench/                # Legacy microbenchmarking pipeline
├── experiments/
│   ├── sweep/                     # Parameter sweep orchestration
│   └── rebuttal/
│       └── slurm/                 # SLURM scripts for rebuttal experiments
├── slurm/
│   └── run_job.sh                 # SLURM job template
├── env.sh                         # Environment setup & helper functions
├── FORMULAS.md                    # Comprehensive formula reference
└── README.md
```

## Docker

```bash
# Build the image
docker compose build soda

# Run standard profiling
docker compose run --rm soda soda-cli --model gpt2 --seq-len 512

# Enhanced TaxBreak (two-stage)
docker compose run --rm soda soda-cli -m gpt2 --kernel-db
docker compose run --rm soda soda-cli --taxbreak --kernel-db-path output/gpt2_eager_bfloat16_bs1_sl128_mt1/kernel_database.json

# Interactive shell
docker compose run --rm soda /bin/bash

# For gated models
export HF_TOKEN=hf_your_token_here
docker compose run --rm soda soda-cli --model meta-llama/Llama-3.2-1B
```

Output is persisted to `./output` via volume mount. HuggingFace model weights are cached in a named Docker volume (`hf-cache`) across runs.

## Building the Baremetal Binary

```bash
source env.sh
build  # Uses env.sh helper function
```

## Formulas Reference

See [`FORMULAS.md`](FORMULAS.md) for the complete mathematical reference of all metrics computed by SODA, including HDBI, kernel taxes, GPU utilization, inference throughput, roofline analysis, and memory profiling formulas.

## Citation

```bibtex
@INPROCEEDINGS{11096369,
  author={Vellaisamy, Prabhu and Labonte, Thomas and Chakraborty, Sourav and Turner, Matt and Sury, Samantika and Shen, John Paul},
  booktitle={2025 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)},
  title={Characterizing and Optimizing LLM Inference Workloads on CPU-GPU Coupled Architectures},
  year={2025},
  pages={49-61},
  doi={10.1109/ISPASS64960.2025.00015}}
```
