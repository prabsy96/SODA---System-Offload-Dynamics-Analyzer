# SODA — System Offload Dynamics Analyzer

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-enabled-green?logo=nvidia&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)

SODA implements the **TaxBreak methodology** for decomposing host-side overhead in LLM inference. It parses PyTorch profiler traces to quantify where inference time is spent across the CPU-GPU execution stack.

## What SODA Measures

SODA profiles a model over 150 inference runs and reports the following:

**Standard mode** (no kernel database required):

| Metric | Description |
|--------|-------------|
| **GPU Utilization** | Percentage of the inference span with at least one kernel actively executing (concurrent-aware). |
| **Inference Latency** | Wall-clock inference time. Auto-labeled as TTFT (prefill) or TPOT (decode) based on `--max-new-tokens`. |
| **TKLQT** | Top-K Launch Queue Tax — total observable CPU→GPU launch overhead for the most frequent kernels. |
| **Throughput** | Tokens/sec (system) and tokens/sec/user (interactivity), with Pareto frontier plot across batch configs. |
| **Memory** | Model footprint, peak allocated/reserved, inference delta, empirical KV cache size (GQA/MLA-aware), memcpy/memset. |
| **Kernel Fragmentation** | Total launches, unique kernel count, diversity ratio, per-token dispatch rate. Useful for MoE models. |
| **Carbon Footprint** | Estimated gCO₂eq per inference from GPU TDP × utilization × grid carbon intensity × PUE. |
| **Top-K Kernels** | Highest-frequency and longest-duration kernels across the trace. |
| **Per-Stream Analysis** | Kernel count, op count, and busy time per CUDA stream. |

**Enhanced TaxBreak pipeline** (Stage 2, requires kernel database):

| Metric | Description |
|--------|-------------|
| **HDBI** | Host-Device Balance Index — isolation-replay taxes decomposed into FT + CT + CudaT + KT components with `i_lib` gating. |
| **Launch Tax (KT)** | Per-kernel isolation-replay launch overhead with dynamic T_sys floor subtracted. |
| **ATen Translation Tax (CT)** | Per-kernel overhead between ATen dispatch and CUDA runtime entry. |
| **GPU Roofline** | Arithmetic intensity, achieved GFLOP/s, bound classification, and roofline plot (requires `--ncu`). |

## Output Files

Each run writes an experiment directory under `<output-dir>/<model>_<precision>_bs<B>_sl<S>_mt<T>/` (single GPU) or `…_mt<T>_gpu<N>/` (multi-GPU, N > 1):

| File | Description |
|------|-------------|
| `trace.json` | Raw PyTorch profiler trace (Chrome trace format). Open in `chrome://tracing` or Perfetto. |
| `report.json` | Full analysis: metadata, performance metrics, per-stream analysis, top-K kernels. |
| `summary.md` | Compact human-readable summary of the run. |
| `env_metadata.json` | Runtime environment snapshot (GPU, PyTorch, CUDA versions). |
| `kernel_database.json` | Op-to-kernel mapping (Stage 1, with `--kernel-db`). Required for Stage 2. |
| `taxbreak/enhanced_taxbreak.json` | Per-kernel isolation-replay breakdown with optional ncu metrics. |
| `taxbreak/roofline.png` | GPU roofline plot (Stage 2 + `--ncu`). |
| `pareto.png` | Throughput–interactivity Pareto plot across batch configurations. |

## Installation

```bash
conda create -y -n soda-311 python=3.11
conda activate soda-311

# CUDA-enabled PyTorch
conda install -y -c pytorch -c nvidia pytorch pytorch-cuda=12.1

# Clone and install
git clone https://github.com/prabsy96/soda.git
cd SODA---System-Offload-Dynamics-Analyzer
pip install -e .
```

## Quick Start

```bash
# Load environment (required before every run)
source env.sh

# Profile a model
soda-cli --model gpt2 --output-dir output/ --seq-len 512 --batch-size 1

# With verbose expert output
soda-cli --model gpt2 --output-dir output/ --seq-len 512 --batch-size 1 --verbose

# Distribute model across 2 GPUs (model parallelism)
soda-cli --model gpt2 --output-dir output/ --seq-len 512 --batch-size 1 --num-gpus 2
```

## CLI Reference

### Core Profiling

| Argument | Default | Description |
|----------|---------|-------------|
| `-m`, `--model` | *(required)* | HuggingFace model name or local path |
| `--output-dir` | `$SODA_OUTPUT` | Output directory for traces and reports |
| `-p`, `--precision` | `bfloat16` | Weight precision: `float32`, `float16`, `bfloat16`, `float8_e4m3fn` |
| `-sl`, `--seq-len` | `128` | Input sequence length |
| `-bs`, `--batch-size` | `1` | Batch size |
| `--max-new-tokens` | `1` | Tokens to generate (1 = TTFT/prefill; >1 = TPOT/decode) |
| `-c`, `--compile-type` | `eager` | Execution mode: `eager`, `torch.compile`, `flash-attention` |
| `--runs` | `150` | Number of profiled inference iterations |
| `--warmup` | `50` | Warmup iterations before profiling |
| `--seed` | `42` | Random seed |

### Analysis Options

| Argument | Description |
|----------|-------------|
| `--verbose` | Print full expert output (per-kernel tables, derivation details). Default shows compact summary. |
| `-f`, `--fusion` | Kernel chain lengths to check for fusion candidates (e.g., `-f 2 3`) |
| `--kernel-db` | Generate kernel database after profiling (required for Stage 2) |

### Carbon & Environment

| Argument | Default | Description |
|----------|---------|-------------|
| `--carbon-intensity` | `400` | Grid carbon intensity in gCO₂eq/kWh. Presets: FR=58, EU=295, US=386, CN=581 |
| `--pue` | `1.1` | Power Usage Effectiveness of the data centre (1.05–1.6 typical) |

### Multi-GPU

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-gpus` | `1` | Number of GPUs to use via `device_map="balanced"` (model parallelism). Values exceeding available GPU count are clamped. Single-GPU behaviour is unchanged. |

### Enhanced TaxBreak (Stage 2)

| Argument | Description |
|----------|-------------|
| `--taxbreak` | Run the enhanced TaxBreak pipeline (no model loading required) |
| `--kernel-db-path` | Path to `kernel_database.json` produced by Stage 1 |
| `--ncu` | Enable ncu profiling on top-N kernels |
| `--ncu-top-n` | Number of kernels to profile with ncu (default: 10) |

## Enhanced TaxBreak Pipeline

The pipeline replaces hardcoded baselines with dynamic per-kernel measurements. It runs in two decoupled stages:

```bash
# Stage 1: Profile model and generate kernel database (single GPU)
soda-cli -m gpt2 --output-dir output/ --kernel-db

# Stage 2: Run enhanced TaxBreak (reads kernel database, no model loading)
soda-cli --taxbreak --kernel-db-path output/gpt2_eager_bfloat16_bs1_sl128_mt1/kernel_database.json

# With ncu profiling on the top 5 kernels
soda-cli --taxbreak \
  --kernel-db-path output/gpt2_eager_bfloat16_bs1_sl128_mt1/kernel_database.json \
  --ncu --ncu-top-n 5

# Multi-GPU TaxBreak: Stage 1 with 2 GPUs, then Stage 2 (T_sys measured on both GPUs)
soda-cli -m gpt2 --output-dir output/ --kernel-db --num-gpus 2
soda-cli --taxbreak --kernel-db-path output/gpt2_eager_bfloat16_bs1_sl128_mt1_gpu2/kernel_database.json
```

Stage 2 performs:
1. **Dynamic system floor** — measures null-kernel launch overhead via nsys (no hardcoded constants)
2. **Isolation replay** — replays each kernel individually under nsys for clean launch-tax measurement
3. **ncu profiling** (optional) — collects L1/L2 hit rates, DRAM throughput, compute utilization
4. **Roofline analysis** (with ncu) — classifies kernels as compute/memory bound, generates `roofline.png`
5. **Enhanced report** — writes `taxbreak/enhanced_taxbreak.json` with per-kernel breakdown

## Sweep & Comparative Report

Run a parameter sweep across batch sizes and sequence lengths, then generate a comparative HTML report.

```bash
# 1. Run the sweep (edits config.py to change the parameter grid)
python experiments/sweep/soda_sweep.py --model gpt2 --output-dir output/sweep/

# 2. Generate comparative HTML report and heatmaps
python experiments/sweep/summarize_soda_sweep.py output/sweep/
```

The summarizer writes `output/sweep/summary/` containing:

| File | Description |
|------|-------------|
| `comparative_report.html` | Self-contained dark-mode HTML report with per-group heatmaps (base64-embedded) and a sortable run comparison table (inference time, throughput, GPU util, TKLQT, peak memory, KV cache). Opens from any path. |
| `*_inference_heatmap.png` | Inference time heatmap (batch × seq-len grid) |
| `*_gpu_active_heatmap.png` | GPU active time heatmap |
| `*_t_exposed_heatmap.png` | T_exposed (GPU idle time) heatmap |
| `*_tklqt_heatmap.png` | TKLQT launch overhead heatmap |
| `*_peak_memory_heatmap.png` | Peak memory heatmap |
| `*_pivot.csv` | Raw pivot tables for each metric |

GPU utilization is color-coded: green ≥60%, yellow 30–60%, red <30%. OOM cells are shown in red. All table columns are client-side sortable with no external dependencies.

## Running on SLURM

Copy the template and edit the configuration variables at the top:

```bash
cp slurm/sbatch_template.sh slurm/my_job.sh
# Edit partition, GPU count, model, and job commands
sbatch slurm/my_job.sh
```

## Docker

```bash
# Build the image
docker compose build soda

# Standard profiling
docker compose run --rm soda soda-cli --model gpt2 --seq-len 512

# Enhanced TaxBreak (two-stage)
docker compose run --rm soda soda-cli -m gpt2 --kernel-db
docker compose run --rm soda soda-cli --taxbreak \
  --kernel-db-path output/gpt2_eager_bfloat16_bs1_sl128_mt1/kernel_database.json

# Interactive shell
docker compose run --rm soda /bin/bash

# Gated models (e.g., Llama)
export HF_TOKEN=hf_your_token_here
docker compose run --rm soda soda-cli --model meta-llama/Llama-3.2-1B
```

Output is persisted to `./output` via volume mount. HuggingFace weights are cached in a named Docker volume (`hf-cache`) across runs.

## Project Structure

```
src/soda/
├── __init__.py            # SodaAnalyzer, ModelTracer, main()
├── kerneldb.py            # Kernel database generator (Stage 1)
├── ncu.py                 # NVIDIA Compute Profiler integration
├── roofline.py            # Roofline and Pareto plot generation
├── carbon.py              # Carbon footprint computation
├── taxbreak/              # Enhanced TaxBreak pipeline (Stage 2)
│   ├── null_kernel.py     # Dynamic system floor measurement
│   ├── nsys_replay.py     # Per-kernel nsys isolation replay
│   ├── pipeline.py        # Pipeline orchestrator
│   └── report.py          # Enhanced report generation
├── common/                # Shared utilities, data structures, trace parsing
└── microbench/            # Legacy baremetal + PyTorch GEMM replay pipeline

experiments/
├── sweep/
│   ├── config.py                   # Parameter grids (batch × seq-len)
│   ├── soda_sweep.py               # Sweep runner
│   └── summarize_soda_sweep.py     # HTML report + heatmap generator
└── rebuttal/              # One-off experiments for paper reviewers

slurm/
└── sbatch_template.sh     # SLURM job template
```

## Development & Testing

### Running the unit tests

SODA includes a pytest suite covering 225 tests across 7 modules. No GPU is required.

```bash
# Python 3.11+ is required (login node Python 3.9 fails on str|Path unions)
# The conda base environment at /home/pvellais/miniconda3 has all dependencies.

PYTHONPATH=src /home/pvellais/miniconda3/bin/python3.13 -m pytest
```

Expected output:
```
225 passed in ~4.5s
```

Install pytest once if needed:
```bash
/home/pvellais/miniconda3/bin/python3.13 -m pip install pytest
```

### Test coverage summary

| File | Tests | Area |
|---|---|---|
| `tests/test_data.py` | 33 | `clean_kernel_name`, `Kernel`, `ATenOp`, `Sequence` |
| `tests/test_utils.py` | 77 | Conversions, GEMM detection, sequence parsing, tax metrics, HDBI, GPU utilization, fragmentation |
| `tests/test_carbon.py` | 12 | TDP lookup, carbon intensity presets, `compute_carbon_footprint` |
| `tests/test_roofline.py` | 15 | GPU specs, `compute_gemm_flops`, Pareto frontier |
| `tests/test_summary_report.py` | 21 | Formatting helpers, Rich table builders |
| `tests/test_kerneldb.py` | 20 | Vendor-replayability, three-way kernel class, last-run extraction |
| `tests/test_multi_gpu.py` | 47 | `num_gpus` naming, clamping, `device_map` selection, multi-GPU metadata |

pytest configuration is in `pyproject.toml` under `[tool.pytest.ini_options]`.

### Syntax check (works on Python 3.9)

```bash
python3 -c "import py_compile; py_compile.compile('src/soda/<file>.py', doraise=True)"
```

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
