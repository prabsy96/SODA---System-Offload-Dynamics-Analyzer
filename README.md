# SODA — System Offload Dynamics Analyzer

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║  ███████╗ ██████╗ ██████╗   █████╗                           ║
║  ██╔════╝██╔═══██╗██╔══██╗██╔══██╗                           ║
║  ███████╗██║   ██║██║  ██║███████║                           ║
║  ╚════██║██║   ██║██║  ██║██╔══██║                           ║
║  ███████║╚██████╔╝██████╔╝██║  ██║                           ║
║  ╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝                          ║
║                                                               ║
║           System Offload Dynamics Analyzer                   ║
║                 Prabhu Vellaisamy                            ║
║              Carnegie Mellon University                      ║
║                  © Apache 2.0 License                        ║
╚═══════════════════════════════════════════════════════════════╝
```

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-enabled-green?logo=nvidia&logoColor=white)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)

SODA implements the **TaxBreak methodology** for decomposing host-side overhead in LLM inference. It parses PyTorch profiler traces to quantify where inference time is spent across the CPU-GPU execution stack.

## What SODA Measures

**Standard mode** — GPU utilization, inference latency (TTFT/TPOT), TKLQT launch tax, throughput + Pareto plot, memory (model/peak/KV cache/memcpy), kernel fragmentation, carbon footprint, top-K kernels, per-stream analysis.

**Stage 2 (TaxBreak)** — HDBI decomposed into FT_python + FT_dispatch + δCT + KT with `i_lib` gating; per-kernel isolation-replay launch tax (raw + floor-adjusted); ATen translation tax; GPU roofline (with `--ncu`).

**Stage 3 (MoE)** — per-kernel per-layer op profile with `is_shared_expert` flag; NCU isolation HBM bytes per expert type; optional NVBit in-context cache reuse.

## Output Files

Experiment directory: `<output-dir>/<model>_<precision>_bs<B>_sl<S>_mt<T>/`

| File | When | Description |
|------|------|-------------|
| `report.json` | always | Full metrics: performance, per-stream, top-K kernels |
| `summary.md` | always | Human-readable summary |
| `pareto.png` | always | Throughput–interactivity Pareto plot |
| `trace.json` | always | Raw Chrome trace (open in Perfetto) |
| `kernel_database.json` | `--kernel-db` | Op-to-kernel mapping; required for Stage 2 |
| `taxbreak/enhanced_taxbreak.json` | Stage 2 | Per-kernel isolation-replay breakdown |
| `taxbreak/roofline.png` | Stage 2 + `--ncu` | GPU roofline plot |
| `moe_profile/op_profile.json` | Stage 3 | Per-kernel per-layer records with `is_shared_expert` |
| `moe_profile/moe_profile.json` | Stage 3 | NCU isolation + NVBit reuse metrics per expert type |

## Installation

```bash
conda create -y -n soda-311 python=3.11
conda activate soda-311
conda install -y -c pytorch -c nvidia pytorch pytorch-cuda=12.1
git clone https://github.com/prabsy96/SODA---System-Offload-Dynamics-Analyzer.git
cd SODA---System-Offload-Dynamics-Analyzer
pip install -e .
```

### Optional: NVBit Tool Build (for MoE in-context cache reuse)

The NVBit pass is optional and only used when running MoE profiling with `--nvbit-lib`.

```bash
# From repo root
cd src/soda/moe/nvbit_tool

# Uses bundled NVBit SDK by default
make ARCH=all

# Output shared library used by --nvbit-lib
ls -lh mem_reuse_tracker.so
```

Build from downloaded NVBit SDK tarball (`nvbit-Linux-x86_64-1.7.7.3.tar.bz2`):

```bash
tar -xjf nvbit-Linux-x86_64-1.7.7.3.tar.bz2
# Produces: nvbit_release_x86_64/

cd src/soda/moe/nvbit_tool
make clean
make ARCH=all NVBIT_SDK_PATH=/path/to/nvbit_release_x86_64
```

Notes:

- `NVBIT_SDK_PATH` must point to the extracted `nvbit_release_x86_64` root (the Makefile internally uses `<root>/core`).
- If `NVBIT_SDK_PATH` is not set, the Makefile falls back to the repo-bundled SDK at `src/soda/moe/nvbit_tool/nvbit_release_x86_64/core`.

## Quick Start

```bash
source env.sh   # required before every run

soda-cli --model gpt2 --output-dir output/ --seq-len 512 --batch-size 1
soda-cli --model gpt2 --output-dir output/ --seq-len 512 --batch-size 1 --verbose
soda-cli --model gpt2 --output-dir output/ --seq-len 512 --batch-size 1 --num-gpus 2
```

## CLI Reference

### Core Profiling

| Argument | Default | Description |
|----------|---------|-------------|
| `-m`, `--model` | *(required)* | HuggingFace model name or local path |
| `--output-dir` | `$SODA_OUTPUT` | Output directory |
| `-p`, `--precision` | `bfloat16` | `float32`, `float16`, `bfloat16`, `float8_e4m3fn` |
| `-sl`, `--seq-len` | `128` | Input sequence length |
| `-bs`, `--batch-size` | `1` | Batch size |
| `--max-new-tokens` | `1` | Tokens to generate (1 = TTFT; >1 = TPOT) |
| `-c`, `--compile-type` | `eager` | `eager`, `torch.compile`, `flash-attention` |
| `--runs` / `--warmup` | `150` / `50` | Profiled and warmup iterations |
| `--num-gpus` | `1` | GPUs for model parallelism (`device_map="balanced"`) |
| `--verbose` | off | Full expert tables (per-kernel, derivation details) |
| `--kernel-db` | off | Generate kernel database (required for Stage 2) |
| `--carbon-intensity` | `400` | gCO₂eq/kWh — presets: FR=58, EU=295, US=386, CN=581 |
| `--pue` | `1.1` | Power Usage Effectiveness |

### Stage 2: Enhanced TaxBreak

| Argument | Description |
|----------|-------------|
| `--taxbreak` | Run TaxBreak pipeline (no model loading required) |
| `--kernel-db-path` | Path to `kernel_database.json` from Stage 1 |
| `--ncu` / `--ncu-top-n` | Enable ncu profiling on top-N kernels (default N=10) |

### Stage 3: MoE Per-Expert Profiling

| Argument | Default | Description |
|----------|---------|-------------|
| `--moe-profile` | — | Run MoE profiling (no model loading or GPU required) |
| `--kernel-db-path` | — | Path to `kernel_database.json` from Stage 1 |
| `--moe-shared-dim` | auto | Shared expert intermediate dimension override |
| `--moe-routed-dim` | auto | Routed expert intermediate dimension override |
| `--moe-num-layers` | auto | Number of transformer layers override |
| `--nvbit-lib` | — | Path to `mem_reuse_tracker.so` for NVBit in-context pass |

## Pipelines

### Stage 1 → Stage 2: TaxBreak

```bash
# Stage 1: profile + build kernel database
soda-cli -m gpt2 --output-dir output/ --kernel-db

# Stage 2: isolation replay + optional ncu
soda-cli --taxbreak \
  --kernel-db-path output/gpt2_eager_bfloat16_bs1_sl128_mt1/kernel_database.json \
  --ncu --ncu-top-n 5

# Multi-GPU
soda-cli -m gpt2 --output-dir output/ --kernel-db --num-gpus 2
soda-cli --taxbreak \
  --kernel-db-path output/gpt2_eager_bfloat16_bs1_sl128_mt1_gpu2/kernel_database.json
```

### Stage 1 → Stage 3: MoE Op Profile

Generates `op_profile.json` with one record per kernel per layer. Every kernel is included; shared expert kernels are flagged with `"is_shared_expert": true`.

```bash
# Stage 1
soda-cli -m Qwen/Qwen1.5-MoE-A2.7B --output-dir output/ \
  --seq-len 4096 --batch-size 4 --kernel-db

# Stage 3 (auto-detects dimensions and num_layers from kernel DB)
soda-cli --moe-profile \
  --kernel-db-path output/Qwen_Qwen1.5-MoE-A2.7B_eager_bfloat16_bs4_sl4096_mt1/kernel_database.json
```

Records with `layer_id == -1` are not layer-specific (embedding, LM head, variable-shape routed experts). `num_layers` auto-detection priority: `--moe-num-layers` → HF `AutoConfig` → GCD of shared expert frequencies.

## Running on SLURM

```bash
cp slurm/sbatch_template.sh slurm/my_job.sh
# Edit partition, GPU count, model, and job commands
sbatch slurm/my_job.sh
```

For the NVBit MoE shared-expert pipeline example, use the dedicated template:

```bash
cp slurm/sbatch_moe_shared_expert_template.sh slurm/my_moe_job.sh
# Edit configurable paths/modules/model settings in USER CONFIGURATION
sbatch slurm/my_moe_job.sh
```

## Docker

```bash
docker compose build soda
docker compose run --rm soda soda-cli --model gpt2 --seq-len 512

# Two-stage TaxBreak
docker compose run --rm soda soda-cli -m gpt2 --kernel-db
docker compose run --rm soda soda-cli --taxbreak \
  --kernel-db-path output/gpt2_eager_bfloat16_bs1_sl128_mt1/kernel_database.json
```

Output persists to `./output` via volume mount; HF weights cached in `hf-cache` Docker volume.

## Project Structure

```
src/soda/
├── __init__.py       # SodaAnalyzer, ModelTracer, main()
├── kerneldb.py       # Kernel database generator (Stage 1)
├── taxbreak/         # TaxBreak pipeline (Stage 2)
├── moe/              # MoE per-expert profiling (Stage 3)
│   ├── detect.py     # Expert type classification
│   ├── op_profile.py # op_profile.json with is_shared_expert flag
│   ├── pipeline.py   # Orchestrator (NCU + optional NVBit)
│   └── report.py     # moe_profile.json
├── common/           # Utilities, trace parsing, CLI args
├── ncu.py            # NVIDIA Compute Profiler integration
├── roofline.py       # Roofline and Pareto plots
└── carbon.py         # Carbon footprint computation
```

## Testing

350 tests, no GPU required:

```bash
PYTHONPATH=src python -m pytest
```

## Citation

SODA implements the methodology from our ISPASS 2025 paper (TKLQT characterization) and ISPASS 2026 paper (TaxBreak decomposition). If you use SODA or build upon it in your research or product, **you must cite the following papers**:

**ISPASS 2025 — TKLQT Characterization:**

```bibtex
@INPROCEEDINGS{11096369,
  author={Vellaisamy, Prabhu and Labonte, Thomas and Chakraborty, Sourav and Turner, Matt and Sury, Samantika and Shen, John Paul},
  booktitle={2025 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)},
  title={Characterizing and Optimizing LLM Inference Workloads on CPU-GPU Coupled Architectures},
  year={2025},
  pages={49-61},
  doi={10.1109/ISPASS64960.2025.00015}}
```

**ISPASS 2026 — TaxBreak:**

*Citation forthcoming.*

## NVBit MoE Shared-Expert Pipeline (Separate from SODA Flow)

This is a separate workflow that should be run independently from standard SODA analysis and TaxBreak runs.

```bash
# 0) Required environment
source env.sh

# 1) Build NVBit tool once (if not already built)
cd src/soda/moe/nvbit_tool
make ARCH=all
cd ../../../../

# 2) Stage 1: build kernel DB from a profiled run
soda-cli -m Qwen/Qwen1.5-MoE-A2.7B \
  --output-dir output/ \
  --seq-len 1024 --batch-size 1 --max-new-tokens 1 \
  --precision bfloat16 \
  --kernel-db

# 3) Stage 3: MoE profile with NVBit shared library
soda-cli --moe-profile \
  --kernel-db-path output/Qwen_Qwen1.5-MoE-A2.7B_eager_bfloat16_bs1_sl1024_mt1/kernel_database.json \
  --nvbit-lib src/soda/moe/nvbit_tool/mem_reuse_tracker.so
```

Outputs for this separate pipeline:

- `moe_profile/op_profile.json` — per-kernel per-layer records with `is_shared_expert`
- `moe_profile/moe_profile.json` — NCU isolation metrics + NVBit in-context reuse metrics

If `--nvbit-lib` is omitted, MoE profiling still runs, but NVBit/data-reuse fields are not populated.
