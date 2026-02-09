# SODA — System Offload Dynamics Analyzer

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-enabled-green?logo=nvidia&logoColor=white)
![Profiler](https://img.shields.io/badge/PyTorch%20Profiler-supported-blueviolet)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![System Analysis](https://img.shields.io/badge/System%20Profiler-Enabled-blue)

SODA implements the **TaxBreak methodology** for decomposing host-side overhead in LLM inference. It parses PyTorch profiler traces to quantify where inference time is spent across the CPU-GPU execution stack.

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

## Running on SLURM Clusters

A template SLURM sbatch script is provided at [`slurm/run_job.sh`](slurm/sbatch_template.sh). Edit the configuration variables at the top and the job commands at the bottom, then submit:

```bash
sbatch slurm/sbatch_template.sh
```

## Project Structure

```
SODA---System-Offload-Dynamics-Analyzer/
├── src/soda/                      # Python package
│   ├── __init__.py                # SodaAnalyzer, ModelTracer, main()
│   ├── __main__.py                # python -m soda support
│   ├── common/                    # Utilities, data structures, trace parsing
│   └── microbench/                # Microbenchmarking pipeline
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

## Building the Baremetal Binary

```bash
source env.sh
build  # Uses env.sh helper function
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
