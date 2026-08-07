# TaxBreak Figure 5 Comparison Benchmark (`experiments/fig5`)

This experiment compares **Cerebras CS-3 Wafer-Scale Engine** prefill and decode performance against baseline **NVIDIA H100 and H200** GPU results extracted from Figure 5 of *TaxBreak: Unmasking the Hidden Costs of LLM Inference Through Overhead Decomposition* (arXiv:2603.12465v1).

## Target Models & Protocols
- **Models**:
  - `GPT-2` (137M)
  - `Llama-3.2-1B` (1.2B)
  - `Llama-3.2-3B` (3B)
  - `OLMoE-1B/7B` (MoE)
- **Sequence Length**: $SL = 512$
- **Phases**:
  - **Prefill ($m=1$)**: Single-step prompt latency across batch sizes ($BS \in \{1, 2, 4, 8, 16, 32, 64, 128, 256\}$).
  - **Decode ($m=10$)**: Aggregated 10-token decode window latency across batch sizes.

## Directory Structure
- `export_fig5_results.py`: Script to extract CS-3 metrics, combine with TaxBreak Fig. 5 baseline data, and generate output reports/plots.
- `results/`: Output directory containing:
  - `taxbreak_fig5_report.md`: Full markdown comparison report.
  - `taxbreak_fig5_metrics_summary.json`: Aggregated metrics JSON export.
  - `taxbreak_fig5_metrics_table.csv`: Aggregated CSV metrics table.
  - `fig5_prefill_latency.png`: Prefill latency comparison plot.
  - `fig5_decode_latency.png`: Decode latency comparison plot.
  - `fig5_host_idle_time.png`: Host idle time / orchestration overhead plot.
  - `fig5_compilation_time.png`: AOT compilation time plot.

## How to Run
```bash
/cra-1684/tutorial/pvellaisamy/venvs/r2.10/bin/python experiments/fig5/export_fig5_results.py
```
