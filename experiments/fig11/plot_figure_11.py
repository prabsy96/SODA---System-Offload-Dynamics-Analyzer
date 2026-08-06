#!/usr/bin/env python3
"""Render the six Figure 11-style panels from benchmark_prefill.py results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


MODELS = ("gpt2", "llama-3.2-1b")
METRICS = (
    ("latency_ms", "p50", "Prefill latency (ms)"),
    ("trace_metrics", "gpu_idle_ms", "GPU idle time (ms)"),
    ("trace_metrics", "cpu_idle_proxy_ms", "CPU idle proxy (ms)"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_results(path: Path) -> list[dict]:
    records = []
    for result_path in path.rglob("result.json"):
        record = json.loads(result_path.read_text())
        if record.get("experiment") == "figure_11_prefill":
            record["_path"] = result_path
            records.append(record)
    if not records:
        raise FileNotFoundError(f"No Figure 11 result.json files beneath {path}")
    return records


def main() -> int:
    args = parse_args()
    records = load_results(args.results_dir)
    platforms = sorted({record["platform"].get("label", record["platform"]["gpu"]) for record in records})
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    for col, model in enumerate(MODELS):
        model_records = [record for record in records if record["model"]["label"] == model]
        if not model_records:
            continue
        for row, (section, key, ylabel) in enumerate(METRICS):
            axis = axes[row][col]
            for platform in platforms:
                values = []
                for record in model_records:
                    if record["platform"].get("label", record["platform"]["gpu"]) != platform:
                        continue
                    value = record.get(section, {}).get(key)
                    if value is not None:
                        values.append((record["workload"]["batch_size"], value))
                if values:
                    values.sort()
                    axis.plot(*zip(*values), marker="o", label=platform)
            axis.set_title(model)
            axis.set_xlabel("Batch size")
            axis.set_ylabel(ylabel)
            axis.set_xticks((1, 2, 4, 8, 16))
            axis.grid(alpha=0.25)
            if row == 0:
                axis.legend(fontsize="small")
    fig.suptitle("Figure 11 reproduction: eager FP16 prefill, sequence length 512")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
