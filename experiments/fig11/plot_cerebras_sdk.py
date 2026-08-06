#!/usr/bin/env python3
"""Plot Cerebras SDK Figure-11-compatible prefill results with 95% CI bars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = []
    for path in args.results_dir.rglob("result.json"):
        item = json.loads(path.read_text())
        if item.get("experiment") == "figure_11_cerebras_sdk_prefill":
            records.append(item)
    if not records:
        raise FileNotFoundError("No Cerebras SDK Figure 11 results found.")
    records.sort(key=lambda row: row["workload"]["batch_size"])
    batch_sizes = [row["workload"]["batch_size"] for row in records]
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True, constrained_layout=True)
    for axis, key, title in (
        (axes[0], "server_prefill_ms", "CS-3 prefill latency (server prompt time)"),
        (axes[1], "server_queue_ms", "CS-3 service queue delay"),
    ):
        rows = [(batch, item["summary"].get(key)) for batch, item in zip(batch_sizes, records)]
        rows = [(batch, value) for batch, value in rows if value]
        x = [batch for batch, _ in rows]
        y = [value["mean"] for _, value in rows]
        lower = [value["mean"] - value["ci95_low"] for _, value in rows]
        upper = [value["ci95_high"] - value["mean"] for _, value in rows]
        axis.errorbar(x, y, yerr=[lower, upper], marker="o", capsize=4)
        axis.set_ylabel("Milliseconds")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[1].set_xlabel("Batch size (prompts per SDK request)")
    axes[1].set_xticks((1, 2, 4, 8, 16))
    fig.suptitle("Cerebras Figure 11-compatible scaling (mean ± bootstrap 95% CI)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
