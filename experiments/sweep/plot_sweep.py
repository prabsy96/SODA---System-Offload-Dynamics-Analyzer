#!/usr/bin/env python3
"""
Generate plots from SODA sweep outputs.

Run from repo root:
    python -m experiments.plot_sweep output/<run_dir>

Outputs one heatmap PNG per (model, compile_type, precision) group, showing
seq_len on the Y axis, batch_size on the X axis, colored by inference_time_ms.
Optionally, with --bar-batch-size, also emits bar charts of inference_time_ms
per seq_len for the specified batch size.
"""

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

# Allow running directly (python experiments/plot_sweep.py …) or via -m.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from summarize_sweep import collect_reports, slugify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SODA sweep outputs.")
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Path to sweep output directory (default: current directory).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write plots (default: root).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output image DPI.",
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colormap to use.",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Log10 scale the color values (useful if times differ by orders of magnitude).",
    )
    parser.add_argument(
        "--bar-batch-size",
        type=int,
        default=None,
        help="If set, generate a bar chart per model group for this batch size (x=seq_len, y=inference_time_ms).",
    )
    return parser.parse_args()


def build_matrix(items: List[Dict]) -> Tuple[List[int], List[int], List[List[Optional[float]]], List[List[Optional[str]]]]:
    batch_sizes = sorted({r.get("batch_size") for r in items if r.get("batch_size") is not None})
    seq_lens = sorted({r.get("seq_len") for r in items if r.get("seq_len") is not None})
    lookup = {(r.get("seq_len"), r.get("batch_size")): r.get("inference_time_ms") for r in items}
    status_lookup = {(r.get("seq_len"), r.get("batch_size")): r.get("status") for r in items}

    matrix: List[List[Optional[float]]] = []
    status_matrix: List[List[Optional[str]]] = []
    for sl in seq_lens:
        row: List[Optional[float]] = []
        status_row: List[Optional[str]] = []
        for bs in batch_sizes:
            key = (sl, bs)
            row.append(lookup.get(key))
            status_row.append(status_lookup.get(key))
        matrix.append(row)
        status_matrix.append(status_row)
    return batch_sizes, seq_lens, matrix, status_matrix


def plot_heatmap(
    key: Tuple[str, str, str],
    batch_sizes: List[int],
    seq_lens: List[int],
    matrix: List[List[Optional[float]]],
    status_matrix: List[List[Optional[str]]],
    out_path: Path,
    cmap: str,
    use_log: bool,
    dpi: int,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not available. Please install it (pip install matplotlib).")

    model, compile_type, precision = key
    # Replace None with NaN for masking
    values = [[(float("nan") if v is None else v) for v in row] for row in matrix]

    fig, ax = plt.subplots(figsize=(6, 4))
    data = values

    if use_log:
        # Avoid log(0) by masking non-positive
        data = [
            [float("nan") if (v is None or v <= 0 or math.isnan(v)) else math.log10(v) for v in row]
            for row in values
        ]

    im = ax.imshow(data, aspect="auto", cmap=cmap)
    norm = im.norm
    cmap_obj = im.get_cmap()

    def pick_text_color(value: float) -> str:
        # Choose black/white for contrast against the cell background.
        r, g, b, _ = cmap_obj(norm(value))
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return "black" if luminance > 0.6 else "white"

    ax.set_xticks(range(len(batch_sizes)))
    ax.set_xticklabels(batch_sizes)
    ax.set_yticks(range(len(seq_lens)))
    ax.set_yticklabels(seq_lens)
    ax.set_xlabel("batch_size")
    ax.set_ylabel("seq_len")

    title = f"{model} | {compile_type} | {precision}"
    ax.set_title(title)
    cbar_label = "log10(inference_time_ms)" if use_log else "inference_time_ms"
    cbar = fig.colorbar(im, ax=ax, label=cbar_label)

    # Annotate with raw ms values for readability on small grids
    for i, sl in enumerate(seq_lens):
        for j, bs in enumerate(batch_sizes):
            val = matrix[i][j]
            status = status_matrix[i][j]
            data_val = data[i][j]
            if status == "oom":
                ax.text(
                    j,
                    i,
                    "OOM",
                    ha="center",
                    va="center",
                    color="red",
                    fontsize=8,
                    weight="bold",
                )
                continue
            if val is None or math.isnan(data_val):
                continue
            text_color = pick_text_color(data_val)
            ax.text(
                j,
                i,
                f"{val:.0f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
                weight="bold",
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_bar(
    key: Tuple[str, str, str],
    seq_lens: List[int],
    values: List[Optional[float]],
    statuses: List[Optional[str]],
    batch_size: int,
    out_path: Path,
    dpi: int,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not available. Please install it (pip install matplotlib).")

    model, compile_type, precision = key

    fig, ax = plt.subplots(figsize=(6, 4))
    bar_values = [v if v is not None else 0.0 for v in values]

    ax.bar([str(sl) for sl in seq_lens], bar_values, color="#4c72b0")
    ax.set_xlabel("seq_len")
    ax.set_ylabel("inference_time_ms")
    ax.set_title(f"{model} | {compile_type} | {precision} | bs={batch_size}")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for i, v in enumerate(values):
        status = statuses[i] if i < len(statuses) else None
        if status == "oom":
            ax.text(i, 0, "OOM", ha="center", va="bottom", fontsize=8, color="red")
            continue
        if v is None:
            continue
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    out_dir = (args.out_dir or root).resolve()

    if plt is None:
        print("matplotlib is not installed. Please install it with `pip install matplotlib`.", file=sys.stderr)
        return 1

    if not root.exists():
        print(f"Root path does not exist: {root}", file=sys.stderr)
        return 1

    rows = collect_reports(root)
    if not rows:
        print(f"No report.json files found under {root}", file=sys.stderr)
        return 1

    grouped: Dict[Tuple[str, str, str], List[Dict]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("model_name") or "unknown_model",
            row.get("compile_type") or "unknown_compile",
            row.get("precision") or "unknown_precision",
        )
        grouped[key].append(row)

    for key, items in grouped.items():
        batch_sizes, seq_lens, matrix, status_matrix = build_matrix(items)
        if not batch_sizes or not seq_lens:
            print(f"Skipping {key}: missing batch_size or seq_len data", file=sys.stderr)
            continue
        filename = f"heatmap_{slugify(key[0])}_{slugify(key[1])}_{slugify(key[2])}.png"
        out_path = out_dir / filename
        try:
            plot_heatmap(
                key,
                batch_sizes,
                seq_lens,
                matrix,
                status_matrix,
                out_path,
                args.cmap,
                args.log,
                args.dpi,
            )
            print(f"Wrote plot: {out_path}")
        except Exception as exc:  # pragma: no cover - runtime plotting issues
            print(f"Failed to plot {key}: {exc}", file=sys.stderr)
            continue

        if args.bar_batch_size is not None:
            bs = args.bar_batch_size
            if bs not in batch_sizes:
                print(f"Skipping bar plot for {key}: batch_size {bs} not found", file=sys.stderr)
                continue
            # Extract values for this batch size in seq_len order
            col_idx = batch_sizes.index(bs)
            vals = [matrix[i][col_idx] for i in range(len(seq_lens))]
            status_vals = [status_matrix[i][col_idx] for i in range(len(seq_lens))]
            bar_name = f"bar_{slugify(key[0])}_{slugify(key[1])}_{slugify(key[2])}_bs{bs}.png"
            bar_path = out_dir / bar_name
            try:
                plot_bar(key, seq_lens, vals, status_vals, bs, bar_path, args.dpi)
                print(f"Wrote plot: {bar_path}")
            except Exception as exc:
                print(f"Failed to plot bar for {key}, bs={bs}: {exc}", file=sys.stderr)
                continue

    return 0


if __name__ == "__main__":
    sys.exit(main())
