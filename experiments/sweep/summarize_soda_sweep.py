#!/usr/bin/env python3
"""
Summarize a sweep directory by producing table pivots and heatmaps.

Usage:
    python -m experiments.sweep.summarize_soda_sweep output/<sweep_dir>
    python experiments/sweep/summarize_soda_sweep.py output/<sweep_dir>

Creates <sweep_dir>/summary containing one CSV pivot and one heatmap per
model/compile/precision group detected under the provided root.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


METRIC_LABEL = "inference time (ms)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a SODA sweep directory.")
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Path to sweep group directory (e.g., output/<model_group>).",
    )
    return parser.parse_args()


def safe_inference_time(report: Dict) -> Optional[float]:
    perf = report.get("performance_metrics") or report.get("metrics") or {}
    value = perf.get("inference_time_ms")
    if value is not None:
        return float(value)
    timing = perf.get("inference_time_breakdown") or perf.get("inference_time") or {}
    for key in ("torch_measured_inference_time_ms", "trace_calculated_inference_time_ms"):
        if key in timing:
            return float(timing[key])
    return None


def parse_from_dirname(path: Path) -> Tuple[Optional[int], Optional[int], Optional[str], Optional[str]]:
    name = path.name
    if "_bs" not in name or "_sl" not in name:
        return None, None, None, None
    try:
        before_bs, rest = name.split("_bs", 1)
        bs_part, after_bs = rest.split("_sl", 1)
        batch_size = int(bs_part)
        digits = []
        for ch in after_bs:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        seq_len = int("".join(digits)) if digits else None

        tokens = before_bs.split("_")
        compile_type = tokens[-2] if len(tokens) >= 2 else tokens[0]
        precision = tokens[-1] if len(tokens) >= 2 else None
        return batch_size, seq_len, compile_type, precision
    except Exception:
        return None, None, None, None


def infer_model_name(dirname: str, compile_type: Optional[str], precision: Optional[str]) -> str:
    if compile_type and precision:
        token = f"_{compile_type}_{precision}_bs"
        if token in dirname:
            return dirname.split(token, 1)[0].replace("_", "/")
    return dirname.replace("_", "/")


def collect_reports(root: Path) -> List[Dict]:
    rows: List[Dict] = []
    seen_dirs: Set[Path] = set()

    for report_path in root.rglob("report.json"):
        try:
            data = json.loads(report_path.read_text())
        except Exception as exc:
            print(f"Skipping {report_path}: unable to read JSON ({exc})", file=sys.stderr)
            continue

        metadata = data.get("metadata", {})
        config = metadata.get("config", {})
        batch_size = config.get("batch_size")
        seq_len = config.get("seq_len")
        precision = config.get("precision")
        compile_type = config.get("compile_type")
        device = config.get("device")

        if any(v is None for v in (batch_size, seq_len, compile_type, precision)):
            bs_guess, sl_guess, ct_guess, prec_guess = parse_from_dirname(report_path.parent)
            batch_size = batch_size or bs_guess
            seq_len = seq_len or sl_guess
            compile_type = compile_type or ct_guess
            precision = precision or prec_guess

        model_name = metadata.get("model_name") or infer_model_name(report_path.parent.name, compile_type, precision)
        inference_time_ms = safe_inference_time(data)

        rows.append(
            {
                "model_name": model_name,
                "compile_type": compile_type,
                "precision": precision,
                "device": device,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "inference_time_ms": inference_time_ms,
                "status": "ok",
            }
        )
        seen_dirs.add(report_path.parent.resolve())

    experiment_dirs: Set[Path] = set()
    for exp_dir in root.rglob("*_bs*_sl*"):
        if exp_dir.is_dir():
            experiment_dirs.add(exp_dir.resolve())

    missing_dirs = sorted(exp_dir for exp_dir in experiment_dirs if exp_dir not in seen_dirs)
    for exp_dir in missing_dirs:
        bs_guess, sl_guess, ct_guess, prec_guess = parse_from_dirname(exp_dir)
        if bs_guess is None or sl_guess is None:
            continue
        rows.append(
            {
                "model_name": infer_model_name(exp_dir.name, ct_guess, prec_guess),
                "compile_type": ct_guess,
                "precision": prec_guess,
                "device": None,
                "batch_size": bs_guess,
                "seq_len": sl_guess,
                "inference_time_ms": None,
                "status": "oom",
            }
        )
    return rows


def build_sections(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, str, str], List[Dict]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("model_name") or "unknown_model",
            row.get("compile_type") or "unknown_compile",
            row.get("precision") or "unknown_precision",
        )
        grouped[key].append(row)

    sections: List[Dict] = []
    for (model, compile_type, precision), items in grouped.items():
        batch_sizes = sorted({r.get("batch_size") for r in items if r.get("batch_size") is not None})
        seq_lens = sorted({r.get("seq_len") for r in items if r.get("seq_len") is not None})
        lookup = {(r.get("seq_len"), r.get("batch_size")): r for r in items}

        values = []
        statuses = []
        for sl in seq_lens:
            value_row: List[Optional[float]] = []
            status_row: List[Optional[str]] = []
            for bs in batch_sizes:
                entry = lookup.get((sl, bs))
                if entry is None:
                    value_row.append(None)
                    status_row.append(None)
                else:
                    value_row.append(entry.get("inference_time_ms"))
                    status_row.append(entry.get("status"))
            values.append(value_row)
            statuses.append(status_row)

        sections.append(
            {
                "model_name": model,
                "compile_type": compile_type,
                "precision": precision,
                "batch_sizes": batch_sizes,
                "seq_lens": seq_lens,
                "values": values,
                "statuses": statuses,
            }
        )
    return sections


def slugify(text: str) -> str:
    return (
        text.replace("/", "-")
        .replace(" ", "_")
        .replace(":", "-")
        .replace(".", "-")
    )


def short_model_name(name: Optional[str]) -> str:
    if not name:
        return "unknown_model"
    tokens = [part for part in name.split("/") if part]
    return tokens[-1] if tokens else name


def write_pivot(section: Dict, out_csv: Path) -> None:
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_name", section["model_name"]])
        writer.writerow(["compile_type", section["compile_type"]])
        writer.writerow(["precision", section["precision"]])
        writer.writerow(["seq_len"] + [str(bs) for bs in section["batch_sizes"]])
        for row_idx, sl in enumerate(section["seq_lens"]):
            row = [sl]
            for col_idx, _ in enumerate(section["batch_sizes"]):
                value = section["values"][row_idx][col_idx]
                status = section["statuses"][row_idx][col_idx]
                if status == "oom":
                    row.append("OOM")
                elif value is None:
                    row.append("")
                else:
                    row.append(f"{value}")
            writer.writerow(row)


def transpose_grid(grid: List[List]) -> List[List]:
    if not grid:
        return []
    return [list(row) for row in zip(*grid)]


def prepare_grids(section: Dict) -> Tuple[bool, List, List, List[List], List[List]]:
    bs = section["batch_sizes"]
    sl = section["seq_lens"]
    rotate_axes = len(bs) != len(sl)
    if rotate_axes:
        x_labels = sl
        y_labels = bs
        value_grid = transpose_grid(section["values"])
        status_grid = transpose_grid(section["statuses"])
    else:
        x_labels = bs
        y_labels = sl
        value_grid = section["values"]
        status_grid = section["statuses"]
    return rotate_axes, x_labels, y_labels, value_grid, status_grid


def compute_figsize(x_labels: List, y_labels: List, rotate_axes: bool) -> Tuple[float, float]:
    x_tile = 0.6
    y_tile = 0.6
    min_height = 1.8 if rotate_axes else 4
    min_width = 4
    max_tiles = max(len(x_labels), len(y_labels), 1)
    base_dim = max(min_width, min_height, max_tiles * max(x_tile, y_tile))
    fig_width = max(min_width, base_dim * (len(x_labels) / max_tiles if max_tiles else 1), len(x_labels) * x_tile + 1.2)
    fig_height = max(min_height, base_dim * (len(y_labels) / max_tiles if max_tiles else 1), len(y_labels) * y_tile + 1.0)
    return fig_width, fig_height


def format_model_label(name: Optional[str]) -> str:
    clean = short_model_name(name)
    return clean.replace("-", " ")


def apply_colorbar(fig, ax, im, data: np.ndarray) -> None:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = fig.colorbar(im, cax=cax)
    finite_values = data[np.isfinite(data)]
    if finite_values.size:
        vmin = float(finite_values.min())
        vmax = float(finite_values.max())
        im.set_clim(vmin=vmin, vmax=vmax)
        if vmin == vmax:
            cbar.set_ticks([vmin])
        else:
            cbar.set_ticks([vmin, vmax])


def annotate_cells(ax, x_labels: List, y_labels: List, value_grid: List[List], status_grid: List[List]) -> None:
    for i, _ in enumerate(y_labels):
        for j, _ in enumerate(x_labels):
            status = status_grid[i][j]
            value = value_grid[i][j]
            is_number = value is not None and not (isinstance(value, float) and np.isnan(value))
            if status == "oom":
                ax.text(j, i, "OOM", ha="center", va="center", color="red", fontsize=8, fontweight="bold")
            elif is_number:
                ax.text(j, i, f"{float(value):.0f}", ha="center", va="center", color="white", fontsize=8)
            else:
                ax.text(j, i, "DNH", ha="center", va="center", color="#9e9e9e", fontsize=8, fontweight="bold")


def plot_heatmap(section: Dict, out_paths: List[Path]) -> None:
    rotate_axes, x_labels, y_labels, value_grid, status_grid = prepare_grids(section)
    data = np.array(
        [
            [np.nan if v is None else v for v in row]
            for row in value_grid
        ],
        dtype=float,
    )

    fig_width, fig_height = compute_figsize(x_labels, y_labels, rotate_axes)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(data, aspect="equal", cmap="viridis")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    if x_labels and y_labels:
        ax.set_box_aspect(len(y_labels) / len(x_labels))
    if rotate_axes:
        ax.set_xlabel("sequence length")
        ax.set_ylabel("batch size")
    else:
        ax.set_xlabel("batch size")
        ax.set_ylabel("sequence length")
    metric_title = METRIC_LABEL[:1].upper() + METRIC_LABEL[1:]
    model_label = format_model_label(section.get("model_name"))
    precision_label = section.get("precision") or "unknown_precision"
    ax.set_title(f"{metric_title}\n{model_label}, {precision_label}")

    apply_colorbar(fig, ax, im, data)
    annotate_cells(ax, x_labels, y_labels, value_grid, status_grid)

    fig.tight_layout()

    for out_path in out_paths:
        suffix = out_path.suffix.lower()
        save_kwargs = {"dpi": 150} if suffix in {".png", ".jpg", ".jpeg"} else {}
        fig.savefig(out_path, **save_kwargs)
    plt.close(fig)


def summarize(root: Path) -> None:
    rows = collect_reports(root)
    if not rows:
        raise RuntimeError(f"No report.json files found under {root}")

    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    sections = build_sections(rows)
    for section in sections:
        slug = slugify(f"{section['model_name']}_{section['compile_type']}_{section['precision']}")
        csv_path = summary_dir / f"{slug}_pivot.csv"
        png_path = summary_dir / f"{slug}_heatmap.png"
        pdf_path = summary_dir / f"{slug}_heatmap.pdf"
        write_pivot(section, csv_path)
        plot_heatmap(section, [png_path, pdf_path])
        print("Wrote")
        print(f"* Summary to {csv_path}")
        print(f"* Heatmap to {png_path}")
        print(f"* Heatmap to {pdf_path}")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not root.exists():
        print(f"Path does not exist: {root}", file=sys.stderr)
        return 1
    try:
        summarize(root)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
