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
T_EXPOSED_LABEL = "T_exposed (ms)"

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

def safe_t_exposed(report: Dict) -> Optional[float]:
    """Extract T_exposed from report with fallbacks."""
    perf = report.get("performance_metrics") or report.get("metrics") or {}
    
    # Try framework_overhead dict first (new structure)
    framework = perf.get("framework_overhead", {})
    t_exposed = framework.get("T_exposed_ms")
    if t_exposed is not None:
        return float(t_exposed)
    
    # Fallback: Try direct key (old structure)
    t_exposed = perf.get("T_exposed_ms")
    if t_exposed is not None:
        return float(t_exposed)
    
    return None

def get_gpu_name(data: Dict) -> str:
    """Extract GPU name from report data with fallbacks."""
    meta = data.get("metadata", {})
    if "gpu_name" in meta: return meta["gpu_name"]
    if "device_name" in meta: return meta["device_name"]
    
    env = data.get("environment", {})
    if "gpu" in env: return env["gpu"]
    if "gpu_name" in env: return env["gpu_name"]
    
    return "gpu"


def short_gpu_name(name: str) -> str:
    """Shorten GPU name for filenames and titles (e.g. 'NVIDIA H100 80GB' -> 'H100')."""
    if not name or name.lower() == "gpu":
        return ""
    upper = name.upper()
    # Check for common architectures
    for key in ["H100", "H200", "A100", "V100", "T4", "L4", "4090", "3090"]:
        if key in upper:
            return key
    # Fallback: remove NVIDIA and take first word
    return name.replace("NVIDIA", "").replace("nvidia", "").strip().split(" ")[0]


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
        t_exposed_ms = safe_t_exposed(data)
        gpu_name = get_gpu_name(data)  # FIX: Extract GPU name from report

        rows.append(
            {
                "model_name": model_name,
                "compile_type": compile_type,
                "precision": precision,
                "device": device,
                "gpu_name": gpu_name,  # FIX: Add gpu_name to row
                "batch_size": batch_size,
                "seq_len": seq_len,
                "inference_time_ms": inference_time_ms,
                "t_exposed_ms": t_exposed_ms,
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
                "gpu_name": None,  # FIX: OOM dirs don't have GPU name
                "batch_size": bs_guess,
                "seq_len": sl_guess,
                "inference_time_ms": None,
                "t_exposed_ms": None,
                "status": "oom",
            }
        )
    return rows


def build_sections(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, str, str, str], List[Dict]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("model_name") or "unknown_model",
            row.get("compile_type") or "unknown_compile",
            row.get("precision") or "unknown_precision",
            row.get("gpu_name") or "unknown_gpu",
        )
        grouped[key].append(row)

    sections: List[Dict] = []
    for (model, compile_type, precision, gpu_name), items in grouped.items():
        batch_sizes = sorted({r.get("batch_size") for r in items if r.get("batch_size") is not None})
        seq_lens = sorted({r.get("seq_len") for r in items if r.get("seq_len") is not None})
        lookup = {(r.get("seq_len"), r.get("batch_size")): r for r in items}

        values = []
        statuses = []
        t_exposed_values = [] 
        for sl in seq_lens:
            value_row: List[Optional[float]] = []
            status_row: List[Optional[str]] = []
            t_exposed_row: List[Optional[float]] = [] 
            for bs in batch_sizes:
                entry = lookup.get((sl, bs))
                if entry is None:
                    value_row.append(None)
                    status_row.append(None)
                    t_exposed_row.append(None) 
                else:
                    value_row.append(entry.get("inference_time_ms"))
                    status_row.append(entry.get("status"))
                    t_exposed_row.append(entry.get("t_exposed_ms"))
            values.append(value_row)
            statuses.append(status_row)
            t_exposed_values.append(t_exposed_row)

        sections.append(
            {
                "model_name": model,
                "compile_type": compile_type,
                "precision": precision,
                "gpu_name": gpu_name, # Ensure this is passed to section dict
                "batch_sizes": batch_sizes,
                "seq_lens": seq_lens,
                "values": values,
                "statuses": statuses,
                "t_exposed_values": t_exposed_values,
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


def write_pivot(section: Dict, out_csv: Path, metric_key: str = "values", metric_name: str = "inference_time") -> None:
    """Write pivot table to CSV. Can be used for inference_time or t_exposed."""
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_name", section["model_name"]])
        writer.writerow(["compile_type", section["compile_type"]])
        writer.writerow(["precision", section["precision"]])
        writer.writerow(["metric", metric_name])  # ADD THIS
        writer.writerow(["seq_len"] + [str(bs) for bs in section["batch_sizes"]])
        for row_idx, sl in enumerate(section["seq_lens"]):
            row = [sl]
            for col_idx, _ in enumerate(section["batch_sizes"]):
                value = section[metric_key][row_idx][col_idx]
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


def prepare_grids(section: Dict, metric_key: str = "values") -> Tuple[bool, List, List, List[List], List[List]]:
    """Prepare grids for plotting. Supports both inference_time and t_exposed."""
    bs = section["batch_sizes"]
    sl = section["seq_lens"]
    rotate_axes = len(bs) != len(sl)
    if rotate_axes:
        x_labels = sl
        y_labels = bs
        value_grid = transpose_grid(section[metric_key])
        status_grid = transpose_grid(section["statuses"])
    else:
        x_labels = bs
        y_labels = sl
        value_grid = section[metric_key]
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
                ax.text(j, i, "OOM", ha="center", va="center", color="red", fontsize=6, fontweight="bold")
            elif is_number:
                ax.text(j, i, f"{float(value):.0f}", ha="center", va="center", color="white", fontsize=6)
            else:
                ax.text(j, i, "DNH", ha="center", va="center", color="#9e9e9e", fontsize=6, fontweight="bold")


def plot_heatmap(section: Dict, out_paths: List[Path], metric_key: str = "values", metric_label: str = METRIC_LABEL) -> None:
    """Plot heatmap. Can be used for inference_time or t_exposed."""
    rotate_axes, x_labels, y_labels, value_grid, status_grid = prepare_grids(section, metric_key)
    data = np.array(
        [
            [np.nan if v is None else v for v in row]
            for row in value_grid
        ],
        dtype=float,
    )

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "font.size": 8,
        "axes.labelsize": 7,
        "axes.titlesize": 8, 
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6,
        "axes.linewidth": 1.5,
        "lines.linewidth": 3,
        "lines.markersize": 6,
        "grid.linewidth": 0.6,
        "grid.linestyle": ":",
        "mathtext.fontset": "stix",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
    })

    IEEE_COL_WIDTH = 3.5 
    n_rows = len(y_labels)
    cell_height = 0.2
    base_height = n_rows * cell_height
    fig_height = base_height + 0.8

    fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH, fig_height))

    im = ax.imshow(data, aspect="auto", cmap="viridis")
    
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Sequence Length")
    
    metric_title = metric_label[:1].upper() + metric_label[1:]
    model_label = format_model_label(section.get("model_name"))
    precision_label = section.get("precision") or "unknown_precision"
    gpu_short = short_gpu_name(section.get("gpu_name"))
    title_str = f"{metric_title}\n{model_label} ({precision_label})"
    if gpu_short:
        title_str += f" [{gpu_short}]"
    ax.set_title(title_str, pad=8)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)

    finite_values = data[np.isfinite(data)]
    if finite_values.size:
        vmin = float(finite_values.min())
        vmax = float(finite_values.max())
        im.set_clim(vmin=vmin, vmax=vmax)
        cbar.set_ticks([vmin, vmax])
        cbar.ax.set_yticklabels([f"{vmin:.0f}", f"{vmax:.0f}"])

    annotate_cells(ax, x_labels, y_labels, value_grid, status_grid)

    fig.tight_layout()

    for out_path in out_paths:
        suffix = out_path.suffix.lower()
        save_kwargs = {"dpi": 600} if suffix in {".png", ".jpg", ".jpeg"} else {}
        fig.savefig(out_path, **save_kwargs, bbox_inches="tight")
    plt.close(fig)


def summarize(root: Path, gpu_name_override: Optional[str] = None, max_tok_override: Optional[str] = None) -> None:
    rows = collect_reports(root)
    if not rows:
        raise RuntimeError(f"No report.json files found under {root}")

    if gpu_name_override:
        for row in rows:
            row["gpu_name"] = gpu_name_override

    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    sections = build_sections(rows)
    for section in sections:
        gpu_suffix = short_gpu_name(section['gpu_name'])
        slug_str = f"{section['model_name']}_{section['compile_type']}_{section['precision']}"
        
        if max_tok_override:
            slug_str += f"_{max_tok_override}"
        
        if gpu_suffix:
            slug_str += f"_{gpu_suffix}"
        
        slug = slugify(slug_str)
        
        # Inference time outputs
        csv_path = summary_dir / f"{slug}_pivot.csv"
        png_path = summary_dir / f"{slug}_heatmap.png"
        pdf_path = summary_dir / f"{slug}_heatmap.pdf"
        
        write_pivot(section, csv_path, metric_key="values", metric_name="inference_time_ms")
        plot_heatmap(section, [png_path, pdf_path], metric_key="values", metric_label=METRIC_LABEL)
        
        # T_exposed outputs (ADD THIS BLOCK)
        t_exposed_csv = summary_dir / f"{slug}_t_exposed_pivot.csv"
        t_exposed_png = summary_dir / f"{slug}_t_exposed_heatmap.png"
        t_exposed_pdf = summary_dir / f"{slug}_t_exposed_heatmap.pdf"
        
        write_pivot(section, t_exposed_csv, metric_key="t_exposed_values", metric_name="t_exposed_ms")
        plot_heatmap(section, [t_exposed_png, t_exposed_pdf], metric_key="t_exposed_values", metric_label=T_EXPOSED_LABEL)
        
        print("Wrote")
        print(f"* Inference time summary to {csv_path}")
        print(f"* Inference time heatmap to {png_path}, {pdf_path}")
        print(f"* T_exposed summary to {t_exposed_csv}")
        print(f"* T_exposed heatmap to {t_exposed_png}, {t_exposed_pdf}")


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