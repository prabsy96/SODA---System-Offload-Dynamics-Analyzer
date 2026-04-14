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
from typing import Dict, List, Optional, Tuple, Set, Union

METRIC_LABEL = "inference time (ms)"
T_EXPOSED_LABEL = "T_exposed (ms)"
GPU_ACTIVE_LABEL = "GPU Active Time (ms)"
TKLQT_LABEL = "TKLQT (µs)"
PEAK_MEMORY_LABEL = "Peak Memory (MB)"

_SWEEP_HTML_CSS = """
:root {
  --bg: #0d1117; --bg-alt: #161b22; --bg-card: #1c2128;
  --text: #c9d1d9; --text-dim: #8b949e; --border: #30363d;
  --green: #3fb950; --yellow: #d29922; --red: #f85149;
  --cyan: #79c0ff; --blue: #388bfd;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Liberation Sans", sans-serif;
  font-size: 14px; line-height: 1.6; padding: 24px;
}
h1 { font-size: 20px; margin-bottom: 6px; color: var(--cyan); }
h2 { font-size: 16px; margin: 28px 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--border); }
.meta { color: var(--text-dim); font-size: 12px; margin-bottom: 24px; }
.heatmap-row { display: flex; gap: 20px; overflow-x: auto; margin-bottom: 28px; padding-bottom: 4px; }
.heatmap-item { flex: 0 0 auto; text-align: center; }
.heatmap-item img { max-height: 340px; border: 1px solid var(--border); border-radius: 4px; }
.heatmap-label { font-size: 11px; color: var(--text-dim); margin-top: 4px; }
table { width: 100%; border-collapse: collapse; margin-bottom: 28px; font-size: 13px; }
thead tr { background: var(--bg-alt); }
th { padding: 8px 12px; text-align: left; font-weight: 600; border-bottom: 2px solid var(--border); white-space: nowrap; }
td { padding: 7px 12px; border-bottom: 1px solid var(--border); }
tr:nth-child(even) { background: var(--bg-alt); }
tr:hover { background: var(--bg-card); }
th.sortable { cursor: pointer; user-select: none; }
th.sortable:hover { color: var(--cyan); }
th.asc::after { content: " ▲"; }
th.desc::after { content: " ▼"; }
.oom { color: var(--red);   font-weight: 600; }
.na  { color: var(--text-dim); }
.gpu-high { color: var(--green); }
.gpu-mid  { color: var(--yellow); }
.gpu-low  { color: var(--red); }
footer { margin-top: 32px; padding-top: 12px; border-top: 1px solid var(--border); color: var(--text-dim); font-size: 11px; }
"""

_SWEEP_HTML_JS = """
(function () {
  function sortTable(table, colIdx, asc) {
    var tbody = table.querySelector('tbody');
    var rows = Array.from(tbody.querySelectorAll('tr'));
    rows.sort(function (a, b) {
      var ca = a.cells[colIdx], cb = b.cells[colIdx];
      var va = ca ? (ca.getAttribute('data-val') || ca.textContent.trim()) : '';
      var vb = cb ? (cb.getAttribute('data-val') || cb.textContent.trim()) : '';
      var na = parseFloat(va), nb = parseFloat(vb);
      if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
      return asc ? va.localeCompare(vb) : vb.localeCompare(va);
    });
    rows.forEach(function (r) { tbody.appendChild(r); });
  }
  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('th.sortable').forEach(function (th) {
      th.addEventListener('click', function () {
        var table = th.closest('table');
        var idx = Array.from(th.parentNode.children).indexOf(th);
        var asc = !th.classList.contains('asc');
        th.parentNode.querySelectorAll('th').forEach(function (h) {
          h.classList.remove('asc', 'desc');
        });
        th.classList.add(asc ? 'asc' : 'desc');
        sortTable(table, idx, asc);
      });
    });
  });
}());
"""

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


def safe_inference_time(report: Dict) -> Union[float, str, None]:
    perf = report.get("performance_metrics") or report.get("metrics") or {}
    value = perf.get("inference_time_ms")
    
    # FIX: Handle "OOM" string explicitly
    if value == "OOM":
        return "OOM"
        
    if value is not None:
        try:
            return float(value)
        except (ValueError, TypeError):
            pass

    timing = perf.get("inference_time_breakdown") or perf.get("inference_time") or {}
    for key in ("torch_measured_inference_time_ms", "trace_calculated_inference_time_ms"):
        if key in timing:
            try:
                return float(timing[key])
            except (ValueError, TypeError):
                continue
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

def safe_kv_cache_mb(report: Dict) -> Optional[float]:
    """Extract empirical KV cache size from report."""
    perf = report.get("performance_metrics") or report.get("metrics") or {}
    mem = perf.get("memory_metrics", {})
    value = mem.get("kv_cache_mb")
    if value is not None:
        try:
            return float(value)
        except (ValueError, TypeError):
            pass
    return None


def safe_peak_memory(report: Dict) -> Union[float, str, None]:
    """Extract peak memory allocated from report."""
    perf = report.get("performance_metrics") or report.get("metrics") or {}
    mem = perf.get("memory_metrics", {})
    value = mem.get("peak_memory_allocated_mb")
    if value == "OOM":
        return "OOM"
    if value is not None:
        try:
            return float(value)
        except (ValueError, TypeError):
            pass
    return None

def get_gpu_name(data: Dict) -> str:
    """Extract GPU name from report data with fallbacks."""
    meta = data.get("metadata", {})
    # Primary: stored in metadata.config (current SODA schema)
    config = meta.get("config", {})
    if config.get("gpu_name"):
        return config["gpu_name"]
    # Legacy: direct metadata keys
    if "gpu_name" in meta: return meta["gpu_name"]
    if "device_name" in meta: return meta["device_name"]
    # Legacy: environment block
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
        status = "ok"
        if inference_time_ms == "OOM":
            status = "oom"
            inference_time_ms = None

        t_exposed_ms = safe_t_exposed(data)
        gpu_name = get_gpu_name(data)

        perf = data.get("performance_metrics") or data.get("metrics") or {}
        gpu_active_us = perf.get("true_gpu_busy_time_us") or perf.get("gpu_busy_time_us") or 0
        gpu_active_ms = gpu_active_us / 1000.0 if gpu_active_us else None

        tklqt_data = perf.get("tklqt", {})
        tklqt_us = tklqt_data.get("total", None) if isinstance(tklqt_data, dict) else None

        peak_memory_mb = safe_peak_memory(data)
        if peak_memory_mb == "OOM":
            peak_memory_mb = None

        kv_cache_mb = safe_kv_cache_mb(data)

        throughput_info = perf.get("inference_throughput", {})
        throughput_tok_s = throughput_info.get("throughput_tok_s") if isinstance(throughput_info, dict) else None
        gpu_util_pct = perf.get("gpu_utilization_percent")

        rows.append(
            {
                "model_name": model_name,
                "compile_type": compile_type,
                "precision": precision,
                "device": device,
                "gpu_name": gpu_name,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "inference_time_ms": inference_time_ms,
                "t_exposed_ms": t_exposed_ms,
                "gpu_active_ms": gpu_active_ms,
                "tklqt_us": tklqt_us,
                "peak_memory_mb": peak_memory_mb,
                "kv_cache_mb": kv_cache_mb,
                "throughput_tok_s": throughput_tok_s,
                "gpu_util_pct": gpu_util_pct,
                "status": status,
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
                "gpu_name": None,
                "batch_size": bs_guess,
                "seq_len": sl_guess,
                "inference_time_ms": None,
                "t_exposed_ms": None,
                "gpu_active_ms": None,
                "tklqt_us": None,
                "peak_memory_mb": None,
                "kv_cache_mb": None,
                "throughput_tok_s": None,
                "gpu_util_pct": None,
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
        gpu_active_values = []
        t_exposed_values = []
        tklqt_values = []
        peak_memory_values = []

        for sl in seq_lens:
            value_row: List[Optional[float]] = []
            status_row: List[Optional[str]] = []
            t_exposed_row: List[Optional[float]] = []
            gpu_active_row: List[Optional[float]] = []
            tklqt_row: List[Optional[float]] = []
            peak_memory_row: List[Optional[float]] = []

            for bs in batch_sizes:
                entry = lookup.get((sl, bs))
                if entry is None:
                    value_row.append(None)
                    status_row.append(None)
                    t_exposed_row.append(None)
                    gpu_active_row.append(None)
                    tklqt_row.append(None)
                    peak_memory_row.append(None)
                else:
                    value_row.append(entry.get("inference_time_ms"))
                    status_row.append(entry.get("status"))
                    t_exposed_row.append(entry.get("t_exposed_ms"))
                    gpu_active_row.append(entry.get("gpu_active_ms"))
                    tklqt_row.append(entry.get("tklqt_us"))
                    peak_memory_row.append(entry.get("peak_memory_mb"))
            values.append(value_row)
            statuses.append(status_row)
            t_exposed_values.append(t_exposed_row)
            gpu_active_values.append(gpu_active_row)
            tklqt_values.append(tklqt_row)
            peak_memory_values.append(peak_memory_row)

        sections.append(
            {
                "model_name": model,
                "compile_type": compile_type,
                "precision": precision,
                "gpu_name": gpu_name,
                "batch_sizes": batch_sizes,
                "seq_lens": seq_lens,
                "values": values,
                "statuses": statuses,
                "t_exposed_values": t_exposed_values,
                "gpu_active_values": gpu_active_values,
                "tklqt_values": tklqt_values,
                "peak_memory_values": peak_memory_values,
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
    # Drop the org prefix (e.g. "meta-llama/") — just keep the model slug
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
    """Prepare grids for plotting. x=batch_size, y=seq_len always."""
    bs = section["batch_sizes"]
    sl = section["seq_lens"]
    x_labels = bs
    y_labels = sl
    value_grid = section[metric_key]
    status_grid = section["statuses"]
    return False, x_labels, y_labels, value_grid, status_grid


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
                ax.text(j, i, f"{float(value):.0f}", ha="center", va="center", color="black", fontsize=7, fontweight="bold")
            else:
                ax.text(j, i, "DNH", ha="center", va="center", color="#9e9e9e", fontsize=7, fontweight="bold")


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

    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r")
    
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)

    ax.set_xlabel("Batch Size", fontsize=8, fontweight="bold")
    ax.set_ylabel("Seq Len (tokens)", fontsize=8, fontweight="bold")

    # Single-line title: ModelName · precision · GPU · (unit)
    import re
    model_short   = short_model_name(section.get("model_name"))
    precision_str = section.get("precision") or ""
    gpu_short     = short_gpu_name(section.get("gpu_name") or "")
    unit_match    = re.search(r"\(.*?\)", metric_label)
    unit_str      = unit_match.group(0) if unit_match else ""

    title_parts = [p for p in [model_short, precision_str, gpu_short, unit_str] if p]
    title_str   = " · ".join(title_parts)

    ax.set_title(title_str, fontsize=8, pad=3, fontweight="bold")

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


def _build_section_slug(section: Dict, max_tok_override: Optional[str] = None) -> str:
    """Reconstruct the slug string used when writing PNGs for a section."""
    gpu_suffix = short_gpu_name(section.get("gpu_name") or "")
    slug_str = f"{section['model_name']}_{section['compile_type']}_{section['precision']}"
    if max_tok_override:
        slug_str += f"_{max_tok_override}"
    if gpu_suffix:
        slug_str += f"_{gpu_suffix}"
    return slugify(slug_str)


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _fmt_val(value: Optional[float], fmt: str = "{:.1f}", suffix: str = "") -> str:
    if value is None:
        return '<span class="na">–</span>'
    return f"{fmt.format(value)}{suffix}"


def _td(value: Optional[float], is_oom: bool = False, fmt: str = "{:.1f}", suffix: str = "") -> str:
    if is_oom and value is None:
        return '<td class="oom">OOM</td>'
    dv = value if value is not None else ""
    return f'<td data-val="{dv}">{_fmt_val(value, fmt, suffix)}</td>'


def _td_gpu_util(value: Optional[float], is_oom: bool = False) -> str:
    if is_oom and value is None:
        return '<td class="oom">OOM</td>'
    if value is None:
        return '<td class="na">–</td>'
    cls = "gpu-high" if value >= 60 else ("gpu-mid" if value >= 30 else "gpu-low")
    return f'<td data-val="{value}"><span class="{cls}">{value:.1f}%</span></td>'


def _img_data_uri(path: Path) -> str:
    """Return a base64 data URI for a PNG so the HTML is fully self-contained."""
    import base64
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _html_heatmap_row(heatmap_files: List[Tuple[str, str]], summary_dir: Path) -> str:
    items = []
    for filename, label in heatmap_files:
        img_path = summary_dir / filename
        if img_path.exists():
            src = _img_data_uri(img_path)
            items.append(
                f'<div class="heatmap-item">'
                f'<img src="{src}" alt="{_html_escape(label)}">'
                f'<div class="heatmap-label">{_html_escape(label)}</div>'
                f'</div>'
            )
    if not items:
        return '<p class="na" style="margin-bottom:28px">No heatmap images found.</p>\n'
    return '<div class="heatmap-row">' + "".join(items) + '</div>\n'


def _html_run_table(section_rows: List[Dict]) -> str:
    sorted_rows = sorted(
        section_rows,
        key=lambda r: (r.get("seq_len") or 0, r.get("batch_size") or 0),
    )
    headers = [
        ("Seq Len",          True),
        ("Batch Size",       True),
        ("Inference (ms)",   True),
        ("Throughput (tok/s)", True),
        ("GPU Util (%)",     True),
        ("GPU Active (ms)",  True),
        ("TKLQT (µs)",       True),
        ("Peak Mem (MB)",    True),
        ("KV Cache (MB)",    True),
    ]
    th_parts = []
    for label, sortable in headers:
        cls = ' class="sortable"' if sortable else ""
        th_parts.append(f"<th{cls}>{_html_escape(label)}</th>")
    rows_html = []
    for row in sorted_rows:
        status  = row.get("status", "ok")
        is_oom  = status == "oom"
        sl      = row.get("seq_len")
        bs      = row.get("batch_size")
        rows_html.append(
            "<tr>"
            f'<td data-val="{sl or 0}">{sl if sl is not None else "–"}</td>'
            f'<td data-val="{bs or 0}">{bs if bs is not None else "–"}</td>'
            + _td(row.get("inference_time_ms"), is_oom)
            + _td(row.get("throughput_tok_s"),  is_oom)
            + _td_gpu_util(row.get("gpu_util_pct"), is_oom)
            + _td(row.get("gpu_active_ms"),     is_oom)
            + _td(row.get("tklqt_us"),          is_oom, fmt="{:.0f}")
            + _td(row.get("peak_memory_mb"),    is_oom, fmt="{:.0f}")
            + _td(row.get("kv_cache_mb"),       False,  fmt="{:.1f}")
            + "</tr>\n"
        )
    return (
        f'<table>\n<thead><tr>{"".join(th_parts)}</tr></thead>\n'
        f'<tbody>\n{"".join(rows_html)}</tbody>\n</table>\n'
    )


def generate_comparative_html(
    sections: List[Dict],
    rows: List[Dict],
    summary_dir: Path,
    max_tok_override: Optional[str] = None,
) -> None:
    """Write summary/comparative_report.html referencing the already-written PNGs."""
    from datetime import datetime
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    parts = [
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"UTF-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
        "<title>SODA Sweep Report</title>\n"
        f"<style>{_SWEEP_HTML_CSS}</style>\n"
        "</head>\n<body>\n"
        "<h1>SODA Sweep Report</h1>\n"
        f'<div class="meta">Generated: {_html_escape(now_str)}'
        f" &nbsp;·&nbsp; {len(rows)} run(s)"
        f" &nbsp;·&nbsp; {len(sections)} group(s)</div>\n",
    ]

    for section in sections:
        slug = _build_section_slug(section, max_tok_override)
        model        = section.get("model_name") or "unknown"
        compile_type = section.get("compile_type") or ""
        precision    = section.get("precision") or ""
        gpu_name     = section.get("gpu_name") or ""
        gpu_short    = short_gpu_name(gpu_name)

        meta_parts = [p for p in [compile_type, precision, gpu_short or gpu_name] if p]
        meta_span  = (
            f" &nbsp;<span style='color:var(--text-dim);font-weight:400;font-size:13px'>"
            f"({_html_escape(' · '.join(meta_parts))})</span>"
            if meta_parts else ""
        )
        parts.append(f"<h2>{_html_escape(model)}{meta_span}</h2>\n")

        heatmap_files: List[Tuple[str, str]] = [
            (f"{slug}_heatmap.png",             "Inference Time (ms)"),
            (f"{slug}_gpu_active_heatmap.png",   "GPU Active (ms)"),
            (f"{slug}_t_exposed_heatmap.png",    "T_exposed (ms)"),
            (f"{slug}_tklqt_heatmap.png",        "TKLQT (µs)"),
            (f"{slug}_peak_memory_heatmap.png",  "Peak Memory (MB)"),
        ]
        parts.append(_html_heatmap_row(heatmap_files, summary_dir))

        section_rows = [
            r for r in rows
            if (r.get("model_name")    or "unknown_model")    == (section.get("model_name")    or "unknown_model")
            and (r.get("compile_type") or "unknown_compile")  == (section.get("compile_type")  or "unknown_compile")
            and (r.get("precision")    or "unknown_precision") == (section.get("precision")    or "unknown_precision")
            and (r.get("gpu_name")     or "unknown_gpu")       == (section.get("gpu_name")     or "unknown_gpu")
        ]
        parts.append(_html_run_table(section_rows))

    parts.append(
        f'<footer>Generated by SODA summarize_soda_sweep.py &nbsp;·&nbsp; {_html_escape(now_str)}</footer>\n'
        f"<script>{_SWEEP_HTML_JS}</script>\n"
        "</body>\n</html>\n"
    )

    out_path = summary_dir / "comparative_report.html"
    out_path.write_text("".join(parts), encoding="utf-8")


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
        
        # 1. Inference time outputs
        csv_path = summary_dir / f"{slug}_pivot.csv"
        png_path = summary_dir / f"{slug}_heatmap.png"
        pdf_path = summary_dir / f"{slug}_heatmap.pdf"
        
        write_pivot(section, csv_path, metric_key="values", metric_name="inference_time_ms")
        plot_heatmap(section, [png_path, pdf_path], metric_key="values", metric_label=METRIC_LABEL)
        
        # 2. T_exposed (GPU Idle) outputs
        t_exposed_csv = summary_dir / f"{slug}_t_exposed_pivot.csv"
        t_exposed_png = summary_dir / f"{slug}_t_exposed_heatmap.png"
        t_exposed_pdf = summary_dir / f"{slug}_t_exposed_heatmap.pdf"
        
        write_pivot(section, t_exposed_csv, metric_key="t_exposed_values", metric_name="t_exposed_ms")
        plot_heatmap(section, [t_exposed_png, t_exposed_pdf], metric_key="t_exposed_values", metric_label=T_EXPOSED_LABEL)
        
        # 3. GPU Active outputs (ADD THIS BLOCK)
        gpu_active_csv = summary_dir / f"{slug}_gpu_active_pivot.csv"
        gpu_active_png = summary_dir / f"{slug}_gpu_active_heatmap.png"
        gpu_active_pdf = summary_dir / f"{slug}_gpu_active_heatmap.pdf"
        
        write_pivot(section, gpu_active_csv, metric_key="gpu_active_values", metric_name="gpu_active_ms")
        plot_heatmap(section, [gpu_active_png, gpu_active_pdf], metric_key="gpu_active_values", metric_label=GPU_ACTIVE_LABEL)
        
        # 4. TKLQT outputs (ADD THIS BLOCK)
        tklqt_csv = summary_dir / f"{slug}_tklqt_pivot.csv"
        tklqt_png = summary_dir / f"{slug}_tklqt_heatmap.png"
        tklqt_pdf = summary_dir / f"{slug}_tklqt_heatmap.pdf"
        
        write_pivot(section, tklqt_csv, metric_key="tklqt_values", metric_name="tklqt_us")
        plot_heatmap(section, [tklqt_png, tklqt_pdf], metric_key="tklqt_values", metric_label=TKLQT_LABEL)

        # 5. Peak Memory outputs
        peak_mem_csv = summary_dir / f"{slug}_peak_memory_pivot.csv"
        peak_mem_png = summary_dir / f"{slug}_peak_memory_heatmap.png"
        peak_mem_pdf = summary_dir / f"{slug}_peak_memory_heatmap.pdf"

        write_pivot(section, peak_mem_csv, metric_key="peak_memory_values", metric_name="peak_memory_mb")
        plot_heatmap(section, [peak_mem_png, peak_mem_pdf], metric_key="peak_memory_values", metric_label=PEAK_MEMORY_LABEL)

        print("Wrote")
        print(f"* Inference time: {csv_path.name}, {png_path.name}")
        print(f"* GPU Idle (T_exposed): {t_exposed_csv.name}, {t_exposed_png.name}")
        print(f"* GPU Active: {gpu_active_csv.name}, {gpu_active_png.name}")
        print(f"* TKLQT: {tklqt_csv.name}, {tklqt_png.name}")
        print(f"* Peak Memory: {peak_mem_csv.name}, {peak_mem_png.name}")

    generate_comparative_html(sections, rows, summary_dir, max_tok_override)
    print(f"* Comparative HTML: comparative_report.html")


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