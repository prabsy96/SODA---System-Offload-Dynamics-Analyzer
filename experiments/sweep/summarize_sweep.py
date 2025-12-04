#!/usr/bin/env python3
"""
Summarize a sweep directory by producing table pivots and heatmaps.

Usage:
    python -m experiments.sweep.summarize_sweep output/<sweep_dir>
    python experiments/sweep/summarize_sweep.py output/<sweep_dir>

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


def plot_heatmap(section: Dict, out_path: Path) -> None:
    bs = section["batch_sizes"]
    sl = section["seq_lens"]
    data = np.array(
        [
            [np.nan if v is None else v for v in row]
            for row in section["values"]
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(max(4, len(bs) * 0.6), max(4, len(sl) * 0.6)))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(bs)))
    ax.set_xticklabels(bs)
    ax.set_yticks(range(len(sl)))
    ax.set_yticklabels(sl)
    ax.set_xlabel("batch_size")
    ax.set_ylabel("seq_len")
    ax.set_title(f"{section['model_name']} | {section['compile_type']} | {section['precision']}")

    cbar = fig.colorbar(im, ax=ax, label="inference_time_ms")

    for i, seq in enumerate(sl):
        for j, batch in enumerate(bs):
            status = section["statuses"][i][j]
            value = section["values"][i][j]
            if status == "oom":
                ax.text(j, i, "OOM", ha="center", va="center", color="red", fontsize=8, fontweight="bold")
            elif value is not None and not np.isnan(value):
                ax.text(j, i, f"{value:.0f}", ha="center", va="center", color="white", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
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
        write_pivot(section, csv_path)
        plot_heatmap(section, png_path)
        print("Wrote")
        print(f"* Summary to {csv_path}")
        print(f"* Heatmap to {png_path}")


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
