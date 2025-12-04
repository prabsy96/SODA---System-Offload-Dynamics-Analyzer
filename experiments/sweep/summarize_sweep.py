#!/usr/bin/env python3
"""
Summarize SODA sweep outputs into a single CSV.

Run from repo root:
    python -m experiments.summarize_sweep output/<run_dir>

Outputs sweep_summary.csv containing, for each model/compile/precision group:
  * Three metadata rows (model_name, compile_type, precision).
  * A pivot table with seq_len rows and batch_size columns.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SODA sweep outputs.")
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Path to sweep output directory (default: current directory).",
    )
    return parser.parse_args()


def safe_inference_time(report: Dict) -> Optional[float]:
    """Extract inference_time_ms from known report layouts."""
    perf = report.get("performance_metrics") or report.get("metrics") or {}
    value = perf.get("inference_time_ms")
    if value is not None:
        return float(value)
    # Some reports might only have torch_measured_inference_time_ms
    timing = perf.get("inference_time_breakdown") or perf.get("inference_time") or {}
    for key in ("torch_measured_inference_time_ms", "trace_calculated_inference_time_ms"):
        if key in timing:
            return float(timing[key])
    return None


def slugify(text: str) -> str:
    """Create a filesystem-friendly slug."""
    return (
        text.replace("/", "-")
        .replace(" ", "_")
        .replace(":", "-")
        .replace(".", "-")
    )


def parse_from_dirname(path: Path) -> Tuple[Optional[int], Optional[int], Optional[str], Optional[str]]:
    """
    Best-effort parse of batch_size, seq_len, compile_type, and precision from directory name.
    Expected pattern: <model>_<compile>_<precision>_bsX_slY (precision optional).
    """
    name = path.name
    if "_bs" not in name or "_sl" not in name:
        return None, None, None, None
    try:
        before_bs, rest = name.split("_bs", 1)
        bs_part, after_bs = rest.split("_sl", 1)
        batch_size = int(bs_part)
        # seq_len may be followed by more suffixes, keep leading digits
        seq_digits = []
        for ch in after_bs:
            if ch.isdigit():
                seq_digits.append(ch)
            else:
                break
        seq_len = int("".join(seq_digits)) if seq_digits else None

        # Attempt to parse compile_type and precision from before_bs tail
        tokens = before_bs.split("_")
        compile_type = None
        precision = None
        if len(tokens) >= 2:
            compile_type = tokens[-2]
            precision = tokens[-1]
        elif len(tokens) == 1:
            compile_type = tokens[0]
        return batch_size, seq_len, compile_type, precision
    except Exception:
        return None, None, None, None


def infer_model_name(dirname: str, compile_type: Optional[str], precision: Optional[str]) -> str:
    """Infer model slug from experiment directory name."""
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
            with report_path.open("r") as f:
                data = json.load(f)
        except Exception as exc:  # pragma: no cover - defensive
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

        row = {
            "model_name": model_name,
            "compile_type": compile_type,
            "precision": precision,
            "device": device,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "inference_time_ms": inference_time_ms,
            "report_path": str(report_path),
            "status": "ok",
        }
        rows.append(row)
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
        row = {
            "model_name": infer_model_name(exp_dir.name, ct_guess, prec_guess),
            "compile_type": ct_guess,
            "precision": prec_guess,
            "device": None,
            "batch_size": bs_guess,
            "seq_len": sl_guess,
            "inference_time_ms": None,
            "report_path": str(exp_dir / "report.json"),
            "status": "oom",
        }
        rows.append(row)
    return rows


def build_pivot_sections(rows: List[Dict]) -> List[Dict]:
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
        value_lookup = {
            (r.get("seq_len"), r.get("batch_size")): r.get("inference_time_ms")
            for r in items
            if isinstance(r.get("inference_time_ms"), (int, float))
        }
        status_lookup = {
            (r.get("seq_len"), r.get("batch_size")): r.get("status")
            for r in items
        }

        values: List[List[Optional[float]]] = []
        statuses: List[List[Optional[str]]] = []
        for sl in seq_lens:
            val_row: List[Optional[float]] = []
            status_row: List[Optional[str]] = []
            for bs in batch_sizes:
                key = (sl, bs)
                val_row.append(value_lookup.get(key))
                status_row.append(status_lookup.get(key))
            values.append(val_row)
            statuses.append(status_row)

        sections.append({
            "model_name": model,
            "compile_type": compile_type,
            "precision": precision,
            "batch_sizes": batch_sizes,
            "seq_lens": seq_lens,
            "values": values,
            "statuses": statuses,
        })
    return sections


def write_pivot_csv(sections: List[Dict], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        for idx, section in enumerate(sections):
            writer.writerow(["model_name", section["model_name"]])
            writer.writerow(["compile_type", section["compile_type"]])
            writer.writerow(["precision", section["precision"]])
            writer.writerow(["seq_len"] + [str(bs) for bs in section["batch_sizes"]])
            for row_idx, sl in enumerate(section["seq_lens"]):
                row = [sl]
                for col_idx, _ in enumerate(section["batch_sizes"]):
                    val = section["values"][row_idx][col_idx]
                    status = section["statuses"][row_idx][col_idx]
                    if status == "oom":
                        row.append("OOM")
                    elif val is None:
                        row.append("")
                    else:
                        row.append(f"{val}")
                writer.writerow(row)
            if idx != len(sections) - 1:
                writer.writerow([])


def main() -> int:
    args = parse_args()
    root = args.root.resolve()

    if not root.exists():
        print(f"Root path does not exist: {root}", file=sys.stderr)
        return 1

    rows = collect_reports(root)
    if not rows:
        print(f"No report.json files found under {root}", file=sys.stderr)
        return 1

    out_dir = root
    out_dir.mkdir(parents=True, exist_ok=True)
    sections = build_pivot_sections(rows)
    csv_path = out_dir / "sweep_summary.csv"
    write_pivot_csv(sections, csv_path)

    print(f"Wrote pivot CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
