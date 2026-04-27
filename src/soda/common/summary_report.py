"""
summary_report.py — Production-grade summary output for SODA.

Two public functions:
  render_main_analysis(results, args, output_dir)     — for SodaAnalyzer
  render_taxbreak_analysis(report, args, output_dir)  — for TaxBreakPipeline

Both print a compact Rich console summary and write report.html to output_dir.
Full expert tables are controlled by args.verbose in the calling code.
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

_console = Console()

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _bar(fraction: float, width: int = 20) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = round(fraction * width)
    return "█" * filled + "░" * (width - filled)


def _bar_html(fraction: float, width_px: int = 160) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = round(fraction * width_px)
    return (
        f'<div class="bar-bg" style="width:{width_px}px">'
        f'<div class="bar-fill" style="width:{filled}px"></div>'
        f'</div>'
    )


def _pct(part: float, total: float) -> float:
    return (part / total * 100.0) if total > 0 else 0.0


def _fmt_ms(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    if ms >= 1:
        return f"{ms:.2f} ms"
    return f"{ms * 1000:.1f} us"


def _fmt_mb(mb: float) -> str:
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"


def _gpu_name(num_gpus: int = 1) -> str:
    """Return a display name for the GPU(s) used in this run."""
    try:
        import torch
        if num_gpus <= 1:
            return torch.cuda.get_device_name(0)
        names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        # If all GPUs are the same model, show "N× <name>"
        if len(set(names)) == 1:
            return f"{num_gpus}× {names[0]}"
        return " + ".join(names)
    except Exception:
        return "N/A"


def _fmt_energy(mwh: float) -> str:
    if mwh >= 1.0:
        return f"{mwh:.3f} mWh"
    if mwh >= 0.001:
        return f"{mwh * 1000:.3f} uWh"
    return f"{mwh * 1e6:.3f} nWh"


def _fmt_carbon(mgco2eq: float) -> str:
    if mgco2eq >= 1000.0:
        return f"{mgco2eq / 1000:.3f} g CO2eq"
    if mgco2eq >= 1.0:
        return f"{mgco2eq:.3f} mg CO2eq"
    if mgco2eq >= 0.001:
        return f"{mgco2eq * 1000:.3f} ug CO2eq"
    return f"{mgco2eq * 1e6:.3f} ng CO2eq"


# ---------------------------------------------------------------------------
# Rich console table builders
# ---------------------------------------------------------------------------

def _build_overhead_table(
    components: List[Dict[str, Any]], total_ms: float, title: str = "Host Overhead Breakdown"
) -> Table:
    tbl = Table(
        title=title, box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False,
        title_style="bold", expand=False,
    )
    tbl.add_column("Component", style="cyan", no_wrap=True, min_width=16)
    tbl.add_column("Time", justify="right", min_width=10)
    tbl.add_column("%", justify="right", min_width=6)
    tbl.add_column("Bar", no_wrap=True, min_width=20)
    for c in components:
        pct = _pct(c["ms"], total_ms)
        tbl.add_row(c["name"], _fmt_ms(c["ms"]), f"{pct:.0f}%", _bar(pct / 100.0))
    return tbl


def _build_kernel_table(kernels: List[Dict[str, Any]], title: str = "Top Kernels") -> Table:
    tbl = Table(
        title=title, box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False,
        title_style="bold", expand=False,
    )
    tbl.add_column("#", style="dim", no_wrap=True, min_width=3)
    tbl.add_column("Kernel", no_wrap=True, min_width=30)
    tbl.add_column("Count", justify="right", min_width=6)
    tbl.add_column("Total", justify="right", min_width=10)
    for i, k in enumerate(kernels, 1):
        tbl.add_row(
            str(i), str(k.get("name", ""))[:40],
            str(k.get("frequency", "")), _fmt_ms(k.get("total_duration_ms", 0.0)),
        )
    return tbl


def _build_speed_table(metrics: Dict[str, Any]) -> Table:
    tbl = Table(
        title="Inference", box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False,
        title_style="bold", expand=False, show_header=False,
    )
    tbl.add_column("Metric", style="cyan", no_wrap=True, min_width=22)
    tbl.add_column("Value", justify="right", min_width=16)
    tbl.add_column("", min_width=6)
    inf_ms = metrics.get("inference_time_ms")
    if inf_ms is not None:
        tbl.add_row("Inference Time", _fmt_ms(inf_ms), "")
    td = metrics.get("inference_throughput", {})
    tpot = td.get("tpot_ms")
    throughput = td.get("throughput_tok_s")
    interactivity = td.get("interactivity_tok_s")
    is_ttft = td.get("is_ttft_run", False)
    output_tokens = td.get("output_tokens", 1)
    if tpot is not None:
        tbl.add_row("TTFT" if is_ttft else "TPOT", f"{tpot:.2f} ms", "")
    if throughput is not None:
        tbl.add_row("Throughput", f"{throughput:,.0f} tok/s", "")
    if interactivity is not None:
        tbl.add_row("Interactivity", f"{interactivity:,.0f} tok/s/user", "")
    if output_tokens is not None:
        tbl.add_row("Output Tokens", str(output_tokens), "[dim]TTFT[/dim]" if is_ttft else "")
    return tbl


def _build_gpu_table(metrics: Dict[str, Any], args=None) -> Table:
    tbl = Table(
        title="GPU", box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False,
        title_style="bold", expand=False, show_header=False,
    )
    tbl.add_column("Metric", style="cyan", no_wrap=True, min_width=22)
    tbl.add_column("Value", justify="right", min_width=16)
    tbl.add_column("", no_wrap=True, min_width=6)
    num_gpus = getattr(args, "num_gpus", 1) if args is not None else 1
    multi = num_gpus > 1
    if multi:
        tbl.add_row("GPUs", str(num_gpus), "model parallel (device_map=balanced)")
    # Utilization and busy/idle time are per-GPU averages for multi-GPU runs
    avg_suffix = " (avg/GPU)" if multi else ""
    gpu_util = metrics.get("gpu_utilization_percent")
    if gpu_util is not None:
        tbl.add_row("Utilization" + avg_suffix, f"{gpu_util:.1f}%", "")
    for key, label in [("gpu_busy_time_ms", "Busy Time"), ("gpu_idle_time_ms", "Idle Time")]:
        v = metrics.get(key)
        if v is not None:
            tbl.add_row(label + avg_suffix, _fmt_ms(v), "")
    # Per-device breakdown (verbose mode or always for multi-GPU?)
    per_device = metrics.get("per_device_gpu_metrics", {})
    if multi and per_device:
        for dev_id in sorted(per_device):
            d = per_device[dev_id]
            tbl.add_row(
                f"  GPU {dev_id} util",
                f"{d.get('utilization_pct', 0.0):.1f}%",
                f"busy {_fmt_ms(d.get('busy_us', 0.0) / 1000.0)}",
            )
    n_kernels = metrics.get("num_total_kernels")
    if n_kernels is not None:
        tbl.add_row("Kernels", f"{n_kernels:,}", "")
    streams = metrics.get("active_streams")
    if streams is not None:
        tbl.add_row("Active Streams", str(streams), "")
    return tbl


def _build_memory_table(metrics: Dict[str, Any]) -> Table:
    tbl = Table(
        title="Memory", box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False,
        title_style="bold", expand=False, show_header=False,
    )
    tbl.add_column("Metric", style="cyan", no_wrap=True, min_width=22)
    tbl.add_column("Value", justify="right", min_width=16)
    tbl.add_column("", no_wrap=True, min_width=6)
    mem = metrics.get("memory_metrics", {})
    if not mem:
        return tbl
    for key, label, fmt in [
        ("model_memory_mb", "Model Weights", "mb"), ("peak_memory_allocated_mb", "Peak Allocated", "mb"),
        ("peak_memory_reserved_mb", "Peak Reserved", "mb"), ("memory_delta_mb", "Inference Delta", "mb"),
        ("kv_cache_mb", "KV Cache", "mb"),
        ("num_memcpy_memset_ops", "Memcpy/Memset Ops", "int"),
        ("total_memcpy_memset_time_ms", "Memcpy/Memset Time", "ms"),
    ]:
        v = mem.get(key)
        if v is not None and v != 0:
            if fmt == "mb":
                tbl.add_row(label, _fmt_mb(abs(v)), "")
            elif fmt == "ms":
                tbl.add_row(label, _fmt_ms(v), "")
            else:
                tbl.add_row(label, f"{v:,}", "")
    return tbl


def _build_carbon_table(carbon: Dict[str, Any]) -> Table:
    tbl = Table(
        title="Carbon Footprint", box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False,
        title_style="bold", expand=False, show_header=False,
    )
    tbl.add_column("Metric", style="cyan", no_wrap=True, min_width=22)
    tbl.add_column("Value", justify="right", min_width=16)
    for key, label, fmt in [
        ("inference_energy_mwh", "Energy/Inference", "energy"),
        ("inference_carbon_mgco2eq", "Carbon/Inference", "carbon"),
        ("carbon_per_token_mgco2eq", "Carbon/Token", "carbon"),
        ("carbon_intensity_g_kwh", "Grid Intensity", "intensity"),
        ("gpu_tdp_w", "GPU TDP", "tdp"),
    ]:
        v = carbon.get(key)
        if v is not None:
            if fmt == "energy":
                tbl.add_row(label, _fmt_energy(v))
            elif fmt == "carbon":
                tbl.add_row(label, _fmt_carbon(v))
            elif fmt == "intensity":
                tbl.add_row(label, f"{v:.0f} gCO2/kWh")
            elif fmt == "tdp":
                tbl.add_row(label, f"{v:.0f} W")
    # Measured power rows (only present when --power-sample was used)
    measured = carbon.get("measured_power_w")
    if measured is not None:
        tbl.add_row("Measured Mean Power", f"{measured:.1f} W [measured]")
        peak = carbon.get("peak_power_w")
        if peak is not None:
            tbl.add_row("Measured Peak Power", f"{peak:.1f} W [measured]")
        n = carbon.get("power_sample_count")
        backend = carbon.get("power_backend", "")
        if n is not None:
            tbl.add_row("Power Samples", f"{n} ({backend})")
    return tbl


def _build_tklqt_table(tklqt: Dict[str, Any]) -> Table:
    tbl = Table(
        title="TKLQT Metrics", box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False,
        title_style="bold", expand=False, show_header=False,
    )
    tbl.add_column("Metric", style="cyan", no_wrap=True, min_width=22)
    tbl.add_column("Value", justify="right", min_width=16)
    for k, v in tklqt.items():
        tbl.add_row(str(k), f"{v:.4f}" if isinstance(v, float) else str(v))
    return tbl


# ---------------------------------------------------------------------------
# HTML report generator
# ---------------------------------------------------------------------------

_HTML_CSS = """\
:root {
  --bg: #0d1117; --surface: #161b22; --border: #30363d;
  --text: #e6edf3; --text-dim: #8b949e; --cyan: #58a6ff;
  --green: #3fb950; --yellow: #d29922; --red: #f85149;
  --bar-bg: #21262d; --bar-fill: #58a6ff;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Consolas', monospace;
  background: var(--bg); color: var(--text);
  padding: 24px 32px; max-width: 900px; margin: 0 auto;
  font-size: 13px; line-height: 1.5;
}
.header {
  border: 1px solid var(--border); border-radius: 8px;
  padding: 16px 20px; margin-bottom: 20px; background: var(--surface);
}
.header h1 { font-size: 14px; font-weight: 600; color: var(--cyan); margin-bottom: 4px; }
.header .sub { color: var(--text-dim); font-size: 12px; }
.section { margin-bottom: 16px; }
.section h2 {
  font-size: 12px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.05em; color: var(--text-dim);
  border-bottom: 1px solid var(--border); padding-bottom: 4px; margin-bottom: 8px;
}
table { border-collapse: collapse; width: 100%; }
th {
  text-align: left; font-size: 11px; font-weight: 600;
  color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.04em;
  padding: 4px 12px 4px 0; border-bottom: 2px solid var(--border);
}
td { padding: 3px 12px 3px 0; border-bottom: 1px solid var(--border); vertical-align: middle; }
td.label { color: var(--cyan); white-space: nowrap; }
td.value { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.dim { color: var(--text-dim); }
.bar-bg {
  height: 10px; background: var(--bar-bg); border-radius: 3px;
  display: inline-block; vertical-align: middle;
}
.bar-fill { height: 10px; background: var(--bar-fill); border-radius: 3px; }
.kernel-name {
  max-width: 340px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  display: inline-block; vertical-align: bottom;
}
.hdbi-badge {
  display: inline-block; padding: 2px 8px; border-radius: 4px;
  font-size: 12px; font-weight: 600;
}
.hdbi-host { background: rgba(248,81,73,0.15); color: var(--red); }
.hdbi-balanced { background: rgba(210,153,34,0.15); color: var(--yellow); }
.hdbi-device { background: rgba(63,185,80,0.15); color: var(--green); }
footer { margin-top: 24px; padding-top: 12px; border-top: 1px solid var(--border); color: var(--text-dim); font-size: 11px; }
"""


def _html_kv_table(rows: List[Dict[str, Any]], title: str) -> str:
    """Render a key-value section as HTML table. Each row: {label, value}."""
    lines = [f'<div class="section"><h2>{title}</h2><table>']
    for r in rows:
        lines.append(
            f'<tr><td class="label">{r["label"]}</td>'
            f'<td class="value">{r["value"]}</td></tr>'
        )
    lines.append('</table></div>')
    return "\n".join(lines)


def _html_col_table(
    headers: List[str], rows: List[List[str]], title: str,
    note: Optional[str] = None,
) -> str:
    """Render a multi-column table section."""
    lines = [f'<div class="section"><h2>{title}</h2><table><tr>']
    for h in headers:
        align = 'style="text-align:right"' if h in ("Time", "%", "Count", "Total", "Launch Tax", "L1 Hit") else ""
        lines.append(f"<th {align}>{h}</th>")
    lines.append("</tr>")
    for row in rows:
        lines.append("<tr>")
        for i, cell in enumerate(row):
            cls = "value" if i > 0 else "label"
            lines.append(f'<td class="{cls}">{cell}</td>')
        lines.append("</tr>")
    lines.append("</table>")
    if note:
        lines.append(f'<p class="dim" style="margin-top:6px;font-size:11px">{note}</p>')
    lines.append("</div>")
    return "\n".join(lines)


def _to_html_main(
    model: str, gpu: str, precision: str, batch: int, seq_len: int,
    date: str, metrics: dict, top_kernels: list,
    carbon: Optional[dict] = None, hdbi_info: Optional[dict] = None,
) -> str:
    parts: List[str] = []

    # Header
    parts.append(
        f'<div class="header">'
        f'<h1>SODA Report</h1>'
        f'<div class="sub">{model} &middot; {gpu} &middot; {precision} '
        f'&middot; batch={batch} &middot; seq={seq_len} &middot; {date}</div>'
        f'</div>'
    )

    # Inference
    inf_rows: List[Dict[str, Any]] = []
    inf_ms = metrics.get("inference_time_ms")
    if inf_ms is not None:
        inf_rows.append({"label": "Inference Time", "value": _fmt_ms(inf_ms)})
    td = metrics.get("inference_throughput", {})
    tpot = td.get("tpot_ms")
    is_ttft = td.get("is_ttft_run", False)
    if tpot is not None:
        inf_rows.append({"label": "TTFT" if is_ttft else "TPOT", "value": f"{tpot:.2f} ms"})
    throughput = td.get("throughput_tok_s")
    if throughput is not None:
        inf_rows.append({"label": "Throughput", "value": f"{throughput:,.0f} tok/s"})
    interactivity = td.get("interactivity_tok_s")
    if interactivity is not None:
        inf_rows.append({"label": "Interactivity", "value": f"{interactivity:,.0f} tok/s/user"})
    output_tokens = td.get("output_tokens")
    if output_tokens is not None:
        suffix = ' <span class="dim">(TTFT)</span>' if is_ttft else ""
        inf_rows.append({"label": "Output Tokens", "value": f"{output_tokens}{suffix}"})
    if inf_rows:
        parts.append(_html_kv_table(inf_rows, "Inference"))

    # GPU
    gpu_rows: List[Dict[str, Any]] = []
    gpu_util = metrics.get("gpu_utilization_percent")
    if gpu_util is not None:
        gpu_rows.append({"label": "Utilization", "value": f"{gpu_util:.1f}%"})
    for key, label in [("gpu_busy_time_ms", "Busy Time"), ("gpu_idle_time_ms", "Idle Time")]:
        v = metrics.get(key)
        if v is not None:
            gpu_rows.append({"label": label, "value": _fmt_ms(v)})
    n_kernels = metrics.get("num_total_kernels")
    if n_kernels is not None:
        gpu_rows.append({"label": "Kernels", "value": f"{n_kernels:,}"})
    streams = metrics.get("active_streams")
    if streams is not None:
        gpu_rows.append({"label": "Active Streams", "value": str(streams)})
    if gpu_rows:
        parts.append(_html_kv_table(gpu_rows, "GPU"))

    # Memory
    mem = metrics.get("memory_metrics", {})
    if mem:
        mem_rows: List[Dict[str, Any]] = []
        for key, label, fmt in [
            ("model_memory_mb", "Model Weights", "mb"),
            ("peak_memory_allocated_mb", "Peak Allocated", "mb"),
            ("peak_memory_reserved_mb", "Peak Reserved", "mb"),
            ("memory_delta_mb", "Inference Delta", "mb"),
            ("num_memcpy_memset_ops", "Memcpy/Memset Ops", "int"),
            ("total_memcpy_memset_time_ms", "Memcpy/Memset Time", "ms"),
        ]:
            v = mem.get(key)
            if v is not None and v != 0:
                if fmt == "mb":
                    mem_rows.append({"label": label, "value": _fmt_mb(abs(v))})
                elif fmt == "ms":
                    mem_rows.append({"label": label, "value": _fmt_ms(v)})
                else:
                    mem_rows.append({"label": label, "value": f"{v:,}"})
        if mem_rows:
            parts.append(_html_kv_table(mem_rows, "Memory"))

    # TKLQT
    tklqt = metrics.get("tklqt")
    if tklqt:
        tklqt_rows = [{"label": str(k), "value": f"{v:.4f}" if isinstance(v, float) else str(v)} for k, v in tklqt.items()]
        parts.append(_html_kv_table(tklqt_rows, "TKLQT Metrics"))

    # HDBI (approximate)
    if hdbi_info:
        val = hdbi_info.get("value", 0.0)
        cls_name = hdbi_info.get("classification", "balanced")
        css_cls = {"host-bound": "hdbi-host", "balanced": "hdbi-balanced", "device-bound": "hdbi-device"}.get(cls_name, "hdbi-balanced")
        t_sys = hdbi_info.get("t_sys_us", 0.0)
        source = hdbi_info.get("source", "")
        parts.append(
            f'<div class="section"><h2>HDBI (approximate)</h2>'
            f'<p><strong>{val:.3f}</strong> '
            f'<span class="hdbi-badge {css_cls}">{cls_name}</span></p>'
            f'<p class="dim">T_sys = {t_sys:.2f} us ({source})</p>'
            f'</div>'
        )

    # Top kernels
    if top_kernels:
        k_rows: List[List[str]] = []
        for i, k in enumerate(top_kernels, 1):
            name = str(k.get("name", ""))
            display = f'<span class="kernel-name" title="{name}">{name[:42]}</span>'
            k_rows.append([
                str(i), display, str(k.get("frequency", "")),
                _fmt_ms(k.get("total_duration_ms", 0.0)),
            ])
        parts.append(_html_col_table(["#", "Kernel", "Count", "Total"], k_rows, "Top Kernels by Duration"))

    # Carbon
    if carbon:
        c_rows: List[Dict[str, Any]] = []
        for key, label, fmt in [
            ("inference_energy_mwh", "Energy/Inference", "energy"),
            ("inference_carbon_mgco2eq", "Carbon/Inference", "carbon"),
            ("carbon_per_token_mgco2eq", "Carbon/Token", "carbon"),
            ("carbon_intensity_g_kwh", "Grid Intensity", "intensity"),
            ("gpu_tdp_w", "GPU TDP", "tdp"),
        ]:
            v = carbon.get(key)
            if v is not None:
                if fmt == "energy":
                    c_rows.append({"label": label, "value": _fmt_energy(v)})
                elif fmt == "carbon":
                    c_rows.append({"label": label, "value": _fmt_carbon(v)})
                elif fmt == "intensity":
                    c_rows.append({"label": label, "value": f"{v:.0f} gCO2/kWh"})
                elif fmt == "tdp":
                    c_rows.append({"label": label, "value": f"{v:.0f} W"})
        if c_rows:
            parts.append(_html_kv_table(c_rows, "Carbon Footprint"))

    # Footer
    parts.append(
        f'<footer>Generated by SODA &middot; System Offload Dynamics Analyzer '
        f'&middot; Carnegie Mellon University &middot; {date}</footer>'
    )

    body = "\n".join(parts)
    return (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>SODA Report — {model}</title>'
        f'<style>{_HTML_CSS}</style></head>'
        f'<body>{body}</body></html>'
    )


def _to_html_taxbreak(
    model: str, gpu: str, precision: str, date: str,
    floor: dict, overhead: dict, top_kernels: list,
    profiled: int, total: int, hdbi: Optional[dict] = None,
) -> str:
    parts: List[str] = []

    parts.append(
        f'<div class="header">'
        f'<h1>SODA TaxBreak Report</h1>'
        f'<div class="sub">{model} &middot; {gpu} &middot; {precision} &middot; {date}</div>'
        f'</div>'
    )

    # System floor
    floor_rows = [{"label": k, "value": v} for k, v in floor.items()]
    parts.append(_html_kv_table(floor_rows, "System Floor"))

    # Overhead
    total_ms = overhead.get("total_ms", 0.0)
    if total_ms > 0:
        oh_rows: List[List[str]] = []
        for c in overhead.get("components", []):
            pct = _pct(c["ms"], total_ms)
            oh_rows.append([c["name"], _fmt_ms(c["ms"]), f"{pct:.0f}%", _bar_html(pct / 100.0)])
        parts.append(_html_col_table(
            ["Component", "Time", "%", ""],
            oh_rows,
            f"Structural Overhead ({_fmt_ms(total_ms)} total)",
        ))

    # HDBI
    if hdbi:
        val = float(hdbi.get("value", "0").replace(",", ""))
        if val > 1.0:
            cls_name, css_cls = "host-bound", "hdbi-host"
        elif val > 0.5:
            cls_name, css_cls = "balanced", "hdbi-balanced"
        else:
            cls_name, css_cls = "device-bound", "hdbi-device"
        parts.append(
            f'<div class="section"><h2>HDBI</h2>'
            f'<p><strong>{val:.3f}</strong> '
            f'<span class="hdbi-badge {css_cls}">{cls_name}</span></p>'
            f'</div>'
        )

    # Top kernels
    if top_kernels:
        k_rows: List[List[str]] = []
        for k in top_kernels:
            l1 = k.get("l1_pct")
            l1_str = f"{l1:.0f}%" if l1 is not None else "--"
            name = str(k.get("kernel_name", ""))
            display = f'<span class="kernel-name" title="{name}">{name[:32]}</span>'
            k_rows.append([
                str(k.get("id", "")), str(k.get("aten_op", ""))[:20], display,
                f"{k.get('launch_tax_us', 0.0):.1f} us", l1_str,
            ])
        parts.append(_html_col_table(
            ["ID", "ATen Op", "Kernel", "Launch Tax", "L1 Hit"],
            k_rows, f"Top {len(k_rows)} Kernels by Isolation Launch Tax",
        ))

    parts.append(f'<p class="dim" style="margin-top:8px">Coverage: {profiled}/{total} unique kernels profiled</p>')
    parts.append(
        f'<footer>Generated by SODA &middot; System Offload Dynamics Analyzer '
        f'&middot; Carnegie Mellon University &middot; {date}</footer>'
    )

    body = "\n".join(parts)
    return (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>SODA TaxBreak Report — {model}</title>'
        f'<style>{_HTML_CSS}</style></head>'
        f'<body>{body}</body></html>'
    )


# ---------------------------------------------------------------------------
# File writer
# ---------------------------------------------------------------------------

def _write_report(html: str, output_dir: Path, filename: str = "report.html") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / filename
    dest.write_text(html, encoding="utf-8")
    _console.print(f"[dim]Report written to {dest}[/dim]")


# ---------------------------------------------------------------------------
# Main analysis summary
# ---------------------------------------------------------------------------

def render_main_analysis(results: dict, args, output_dir: Path) -> str:
    """Render console summary for SodaAnalyzer results. Returns HTML string."""
    metrics = results.get("metrics", {})
    model = getattr(args, "model", "unknown")
    precision = getattr(args, "precision", "fp16")
    batch = getattr(args, "batch_size", 1)
    seq_len = getattr(args, "seq_len", 512)
    gpu = _gpu_name(getattr(args, "num_gpus", 1))
    date = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Console output ──
    header = f"[bold]{model}[/bold] | {gpu} | {precision} | batch={batch} seq={seq_len}"
    _console.print(Panel(header, title="SODA Report", box=box.ROUNDED, expand=False))

    tbl_speed = _build_speed_table(metrics)
    if tbl_speed.row_count > 0:
        _console.print(tbl_speed)

    tbl_gpu = _build_gpu_table(metrics, args=args)
    if tbl_gpu.row_count > 0:
        _console.print(tbl_gpu)

    tbl_mem = _build_memory_table(metrics)
    if tbl_mem.row_count > 0:
        _console.print(tbl_mem)

    tklqt = metrics.get("tklqt")
    if tklqt:
        _console.print(_build_tklqt_table(tklqt))

    # HDBI (approximate) — display if available in results
    hdbi_info = results.get("hdbi_approx")
    if hdbi_info:
        val = hdbi_info.get("value", 0.0)
        cls_name = hdbi_info.get("classification", "balanced")
        t_sys = hdbi_info.get("t_sys_us", 0.0)
        source = hdbi_info.get("source", "")
        color_map = {"host-bound": "red", "balanced": "yellow", "device-bound": "green"}
        color = color_map.get(cls_name, "yellow")
        _console.print(
            f"  [bold]HDBI (approx):[/bold] {val:.3f}  [{color}]{cls_name}[/{color}]"
            f"  [dim](T_sys={t_sys:.2f} us via {source})[/dim]"
        )

    # Top kernels
    top_k_raw = results.get("top_k_kernels", {})
    top_by_dur = top_k_raw.get("by_duration", [])
    top_by_freq = top_k_raw.get("by_frequency", [])
    source_list = top_by_dur if top_by_dur else top_by_freq
    top_k_entries: List[Dict[str, Any]] = []
    for name, data in source_list[:5]:
        from soda.common import utils as _u
        top_k_entries.append({
            "name": name,
            "frequency": int(data.get("frequency", 0)),
            "total_duration_ms": _u.us_to_ms(data.get("duration", 0.0)),
        })
    if top_k_entries:
        _console.print(_build_kernel_table(top_k_entries, title="Top Kernels by Duration"))

    carbon = metrics.get("carbon_footprint")
    if carbon:
        _console.print(_build_carbon_table(carbon))

    _console.print()

    # ── HTML report ──
    html = _to_html_main(
        model=model, gpu=gpu, precision=precision, batch=batch,
        seq_len=seq_len, date=date, metrics=metrics,
        top_kernels=top_k_entries, carbon=carbon, hdbi_info=hdbi_info,
    )
    _write_report(html, output_dir)
    return html


# ---------------------------------------------------------------------------
# Enhanced TaxBreak summary
# ---------------------------------------------------------------------------

def render_taxbreak_analysis(report: dict, args, output_dir: Path) -> str:
    """Render console summary for TaxBreak pipeline results. Returns HTML string."""
    model = getattr(args, "model", "unknown")
    precision = getattr(args, "precision", "fp16")
    gpu = _gpu_name(getattr(args, "num_gpus", 1))
    date = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Console output ──
    header = f"[bold]{model}[/bold] | {gpu} | {precision} | TaxBreak Enhanced"
    _console.print(Panel(header, title="SODA TaxBreak Report", box=box.ROUNDED, expand=False))

    floor = report.get("system_floor", {})
    floor_avg = floor.get("avg_us", 0.0)
    floor_std = floor.get("std_us", 0.0)
    floor_md = {"avg": f"{floor_avg:.2f} us", "std": f"{floor_std:.2f} us"}

    floor_tbl = Table(
        title="System Floor", box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False,
        title_style="bold", expand=False, show_header=False,
    )
    floor_tbl.add_column("Metric", style="cyan", no_wrap=True, min_width=22)
    floor_tbl.add_column("Value", justify="right", min_width=16)
    floor_tbl.add_row("Floor (avg)", f"{floor_avg:.2f} us")
    floor_tbl.add_row("Floor (std)", f"{floor_std:.2f} us")
    _console.print(floor_tbl)

    agg = report.get("aggregate", {})
    breakdown = agg.get("breakdown_mean", {})
    t_structural_ms = agg.get("T_host_observed_ms", 0.0)
    py_ms = breakdown.get("delta_FT_py_ms", 0.0)
    ft_disp_ms = breakdown.get("delta_FT_dispatch_ms", 0.0)
    delta_ct_ms = breakdown.get("delta_CT_ms", 0.0)
    launch_ms = breakdown.get("T_launch_raw_ms", 0.0)
    components = [
        {"name": "Python layer (\u0394FT_py)", "ms": py_ms},
        {"name": "ATen dispatch (\u0394FT_disp)", "ms": ft_disp_ms},
        {"name": "CUDA lib overhead (\u0394CT)", "ms": delta_ct_ms},
        {"name": "Kernel launch (\u0394KT)", "ms": launch_ms},
    ]
    if t_structural_ms > 0:
        _console.print(_build_overhead_table(
            components, t_structural_ms,
            title=f"Structural Overhead ({_fmt_ms(t_structural_ms)} total)"
        ))

    hdbi_data = agg.get("hdbi")
    hdbi_md = None
    if hdbi_data:
        hdbi_val = hdbi_data if isinstance(hdbi_data, (int, float)) else hdbi_data.get("value", 0.0)
        hdbi_md = {"value": f"{hdbi_val:.3f}"}
        label = "[bold]HDBI:[/bold]"
        if hdbi_val > 1.0:
            _console.print(f"  {label} {hdbi_val:.3f}  [red]host-bound[/red]")
        elif hdbi_val > 0.5:
            _console.print(f"  {label} {hdbi_val:.3f}  [yellow]balanced[/yellow]")
        else:
            _console.print(f"  {label} {hdbi_val:.3f}  [green]device-bound[/green]")

    per_kernel = report.get("per_kernel", [])
    top5 = sorted(
        per_kernel,
        key=lambda k: k.get("taxes", {}).get("launch_tax_us", {}).get("avg_us", 0),
        reverse=True,
    )[:5]
    top_k_md: List[Dict[str, Any]] = []
    if top5:
        k_entries: List[Dict[str, Any]] = []
        for k in top5:
            entry = {
                "id": k.get("id", ""),
                "aten_op": k.get("aten_op", ""),
                "kernel_name": k.get("kernel_name", ""),
                "launch_tax_us": k.get("taxes", {}).get("launch_tax_us", {}).get("avg_us", 0.0),
                "l1_pct": (k.get("ncu") or {}).get("l1_hit_rate_pct"),
            }
            k_entries.append(entry)
            top_k_md.append(entry)

        ktbl = Table(
            title=f"Top {len(k_entries)} Kernels by Isolation Launch Tax",
            box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False,
            title_style="bold", expand=False,
        )
        ktbl.add_column("ID", style="dim", no_wrap=True, min_width=5)
        ktbl.add_column("ATen Op", style="cyan", no_wrap=True, min_width=18)
        ktbl.add_column("Kernel", no_wrap=True, min_width=26)
        ktbl.add_column("Launch Tax", justify="right", min_width=10)
        ktbl.add_column("L1 Hit", justify="right", min_width=7)
        for entry in k_entries:
            l1 = entry.get("l1_pct")
            l1_str = f"{l1:.0f}%" if l1 is not None else "--"
            ktbl.add_row(
                str(entry.get("id", "")), str(entry.get("aten_op", ""))[:20],
                str(entry.get("kernel_name", ""))[:30],
                f"{entry.get('launch_tax_us', 0.0):.1f} us", l1_str,
            )
        _console.print(ktbl)

    summary = report.get("summary", {})
    total_k = summary.get("total_unique_kernels", 0)
    nsys_k = summary.get("kernels_with_nsys", 0)
    ncu_k = summary.get("kernels_with_ncu", 0)
    power_k = summary.get("kernels_with_power_replay", 0)
    _console.print(f"  [dim]Coverage: {nsys_k}/{total_k} nsys, {ncu_k}/{total_k} ncu"
                   + (f", {power_k}/{total_k} power" if power_k else "") + "[/dim]")

    # Power replay summary (only when --power-replay was used)
    if power_k:
        idle_w = summary.get("idle_baseline_w", 0.0)
        total_nj = summary.get("total_measured_energy_nj", 0.0)
        infer_pwr = summary.get("inference_power", {})
        recon_w = infer_pwr.get("reconstructed_inference_power_w")
        total_energy_uj = infer_pwr.get("total_inference_energy_uj")

        pw_tbl = Table(
            title="⚡ Per-Kernel Power Replay",
            box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False,
            title_style="bold", expand=False, show_header=False,
        )
        pw_tbl.add_column("Metric", style="cyan", no_wrap=True, min_width=30)
        pw_tbl.add_column("Value", justify="right", min_width=18)
        if recon_w is not None:
            pw_tbl.add_row(
                "Inference power (reconstructed)",
                f"[bold]{recon_w:.2f} W[/bold]",
            )
        # Validation: ground truth from Stage 1 energy counter
        v = summary.get("inference_power", {})
        # validation is in the top-level report, not nested in inference_power
        v_block = None
        # Check the report-level validation field (set by _write_power_report via inference_power dict)
        for _vkey in ("validation",):
            _vdata = report.get(_vkey)
            if _vdata and _vdata.get("measured_inference_power_w"):
                v_block = _vdata
                break
        if v_block is None:
            # Also check inside inference_power (where it's surfaced in summary)
            _vdata = infer_pwr.get("validation") or summary.get("validation")
            if _vdata and _vdata.get("measured_inference_power_w"):
                v_block = _vdata
        if v_block:
            gt_w = v_block["measured_inference_power_w"]
            err_pct = v_block.get("error_pct", 0.0)
            err_color = "green" if abs(err_pct) < 10 else ("yellow" if abs(err_pct) < 25 else "red")
            pw_tbl.add_row(
                "Inference power (measured)",
                f"[bold]{gt_w:.2f} W[/bold]",
            )
            sign = "+" if err_pct >= 0 else ""
            pw_tbl.add_row(
                "  Validation error",
                f"[{err_color}]{sign}{err_pct:.1f}%[/{err_color}]",
            )
        pw_tbl.add_row("  Idle baseline", f"{idle_w:.2f} W")
        net_w = infer_pwr.get("net_kernel_power_w")
        if net_w is not None:
            pw_tbl.add_row("  Net kernel power (avg)", f"{net_w:.2f} W")
        if total_energy_uj is not None:
            pw_tbl.add_row(
                "Inference energy (kernel-active)",
                f"{total_energy_uj / 1e3:.4f} mJ",
            )
        pw_tbl.add_row("Kernels profiled", str(power_k))

        # Top-3 kernels by net power (read from per_kernel list)
        pw_kernels = [
            k for k in report.get("per_kernel", [])
            if k.get("power_replay") is not None
        ]
        pw_kernels.sort(
            key=lambda k: k["power_replay"].get("net_power_w", 0.0), reverse=True
        )
        if pw_kernels:
            pw_tbl.add_row("Top kernels by net power", "")
        for k in pw_kernels[:3]:
            pr = k["power_replay"]
            label = (k.get("aten_op") or k.get("kernel_name") or k.get("id", ""))[:28]
            reliable = "" if pr.get("is_reliable", True) else " [yellow]![/yellow]"
            pw_tbl.add_row(
                f"  {label}",
                f"{pr.get('net_power_w', 0.0):.2f} W{reliable}",
            )
        _console.print(pw_tbl)
        _console.print(f"  [dim]Full breakdown: taxbreak/power_report.json[/dim]")

    _console.print()

    # ── HTML report ──
    overhead_dict = {"total_ms": t_structural_ms, "components": components}
    html = _to_html_taxbreak(
        model=model, gpu=gpu, precision=precision, date=date,
        floor=floor_md, overhead=overhead_dict, top_kernels=top_k_md,
        profiled=nsys_k, total=total_k, hdbi=hdbi_md,
    )
    _write_report(html, output_dir, filename="report.html")
    return html