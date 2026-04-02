"""
GPU Roofline Analysis and Throughput–Interactivity Pareto Plot

Provides:
  - GPU hardware specs lookup (get_gpu_specs)
  - GEMM FLOPs derivation (compute_gemm_flops)
  - Roofline data computation (compute_roofline_data)
  - Improved roofline plot with iso-efficiency contours, region shading,
    frequency-scaled markers, and smart annotations (generate_roofline_plot)
  - Pareto frontier computation (compute_pareto_frontier)
  - Throughput–Interactivity Pareto plot with iso-batch reference lines
    (generate_pareto_plot)
"""

import re
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------
# GPU hardware specifications
# peak_tflops_fp16: FP16 tensor core peak (TFLOP/s)
# peak_bw_tb_s: HBM peak bandwidth (TB/s)
# ---------------------------------------------------------------

GPU_SPECS: Dict[str, Dict[str, float]] = {
    "H100 SXM": {"peak_tflops_fp16": 989.5, "peak_bw_tb_s": 3.35},
    "H100 PCIe": {"peak_tflops_fp16": 756.5, "peak_bw_tb_s": 2.0},
    "H200 SXM": {"peak_tflops_fp16": 989.5, "peak_bw_tb_s": 4.8},
    "H200 NVL": {"peak_tflops_fp16": 989.5, "peak_bw_tb_s": 3.35},  # PCIe form-factor, 96 GB HBM3e
    "A100 SXM": {"peak_tflops_fp16": 312.0, "peak_bw_tb_s": 2.0},
    "A100 PCIe": {"peak_tflops_fp16": 312.0, "peak_bw_tb_s": 1.555},
    "V100 SXM2": {"peak_tflops_fp16": 125.0, "peak_bw_tb_s": 0.9},
    "A6000": {"peak_tflops_fp16": 155.0, "peak_bw_tb_s": 0.768},
    "L40S": {"peak_tflops_fp16": 362.0, "peak_bw_tb_s": 0.864},
    # Blackwell workstation — RTX 6000 Blackwell (96 GB GDDR7, 600 W TDP)
    # FP16 Tensor (dense) = 2× FP32 CUDA (125 TFLOPS); same multiplier as RTX 6000 Ada.
    "RTX 6000 Blackwell": {"peak_tflops_fp16": 250.0, "peak_bw_tb_s": 1.792},
    # Blackwell data-center — B200 SXM5 (192 GB HBM3e, 8.0 TB/s, 1000 W TDP)
    "B200 SXM": {"peak_tflops_fp16": 2250.0, "peak_bw_tb_s": 8.0},
    # GB200 NVL — per-GPU spec in NVLink fabric (same die as B200 SXM)
    "GB200 NVL": {"peak_tflops_fp16": 2250.0, "peak_bw_tb_s": 8.0},
}

# Patterns for fuzzy-matching torch.cuda.get_device_name() strings
_GPU_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Blackwell — match before any generic H/A patterns
    (re.compile(r"RTX\s*6000.*Blackwell", re.IGNORECASE), "RTX 6000 Blackwell"),
    (re.compile(r"GB200.*NVL|NVL.*GB200", re.IGNORECASE), "GB200 NVL"),
    (re.compile(r"B200.*SXM|B200", re.IGNORECASE), "B200 SXM"),
    # Hopper
    (re.compile(r"H200.*NVL", re.IGNORECASE), "H200 NVL"),
    (re.compile(r"H200.*SXM", re.IGNORECASE), "H200 SXM"),
    # H200 without variant qualifier — default to SXM (most common data-centre config)
    (re.compile(r"H200", re.IGNORECASE), "H200 SXM"),
    (re.compile(r"H100.*SXM", re.IGNORECASE), "H100 SXM"),
    (re.compile(r"H100.*PCIe|H100.*PCI", re.IGNORECASE), "H100 PCIe"),
    (re.compile(r"A100.*SXM", re.IGNORECASE), "A100 SXM"),
    (re.compile(r"A100.*PCIe|A100.*PCI", re.IGNORECASE), "A100 PCIe"),
    (re.compile(r"V100.*SXM", re.IGNORECASE), "V100 SXM2"),
    (re.compile(r"A6000", re.IGNORECASE), "A6000"),
    (re.compile(r"L40S", re.IGNORECASE), "L40S"),
    # Fallback: H100/A100 without variant → assume SXM
    (re.compile(r"H100", re.IGNORECASE), "H100 SXM"),
    (re.compile(r"A100", re.IGNORECASE), "A100 SXM"),
]


def get_gpu_specs(device_name: str) -> Optional[Dict[str, Any]]:
    """Look up GPU peak specs by fuzzy-matching a device name string.

    Args:
        device_name: String from ``torch.cuda.get_device_name()``,
            e.g. ``"NVIDIA H100 80GB HBM3"``.

    Returns:
        Dict with ``peak_tflops_fp16``, ``peak_bw_tb_s``,
        ``peak_gflops`` (derived), ``peak_bw_bytes_s`` (derived),
        ``ridge_point``, and ``gpu_key``; or ``None`` if unknown.
    """
    for pattern, key in _GPU_PATTERNS:
        if pattern.search(device_name):
            raw = GPU_SPECS[key]
            peak_gflops = raw["peak_tflops_fp16"] * 1000.0
            peak_bw_bytes = raw["peak_bw_tb_s"] * 1e12
            ridge = peak_gflops / (peak_bw_bytes / 1e9)  # GFLOP/s / GB/s = FLOP/byte
            return {
                "gpu_key": key,
                "peak_tflops_fp16": raw["peak_tflops_fp16"],
                "peak_bw_tb_s": raw["peak_bw_tb_s"],
                "peak_gflops": peak_gflops,
                "peak_bw_bytes_s": peak_bw_bytes,
                "ridge_point": round(ridge, 4),
            }
    return None


# ---------------------------------------------------------------
# GEMM FLOPs derivation
# ---------------------------------------------------------------

def compute_gemm_flops(aten_op_name: str, input_dims: List) -> Optional[int]:
    """Derive GEMM FLOPs (``2*M*N*K``) from ATen op name and input dimensions.

    Supports: ``aten::mm``, ``aten::bmm``, ``aten::addmm``,
    ``aten::linear``, ``aten::matmul``, ``aten::_scaled_mm``.

    Returns:
        Integer FLOPs or ``None`` if dimensions cannot be derived.
    """
    if not input_dims:
        return None

    try:
        if aten_op_name in ("aten::mm", "aten::matmul"):
            # A[M,K] @ B[K,N]
            if len(input_dims) >= 2:
                a, b = input_dims[0], input_dims[1]
                if len(a) == 2 and len(b) == 2:
                    m, k = a
                    _, n = b
                    return 2 * m * n * k
                # Batched matmul: A[B,M,K] @ B[B,K,N]
                if len(a) == 3 and len(b) == 3:
                    batch, m, k = a
                    _, _, n = b
                    return 2 * batch * m * n * k

        elif aten_op_name == "aten::bmm":
            # A[B,M,K] @ B[B,K,N]
            if len(input_dims) >= 2:
                a, b = input_dims[0], input_dims[1]
                if len(a) == 3 and len(b) == 3:
                    batch, m, k = a
                    _, _, n = b
                    return 2 * batch * m * n * k

        elif aten_op_name == "aten::addmm":
            # bias, A[M,K], B[K,N]
            if len(input_dims) >= 3:
                a, b = input_dims[1], input_dims[2]
                if len(a) == 2 and len(b) == 2:
                    m, k = a
                    _, n = b
                    return 2 * m * n * k

        elif aten_op_name == "aten::linear":
            # input[...,M,K], weight[N,K]
            if len(input_dims) >= 2:
                inp = input_dims[0]
                weight = input_dims[1]
                if len(inp) >= 2 and len(weight) == 2:
                    m = inp[-2] if len(inp) >= 2 else inp[0]
                    k = inp[-1]
                    n = weight[0]
                    batch = 1
                    for d in inp[:-2]:
                        batch *= d
                    return 2 * batch * m * n * k

        elif aten_op_name == "aten::_scaled_mm":
            # A[M,K] @ B[K,N] (+ scale args)
            if len(input_dims) >= 2:
                a, b = input_dims[0], input_dims[1]
                if len(a) == 2 and len(b) == 2:
                    m, k = a
                    _, n = b
                    return 2 * m * n * k

    except (IndexError, TypeError, ValueError):
        return None

    return None


# ---------------------------------------------------------------
# Roofline data computation
# ---------------------------------------------------------------

def compute_roofline_data(
    per_kernel: List[Dict[str, Any]],
    gpu_specs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Compute roofline data points for ncu-profiled kernels.

    For GEMMs, FLOPs are derived from ``compute_gemm_flops()``.
    For non-GEMMs, achieved GFLOP/s is estimated from ncu
    ``compute_throughput_pct``.

    Args:
        per_kernel: Per-kernel entries from the enhanced report
            (must have ``ncu`` and ``_aten_op_full`` temp fields).
        gpu_specs: GPU specs dict from ``get_gpu_specs()``.

    Returns:
        List of roofline data dicts, each with:
        ``id``, ``kernel_name``, ``is_library_mediated``, ``ai``,
        ``achieved_gflops``, ``bound``, ``efficiency_pct``,
        ``frequency`` (invocations per inference, for marker sizing).
    """
    peak_gflops = gpu_specs["peak_gflops"]
    ridge_point = gpu_specs["ridge_point"]

    roofline_data: List[Dict[str, Any]] = []

    for entry in per_kernel:
        ncu = entry.get("ncu")
        if not ncu:
            continue

        # DRAM bytes
        dram_read = ncu.get("dram_bytes_read", 0) or 0
        dram_write = ncu.get("dram_bytes_write", 0) or 0
        total_bytes = dram_read + dram_write
        if total_bytes <= 0:
            continue

        kid = entry["id"]
        is_lib_mediated = entry["classification"].get(
            "is_library_mediated", entry["classification"].get("is_gemm", False)
        )
        duration_us = entry.get("kernel_duration_us", 0)
        if duration_us <= 0:
            continue
        duration_s = duration_us / 1e6

        aten_op_full = entry.get("_aten_op_full", {})
        aten_op_name = aten_op_full.get("name", "") if isinstance(aten_op_full, dict) else ""
        input_dims = aten_op_full.get("input_dims", []) if isinstance(aten_op_full, dict) else []

        # Compute achieved GFLOP/s
        flops = None
        if is_lib_mediated:
            flops = compute_gemm_flops(aten_op_name, input_dims)

        if flops is not None:
            achieved_gflops = (flops / duration_s) / 1e9
            ai = flops / total_bytes
        else:
            # Framework-native or FLOPs unavailable: estimate from compute throughput
            compute_pct = ncu.get("compute_throughput_pct", 0) or 0
            achieved_gflops = (compute_pct / 100.0) * peak_gflops
            if achieved_gflops <= 0:
                continue
            # Back-derive FLOPs from achieved GFLOP/s
            flops = achieved_gflops * 1e9 * duration_s
            ai = flops / total_bytes

        bound = "compute" if ai >= ridge_point else "memory"
        efficiency = (achieved_gflops / peak_gflops) * 100.0

        roofline_data.append({
            "id": kid,
            "kernel_name": entry["kernel_name"],
            "is_library_mediated": is_lib_mediated,
            # Backward-compatible alias (deprecated)
            "is_gemm": is_lib_mediated,
            "ai": round(ai, 4),
            "achieved_gflops": round(achieved_gflops, 2),
            "bound": bound,
            "efficiency_pct": round(efficiency, 2),
            "frequency": entry.get("frequency", 1),
        })

    return roofline_data


# ---------------------------------------------------------------
# Roofline plot — improved
# ---------------------------------------------------------------

def generate_roofline_plot(
    roofline_data: List[Dict[str, Any]],
    gpu_specs: Dict[str, Any],
    output_path: str,
    model_name: str = "",
) -> None:
    """Generate an improved roofline plot and save as PNG.

    Improvements over the original:
      - Iso-efficiency contours at 25 %, 50 %, 75 % of peak compute
      - Memory-bound / compute-bound region shading
      - Marker size scaled by kernel invocation frequency
      - Markers colored by bound type, shaped by kernel class (GEMM / non-GEMM)
      - Alternating annotation offsets to reduce overlap
      - Efficiency % shown in each annotation
      - Cleaner grid (major + minor), white figure background

    Args:
        roofline_data: Output from ``compute_roofline_data()``.
        gpu_specs: GPU specs from ``get_gpu_specs()``.
        output_path: Path to save the PNG file.
        model_name: Optional model name for the plot title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    # -- Style -------------------------------------------------
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "#f9f9f9",
        "axes.edgecolor": "#cccccc",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    peak_gflops = gpu_specs["peak_gflops"]
    peak_bw_gb_s = gpu_specs["peak_bw_bytes_s"] / 1e9
    ridge = gpu_specs["ridge_point"]
    gpu_key = gpu_specs["gpu_key"]

    fig, ax = plt.subplots(figsize=(12, 8))

    ai_range = np.logspace(-2, 4, 1000)

    # -- Region shading ----------------------------------------
    # Memory-bound region: shade from near-zero up to memory BW roof
    mem_mask = ai_range < ridge
    ax.fill_between(
        ai_range[mem_mask],
        1e-2,
        peak_bw_gb_s * ai_range[mem_mask],
        alpha=0.07, color="#4477CC", zorder=0,
    )
    # Compute-bound region: shade from near-zero up to compute peak
    comp_mask = ai_range >= ridge
    ax.fill_between(
        ai_range[comp_mask],
        1e-2,
        peak_gflops,
        alpha=0.07, color="#CC4444", zorder=0,
    )

    # -- Iso-efficiency contours --------------------------------
    for eff_pct, ls in [(25, ":"), (50, "--"), (75, "-.")]:
        y = peak_gflops * eff_pct / 100.0
        ax.axhline(y, color="#bbbbbb", linestyle=ls, linewidth=0.9, zorder=1)
        ax.text(
            ai_range[-1] * 0.8, y * 1.05,
            f"{eff_pct}% peak",
            fontsize=7, color="#999999", ha="right", va="bottom",
        )

    # -- Roofline ceilings -------------------------------------
    attainable = np.minimum(peak_gflops, peak_bw_gb_s * ai_range)
    ax.loglog(ai_range, attainable, "k-", linewidth=2.5, zorder=3,
              label="Roofline ceiling")

    # Memory bandwidth diagonal
    ax.loglog(ai_range, peak_bw_gb_s * ai_range,
              linestyle="--", color="#4477CC", linewidth=1.3, alpha=0.65,
              label=f"Mem BW  {peak_bw_gb_s:.0f} GB/s", zorder=2)

    # Compute peak horizontal
    ax.axhline(peak_gflops, color="#CC4444", linestyle="--",
               linewidth=1.3, alpha=0.65,
               label=f"Compute  {peak_gflops:.0f} GFLOP/s", zorder=2)

    # Ridge point
    ax.axvline(ridge, color="#888888", linestyle=":", linewidth=0.9, alpha=0.7, zorder=2)
    ax.annotate(
        f"Ridge\n{ridge:.0f} FLOP/B",
        xy=(ridge, peak_gflops * 0.82),
        ha="center", fontsize=8, color="#666666",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.9),
        zorder=4,
    )

    # -- Kernel data points ------------------------------------
    max_freq = max((d.get("frequency", 1) for d in roofline_data), default=1)

    lib_mediated_pts = [d for d in roofline_data if d.get("is_library_mediated", d.get("is_gemm", False))]
    fw_native_pts = [d for d in roofline_data if not d.get("is_library_mediated", d.get("is_gemm", False))]

    _COLORS = {"compute": "#e63946", "memory": "#2196f3"}
    _EDGE   = {"compute": "#9b0000", "memory": "#0d47a1"}

    def _plot_group(pts, marker, label_suffix):
        if not pts:
            return
        for d in pts:
            freq = d.get("frequency", 1)
            size = 80 + (freq / max_freq) * 150  # 80 … 230 pt²
            color = _COLORS.get(d["bound"], "#888888")
            edge  = _EDGE.get(d["bound"], "#444444")
            ax.scatter(
                d["ai"], d["achieved_gflops"],
                c=color, marker=marker, s=size, zorder=5,
                edgecolors=edge, linewidths=0.8, alpha=0.88,
            )

    _plot_group(lib_mediated_pts, "^", "Library-mediated")
    _plot_group(fw_native_pts, "o", "Framework-native")

    # Annotations — alternate y-offsets to reduce overlap
    annotated = sorted(roofline_data, key=lambda d: d["ai"])
    for i, d in enumerate(annotated):
        oy = 9 if i % 2 == 0 else -16
        label = f"{d['id']} {d['efficiency_pct']:.0f}%"
        ax.annotate(
            label,
            (d["ai"], d["achieved_gflops"]),
            fontsize=6.5, color="#333333",
            textcoords="offset points", xytext=(5, oy),
            arrowprops=(
                dict(arrowstyle="-", color="#aaaaaa", lw=0.5)
                if abs(oy) >= 14 else None
            ),
            zorder=6,
        )

    # -- Legend -----------------------------------------------
    legend_patches = [
        mpatches.Patch(color=_COLORS["compute"], alpha=0.75, label="Compute-bound"),
        mpatches.Patch(color=_COLORS["memory"],  alpha=0.75, label="Memory-bound"),
        plt.scatter([], [], marker="^", c="#888888", s=80,
                    edgecolors="#444444", linewidths=0.7, label=f"Library-mediated ({len(lib_mediated_pts)})"),
        plt.scatter([], [], marker="o", c="#888888", s=60,
                    edgecolors="#444444", linewidths=0.7, label=f"Framework-native ({len(fw_native_pts)})"),
    ]
    line_handles, line_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=line_handles + legend_patches,
        fontsize=8.5, loc="lower right",
        framealpha=0.92, edgecolor="#dddddd",
    )

    # Region text labels
    ax.text(0.03, 0.93, "Memory-bound", transform=ax.transAxes,
            ha="left", va="top", fontsize=9,
            color=_COLORS["memory"], fontweight="bold", alpha=0.75)
    ax.text(0.97, 0.93, "Compute-bound", transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            color=_COLORS["compute"], fontweight="bold", alpha=0.75)

    # -- Axes & formatting ------------------------------------
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=11)
    ax.set_ylabel("Achieved Performance (GFLOP/s)", fontsize=11)

    title = f"GPU Roofline — {gpu_key}"
    if model_name:
        title = f"{model_name}  ·  {title}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # Dynamic axis limits
    if roofline_data:
        ai_vals   = [d["ai"] for d in roofline_data]
        gflop_vals = [d["achieved_gflops"] for d in roofline_data]
        ax.set_xlim(min(min(ai_vals) * 0.4, 0.05), max(max(ai_vals) * 3.0, 300))
        ax.set_ylim(max(min(gflop_vals) * 0.4, 0.5), peak_gflops * 2.2)
    else:
        ax.set_xlim(0.01, 10000)
        ax.set_ylim(0.5, peak_gflops * 2.2)

    ax.grid(True, which="major", alpha=0.35, linewidth=0.8, color="#cccccc")
    ax.grid(True, which="minor", alpha=0.15, linewidth=0.4, color="#dddddd")

    # Size-legend note
    if max_freq > 1:
        ax.text(0.03, 0.03,
                "Marker size ∝ kernel invocation frequency",
                transform=ax.transAxes,
                fontsize=7, color="#888888", va="bottom")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------
# Pareto frontier computation
# ---------------------------------------------------------------

def compute_pareto_frontier(
    points: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return the Pareto-optimal subset of operating points.

    Maximises both ``throughput_tok_s`` and ``interactivity_tok_s``
    simultaneously.  A point P dominates Q if P is ≥ Q in both
    dimensions and strictly greater in at least one.

    Args:
        points: List of dicts, each with at least
            ``throughput_tok_s`` and ``interactivity_tok_s``.

    Returns:
        Pareto-optimal subset (un-dominated points), sorted by
        ascending ``interactivity_tok_s``.
    """
    if not points:
        return []

    # Sort by throughput descending; sweep by interactivity
    sorted_pts = sorted(points, key=lambda p: p["throughput_tok_s"], reverse=True)
    pareto: List[Dict[str, Any]] = []
    best_interactivity = -1.0
    for p in sorted_pts:
        inter = p.get("interactivity_tok_s", 0.0)
        if inter > best_interactivity:
            pareto.append(p)
            best_interactivity = inter

    return sorted(pareto, key=lambda p: p["interactivity_tok_s"])


# ---------------------------------------------------------------
# Throughput–Interactivity Pareto plot
# ---------------------------------------------------------------

def generate_pareto_plot(
    points: List[Dict[str, Any]],
    output_path: str,
    model_name: str = "",
    gpu_name: str = "",
) -> None:
    """Generate a throughput–interactivity Pareto plot.

    Axes:
      X — Interactivity (tokens/s per user = seq_len / inference_time_s)
      Y — Throughput (total tokens/s = batch × seq_len / inference_time_s)

    For a single operating point the plot shows the point against
    iso-batch reference lines (throughput = batch_size × interactivity)
    so the user can see which "batch regime" they are in.  When
    multiple points from a sweep are supplied, Pareto-optimal points
    are highlighted and connected by the Pareto frontier.

    Args:
        points: List of dicts, each with:
            ``throughput_tok_s``   (float) — total tokens/s
            ``interactivity_tok_s`` (float) — tokens/s per user
            ``label``              (str)   — e.g. "bs=4 sl=512"
            ``is_current``         (bool)  — highlight this point
        output_path: PNG output path.
        model_name: Optional model name for the title.
        gpu_name: Optional GPU name for the title.
    """
    if not points:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "#f9f9f9",
        "axes.edgecolor": "#cccccc",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(11, 7))

    # Compute Pareto frontier
    pareto = compute_pareto_frontier(points)
    pareto_set = {id(p) for p in pareto}

    # Axis ranges (with padding)
    all_x = [p["interactivity_tok_s"] for p in points]
    all_y = [p["throughput_tok_s"] for p in points]
    x_min = min(all_x) * 0.3
    x_max = max(all_x) * 4.0
    y_min = min(all_y) * 0.3
    y_max = max(all_y) * 4.0

    # -- Iso-batch reference lines ----------------------------
    # throughput = bs × interactivity  →  straight line through origin in log-log
    x_ref = np.logspace(np.log10(x_min), np.log10(x_max), 200)
    for bs, lw in [(1, 1.2), (2, 0.9), (4, 0.9), (8, 0.9), (16, 0.9), (32, 0.9), (64, 0.7)]:
        y_ref = bs * x_ref
        # Only draw if line crosses the visible y range
        if y_ref.max() < y_min or y_ref.min() > y_max:
            continue
        ax.plot(x_ref, y_ref, color="#cccccc", linestyle="--", linewidth=lw,
                alpha=0.7, zorder=1)
        # Label at top of line within plot bounds
        for xi, yi in zip(x_ref[::-1], y_ref[::-1]):
            if yi <= y_max * 0.95:
                ax.text(xi * 1.03, yi, f"bs={bs}", fontsize=7,
                        color="#bbbbbb", va="center", zorder=2)
                break

    # -- Zone shading -----------------------------------------
    # Bottom-right: Interactive mode (high x, low y)  — right portion, low
    # Top-left: Throughput mode (low x, high y)        — left portion, high
    ax.text(0.03, 0.97, "Throughput\nmode", transform=ax.transAxes,
            ha="left", va="top", fontsize=9,
            color="#888888", fontstyle="italic", zorder=3)
    ax.text(0.97, 0.03, "Interactive\nmode", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9,
            color="#888888", fontstyle="italic", zorder=3)
    ax.text(0.97, 0.97, "Ideal\n(high both)", transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            color="#2b9348", fontstyle="italic", alpha=0.6, zorder=3)

    # -- Pareto frontier line ---------------------------------
    if len(pareto) > 1:
        ax.plot(
            [p["interactivity_tok_s"] for p in pareto],
            [p["throughput_tok_s"] for p in pareto],
            color="#2b9348", linewidth=2.0, alpha=0.75, zorder=4,
            label="Pareto frontier",
        )

    # -- Data points ------------------------------------------
    for p in points:
        is_current = p.get("is_current", False)
        on_frontier = id(p) in pareto_set
        x, y = p["interactivity_tok_s"], p["throughput_tok_s"]
        label = p.get("label", "")

        if is_current:
            color, edge, size, marker, zorder = "#e63946", "#9b0000", 220, "*", 7
        elif on_frontier:
            color, edge, size, marker, zorder = "#2b9348", "#1b5e20", 120, "D", 6
        else:
            color, edge, size, marker, zorder = "#555555", "#222222",  70, "o", 5

        ax.scatter(x, y, c=color, s=size, marker=marker, zorder=zorder,
                   edgecolors=edge, linewidths=0.8, alpha=0.90)
        ax.annotate(
            label, (x, y),
            fontsize=8, textcoords="offset points", xytext=(6, 5),
            color="#333333", zorder=zorder + 1,
        )

    # -- Legend -----------------------------------------------
    legend_handles = []
    if any(p.get("is_current") for p in points):
        legend_handles.append(
            plt.scatter([], [], marker="*", c="#e63946", s=160,
                        edgecolors="#9b0000", linewidths=0.7, label="Current run")
        )
    if any(id(p) in pareto_set and not p.get("is_current") for p in points):
        legend_handles.append(
            plt.scatter([], [], marker="D", c="#2b9348", s=90,
                        edgecolors="#1b5e20", linewidths=0.7, label="Pareto-optimal")
        )
    if any(id(p) not in pareto_set for p in points):
        legend_handles.append(
            plt.scatter([], [], marker="o", c="#555555", s=60,
                        edgecolors="#222222", linewidths=0.7, label="Sub-optimal")
        )
    if len(pareto) > 1:
        from matplotlib.lines import Line2D
        legend_handles.append(
            Line2D([0], [0], color="#2b9348", linewidth=2.0,
                   alpha=0.75, label="Pareto frontier")
        )
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=8.5, loc="upper left",
                  framealpha=0.92, edgecolor="#dddddd")

    # -- Axes & formatting ------------------------------------
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Interactivity  (tokens / s / user)", fontsize=11)
    ax.set_ylabel("Throughput  (total tokens / s)", fontsize=11)

    title_parts = ["Throughput – Interactivity"]
    if model_name:
        title_parts.append(model_name)
    if gpu_name:
        title_parts.append(gpu_name)
    ax.set_title("  ·  ".join(title_parts), fontsize=13, fontweight="bold", pad=12)

    ax.grid(True, which="major", alpha=0.35, linewidth=0.8, color="#cccccc")
    ax.grid(True, which="minor", alpha=0.15, linewidth=0.4, color="#dddddd")

    ax.text(0.03, 0.03,
            "Dashed lines = iso-batch contours  (throughput = batch_size × interactivity)",
            transform=ax.transAxes, fontsize=7, color="#aaaaaa", va="bottom")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
