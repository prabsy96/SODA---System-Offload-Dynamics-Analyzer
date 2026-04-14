"""Custom MoE architecture diagram with dimension annotations.

Generates a publication-quality diagram showing the MoE data flow:
input tokens → attention → gate → shared/routed experts → output,
annotated with weight dimensions, HBM bytes, and kernel counts from
the op profile and detected MoE config.

Usage::

    from soda.moe.arch_diagram import generate_moe_arch_diagram

    generate_moe_arch_diagram(
        moe_config=config,
        op_profile=profile,
        output_path=Path("moe_architecture.png"),
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _import_matplotlib():
    """Import matplotlib with Agg backend (no display required)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    return plt, mpatches


# ---------------------------------------------------------------------------
# Box drawing helpers
# ---------------------------------------------------------------------------

def _draw_box(
    ax: Any,
    center: Tuple[float, float],
    width: float,
    height: float,
    label: str,
    sublabel: str = "",
    color: str = "#E8F0FE",
    edgecolor: str = "#4285F4",
    fontsize: int = 9,
    sublabel_fontsize: int = 7,
) -> None:
    """Draw a rounded rectangle with centered text."""
    _, mpatches = _import_matplotlib()

    x = center[0] - width / 2
    y = center[1] - height / 2

    box = mpatches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=1.5,
    )
    ax.add_patch(box)

    if sublabel:
        ax.text(
            center[0], center[1] + 0.02,
            label,
            ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold",
        )
        ax.text(
            center[0], center[1] - 0.02,
            sublabel,
            ha="center", va="top",
            fontsize=sublabel_fontsize, color="#555555",
            fontstyle="italic",
        )
    else:
        ax.text(
            center[0], center[1],
            label,
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
        )


def _draw_arrow(
    ax: Any,
    start: Tuple[float, float],
    end: Tuple[float, float],
    label: str = "",
    color: str = "#666666",
    fontsize: int = 7,
) -> None:
    """Draw an arrow between two points with an optional label."""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            linewidth=1.2,
            connectionstyle="arc3,rad=0",
        ),
    )
    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(
            mid_x + 0.02, mid_y,
            label,
            ha="left", va="center",
            fontsize=fontsize, color="#888888",
        )


# ---------------------------------------------------------------------------
# Op profile aggregation
# ---------------------------------------------------------------------------

def _aggregate_op_profile(
    op_profile: List[Dict],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate op profile entries by op_name for annotation.

    Returns:
        Dict mapping op_name -> {
            "count": int,
            "total_hbm_bytes": int,
            "total_flops": int,
            "avg_latency_us": float,
        }
    """
    agg: Dict[str, Dict[str, Any]] = {}
    for entry in op_profile:
        name = entry.get("op_name", "unknown")
        if name not in agg:
            agg[name] = {
                "count": 0,
                "total_hbm_bytes": 0,
                "total_flops": 0,
                "latencies": [],
            }
        agg[name]["count"] += entry.get("kernel_count", entry.get("count", 1))
        agg[name]["total_hbm_bytes"] += entry.get("hbm_bytes", 0)
        agg[name]["total_flops"] += entry.get("flops", 0)
        lat = entry.get("avg_latency_us", entry.get("latency_us", 0))
        if lat:
            agg[name]["latencies"].append(lat)

    result = {}
    for name, data in agg.items():
        lats = data["latencies"]
        result[name] = {
            "count": data["count"],
            "total_hbm_bytes": data["total_hbm_bytes"],
            "total_flops": data["total_flops"],
            "avg_latency_us": sum(lats) / len(lats) if lats else 0.0,
        }
    return result


def _fmt_bytes(b: int) -> str:
    """Format bytes as human-readable string."""
    if b >= 1e9:
        return f"{b / 1e9:.1f} GB"
    if b >= 1e6:
        return f"{b / 1e6:.1f} MB"
    if b >= 1e3:
        return f"{b / 1e3:.1f} KB"
    return f"{b} B"


def _fmt_dim(dim: Optional[int]) -> str:
    """Format a dimension as string."""
    return str(dim) if dim is not None else "?"


# ---------------------------------------------------------------------------
# Main diagram generator
# ---------------------------------------------------------------------------

def generate_moe_arch_diagram(
    moe_config: Dict,
    op_profile: Optional[List[Dict]] = None,
    output_path: Optional[Path] = None,
    model_name: str = "",
    figsize: Tuple[float, float] = (16, 20),
) -> Path:
    """Generate an annotated MoE architecture diagram.

    Args:
        moe_config: Output of ``detect_moe_config()`` — must contain
            ``hidden_dim``, ``shared_dim``, ``routed_dim``, ``num_experts``.
        op_profile: Optional list of op profile dicts from
            ``generate_op_profile()``.  If provided, HBM bytes and kernel
            counts are annotated on each block.
        output_path: Path for the output PNG.  If None, defaults to
            ``moe_architecture.png`` in the current directory.
        model_name: Model name for the title.
        figsize: Figure size in inches.

    Returns:
        Path to the saved PNG.
    """
    plt, mpatches = _import_matplotlib()

    hidden_dim = moe_config.get("hidden_dim")
    shared_dim = moe_config.get("shared_dim")
    routed_dim = moe_config.get("routed_dim")
    num_experts = moe_config.get("num_experts")

    # Aggregate op profile for annotations.
    op_agg = _aggregate_op_profile(op_profile) if op_profile else {}

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")

    # Layout: vertical flow from top (y=1) to bottom (y=0)
    box_w = 0.5
    box_h = 0.06
    small_box_w = 0.35
    small_box_h = 0.08

    # -- Input tokens --
    y_input = 0.95
    _draw_box(
        ax, (0.5, y_input), box_w, box_h,
        "Input Tokens",
        f"(batch, seq_len, {_fmt_dim(hidden_dim)})",
        color="#E8F5E9", edgecolor="#4CAF50",
    )

    # -- Attention block --
    y_attn = 0.84
    _draw_box(
        ax, (0.5, y_attn), box_w, box_h * 1.4,
        "Multi-Head Attention",
        f"Q/K/V/O: ({_fmt_dim(hidden_dim)} \u2192 {_fmt_dim(hidden_dim)})",
        color="#E3F2FD", edgecolor="#2196F3",
    )
    _draw_arrow(ax, (0.5, y_input - box_h / 2), (0.5, y_attn + box_h * 1.4 / 2))

    # -- LayerNorm / residual (small) --
    y_norm = 0.74
    _draw_box(
        ax, (0.5, y_norm), 0.3, 0.03,
        "LayerNorm + Residual",
        color="#FFF3E0", edgecolor="#FF9800",
        fontsize=7,
    )
    _draw_arrow(ax, (0.5, y_attn - box_h * 1.4 / 2), (0.5, y_norm + 0.015))

    # -- MoE Gate --
    y_gate = 0.66
    gate_sublabel = f"({_fmt_dim(hidden_dim)} \u2192 {_fmt_dim(num_experts)})"
    _draw_box(
        ax, (0.5, y_gate), 0.4, box_h,
        "MoE Gate (Router)",
        f"Linear {gate_sublabel}  \u2192 softmax \u2192 top-k",
        color="#FCE4EC", edgecolor="#E91E63",
    )
    _draw_arrow(ax, (0.5, y_norm - 0.015), (0.5, y_gate + box_h / 2))

    # -- Fork into shared + routed --
    y_fork = y_gate - box_h / 2 - 0.02

    # Shared expert (left)
    x_shared = 0.22
    y_shared = 0.50
    shared_detail = (
        f"gate_proj: {_fmt_dim(hidden_dim)} \u2192 {_fmt_dim(shared_dim)}\n"
        f"up_proj:   {_fmt_dim(hidden_dim)} \u2192 {_fmt_dim(shared_dim)}\n"
        f"SiLU(gate) \u2297 up\n"
        f"down_proj: {_fmt_dim(shared_dim)} \u2192 {_fmt_dim(hidden_dim)}"
    )
    _draw_box(
        ax, (x_shared, y_shared), small_box_w, small_box_h * 1.5,
        "Shared Expert",
        shared_detail,
        color="#E8EAF6", edgecolor="#3F51B5",
        sublabel_fontsize=6,
    )

    # Routed experts (right)
    x_routed = 0.78
    y_routed = 0.50
    routed_detail = (
        f"gate_proj: {_fmt_dim(hidden_dim)} \u2192 {_fmt_dim(routed_dim)}\n"
        f"up_proj:   {_fmt_dim(hidden_dim)} \u2192 {_fmt_dim(routed_dim)}\n"
        f"SiLU(gate) \u2297 up\n"
        f"down_proj: {_fmt_dim(routed_dim)} \u2192 {_fmt_dim(hidden_dim)}"
    )
    n_exp_str = f"\u00d7 {num_experts} experts" if num_experts else "\u00d7 N experts"
    _draw_box(
        ax, (x_routed, y_routed), small_box_w, small_box_h * 1.5,
        f"Routed Expert ({n_exp_str})",
        routed_detail,
        color="#FFF9C4", edgecolor="#FBC02D",
        sublabel_fontsize=6,
    )

    # Arrows from gate to shared/routed
    _draw_arrow(
        ax,
        (0.5 - 0.05, y_fork),
        (x_shared, y_shared + small_box_h * 1.5 / 2),
        label="all tokens",
    )
    _draw_arrow(
        ax,
        (0.5 + 0.05, y_fork),
        (x_routed, y_routed + small_box_h * 1.5 / 2),
        label="top-k tokens",
    )

    # -- Merge / weighted sum --
    y_merge = 0.36
    _draw_box(
        ax, (0.5, y_merge), 0.4, 0.04,
        "Weighted Sum (shared + routed outputs)",
        color="#F3E5F5", edgecolor="#9C27B0",
        fontsize=8,
    )
    _draw_arrow(ax, (x_shared, y_shared - small_box_h * 1.5 / 2), (0.5 - 0.05, y_merge + 0.02))
    _draw_arrow(ax, (x_routed, y_routed - small_box_h * 1.5 / 2), (0.5 + 0.05, y_merge + 0.02))

    # -- Repeat indicator --
    y_repeat = 0.30
    ax.text(
        0.5, y_repeat,
        "\u00d7 num_layers",
        ha="center", va="center",
        fontsize=11, fontweight="bold", color="#9E9E9E",
        fontstyle="italic",
    )

    # -- Output --
    y_output = 0.22
    _draw_box(
        ax, (0.5, y_output), box_w, box_h,
        "Output Logits",
        f"(batch, seq_len, vocab_size)",
        color="#E8F5E9", edgecolor="#4CAF50",
    )
    _draw_arrow(ax, (0.5, y_merge - 0.02), (0.5, y_output + box_h / 2))

    # -- Op profile annotations (right margin) --
    if op_agg:
        y_annot = 0.14
        annot_lines = ["Op Profile Summary:"]

        # Map op_name patterns to display.  Order matters: more specific
        # patterns must come before less specific ones (e.g.,
        # "routed_expert_gate_proj" before "routed_expert_gate_up").
        _name_map = [
            ("shared_expert_gate", "Shared gate_proj"),
            ("shared_expert_up", "Shared up_proj"),
            ("shared_expert_down", "Shared down_proj"),
            ("routed_expert_gate_proj", "Routed gate_proj"),
            ("routed_expert_up_proj", "Routed up_proj"),
            ("routed_expert_gate_up", "Routed gate_up (merged)"),
            ("routed_expert_down", "Routed down_proj"),
            ("gate", "MoE Gate"),
            ("attention", "Attention"),
        ]

        for pattern, display_name in _name_map:
            matched = [v for k, v in op_agg.items() if pattern in k]
            if matched:
                total_hbm = sum(m["total_hbm_bytes"] for m in matched)
                total_count = sum(m["count"] for m in matched)
                avg_lat = sum(m["avg_latency_us"] for m in matched) / len(matched) if matched else 0
                line = f"  {display_name}: {total_count} kernels, {_fmt_bytes(total_hbm)} HBM"
                if avg_lat > 0:
                    line += f", {avg_lat:.1f} \u03bcs avg"
                annot_lines.append(line)

        ax.text(
            0.02, y_annot,
            "\n".join(annot_lines),
            ha="left", va="top",
            fontsize=7, fontfamily="monospace",
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", edgecolor="#BDBDBD"),
        )

    # -- Config summary box --
    y_config = 0.07
    config_lines = []
    if model_name:
        config_lines.append(f"Model: {model_name}")
    config_lines.extend([
        f"hidden_dim: {_fmt_dim(hidden_dim)}",
        f"shared_intermediate: {_fmt_dim(shared_dim)}",
        f"routed_intermediate: {_fmt_dim(routed_dim)}",
        f"num_experts: {_fmt_dim(num_experts)}",
    ])
    ax.text(
        0.98, y_config,
        "\n".join(config_lines),
        ha="right", va="top",
        fontsize=7, fontfamily="monospace",
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F0FE", edgecolor="#4285F4"),
    )

    # Title
    title_text = "MoE Architecture"
    if model_name:
        title_text = f"{model_name} — {title_text}"
    fig.suptitle(title_text, fontsize=14, fontweight="bold", y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path is None:
        output_path = Path("moe_architecture.png")
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
