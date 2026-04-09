"""Routing diversity metrics and visualizations for MoE gate decisions.

Computes per-layer expert load distributions, Shannon entropy (load balance),
and cross-layer expert overlap from captured routing decisions.  Generates
matplotlib plots: routing heatmaps, load-balance bar charts, and cross-input
routing comparisons.

Usage::

    from soda.moe.routing_analysis import compute_routing_metrics, plot_routing_heatmap

    data = RoutingCapture.load("routing_decisions.npz")
    metrics = compute_routing_metrics(data, num_experts=60)
    plot_routing_heatmap(metrics, output_path / "routing_heatmap.png")
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_routing_metrics(
    routing_data: Dict,
    num_experts: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute routing diversity metrics from captured routing decisions.

    Args:
        routing_data: Output of ``RoutingCapture.load()`` — dict with keys
            ``metadata`` (dict) and ``layers`` (dict mapping layer_idx to
            {``experts``: ndarray, ``probs``: ndarray}).
        num_experts: Number of experts.  If None, inferred from metadata or
            the max expert ID in the data.

    Returns:
        Dict with keys:
            - ``num_experts`` (int)
            - ``num_layers`` (int)
            - ``layer_indices`` (list[int])
            - ``expert_load_per_layer`` (ndarray, shape [num_layers, num_experts])
                Token count per expert at each layer.
            - ``load_balance_entropy`` (ndarray, shape [num_layers])
                Shannon entropy of the expert selection distribution per layer.
                Higher = more balanced.
            - ``dominant_expert_ratio`` (ndarray, shape [num_layers])
                Fraction of tokens going to the most-used expert per layer.
            - ``expert_correlation`` (ndarray, shape [num_layers, num_layers])
                Pearson correlation of expert load vectors across layers.
    """
    metadata = routing_data.get("metadata", {})
    layers = routing_data.get("layers", {})

    if not layers:
        raise ValueError("routing_data contains no layer records")

    # Resolve num_experts.
    if num_experts is None:
        num_experts = metadata.get("num_experts")
    if num_experts is None:
        # Infer from data.
        max_id = max(
            int(layers[idx]["experts"].max())
            for idx in layers
            if layers[idx]["experts"] is not None and layers[idx]["experts"].size > 0
        )
        num_experts = max_id + 1

    layer_indices = sorted(layers.keys())
    num_layers = len(layer_indices)

    # Expert load: count how many times each expert is selected per layer.
    expert_load = np.zeros((num_layers, num_experts), dtype=np.int64)
    for i, layer_idx in enumerate(layer_indices):
        experts = layers[layer_idx].get("experts")
        if experts is None or experts.size == 0:
            continue
        # experts shape: (num_tokens, top_k) — flatten to count each selection.
        for eid in experts.flat:
            eid = int(eid)
            if 0 <= eid < num_experts:
                expert_load[i, eid] += 1

    # Shannon entropy per layer.
    entropy = np.zeros(num_layers, dtype=np.float64)
    for i in range(num_layers):
        total = expert_load[i].sum()
        if total == 0:
            continue
        probs = expert_load[i].astype(np.float64) / total
        # Mask zeros to avoid log(0).
        nonzero = probs > 0
        entropy[i] = -np.sum(probs[nonzero] * np.log2(probs[nonzero]))

    # Dominant expert ratio per layer.
    dominant_ratio = np.zeros(num_layers, dtype=np.float64)
    for i in range(num_layers):
        total = expert_load[i].sum()
        if total > 0:
            dominant_ratio[i] = expert_load[i].max() / total

    # Cross-layer correlation of expert load vectors.
    # Use Pearson correlation; handle constant rows (all zeros) gracefully.
    correlation = np.eye(num_layers, dtype=np.float64)
    for i in range(num_layers):
        for j in range(i + 1, num_layers):
            vi = expert_load[i].astype(np.float64)
            vj = expert_load[j].astype(np.float64)
            std_i, std_j = vi.std(), vj.std()
            if std_i > 0 and std_j > 0:
                corr = np.corrcoef(vi, vj)[0, 1]
                correlation[i, j] = corr
                correlation[j, i] = corr

    return {
        "num_experts": num_experts,
        "num_layers": num_layers,
        "layer_indices": layer_indices,
        "expert_load_per_layer": expert_load,
        "load_balance_entropy": entropy,
        "dominant_expert_ratio": dominant_ratio,
        "expert_correlation": correlation,
    }


def compute_max_entropy(num_experts: int) -> float:
    """Maximum Shannon entropy for uniform distribution over num_experts."""
    if num_experts <= 0:
        return 0.0
    return np.log2(num_experts)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _import_matplotlib():
    """Import matplotlib with Agg backend (no display required)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_routing_heatmap(
    metrics: Dict[str, Any],
    output_path: Path,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 8),
) -> Path:
    """Generate a routing heatmap: expert ID (x) vs layer (y), colored by token count.

    Args:
        metrics: Output of ``compute_routing_metrics()``.
        output_path: Path for the output PNG.
        title: Optional plot title.  Defaults to "Expert Routing Heatmap".
        figsize: Figure size in inches.

    Returns:
        Path to the saved PNG.
    """
    plt = _import_matplotlib()

    expert_load = metrics["expert_load_per_layer"]
    layer_indices = metrics["layer_indices"]
    num_experts = metrics["num_experts"]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    im = ax.imshow(
        expert_load,
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
        origin="lower",
    )
    fig.colorbar(im, ax=ax, label="Token count")

    ax.set_xlabel("Expert ID")
    ax.set_ylabel("Layer")
    ax.set_title(title or "Expert Routing Heatmap")

    # Tick labels.
    if num_experts <= 64:
        ax.set_xticks(range(num_experts))
        ax.set_xticklabels(range(num_experts), fontsize=max(4, 8 - num_experts // 16))
    else:
        ax.set_xticks(np.linspace(0, num_experts - 1, min(20, num_experts), dtype=int))

    ax.set_yticks(range(len(layer_indices)))
    ax.set_yticklabels(layer_indices, fontsize=max(5, 8 - len(layer_indices) // 8))

    fig.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_load_balance(
    metrics_by_category: Dict[str, Dict[str, Any]],
    output_path: Path,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 6),
) -> Path:
    """Generate a load-balance bar chart: entropy per layer, grouped by prompt category.

    Args:
        metrics_by_category: Dict mapping category name to metrics from
            ``compute_routing_metrics()``.
        output_path: Path for the output PNG.
        title: Optional plot title.
        figsize: Figure size in inches.

    Returns:
        Path to the saved PNG.
    """
    plt = _import_matplotlib()

    if not metrics_by_category:
        raise ValueError("metrics_by_category is empty")

    categories = sorted(metrics_by_category.keys())
    # Use layer indices from the first category (assumed consistent).
    first = metrics_by_category[categories[0]]
    layer_indices = first["layer_indices"]
    num_layers = len(layer_indices)
    max_ent = compute_max_entropy(first["num_experts"])

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    num_cats = len(categories)
    bar_width = 0.8 / max(num_cats, 1)
    x = np.arange(num_layers)
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(num_cats)

    for ci, cat in enumerate(categories):
        m = metrics_by_category[cat]
        entropy = m["load_balance_entropy"]
        offset = (ci - num_cats / 2 + 0.5) * bar_width
        ax.bar(x + offset, entropy, width=bar_width, label=cat, color=cmap(ci), alpha=0.85)

    # Max entropy reference line.
    if max_ent > 0:
        ax.axhline(max_ent, color="gray", linestyle="--", linewidth=0.8, label=f"Max entropy ({max_ent:.1f})")

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Shannon entropy (bits)")
    ax.set_title(title or "Expert Load Balance (Shannon Entropy per Layer)")
    ax.set_xticks(x)
    ax.set_xticklabels(layer_indices, fontsize=max(5, 8 - num_layers // 8))
    ax.legend(fontsize=7, loc="lower right", ncol=min(4, num_cats))

    fig.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_routing_comparison(
    metrics_by_category: Dict[str, Dict[str, Any]],
    output_path: Path,
    max_cols: int = 4,
    figsize_per_subplot: Tuple[float, float] = (4, 3),
) -> Path:
    """Generate side-by-side routing heatmaps for each prompt category.

    Args:
        metrics_by_category: Dict mapping category name to metrics.
        output_path: Path for the output PNG.
        max_cols: Maximum number of columns in the subplot grid.
        figsize_per_subplot: (width, height) per subplot in inches.

    Returns:
        Path to the saved PNG.
    """
    plt = _import_matplotlib()

    if not metrics_by_category:
        raise ValueError("metrics_by_category is empty")

    categories = sorted(metrics_by_category.keys())
    n = len(categories)
    ncols = min(n, max_cols)
    nrows = (n + ncols - 1) // ncols

    fig_w = figsize_per_subplot[0] * ncols + 1.5
    fig_h = figsize_per_subplot[1] * nrows + 1.0
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    # Shared color scale across all subplots.
    vmax = max(
        m["expert_load_per_layer"].max()
        for m in metrics_by_category.values()
    )
    vmin = 0

    for idx, cat in enumerate(categories):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        m = metrics_by_category[cat]
        load = m["expert_load_per_layer"]

        im = ax.imshow(
            load,
            aspect="auto",
            cmap="YlOrRd",
            interpolation="nearest",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(cat, fontsize=9)
        ax.set_xlabel("Expert", fontsize=7)
        ax.set_ylabel("Layer", fontsize=7)
        ax.tick_params(labelsize=5)

    # Hide unused subplots.
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    # Shared colorbar.
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Token count")

    fig.suptitle("Expert Routing Comparison Across Input Types", fontsize=12, y=1.01)
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_dominant_expert(
    metrics_by_category: Dict[str, Dict[str, Any]],
    output_path: Path,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 6),
) -> Path:
    """Generate a line plot of dominant expert ratio per layer across categories.

    Args:
        metrics_by_category: Dict mapping category name to metrics.
        output_path: Path for the output PNG.
        title: Optional plot title.
        figsize: Figure size in inches.

    Returns:
        Path to the saved PNG.
    """
    plt = _import_matplotlib()

    if not metrics_by_category:
        raise ValueError("metrics_by_category is empty")

    categories = sorted(metrics_by_category.keys())
    first = metrics_by_category[categories[0]]
    layer_indices = first["layer_indices"]
    num_cats = len(categories)
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(num_cats)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for ci, cat in enumerate(categories):
        m = metrics_by_category[cat]
        ax.plot(
            layer_indices,
            m["dominant_expert_ratio"],
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=cat,
            color=cmap(ci),
        )

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Dominant expert ratio")
    ax.set_title(title or "Dominant Expert Ratio per Layer")
    ax.legend(fontsize=7, loc="upper right", ncol=min(4, num_cats))
    ax.set_ylim(0, 1)

    fig.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
