
"""
Summarize HSB sweep results with heatmaps and visualizations.

Creates:
- HSB heatmap (batch_size vs seq_len)
- T_Exposed vs T_Structural comparison plot
- Summary statistics CSV

Usage:
    python -m experiments.sweep.summarize_hsb_sweep output/<sweep_dir>
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plots will not be generated.", file=sys.stderr)

from experiments.sweep.hsb import HSBResult, classify_hsb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize HSB sweep results.")
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Path to HSB sweep directory.",
    )
    parser.add_argument(
        "--gpu-name",
        type=str,
        default=None,
        help="GPU name for plot titles.",
    )
    return parser.parse_args()


def load_hsb_results(root: Path) -> List[HSBResult]:
    """Load HSB results from sweep summary or individual reports."""
    results = []
    
    # Try sweep summary first
    summary_file = root / "hsb_sweep_summary.json"
    if summary_file.exists():
        with open(summary_file, "r") as f:
            summary = json.load(f)
        for r in summary.get("results", []):
            results.append(HSBResult.from_dict(r))
        return results
    
    # Fall back to individual hsb_report.json files
    for hsb_report in root.rglob("hsb_report.json"):
        with open(hsb_report, "r") as f:
            data = json.load(f)
        hsb_metrics = data.get("hsb_metrics", {})
        if hsb_metrics:
            results.append(HSBResult.from_dict(hsb_metrics))
    
    return results


def build_hsb_grid(
    results: List[HSBResult],
) -> Tuple[List[int], List[int], np.ndarray, np.ndarray]:
    """
    Build 2D grids for HSB heatmap.
    
    Returns:
        (batch_sizes, seq_lens, hsb_grid, status_grid)
    """
    if not results:
        return [], [], np.array([]), np.array([])
    
    # Collect unique batch sizes and sequence lengths
    batch_sizes = sorted(set(r.batch_size for r in results))
    seq_lens = sorted(set(r.seq_len for r in results))
    
    # Create lookup
    lookup = {(r.batch_size, r.seq_len): r for r in results}
    
    # Build grids
    hsb_grid = np.full((len(seq_lens), len(batch_sizes)), np.nan)
    status_grid = np.empty((len(seq_lens), len(batch_sizes)), dtype=object)
    
    for i, sl in enumerate(seq_lens):
        for j, bs in enumerate(batch_sizes):
            result = lookup.get((bs, sl))
            if result:
                if result.hsb is not None:
                    hsb_grid[i, j] = result.hsb
                status_grid[i, j] = result.status
            else:
                status_grid[i, j] = "missing"
    
    return batch_sizes, seq_lens, hsb_grid, status_grid


def plot_hsb_heatmap(
    results: List[HSBResult],
    output_path: Path,
    title_suffix: str = "",
) -> None:
    """Generate HSB heatmap visualization."""
    if not HAS_MATPLOTLIB:
        return
    
    batch_sizes, seq_lens, hsb_grid, status_grid = build_hsb_grid(results)
    
    if len(batch_sizes) == 0 or len(seq_lens) == 0:
        print("No data available for HSB heatmap.")
        return
    
    # Configure plot style
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    
    # Create figure
    fig_width = max(4, len(batch_sizes) * 0.8 + 1.5)
    fig_height = max(3, len(seq_lens) * 0.6 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Use diverging colormap centered at 0
    # HSB = 1 (hardware-bound) -> green
    # HSB = 0 (balanced) -> white/yellow
    # HSB < 0 (framework-bound) -> red
    vmin = np.nanmin(hsb_grid) if np.any(~np.isnan(hsb_grid)) else -1
    vmax = np.nanmax(hsb_grid) if np.any(~np.isnan(hsb_grid)) else 1
    
    # Ensure symmetric range around 0 for diverging colormap
    abs_max = max(abs(vmin), abs(vmax), 1.0)
    
    # Create custom norm centered at 0
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=1.0)
    
    # Plot heatmap
    im = ax.imshow(
        hsb_grid,
        aspect="auto",
        cmap="RdYlGn",  # Red (framework-bound) -> Yellow (balanced) -> Green (hardware-bound)
        norm=norm,
    )
    
    # Configure axes
    ax.set_xticks(range(len(batch_sizes)))
    ax.set_xticklabels(batch_sizes)
    ax.set_yticks(range(len(seq_lens)))
    ax.set_yticklabels(seq_lens)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Sequence Length")
    
    # Title
    title = "Hardware-Software Inversion (HSB)"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title)
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("HSB")
    
    # Add text annotations
    for i in range(len(seq_lens)):
        for j in range(len(batch_sizes)):
            val = hsb_grid[i, j]
            status = status_grid[i, j]
            
            if status == "oom":
                text = "OOM"
                color = "black"
            elif np.isnan(val):
                text = "N/A"
                color = "gray"
            else:
                text = f"{val:.2f}"
                # Choose text color based on background
                color = "white" if val < -0.3 or val > 0.7 else "black"
            
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)
    
    # Add legend for interpretation
    legend_text = (
        "HSB Interpretation:\n"
        "  1.0 = Hardware-bound\n"
        "  0.0 = Balanced\n"
        " <0.0 = Framework-bound"
    )
    fig.text(0.02, 0.02, legend_text, fontsize=7, family="monospace",
             verticalalignment="bottom", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved HSB heatmap to: {output_path}")


def plot_t_comparison(
    results: List[HSBResult],
    output_path: Path,
    title_suffix: str = "",
) -> None:
    """Plot T_Exposed vs T_Structural comparison."""
    if not HAS_MATPLOTLIB:
        return
    
    # Filter valid results
    valid_results = [r for r in results if r.t_exposed_us is not None and r.t_structural_us is not None]
    
    if not valid_results:
        print("No valid data for T comparison plot.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Scatter plot of T_Exposed vs T_Structural
    ax1 = axes[0]
    t_exposed = [r.t_exposed_us / 1000.0 for r in valid_results]  # Convert to ms
    t_structural = [r.t_structural_us / 1000.0 for r in valid_results]
    hsb_values = [r.hsb for r in valid_results]
    
    scatter = ax1.scatter(t_structural, t_exposed, c=hsb_values, cmap="RdYlGn", 
                          s=50, alpha=0.7, edgecolors="black", linewidths=0.5)
    
    # Add diagonal line (T_Exposed = T_Structural, i.e., HSB = 0)
    max_val = max(max(t_exposed), max(t_structural))
    ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label="HSB = 0 (balanced)")
    
    ax1.set_xlabel("T_Structural (ms)")
    ax1.set_ylabel("T_Exposed (ms)")
    ax1.set_title("Framework Overhead: Exposed vs Structural")
    ax1.legend()
    
    cbar1 = fig.colorbar(scatter, ax=ax1)
    cbar1.set_label("HSB")
    
    # Plot 2: HSB distribution histogram
    ax2 = axes[1]
    hsb_array = np.array(hsb_values)
    
    # Classify results
    hw_bound = np.sum(hsb_array >= 0.5)
    balanced = np.sum((hsb_array >= 0) & (hsb_array < 0.5))
    fw_bound = np.sum(hsb_array < 0)
    
    colors = ["green", "gold", "red"]
    labels = [f"Hardware-bound\n(HSB ≥ 0.5): {hw_bound}",
              f"Balanced\n(0 ≤ HSB < 0.5): {balanced}",
              f"Framework-bound\n(HSB < 0): {fw_bound}"]
    counts = [hw_bound, balanced, fw_bound]
    
    bars = ax2.bar(range(3), counts, color=colors, edgecolor="black")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Number of Configurations")
    ax2.set_title("HSB Distribution")
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(count), ha="center", va="bottom", fontweight="bold")
    
    if title_suffix:
        fig.suptitle(title_suffix, fontsize=12, fontweight="bold")
    
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved T comparison plot to: {output_path}")


def write_hsb_csv(
    results: List[HSBResult],
    output_path: Path,
) -> None:
    """Write HSB results to CSV file."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "batch_size", "seq_len", "max_new_tokens",
            "inference_time_ms", "gpu_busy_time_ms", "gpu_idle_time_ms",
            "t_exposed_us", "t_structural_us", "hsb",
            "hsb_classification", "num_kernels", "status"
        ])
        
        for r in sorted(results, key=lambda x: (x.batch_size, x.seq_len)):
            writer.writerow([
                r.batch_size, r.seq_len, r.max_new_tokens,
                r.inference_time_ms, r.gpu_busy_time_ms, r.gpu_idle_time_ms,
                r.t_exposed_us, r.t_structural_us, r.hsb,
                classify_hsb(r.hsb) if r.hsb is not None else "unknown",
                r.num_kernels, r.status
            ])
    
    print(f"Saved HSB CSV to: {output_path}")


def summarize_hsb_sweep(root: Path, gpu_name: Optional[str] = None) -> None:
    """
    Main summarization function for HSB sweep.
    
    Args:
        root: Path to HSB sweep directory
        gpu_name: GPU name for plot titles
    """
    print(f"\n=== Summarizing HSB sweep: {root} ===")
    
    # Load results
    results = load_hsb_results(root)
    
    if not results:
        print(f"No HSB results found in {root}")
        return
    
    print(f"Loaded {len(results)} HSB results")
    
    # Create summary directory
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    # Build title suffix
    title_parts = []
    
    # Try to extract model name from sweep summary
    summary_file = root / "hsb_sweep_summary.json"
    if summary_file.exists():
        with open(summary_file, "r") as f:
            summary = json.load(f)
        model_name = summary.get("model_name", "")
        precision = summary.get("precision", "")
        if model_name:
            short_model = model_name.split("/")[-1]
            title_parts.append(short_model)
        if precision:
            title_parts.append(precision)
    
    if gpu_name:
        title_parts.append(gpu_name)
    
    title_suffix = " | ".join(title_parts) if title_parts else ""
    
    # Generate visualizations
    plot_hsb_heatmap(
        results,
        summary_dir / "hsb_heatmap.png",
        title_suffix=title_suffix,
    )
    
    plot_t_comparison(
        results,
        summary_dir / "t_comparison.png",
        title_suffix=title_suffix,
    )
    
    # Write CSV summary
    write_hsb_csv(results, summary_dir / "hsb_summary.csv")
    
    # Print statistics
    valid_results = [r for r in results if r.hsb is not None]
    if valid_results:
        hsb_values = [r.hsb for r in valid_results]
        print(f"\nHSB Statistics:")
        print(f"  Mean:   {np.mean(hsb_values):.4f}")
        print(f"  Median: {np.median(hsb_values):.4f}")
        print(f"  Min:    {np.min(hsb_values):.4f}")
        print(f"  Max:    {np.max(hsb_values):.4f}")
        print(f"  Std:    {np.std(hsb_values):.4f}")
        
        # Classification breakdown
        hw_bound = sum(1 for h in hsb_values if h >= 0.5)
        balanced = sum(1 for h in hsb_values if 0 <= h < 0.5)
        fw_bound = sum(1 for h in hsb_values if h < 0)
        
        print(f"\nClassification:")
        print(f"  Hardware-bound (HSB ≥ 0.5):  {hw_bound} ({100*hw_bound/len(valid_results):.1f}%)")
        print(f"  Balanced (0 ≤ HSB < 0.5):    {balanced} ({100*balanced/len(valid_results):.1f}%)")
        print(f"  Framework-bound (HSB < 0):   {fw_bound} ({100*fw_bound/len(valid_results):.1f}%)")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    
    if not root.exists():
        print(f"Error: Directory not found: {root}", file=sys.stderr)
        return 1
    
    try:
        summarize_hsb_sweep(root, args.gpu_name)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())