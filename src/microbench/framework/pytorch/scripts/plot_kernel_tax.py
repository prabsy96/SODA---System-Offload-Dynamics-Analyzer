import os
import json
import argparse
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from verify_replayed_kernels import get_clean_kernel_name


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def format_filename(index: int, op_name: str, kernel_name: str) -> str:
    op_short = op_name.replace("::", "_")
    kernel_short = get_clean_kernel_name(kernel_name).strip()
    return f"{index:02d}_{op_short}_{kernel_short}.png"


def plot_kernel_tax(values: List[float], title: str, out_path: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(values) + 1), values, marker="o", markersize=0.3, linewidth=0.8, label="kernel_tax_us")
    if values:
        avg = sum(values) / len(values)
        plt.axhline(avg, color="red", linestyle="--", label=f"avg={avg:.3f} us")
    plt.xlabel("run")
    plt.ylabel("kernel_tax_us")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main(input_file: str, output_dir: str) -> None:
    # Resolve file paths
    pytorch_output = os.environ.get("PYTORCH_OUTPUT", "output")
    if not (os.path.isabs(input_file) or os.path.exists(input_file)):
        input_file = os.path.join(pytorch_output, input_file)

    with open(input_file, "r") as f:
        data = json.load(f)

    chains = data.get("causal_chains", [])

    graphs_dir = os.path.join(pytorch_output, "graphs", "kernel_tax") if output_dir is None else output_dir
    ensure_dir(graphs_dir)

    for idx, chain in enumerate(chains, start=1):
        kernel = chain.get("kernel") or {}
        cpu_op = chain.get("cpu_op") or {}
        meta = chain.get("meta") or {}
        kernel_name = kernel.get("name", "unknown")
        op_name = cpu_op.get("name", "unknown")
        values = meta.get("all_kernel_tax_us") or []

        if not values:
            # Nothing to plot
            continue

        filename = format_filename(idx, op_name, kernel_name)
        out_path = os.path.join(graphs_dir, filename)
        plot_kernel_tax(values, f"{op_name} -> {get_clean_kernel_name(kernel_name)}", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot kernel tax across N runs for replayed GEMM kernels.")
    parser.add_argument("input_file", help="Input JSON file (e.g., output/replayed_gemm_kernel_chains.json)")
    parser.add_argument("--out", default=None, help="Output directory for PNGs (default: output/graphs/kernel_tax)")
    args = parser.parse_args()

    main(args.input_file, args.out)

