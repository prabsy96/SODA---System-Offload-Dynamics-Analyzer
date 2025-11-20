import os
import json
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from soda import SodaProfiler


def format_filename(index: int, op_name: str, kernel_name: str) -> str:
    op_short = op_name.replace("::", "_")
    kernel_short = SodaProfiler.get_clean_kernel_name(kernel_name).strip()
    return f"{index:02d}_{op_short}_{kernel_short}.png"


def plot_kernel_tax(values: List[float], title: str, out_path: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(values) + 1), values, marker="o", markersize=0.3, linewidth=0.8, label="Kernel Tax")
    if values:
        avg = sum(values) / len(values)
        plt.axhline(avg, color="red", linestyle="--", label=f"avg={avg:.3f} us")
    plt.xlabel("run")
    plt.ylabel("Kernel Tax (us)")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    # Load from env var
    input_file = SodaProfiler.get_path("PYTORCH_REPLAYED_KERNELS")
    data = SodaProfiler.load_json(input_file)

    sequences = data.get("sequences", [])

    graphs_dir = SodaProfiler.get_path("PYTORCH_KERNEL_TAX_GRAPHS")
    SodaProfiler.ensure_dir(graphs_dir)

    for idx, sequence in enumerate(sequences, start=1):
        kernel = sequence.get("kernel") or {}
        cpu_op = sequence.get("cpu_op") or {}
        meta = sequence.get("meta") or {}
        kernel_name = kernel.get("name", "unknown")
        op_name = cpu_op.get("name", "unknown")
        values = meta.get("all_kernel_tax") or []

        if not values:
            # Nothing to plot
            continue

        filename = format_filename(idx, op_name, kernel_name)
        out_path = graphs_dir / filename
        plot_kernel_tax(values, f"{op_name} -> {SodaProfiler.get_clean_kernel_name(kernel_name)}", str(out_path))


if __name__ == "__main__":
    main()

