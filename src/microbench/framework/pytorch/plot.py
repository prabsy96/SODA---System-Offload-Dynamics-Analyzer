from pathlib import Path
from typing import Dict, Any, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from soda.common import utils

def plot_pytorch_gemm_sequences(pytorch_gemm_sequences: Dict[str, Any]) -> None:
    """
    Plot launch tax graphs from profiled PyTorch GEMM sequences.
    """
    def plot_launch_tax(values: List[float], title: str, output_file: str) -> None:
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(values) + 1), values, marker="o", markersize=0.3, linewidth=0.8, label="Launch Tax")
        if values:
            avg = sum(values) / len(values)
            plt.axhline(avg, color="red", linestyle="--", label=f"avg={avg:.3f} us")
        plt.xlabel("run")
        plt.ylabel("Launch Tax (us)")
        plt.title(title)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        
    sequences = pytorch_gemm_sequences["sequences"]

    graphs_dir = utils.get_path("PYTORCH_LAUNCH_TAX_GRAPHS")
    utils.ensure_dir(graphs_dir)

    for idx, sequence in enumerate(sequences, start=1):
        kernel = sequence["kernel"]
        cpu_op = sequence["cpu_op"]
        launch_tax = sequence["launch_tax"]
        kernel_name = kernel["name"]
        op_name = cpu_op["name"]
        launch_tax_values = launch_tax["all"] if launch_tax else []

        if not launch_tax_values:
            # Nothing to plot
            continue

        graph_file_name= utils.format_sequence_filename(idx, op_name, kernel_name, extension="png")
        graph_file = graphs_dir / graph_file_name
        plot_launch_tax(
            values=launch_tax_values,
            title=f"{op_name} -> {kernel_name}",
            output_file=str(graph_file)
        )
