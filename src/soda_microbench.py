from pathlib import Path
from typing import Dict, Any, List
import os
import sys
import torch
from torch.profiler import profile, ProfilerActivity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from soda import utils
from microbench.framework.pytorch.verify import verify_event_sequences, print_summary
from microbench.framework.pytorch.replay import replay_sequences_from_cpu_ops

# Add baremetal scripts to path for imports
_baremetal_scripts_dir = Path(__file__).parent / "microbench" / "baremetal" / "scripts"
if str(_baremetal_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_baremetal_scripts_dir))

from gen_bm_jobs import generate_jobs
from search_algorithm_indices import search_algorithm_indices
from profile_baremetal import run_profiling
from compare_kernel_tax import entry_point as compare_entry

class SodaMicrobench:
    
    def __init__(self, tracer: 'ModelTracer', args):
        self.tracer = tracer
        self.args = args
        # Experiment directory is the base for all paths
        self.experiment_dir = tracer.output_dir
    
    def extract_unique_gemm_sequences(self) -> Dict[str, Any]:
        """
        Main pipeline: extract -> save, filter -> save, deduplicate -> save.
        
        Returns:
            Dictionary with unique GEMM sequences data.
        """
        # Calculate kernel tax for event sequences
        event_sequences = utils.calculate_per_seq_launch_tax(list(self.tracer.event_sequences))

        # Save all sequences
        all_sequences_file = utils.get_path("ALL_SEQUENCES", base_path=self.experiment_dir)
        all_sequences_data = {
            "summary": {"count": len(event_sequences)},
            "sequences": event_sequences
        }
        utils.save_json(all_sequences_file, all_sequences_data)
        print(f"Saved {all_sequences_data['summary']['count']} sequences to {all_sequences_file}")

        # Filter and save gemm sequences
        gemm_sequences = utils.filter_gemm_sequences(event_sequences)
        all_gemm_sequences_file = utils.get_path("ALL_GEMM_SEQUENCES", base_path=self.experiment_dir)
        all_gemm_sequences_data = {
            "summary": {"count": len(gemm_sequences)},
            "sequences": gemm_sequences
        }
        utils.save_json(all_gemm_sequences_file, all_gemm_sequences_data)
        print(f"Saved {all_gemm_sequences_data['summary']['count']} GEMM sequences to {all_gemm_sequences_file}")
        
        unique_gemm_sequences = utils.deduplicate_and_aggregate(gemm_sequences)
        unique_gemm_sequences_file = utils.get_path("UNIQUE_GEMM_SEQUENCES", base_path=self.experiment_dir)
        unique_gemm_sequences_data = {
            "summary": {
                "unique_count": len(unique_gemm_sequences),
                "total_count": len(gemm_sequences),
                "deduplication_ratio": f"{len(unique_gemm_sequences)}/{len(gemm_sequences)}"
            },
            "sequences": unique_gemm_sequences
        }
        utils.save_json(unique_gemm_sequences_file, unique_gemm_sequences_data)
        print(f"Saved {unique_gemm_sequences_data['summary']['unique_count']} unique GEMM sequences to {unique_gemm_sequences_file}")
        print(f"\t* Uniqueness key: kernel_name + grid + block + shared_memory + input_dims")
        
        return unique_gemm_sequences_data

    def replay_gemm_sequences(self, unique_gemm_sequences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main pipeline: setup -> replay all -> save.
        
        Args:
            unique_gemm_sequences: Dictionary with unique GEMM sequences data.
        
        Returns:
            Dictionary with replayed GEMM sequences data (same format as saved JSON).
        """
        # Clean previous kernel traces 
        kernel_traces_dir = utils.get_path("PYTORCH_TRACES", base_path=self.experiment_dir)
        utils.ensure_dir(kernel_traces_dir, cleanup=True)
        
        # Replay all event sequences from cpu ops
        replayed_sequences = replay_sequences_from_cpu_ops(
            unique_gemm_sequences["sequences"],
            self.experiment_dir,
            warmup=self.args.warmup, 
            runs=self.args.runs
        )
        
        # Some cpu ops can produce non GEMM kernels as side effects, filter them out
        replayed_gemm_sequences = utils.filter_gemm_sequences(replayed_sequences)

        replayed_gemm_sequences_file = utils.get_path("PYTORCH_REPLAYED_GEMM_SEQUENCES", base_path=self.experiment_dir)
        replayed_gemm_sequences_data = {
            "summary": {"count": len(replayed_gemm_sequences)},
            "sequences": replayed_gemm_sequences
        }
        utils.save_json(replayed_gemm_sequences_file, replayed_gemm_sequences_data)
        return replayed_gemm_sequences_data

    def plot_replayed_sequences(self, replayed_gemm_sequences: Dict[str, Any]) -> None:
        """
        Plot kernel tax graphs from replayed sequences.
        """
        def plot_kernel_tax(values: List[float], title: str, output_file: str) -> None:
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
            plt.savefig(output_file, dpi=150)
            plt.close()
            
        sequences = replayed_gemm_sequences.get("sequences", [])

        graphs_dir = utils.get_path("PYTORCH_KERNEL_TAX_GRAPHS", base_path=self.experiment_dir)
        utils.ensure_dir(graphs_dir)

        for idx, sequence in enumerate(sequences, start=1):
            kernel = sequence.get("kernel", {})
            cpu_op = sequence.get("cpu_op", {})
            meta = sequence.get("meta", {})
            kernel_name = kernel.get("name", "unknown")
            op_name = cpu_op.get("name", "unknown")
            kernel_tax_values = meta.get("all_kernel_tax", [])

            if not kernel_tax_values:
                # Nothing to plot
                continue

            graph_file_name= utils.format_sequence_filename(idx, op_name, kernel_name, extension="png")
            graph_file = graphs_dir / graph_file_name
            plot_kernel_tax(
                values=kernel_tax_values,
                title=f"{op_name} -> {kernel_name}",
                output_file=str(graph_file)
            )

    def run(self) -> None:
        """
        Run the complete microbenchmarking pipeline
        """
        # Extract unique GEMM sequences
        unique_gemm_sequences = self.extract_unique_gemm_sequences()

        # Benchmark pytorch performance
        replayed_gemm_sequences = self.replay_gemm_sequences(unique_gemm_sequences)
        self.verify_replayed_sequences(unique_gemm_sequences, replayed_gemm_sequences)
        self.plot_replayed_sequences(replayed_gemm_sequences)

        # Benchmark baremetal performance
        utils.print_subsection("Generate Baremetal Jobs")
        jobs_file = utils.get_path("BAREMETAL_JOBS", base_path=self.experiment_dir)
        generate_jobs(unique_gemm_sequences, jobs_file, warmup=self.args.warmup, runs=self.args.runs)
        
        utils.print_subsection("Offline Search for Algorithm Indices")
        print("Searching for cuBLASLt algorithm indices for each job...")
        utils.ensure_file(jobs_file)
        search_algorithm_indices(jobs_file)
        
        utils.print_subsection("Profile Baremetal (under nsys profiling)")
        print("This will run nsys profiling for multiple jobs, may take several minutes...")
        output_file = utils.get_path("BAREMETAL_RUNS", base_path=self.experiment_dir)
        run_profiling(jobs_file, output_file)
        print()
        
        utils.print_subsection("Compare PyTorch vs Baremetal")
        compare_entry(self.experiment_dir)
        print()
        
        # Get output paths for summary
        jobs_file = utils.get_path("BAREMETAL_JOBS", base_path=self.experiment_dir)
        runs_file = utils.get_path("BAREMETAL_RUNS", base_path=self.experiment_dir)
        report_file = utils.get_path("BAREMETAL_REPORT", base_path=self.experiment_dir)
        
        print("==============================================")
        print("Done! Check results in:")
        print(f"  - {jobs_file}")
        print(f"  - {runs_file}")
        print(f"  - {report_file}")
        print("==============================================")
    
    def verify_replayed_sequences(self, unique_gemm_sequences: Dict[str, Any], replayed_gemm_sequences: Dict[str, Any]) -> None:
        """
        Main pipeline: verify -> save results.
        
        Args:
            unique_gemm_sequences: Dictionary with unique GEMM sequences data.
            replayed_gemm_sequences: Dictionary with replayed GEMM sequences data.
        """
        # Setup logging
        # Redirect print to both stdout and log file
        log_path = utils.get_path("PYTORCH_VERIFY_LOG", base_path=self.experiment_dir)
        utils.ensure_dir(log_path.parent)
        output_file = open(log_path, "w")
        
        import builtins
        original_print = builtins.print
        
        def print_and_write(*args, **kwargs):
            """Print to stdout and write to file."""
            original_print(*args, **kwargs)
            line = ' '.join(str(arg) for arg in args)
            output_file.write(line + '\n')
            output_file.flush()
        
        builtins.print = print_and_write
        
        # Step 3: Verify event sequences
        print("=" * 80)
        print("Event sequence verification: original vs replayed event sequences")
        print("=" * 80)
        
        print("\n1. Metadata verification:")
        print(f"\tOriginal event sequences:\t{len(unique_gemm_sequences['sequences'])} sequences")
        print(f"\tReplayed event sequences:\t{len(replayed_gemm_sequences['sequences'])} sequences")
        
        print("\n2. Event sequence-by-sequence verification:")
        print("-" * 80)
        
        matches, partial_matches, mismatches = verify_event_sequences(
            unique_gemm_sequences, replayed_gemm_sequences
        )
        
        # Step 4: Print summary
        print_summary(matches, partial_matches, mismatches)
        
        # Restore original print and close file
        builtins.print = original_print
        output_file.close()
        print(f"\nVerification output saved to {log_path}")