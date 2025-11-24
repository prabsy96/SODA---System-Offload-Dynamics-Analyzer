from typing import Dict, Any
from soda import utils
import sys
from pathlib import Path
# Add src to path for common imports
sys.path.insert(0, str(Path(__file__).parent))
from common import print_utils

from microbench.framework.pytorch.profile import profile_pytorch_gemm_sequences
from microbench.framework.pytorch.plot import plot_pytorch_gemm_sequences
from microbench.framework.pytorch.verify import compare_sequences

from microbench.baremetal.generate import generate_jobs
from microbench.baremetal.search import search_algorithm_indices
from microbench.baremetal.profile import profile_baremetal_gemm_kernels
from microbench.baremetal.report import compare

class SodaMicrobench:
    
    def __init__(self, tracer: 'ModelTracer', args):
        self.tracer = tracer
        self.args = args
        self.warmup = args.warmup
        self.runs = args.runs
    
    def extract_unique_gemm_sequences(self) -> Dict[str, Any]:
        """
        Main pipeline: extract -> save, filter -> save, deduplicate -> save.
        
        Returns:
            Dictionary with unique GEMM sequences data.
        """
        # Calculate kernel tax for event sequences
        event_sequences = utils.calculate_per_seq_launch_tax(list(self.tracer.event_sequences))

        # Save all sequences
        all_sequences_file = utils.get_path("ALL_SEQUENCES")
        all_sequences_data = {
            "summary": {"count": len(event_sequences)},
            "sequences": event_sequences
        }
        utils.save_json(all_sequences_file, all_sequences_data)
        print(f"Saved {all_sequences_data['summary']['count']} sequences to {all_sequences_file}")

        # Filter and save gemm sequences
        gemm_sequences = utils.filter_gemm_sequences(event_sequences)
        all_gemm_sequences_file = utils.get_path("ALL_GEMM_SEQUENCES")
        all_gemm_sequences_data = {
            "summary": {"count": len(gemm_sequences)},
            "sequences": gemm_sequences
        }
        utils.save_json(all_gemm_sequences_file, all_gemm_sequences_data)
        print(f"Saved {all_gemm_sequences_data['summary']['count']} GEMM sequences to {all_gemm_sequences_file}")
        
        unique_gemm_sequences = utils.deduplicate_and_aggregate(gemm_sequences)
        unique_gemm_sequences_file = utils.get_path("UNIQUE_GEMM_SEQUENCES")
        target_gemm_sequences_data = {
            "summary": {
                "unique_count": len(unique_gemm_sequences),
                "total_count": len(gemm_sequences),
                "deduplication_ratio": f"{len(unique_gemm_sequences)}/{len(gemm_sequences)}"
            },
            "sequences": unique_gemm_sequences
        }
        utils.save_json(unique_gemm_sequences_file, target_gemm_sequences_data)
        print(f"Saved {target_gemm_sequences_data['summary']['unique_count']} unique target GEMM sequences to {unique_gemm_sequences_file}")
        print(f"\t* Uniqueness key: kernel_name + grid + block + shared_memory + input_dims")
        
        return target_gemm_sequences_data

    def run(self) -> None:
        """
        Run the complete microbenchmarking pipeline
        """
        # Extract target GEMM sequences
        target_gemm_sequences = self.extract_unique_gemm_sequences()

        # Benchmark pytorch performance
        pytorch_gemm_sequences = profile_pytorch_gemm_sequences(
            target_gemm_sequences,
            warmup=self.warmup,
            runs=self.runs
        )
        
        print_utils.subsection("Verifying pytorch gemm sequences", level=0)
        compare_sequences(target_gemm_sequences, pytorch_gemm_sequences)
        plot_pytorch_gemm_sequences(pytorch_gemm_sequences)

        # Benchmark baremetal performance
        print_utils.subsection("Generate Baremetal Jobs", level=0)
        generate_jobs(target_gemm_sequences, warmup=self.warmup, runs=self.runs)
        
        print_utils.subsection("Offline Search for Algorithm Indices", level=0)
        print("Searching for cuBLASLt algorithm indices for each job...")
        search_algorithm_indices()
        
        print_utils.subsection("Profile Baremetal (under nsys profiling)", level=0)
        print("This will run nsys profiling for multiple jobs, may take several minutes...")
        profile_baremetal_gemm_kernels()
        
        print_utils.subsection("Compare PyTorch vs Baremetal", level=0)
        compare()
        
        # Print summary
        jobs_file = utils.get_path("BAREMETAL_JOBS")
        baremetal_gemm_kernels_file = utils.get_path("BAREMETAL_GEMM_KERNELS")
        report_file = utils.get_path("FINAL_REPORT")
        
        print("==============================================")
        print("Done! Check results in:")
        print(f"  - {jobs_file}")
        print(f"  - {baremetal_gemm_kernels_file}")
        print(f"  - {report_file}")
        print("==============================================")