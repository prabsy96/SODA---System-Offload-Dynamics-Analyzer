from typing import Dict, Any

from soda.common import utils
from soda.common import print_utils
from soda.common.data import Sequence

from soda.microbench.framework.pytorch.profile import profile_pytorch_gemm_sequences
from soda.microbench.framework.pytorch.plot import plot_pytorch_gemm_sequences
from soda.microbench.framework.pytorch.verify import compare_sequences

from soda.microbench.baremetal.generate import generate_jobs
from soda.microbench.baremetal.search import search_cublas_algos_offline
from soda.microbench.baremetal.profile import profile_baremetal_gemm_kernels
from soda.microbench.baremetal.report import report

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
        # Calculate launch/xlat taxes for event sequences
        sequences = utils.calculate_sequence_metrics(list(self.tracer.sequences), metrics=["launch_tax", "xlat_tax"])

        # Save all sequences
        all_sequences_file = utils.get_path("ALL_SEQUENCES")
        all_sequences_data = {
            "summary": {"count": len(sequences)},
            "sequences": sequences
        }
        utils.save_json(all_sequences_file, all_sequences_data)
        print(f"Saved {all_sequences_data['summary']['count']} sequences to {all_sequences_file}")

        # Filter and save gemm sequences
        gemm_sequences = utils.filter_gemm_sequences(sequences)
        all_gemm_sequences_file = utils.get_path("ALL_GEMM_SEQUENCES")
        all_gemm_sequences_data = {
            "summary": {"count": len(gemm_sequences)},
            "sequences": gemm_sequences
        }
        utils.save_json(all_gemm_sequences_file, all_gemm_sequences_data)
        print(f"Saved {all_gemm_sequences_data['summary']['count']} GEMM sequences to {all_gemm_sequences_file}")
        
        grouped_seqs_by_id_dict = utils.group_sequences_by_identity(gemm_sequences)
        unique_gemm_sequences = utils.aggregate_sequences(
            grouped_seqs_by_id_dict,
            metrics=["launch_tax", "xlat_tax"],
            event_types=["kernel", "aten_op", "cuda_launch"],
        )
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
        if not self.args.skip_pytorch_profile:
            section = "Profile PyTorch GEMM Kernels"
            print_utils.section_start(section)
            pytorch_gemm_sequences = profile_pytorch_gemm_sequences(
                target_gemm_sequences,
                warmup=self.warmup,
                runs=self.runs
            )
            print_utils.section_end(section)
            
            section = "Verify PyTorch GEMM Sequences"
            print_utils.section_start(section)
            # Convert dict sequences to Sequence objects
            target_seq_objects = [Sequence.from_dict(seq_dict) for seq_dict in target_gemm_sequences["sequences"]]
            pytorch_seq_objects = [Sequence.from_dict(seq_dict) for seq_dict in pytorch_gemm_sequences["sequences"]]
            compare_sequences(target_seq_objects, pytorch_seq_objects, title="Pytorch vs Target")
            plot_pytorch_gemm_sequences(pytorch_gemm_sequences)
            print_utils.section_end(section)
        else:
            section = "Profile PyTorch GEMM Kernels (skipped)"
            print_utils.section_start(section)
            print("Skipping PyTorch GEMM kernel profiling (--skip-pytorch-profile).")
            print_utils.section_end(section)

        # Generate baremetal jobs
        section = "Generate Baremetal Jobs"
        print_utils.section_start(section)
        generate_jobs(target_gemm_sequences, warmup=self.warmup, runs=self.runs)
        print_utils.section_end(section)

        # Search for cuBLASLt algorithms (optional)
        if not self.args.skip_offline_cublas_algo_search:
            section = "Offline Search for cuBLASLt Algorithms"
            print_utils.section_start(section)
            search_cublas_algos_offline()
            print_utils.section_end(section)
        else:
            section = "Offline Search for cuBLASLt Algorithms (skipped)"
            print_utils.section_start(section)
            print("Skipping offline cuBLASLt algorithm search (--skip-offline-cublas-algo-search).")
            print_utils.section_end(section)
        
        # Profile baremetal performance
        if not self.args.skip_baremetal_profile:
            section = "Profile Baremetal GEMM Kernels"
            print_utils.section_start(section)
            print("This will run nsys profiling for multiple jobs, may take several minutes")
            baremetal_gemm_sequences = profile_baremetal_gemm_kernels(
                skip_offline_cublas_algo_search=self.args.skip_offline_cublas_algo_search
            )
            print_utils.section_end(section)
            
            # Verify baremetal sequences
            section = "Verify Baremetal GEMM Sequences"
            print_utils.section_start(section)
            # Align: target[i] -> baremetal[i+1] (skip null kernel at index 0)
            # Convert to Sequence objects (None stays None for skipped jobs)
            target_seq_objects = [Sequence.from_dict(seq_dict) for seq_dict in target_gemm_sequences["sequences"]]
            baremetal_seq_objects = [Sequence.from_dict(seq_dict) 
                                    for seq_dict in baremetal_gemm_sequences["sequences"][1:]]  # Skip null kernel
            compare_sequences(target_seq_objects, baremetal_seq_objects, title="Baremetal vs Target", full=False)
            print_utils.section_end(section)
        else:
            section = "Profile Baremetal GEMM Kernels (skipped)"
            print_utils.section_start(section)
            print("Skipping baremetal GEMM kernel profiling (--skip-baremetal-profile).")
            print_utils.section_end(section)
        
        # Compare PyTorch vs Baremetal
        section = "Report PyTorch vs Baremetal"
        print_utils.section_start(section)
        report()
        print_utils.section_end(section)
