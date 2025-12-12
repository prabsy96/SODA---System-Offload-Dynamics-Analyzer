from typing import Dict, Any
from pathlib import Path
from collections import Counter

from soda.common import utils
from soda.common import print_utils
from soda.common.data import Sequence

from soda.microbench.framework.pytorch.profile import (
    profile_pytorch_gemm_sequences,
    profile_pytorch_all_sequences,
)
from soda.microbench.framework.pytorch.plot import plot_pytorch_gemm_sequences
from soda.microbench.framework.pytorch.verify import compare_sequences

from soda.microbench.baremetal.generate import generate_jobs
from soda.microbench.baremetal.search import search_cublas_algos_offline
from soda.microbench.baremetal.profile import profile_baremetal_gemm_kernels
from soda.microbench.baremetal.report import summarize

def _nested_to_tuple(obj):
    """Convert nested lists to nested tuples for hashability."""
    if isinstance(obj, list):
        return tuple(_nested_to_tuple(item) for item in obj)
    return obj

def _normalize_scalar_value(value, dtype_context=None):
    """
    Normalize a scalar value for consistent comparison.
    """
    if value == "" or value is None:
        return ""
    
    sv = str(value).strip().lower()
    
    # Normalize bool-like values when in bool context
    if dtype_context == "bool":
        if sv in ("false", "0", "0.0"):
            return "False"
        if sv in ("true", "1", "1.0"):
            return "True"
    
    return str(value)


def _normalize_concrete_inputs(concrete_inputs, input_types):
    """
    Normalize concrete_inputs for comparison.
    """
    if concrete_inputs is None:
        return []
    
    if input_types is None:
        input_types = []
    
    # Determine if this is a bool context (first tensor arg is bool)
    is_bool_context = len(input_types) > 0 and input_types[0] == "bool"
    
    out = []
    for i, v in enumerate(concrete_inputs):
        t = input_types[i] if i < len(input_types) else None
        
        if v == "" or v is None:
            out.append("")
            continue
        
        # Normalize Scalar values in bool context
        if t == "Scalar" and is_bool_context:
            out.append(_normalize_scalar_value(v, "bool"))
            continue
        
        out.append(str(v))
    
    return tuple(out)


def _make_sequence_key(seq_dict):
    """
    Create a hashable key for matching sequences between target and profiled.
    
    Includes: name, input_dims, input_type, normalized concrete_inputs
    """
    aten = seq_dict.get("aten_op", {})
    return (
        aten.get("name"),
        _nested_to_tuple(aten.get("input_dims", [])),
        _nested_to_tuple(aten.get("input_type", [])),
        _normalize_concrete_inputs(
            aten.get("concrete_inputs", []),
            aten.get("input_type", []),
        ),
    )
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
        sequences = utils.calculate_sequence_metrics(list(self.tracer.sequences), metrics=["launch_tax", "aten_xlat_tax"])

        print("\n[DEBUG] ATen Op Distribution in Trace (Pre-Filter):")
        from collections import Counter
        op_counts = Counter()
        for s in sequences:
            if s.get("aten_op") and "name" in s["aten_op"]:
                op_counts[s["aten_op"]["name"]] += 1
            else:
                op_counts["<unknown/null>"] += 1
        
        for op, count in op_counts.most_common():
            print(f"  {op}: {count}")
        print("---------------------------------------------------\n")

        # Save all sequences
        all_sequences_file = utils.get_path("ALL_SEQUENCES")
        all_sequences_data = {
            "summary": {"count": len(sequences)},
            "sequences": sequences
        }
        # ============================================================
        # PATH 1: All kernel sequences (for complete T_structural)
        # ============================================================
        all_kernel_sequences = utils.filter_kernel_sequences(sequences)
        
        gemm_count = sum(1 for s in all_kernel_sequences if s.get("is_gemm", False))
        non_gemm_count = len(all_kernel_sequences) - gemm_count

        all_kernel_sequences_file = utils.get_path("ALL_KERNEL_SEQUENCES")
        all_kernel_data = {
            "summary": {
                "count": len(all_kernel_sequences),
                "gemm_count": gemm_count,
                "non_gemm_count": non_gemm_count,
            },
            "sequences": all_kernel_sequences
        }
        utils.save_json(all_kernel_sequences_file, all_kernel_data)
        print(f"Saved {len(all_kernel_sequences)} kernel sequences to {all_kernel_sequences_file}")
        print(f"  - {gemm_count} GEMM sequences")
        print(f"  - {non_gemm_count} non-GEMM sequences")

        # Deduplicate all kernel sequences
        grouped_all_seqs = utils.group_sequences_by_identity(all_kernel_sequences)
        unique_all_sequences = utils.aggregate_sequences(
            grouped_all_seqs,
            metrics=["launch_tax", "aten_xlat_tax"],
            event_types=["kernel", "aten_op", "cuda_launch"],
        )
        
        # Preserve is_gemm classification after aggregation
        for seq in unique_all_sequences:
            aten_name = seq.get("aten_op", {}).get("name", "")
            seq["is_gemm"] = utils.is_gemm_op(aten_name)
        
        unique_gemm_count = sum(1 for s in unique_all_sequences if s.get("is_gemm", False))
        unique_non_gemm_count = len(unique_all_sequences) - unique_gemm_count
        
        unique_all_sequences_file = utils.get_path("UNIQUE_ALL_SEQUENCES")
        unique_all_data = {
            "summary": {
                "unique_count": len(unique_all_sequences),
                "total_count": len(all_kernel_sequences),
                "unique_gemm": unique_gemm_count,
                "unique_non_gemm": unique_non_gemm_count,
                "deduplication_ratio": f"{len(unique_all_sequences)}/{len(all_kernel_sequences)}"
            },
            "sequences": unique_all_sequences
        }
        utils.save_json(unique_all_sequences_file, unique_all_data)
        print(f"Saved {unique_all_data['summary']['unique_count']} unique kernel sequences to {unique_all_sequences_file}")
        print(f"  - {unique_gemm_count} unique GEMM")
        print(f"  - {unique_non_gemm_count} unique non-GEMM")
        print(f"\t* Uniqueness key: kernel_name + grid + block + shared_memory + input_dims")

        # ============================================================
        # PATH 2: GEMM-only sequences (for baremetal comparison)
        # ============================================================
        gemm_sequences = [s for s in all_kernel_sequences if s.get("is_gemm", False)]
        all_gemm_sequences_file = utils.get_path("ALL_GEMM_SEQUENCES")
        all_gemm_sequences_data = {
            "summary": {"count": len(gemm_sequences)},
            "sequences": gemm_sequences
        }
        utils.save_json(all_gemm_sequences_file, all_gemm_sequences_data)
        print(f"Saved {all_gemm_sequences_data['summary']['count']} GEMM sequences to {all_gemm_sequences_file}")
        
        # Deduplicate GEMM sequences
        if gemm_sequences:
            grouped_gemm_seqs = utils.group_sequences_by_identity(gemm_sequences)
            unique_gemm_sequences = utils.aggregate_sequences(
                grouped_gemm_seqs,
                metrics=["launch_tax", "aten_xlat_tax"],
                event_types=["kernel", "aten_op", "cuda_launch"],
            )
            for seq in unique_gemm_sequences:
                seq["is_gemm"] = True
        else:
            unique_gemm_sequences = []
        
        unique_gemm_sequences_file = utils.get_path("UNIQUE_GEMM_SEQUENCES")
        unique_gemm_data = {
            "summary": {
                "unique_count": len(unique_gemm_sequences),
                "total_count": len(gemm_sequences),
                "deduplication_ratio": f"{len(unique_gemm_sequences)}/{len(gemm_sequences)}"
            },
            "sequences": unique_gemm_sequences
        }
        utils.save_json(unique_gemm_sequences_file, unique_gemm_data)
        print(f"Saved {unique_gemm_data['summary']['unique_count']} unique GEMM sequences to {unique_gemm_sequences_file}")
        
        return {
            "unique_gemm_sequences": unique_gemm_data,
            "unique_all_sequences": unique_all_data,
        }

    def run(self) -> None:
        """
        Run the complete microbenchmarking pipeline
        """
        # Extract target GEMM sequences
        extraction_results = self.extract_unique_gemm_sequences()  
        target_gemm_sequences = extraction_results["unique_gemm_sequences"]
        target_all_sequences = extraction_results["unique_all_sequences"]

        # ============================================================
        # Profile ALL PyTorch kernels (GEMM + non-GEMM)
        # ============================================================
        if not self.args.skip_pytorch_profile:
            section = "Profile PyTorch Kernels (All Types)"
            print_utils.section_start(section)
            
            # Profile all unique kernel sequences
            pytorch_all_sequences = profile_pytorch_all_sequences(
                target_all_sequences,
                warmup=self.warmup,
                runs=self.runs,
            )
            print_utils.section_end(section)
            
            section = "Verify PyTorch Sequences"
            print_utils.section_start(section)
            
            if target_all_sequences["sequences"]:
                # Build lookup table from pytorch sequences
                pytorch_by_key = {}
                for seq in pytorch_all_sequences["sequences"]:
                    key = _make_sequence_key(seq)
                    pytorch_by_key[key] = seq
                
                # Debug: print LUT stats
                print(f"[DEBUG] PyTorch LUT: {len(pytorch_by_key)} unique keys from {len(pytorch_all_sequences['sequences'])} sequences")
                
                # Match target sequences to profiled pytorch sequences
                target_seq_objects = []
                pytorch_seq_objects = []
                unmatched_targets = []
                
                for seq_dict in target_all_sequences["sequences"]:
                    key = _make_sequence_key(seq_dict)
                    
                    if key in pytorch_by_key:
                        target_seq_objects.append(Sequence.from_dict(seq_dict))
                        pytorch_seq_objects.append(Sequence.from_dict(pytorch_by_key[key]))
                    else:
                        unmatched_targets.append(seq_dict)
                
                # Report matching stats
                print(f"[DEBUG] Matched: {len(target_seq_objects)}/{len(target_all_sequences['sequences'])} sequences")
                if unmatched_targets:
                    print(f"[DEBUG] Unmatched targets ({len(unmatched_targets)}):")
                    for seq in unmatched_targets[:5]:  # Show first 5
                        aten = seq.get("aten_op", {})
                        print(f"  - {aten.get('name')} dims={aten.get('input_dims')} types={aten.get('input_type')} concrete={aten.get('concrete_inputs')}")
                    if len(unmatched_targets) > 5:
                        print(f"  ... and {len(unmatched_targets) - 5} more")
                
                if target_seq_objects:
                    compare_sequences(target_seq_objects, pytorch_seq_objects, title="PyTorch vs Target (All)")
                else:
                    print("No matched sequences to verify.")
                    
                # Plot GEMM sequences if available
                gemm_pytorch_seqs = [s for s in pytorch_all_sequences["sequences"] if s.get("is_gemm", False)]
                if gemm_pytorch_seqs:
                    plot_pytorch_gemm_sequences({
                        "summary": {"count": len(gemm_pytorch_seqs)},
                        "sequences": gemm_pytorch_seqs
                    })
            else:
                print("No sequences to verify.")
            print_utils.section_end(section)
        else:
            section = "Profile PyTorch Kernels (skipped)"
            print_utils.section_start(section)
            print("Skipping PyTorch kernel profiling (--skip-pytorch-profile).")
            print_utils.section_end(section)

        # ============================================================
        # Generate baremetal jobs (GEMM only)
        # ============================================================
        section = "Generate Baremetal Jobs"
        print_utils.section_start(section)
        if target_gemm_sequences["sequences"]:
            generate_jobs(target_gemm_sequences)
        else:
            print("No GEMM sequences found. Skipping baremetal job generation.")
            print("Note: Non-GEMM kernels cannot be compared with baremetal (cuBLAS only).")
        print_utils.section_end(section)

        # Search for cuBLASLt algorithms (GEMM only)
        if not self.args.skip_offline_cublas_algo_search and target_gemm_sequences["sequences"]:
            section = "Offline Search for cuBLASLt Algorithms"
            print_utils.section_start(section)
            search_cublas_algos_offline()
            print_utils.section_end(section)
        else:
            section = "Offline Search for cuBLASLt Algorithms (skipped)"
            print_utils.section_start(section)
            if not target_gemm_sequences["sequences"]:
                print("No GEMM sequences. Skipping cuBLASLt algorithm search.")
            else:
                print("Skipping offline cuBLASLt algorithm search (--skip-offline-cublas-algo-search).")
            print_utils.section_end(section)
        
        # Profile baremetal performance (GEMM only)
        if not self.args.skip_baremetal_profile and target_gemm_sequences["sequences"]:
            section = "Profile Baremetal GEMM Kernels"
            print_utils.section_start(section)
            print("This will run nsys profiling for multiple jobs, may take several minutes")
            baremetal_gemm_sequences = profile_baremetal_gemm_kernels(
                warmup=self.warmup,
                runs=self.runs,
                skip_offline_cublas_algo_search=self.args.skip_offline_cublas_algo_search
            )

            baremetal_valid_sequences = [s for s in baremetal_gemm_sequences.get("sequences", []) 
                                         if s is not None and s.get("job_id") != "0000"]

            if not baremetal_valid_sequences:
                print("Warning: No baremetal GEMM sequences were profiled.")
                print("         This can happen when PyTorch uses internal kernels instead of cuBLAS.")
                print("         This is common on H100/H200/GB200 where PyTorch uses optimized internal kernels.")
                print_utils.section_end(section)
                
                # Skip comparison but still generate report
                section = "TaxBreak Report"
                print_utils.section_start(section)
                summarize(
                    model=self.args.model, 
                    precision=self.args.precision,
                    include_all_kernels=True
                )
                print_utils.section_end(section)
                return

            print_utils.section_end(section)
            
            # Verify baremetal sequences
            section = "Verify Baremetal GEMM Sequences"
            print_utils.section_start(section)

            baremetal_by_job_id = {s["job_id"]: s for s in baremetal_valid_sequences}

            matched_targets = []
            matched_baremetals = []

            for i, seq_dict in enumerate(target_gemm_sequences["sequences"]):
                job_id = f"{i+1:04d}"
                if job_id in baremetal_by_job_id:
                    matched_targets.append(Sequence.from_dict(seq_dict))
                    matched_baremetals.append(Sequence.from_dict(baremetal_by_job_id[job_id]))
            
            if matched_targets:
                compare_sequences(matched_targets, matched_baremetals, title="Baremetal vs Target", full=False)
            else:
                print("No matched sequences to compare.")
                print("All target kernels appear to be PyTorch internal kernels (not cuBLAS).")
            
            print_utils.section_end(section)
        else:
            section = "Profile Baremetal GEMM Kernels (skipped)"
            print_utils.section_start(section)
            if not target_gemm_sequences["sequences"]:
                print("No GEMM sequences. Skipping baremetal profiling.")
            else:
                print("Skipping baremetal GEMM kernel profiling (--skip-baremetal-profile).")
            print_utils.section_end(section)
        
        # TaxBreak Report (now includes all kernels)
        section = "TaxBreak Report"
        print_utils.section_start(section)
        summarize(
            model=self.args.model,
            precision=self.args.precision,
            include_all_kernels=True
        )
        print_utils.section_end(section)