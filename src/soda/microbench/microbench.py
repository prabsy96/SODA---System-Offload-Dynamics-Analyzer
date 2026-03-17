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
from soda.microbench.baremetal.report import summarize, summarize_from_trace_directly

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
        Extract unique GEMM and non-GEMM sequences from the tracer's collected data.

        This method processes the trace data to identify unique kernel configurations,
        separating GEMM operations (library-mediated) from non-GEMM operations.

        Returns:
            Dictionary containing:
            - unique_gemm_sequences: Dict with summary and list of unique GEMM sequences
            - unique_all_sequences: Dict with summary and list of all unique sequences
        """
        section = "Extract Unique Sequences"
        print_utils.section_start(section)

        # Get sequences from tracer
        sequences = list(self.tracer.sequences)

        # Calculate metrics on all sequences
        sequences_with_metrics = utils.calculate_sequence_metrics(
            sequences,
            metrics=["launch_tax", "aten_xlat_tax", "py_tax"]
        )

        # Filter for kernel sequences
        kernel_sequences = utils.filter_kernel_sequences(sequences_with_metrics)

        lib_med_count = sum(1 for s in kernel_sequences if s.get("is_library_mediated", s.get("is_gemm", False)))
        fw_native_count = len(kernel_sequences) - lib_med_count

        print(f"Found {len(kernel_sequences)} kernel sequences in trace:")
        print(f"  - {lib_med_count} library-mediated sequences")
        print(f"  - {fw_native_count} framework-native sequences")

        # Group by identity and aggregate (deduplicate)
        grouped_seqs = utils.group_sequences_by_identity(kernel_sequences)
        unique_sequences = utils.aggregate_sequences(
            grouped_seqs,
            metrics=["launch_tax", "aten_xlat_tax", "py_tax"],
            event_types=["kernel", "aten_op", "cuda_launch"],
        )

        # Preserve classification and add freq (count) after aggregation
        for seq in unique_sequences:
            aten_name = seq.get("aten_op", {}).get("name", "")
            seq["is_library_mediated"] = utils.is_library_mediated_op(aten_name)
            seq["is_gemm"] = seq["is_library_mediated"]  # backward compat
            # freq is used by replay profiling for matching
            seq["freq"] = seq.get("count", 1)

        # Separate library-mediated and framework-native
        lib_mediated_sequences = [s for s in unique_sequences if s.get("is_library_mediated", s.get("is_gemm", False))]
        fw_native_sequences = [s for s in unique_sequences if not s.get("is_library_mediated", s.get("is_gemm", False))]

        print(f"Deduplicated to {len(unique_sequences)} unique sequences:")
        print(f"  - {len(lib_mediated_sequences)} unique library-mediated")
        print(f"  - {len(fw_native_sequences)} unique framework-native")

        # Save unique sequences for reference
        unique_all_sequences_file = utils.get_path("UNIQUE_ALL_SEQUENCES")
        unique_all_data = {
            "summary": {
                "unique_count": len(unique_sequences),
                "total_count": len(kernel_sequences),
                "unique_library_mediated": len(lib_mediated_sequences),
                "unique_framework_native": len(fw_native_sequences),
                # backward compat
                "unique_gemm": len(lib_mediated_sequences),
                "unique_non_gemm": len(fw_native_sequences),
            },
            "sequences": unique_sequences
        }
        utils.save_json(unique_all_sequences_file, unique_all_data)

        unique_lib_mediated_data = {
            "summary": {
                "count": len(lib_mediated_sequences),
            },
            "sequences": lib_mediated_sequences
        }

        print_utils.section_end(section)

        return {
            "unique_gemm_sequences": unique_lib_mediated_data,  # backward compat key
            "unique_lib_mediated_sequences": unique_lib_mediated_data,
            "unique_all_sequences": unique_all_data,
        }

    def run_direct_from_trace(self) -> None:
        """
        Simplified pipeline: compute T_structural directly from the original trace.
        
        No replay needed - uses the trace data already collected by ModelTracer.
        This provides real-world timing with full coverage.
        """
        section = "Direct Trace Analysis"
        print_utils.section_start(section)
        
        # Get sequences from tracer (already collected during tracing)
        sequences = list(self.tracer.sequences)
        
        # Calculate metrics on all sequences
        sequences_with_metrics = utils.calculate_sequence_metrics(
            sequences, 
            metrics=["launch_tax", "aten_xlat_tax", "py_tax"]
        )
        
        # Filter for kernel sequences (those with both kernel and aten_op)
        kernel_sequences = utils.filter_kernel_sequences(sequences_with_metrics)
        
        lib_med_count = sum(1 for s in kernel_sequences if s.get("is_library_mediated", s.get("is_gemm", False)))
        fw_native_count = len(kernel_sequences) - lib_med_count
        
        print(f"Found {len(kernel_sequences)} kernel sequences in trace:")
        print(f"  - {lib_med_count} library-mediated sequences")
        print(f"  - {fw_native_count} framework-native sequences")
        
        # Group by identity and aggregate (deduplicate)
        grouped_seqs = utils.group_sequences_by_identity(kernel_sequences)
        unique_sequences = utils.aggregate_sequences(
            grouped_seqs,
            metrics=["launch_tax", "aten_xlat_tax", "py_tax"],
            event_types=["kernel", "aten_op", "cuda_launch"],
        )
        
        # Preserve classification after aggregation
        for seq in unique_sequences:
            aten_name = seq.get("aten_op", {}).get("name", "")
            seq["is_library_mediated"] = utils.is_library_mediated_op(aten_name)
            seq["is_gemm"] = seq["is_library_mediated"]  # backward compat
        
        unique_lib_med = sum(1 for s in unique_sequences if s.get("is_library_mediated", s.get("is_gemm", False)))
        unique_fw_native = len(unique_sequences) - unique_lib_med
        
        print(f"Deduplicated to {len(unique_sequences)} unique sequences:")
        print(f"  - {unique_lib_med} unique library-mediated")
        print(f"  - {unique_fw_native} unique framework-native")
        
        # Save unique sequences
        unique_all_sequences_file = utils.get_path("UNIQUE_ALL_SEQUENCES")
        unique_all_data = {
            "summary": {
                "unique_count": len(unique_sequences),
                "total_count": len(kernel_sequences),
                "unique_library_mediated": unique_lib_med,
                "unique_framework_native": unique_fw_native,
                # backward compat
                "unique_gemm": unique_lib_med,
                "unique_non_gemm": unique_fw_native,
            },
            "sequences": unique_sequences
        }
        utils.save_json(unique_all_sequences_file, unique_all_data)
        
        print_utils.section_end(section)
        
        # Generate report directly from trace-derived sequences
        section = "TaxBreak Report (Direct Trace)"
        print_utils.section_start(section)

        # Get num_runs from tracer (defaults to 1 for backwards compatibility)
        num_runs = getattr(self.tracer, 'num_profiled_runs', 1)

        summarize_from_trace_directly(
            unique_sequences=unique_sequences,
            model=self.args.model,
            precision=self.args.precision,
            num_runs=num_runs
        )

        print_utils.section_end(section)

    def run(self) -> None:
        """
        Run the complete microbenchmarking pipeline.
        
        By default, uses direct trace analysis (no replay).
        Set --replay flag to use the original replay-based pipeline.
        """
        # Check if we should use direct trace mode (default) or replay mode
        use_direct_trace = getattr(self.args, 'direct_trace', True)
        
        if use_direct_trace:
            self.run_direct_from_trace()
            return
        
        # --- Original replay-based pipeline below ---
        # Extract target library-mediated (vendor-library) sequences
        extraction_results = self.extract_unique_gemm_sequences()  
        target_gemm_sequences = extraction_results["unique_gemm_sequences"]
        target_all_sequences = extraction_results["unique_all_sequences"]

        # ...rest of original run() method...
        # (Keep existing code for replay-based pipeline)
        
        # ============================================================
        # Profile ALL PyTorch kernels (library-mediated + framework-native)
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
                    
                # Plot library-mediated sequences if available
                gemm_pytorch_seqs = [s for s in pytorch_all_sequences["sequences"] if s.get("is_library_mediated", s.get("is_gemm", False))]
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
        # Generate baremetal jobs (library-mediated only)
        # ============================================================
        section = "Generate Baremetal Jobs"
        print_utils.section_start(section)
        if target_gemm_sequences["sequences"]:
            generate_jobs(target_gemm_sequences)
        else:
            print("No library-mediated sequences found. Skipping baremetal job generation.")
            print("Note: Framework-native kernels cannot be compared with baremetal (cuBLAS only).")
        print_utils.section_end(section)

        # Search for cuBLASLt algorithms (library-mediated only)
        if not self.args.skip_offline_cublas_algo_search and target_gemm_sequences["sequences"]:
            section = "Offline Search for cuBLASLt Algorithms"
            print_utils.section_start(section)
            search_cublas_algos_offline()
            print_utils.section_end(section)
        else:
            section = "Offline Search for cuBLASLt Algorithms (skipped)"
            print_utils.section_start(section)
            if not target_gemm_sequences["sequences"]:
                print("No library-mediated sequences. Skipping cuBLASLt algorithm search.")
            else:
                print("Skipping offline cuBLASLt algorithm search (--skip-offline-cublas-algo-search).")
            print_utils.section_end(section)
        
        # Profile baremetal performance (library-mediated only)
        if not self.args.skip_baremetal_profile and target_gemm_sequences["sequences"]:
            section = "Profile Baremetal Library-mediated Kernels"
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
                print("Warning: No baremetal library-mediated sequences were profiled.")
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
            section = "Verify Baremetal Library-mediated Sequences"
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
                print("All target kernels appear to be PyTorch internal kernels (not vendor-library-mediated).")
            
            print_utils.section_end(section)
        else:
            section = "Profile Baremetal Library-mediated Kernels (skipped)"
            print_utils.section_start(section)
            if not target_gemm_sequences["sequences"]:
                print("No library-mediated sequences. Skipping baremetal profiling.")
            else:
                print("Skipping baremetal library-mediated kernel profiling (--skip-baremetal-profile).")
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