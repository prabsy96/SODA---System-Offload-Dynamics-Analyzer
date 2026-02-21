#!/usr/bin/env python3
"""
Compare PyTorch and baremetal GEMM launch tax.

Joins results from framework/pytorch/output/unique_gemm_sequences.json and
baremetal/output/baremetal_gemm_runs.json, verifies kernel matching,
computes per-kernel launch tax deltas and percentages, and emits
baremetal/output/bm_vs_framework_report.json.
"""
import os
import math
import statistics
from collections import defaultdict
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch

from soda.common import utils, print_utils

# Hardcoded null kernel T_sys baselines (μs) by GPU architecture
# Measured from baremetal profiling with minimal CUDA kernel
NULL_KERNEL_SYS_TAX = {
    "H100": 4.51,
    "H200": 4.30,
    "DEFAULT": 4.50,  # Fallback for unknown GPUs
}

def get_gpu_architecture() -> str:
    """
    Get the GPU architecture name from CUDA device properties.
    
    Returns:
        GPU architecture string (e.g., "H100", "H200", "A100")
    """
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).upper()
            if "H100" in gpu_name:
                return "H100"
            elif "H200" in gpu_name:
                return "H200"
            elif "A100" in gpu_name:
                return "A100"
            elif "GB200" in gpu_name:
                return "GB200"
    except Exception:
        pass
    return "DEFAULT"


def get_null_kernel_sys_tax(dynamic_value: float | None = None) -> float:
    """
    Get the null kernel's system launch tax.

    If *dynamic_value* is provided (from a runtime null-kernel measurement),
    it is returned directly.  Otherwise falls back to the hardcoded
    per-architecture baselines.

    Args:
        dynamic_value: Dynamically measured T_sys in microseconds, or None.

    Returns:
        T_sys for null kernel in microseconds.
    """
    if dynamic_value is not None:
        return dynamic_value
    gpu_arch = get_gpu_architecture()
    return NULL_KERNEL_SYS_TAX.get(gpu_arch, NULL_KERNEL_SYS_TAX["DEFAULT"])


def load_pytorch_results(pytorch_file: str) -> Dict[str, Any]:
    """Load PyTorch GEMM sequences."""
    data = utils.load_json(pytorch_file)
    sequences = data["sequences"]
    results = {}
    
    for idx, sequence in enumerate(sequences):
        if sequence is None:
            continue
        job_id = f"{idx+1:04d}"
        
        results[job_id] = {
            "freq": sequence.get("freq"),
            "kernel": sequence["kernel"]["name"],
            "aten_op": sequence["aten_op"]["name"],
            "launch_tax": sequence["launch_tax"]["avg"],
            "aten_xlat_tax": sequence["aten_xlat_tax"]["avg"],
            "py_tax": sequence["py_tax"]["avg"],
            "is_gemm": sequence.get("is_gemm", False),
        }
    
    return results

def load_all_pytorch_results(pytorch_file: str) -> Dict[str, Any]:
    """Load ALL PyTorch kernel sequences (GEMM and non-GEMM)."""
    if not os.path.exists(pytorch_file):
        return {}
    
    data = utils.load_json(pytorch_file)
    sequences = data.get("sequences", [])
    results = {}
    
    for idx, sequence in enumerate(sequences):
        if sequence is None:
            continue
        job_id = f"{idx+1:04d}"
        
        results[job_id] = {
            "freq": sequence.get("freq"),
            "kernel": sequence["kernel"]["name"],
            "aten_op": sequence["aten_op"]["name"],
            "launch_tax": sequence["launch_tax"]["avg"],
            "aten_xlat_tax": sequence["aten_xlat_tax"]["avg"],
            "py_tax": sequence["py_tax"]["avg"],
            "is_gemm": sequence.get("is_gemm", False),
        }
    
    return results


def load_baremetal_results(baremetal_file: str) -> Dict[str, Any]:
    """Load baremetal results (GEMM only)."""
    if not os.path.exists(baremetal_file):
        return {}
    
    data = utils.load_json(baremetal_file)
    sequences = data["sequences"]
    results = {}
    
    for sequence in sequences:
        if sequence is None:
            continue

        job_id = sequence["job_id"]
        is_null = sequence["kernel"]["name"] == "__null_kernel__"
        results[job_id] = {
            "kernel": sequence["kernel"]["name"],
            "culib_xlat_tax": sequence["culib_xlat_tax"]["avg"] if not is_null else None,
            "culib_setup": sequence["culib"]["setup"]["dur"]["avg"] if not is_null else None,
            "culib_heur": sequence["culib"]["heur"]["dur"]["avg"] if not is_null else None,
            "shim_tax": sequence["shim_tax"]["avg"],
            "launch_tax": sequence["launch_tax"]["avg"],
        }
    
    return results

def generate_framework_summary(show_table: bool = True, include_all_kernels: bool = False):
    """Per Kernel Framework Overhead (μs) @ PyTorch Scope"""
    def fmt_val(v):
        return None if v is None else f"{v:.2f}"

    # Load GEMM sequences
    pytorch_file = utils.get_path("PYTORCH_GEMM_SEQUENCES")
    pytorch_results = {}
    if os.path.exists(pytorch_file):
        pytorch_results = load_pytorch_results(pytorch_file)
        print(f"Loaded {len(pytorch_results)} PyTorch GEMM sequences")

    # Load all kernel sequences if requested
    if include_all_kernels:
        all_kernels_file = utils.get_path("PYTORCH_ALL_SEQUENCES")
        if os.path.exists(all_kernels_file):
            all_kernel_results = load_all_pytorch_results(all_kernels_file)
            print(f"Loaded {len(all_kernel_results)} PyTorch kernel sequences (all types)")
            # Merge, preferring all_kernel_results for complete coverage
            for job_id, result in all_kernel_results.items():
                if job_id not in pytorch_results:
                    pytorch_results[job_id] = result

    if not pytorch_results:
        print("No PyTorch sequences found.")
        return

    framework_summary_table = []
    for job_id, job_result in sorted(pytorch_results.items()):
        py_tax = job_result["py_tax"]
        aten_xlat_tax = job_result["aten_xlat_tax"]
        launch_tax = job_result["launch_tax"]
        is_gemm = job_result.get("is_gemm", False)
        
        fo = py_tax + aten_xlat_tax + launch_tax
        kernel_type = "GEMM" if is_gemm else "other"
        kernel_name = job_result["kernel"]
        
        # Increased truncation length from 20 to 40 characters
        kernel_display = kernel_name[:40] + "..." if len(kernel_name) > 40 else kernel_name
        
        # Truncate ATen op name slightly if too long, but keep it readable
        aten_op = job_result["aten_op"]
        aten_display = aten_op[:25] + "..." if len(aten_op) > 25 else aten_op

        framework_summary_table.append([
            job_id,
            aten_display,
            kernel_display,
            kernel_type,
            fmt_val(fo),
            fmt_val(py_tax),
            fmt_val(aten_xlat_tax),
            fmt_val(launch_tax),
            job_result["freq"],
        ])

    if show_table:
        print_utils.comp_table(
            title=f"Per Kernel Framework Overhead (μs) @ PyTorch Scope",
            headers=["ID", "ATen Op", "Kernel", "Type", "T_fo", "T_py", "T_aten_xlat", "T_sys", "freq"],
            data=framework_summary_table,
        )
        print("Notes:")
        print(" * Type: GEMM (eligible for baremetal comparison) or other (framework overhead only)")
        print(" * T_fo: framework overhead; T_py + T_aten_xlat + T_sys")
        print(" * T_py: python tax; torch_op -> aten_op")
        print(" * T_aten_xlat: ATen+cuBLASLt xlat tax; aten_op -> launch")
        print(" * T_sys: kernel launch tax; launch -> kernel")


def generate_baremetal_summary(show_table: bool = True):
    """Per GEMM Baremetal Overhead (μs)"""
    # NOTE: This function is kept for compatibility but might not be used if baremetal is disabled
    def fmt_val(v):
        return None if v is None else f"{v:.2f}"

    baremetal_file = utils.get_path("BAREMETAL_GEMM_KERNELS")
    if not os.path.exists(baremetal_file):
        print("No baremetal sequences found (expected for non-GEMM kernels).")
        return

    baremetal_results = load_baremetal_results(baremetal_file)
    print(f"Loaded {len(baremetal_results)} baremetal sequences")

    baremetal_summary_data = []
    for job_id, job_result in sorted(baremetal_results.items()):
        res = job_result
        baremetal_summary_data.append([
            job_id,
            job_result["kernel"],
            fmt_val(res["culib_xlat_tax"]),
            fmt_val(res["culib_setup"]),
            fmt_val(res["culib_heur"]),
            fmt_val(res["shim_tax"]),
            fmt_val(res["launch_tax"]),
        ])

    if show_table:
        print_utils.comp_table(
            title="Per GEMM Baremetal Overhead (μs)",
            headers=["ID", "Kernel", "T_culib_xlat", "T_setup", "T_heur", "T_shim", "T_sys"],
            data=baremetal_summary_data,
        )

def generate_final_summary(show_table: bool = True, include_all_kernels: bool = False) -> List[Dict[str, Any]]:
    """
    Combined summary across framework and baremetal.
    
    For GEMM kernels: Full T_structural breakdown
    For non-GEMM kernels: Partial breakdown (framework overhead only)
    """
    def fmt_val(v):
        return None if v is None else f"{v:.2f}"

    # Load PyTorch GEMM sequences
    pytorch_file = utils.get_path("PYTORCH_GEMM_SEQUENCES")
    pytorch_results = {}
    if os.path.exists(pytorch_file):
        pytorch_results = load_pytorch_results(pytorch_file)

    # Load all PyTorch kernel sequences if requested
    if include_all_kernels:
        all_kernels_file = utils.get_path("PYTORCH_ALL_SEQUENCES")
        if os.path.exists(all_kernels_file):
            all_kernel_results = load_all_pytorch_results(all_kernels_file)
            for job_id, result in all_kernel_results.items():
                if job_id not in pytorch_results:
                    pytorch_results[job_id] = result

    # NOTE: Baremetal loading removed for T_cuda calculation, but kept here for structure compatibility
    # if needed for other fields. For now, we rely on PyTorch data.
    
    print(f"Loaded {len(pytorch_results)} PyTorch sequences")

    final_summary_table = []
    final_summary_data = []
    all_job_ids = sorted(pytorch_results.keys())

    for job_id in all_job_ids:
        fw = pytorch_results.get(job_id)
        
        py = fw["py_tax"] if fw else None
        aten_xlat = fw["aten_xlat_tax"] if fw else None
        sys_fw = fw["launch_tax"] if fw else None
        is_gemm = fw.get("is_gemm", False) if fw else False

        # Baremetal fields are None since we disabled dependency
        culib_xlat = None
        shim = None
        sys_bm = None
        setup = None
        heur = None

        sys = sys_fw

        kernel_name = fw["kernel"] if fw else None
        aten_op_name = fw["aten_op"] if fw else None
        freq = fw["freq"] if fw else None

        # T_aten is not calculated here anymore (done in summarize via baseline)
        aten = None

        # Framework overhead calculations
        fo_fw = None
        if all(x is not None for x in (py, aten_xlat, sys_fw)):
            fo_fw = py + aten_xlat + sys_fw

        kernel_type = "GEMM" if is_gemm else "other"

        final_summary_data.append({
            "id": job_id,
            "aten_op": aten_op_name,
            "kernel": kernel_name,
            "kernel_type": kernel_type,
            "is_gemm": is_gemm,
            "T_fo": fo_fw,
            "T_fo_bm": None,
            "T_fo_fw": fo_fw,
            "T_py": py,
            "T_aten_xlat": aten_xlat,  
            "T_aten": aten,
            "T_lib_xlat": culib_xlat,
            "T_lib_setup": setup,
            "T_lib_heur": heur,
            "T_lib_shim": shim,
            "T_sys_bm": sys_bm,
            "T_sys_fw": sys_fw,
            "T_sys": sys,
            "freq": freq,
        })

        kernel_display = kernel_name[:40] + "..." if kernel_name and len(kernel_name) > 40 else kernel_name
        
        final_summary_table.append([
            job_id,
            aten_op_name,
            kernel_display,
            kernel_type,
            fmt_val(fo_fw),
            fmt_val(py),
            fmt_val(aten_xlat),
            fmt_val(sys),
            freq,
        ])

    if show_table:
        print_utils.comp_table(
            title="Framework Tax Break (μs) - All Kernels",
            headers=[
                "ID", "ATen op", "Kernel", "Type",
                "T_fo", "T_py", "T_aten_xlat",
                "T_sys", "freq"
            ],
            data=final_summary_table,
        )
        print("Notes:")
        print(" * Type: GEMM or other")
        print(" * T_fo: framework overhead")
        print(" * T_py: python xlat tax; torch_op -> aten_op")
        print(" * T_aten_xlat: ATen dispatch + potential library overhead")
        print(" * T_sys: kernel launch tax; launch -> kernel")

    return final_summary_data


def plot_per_kernel_taxbreak(final_summary_data: List[Dict], title_suffix: str):
    """Plot stacked bar chart for all kernels."""
    output_path = utils.get_path("TAX_BREAK_PLOT")

    # PyTorch-only components (non-GEMM or fallback)
    pytorch_components = [
        ("T_py", "#4e79a7"),
        ("T_aten_xlat", "#59a14f"),
        ("T_sys_fw", "#af7aa1"),
    ]

    gemm_entries = []
    non_gemm_entries = []

    for row in final_summary_data:
        if row.get("kernel") == "__null_kernel__":
            continue
        
        is_gemm = row.get("is_gemm", False)
        
        vals = [row.get(name) for name, _ in pytorch_components]
        if all(v is not None for v in vals):
            if is_gemm:
                gemm_entries.append((row["id"], row["kernel"], vals, pytorch_components))
            else:
                non_gemm_entries.append((row["id"], row["kernel"], vals, pytorch_components))

    all_entries = gemm_entries + non_gemm_entries

    if not all_entries:
        print("No complete per-kernel taxbreak data to plot.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create subplots if we have both types
    if gemm_entries and non_gemm_entries:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        axes_data = [(ax1, gemm_entries, "GEMM Kernels"), (ax2, non_gemm_entries, "Non-GEMM Kernels")]
    elif gemm_entries:
        fig, ax1 = plt.subplots(figsize=(8, 4))
        axes_data = [(ax1, gemm_entries, "GEMM Kernels")]
    else:
        fig, ax1 = plt.subplots(figsize=(8, 4))
        axes_data = [(ax1, non_gemm_entries, "Non-GEMM Kernels")]

    for ax, entries, subtitle in axes_data:
        if not entries:
            continue
            
        x = list(range(len(entries)))
        job_labels = [f"{jid}\n{kernel[:12]}..." if len(kernel) > 12 else f"{jid}\n{kernel}" 
                      for jid, kernel, _, _ in entries]

        components = entries[0][3]
        
        bottoms = [0.0] * len(entries)
        for idx_component, (name, color) in enumerate(components):
            heights = [vals[idx_component] for _, _, vals, _ in entries]
            ax.bar(x, heights, bottom=bottoms, label=name, color=color)
            bottoms = [b + h for b, h in zip(bottoms, heights)]

        ax.set_ylabel("μs")
        ax.set_title(f"{subtitle}")
        ax.set_xlabel("Kernel")
        ax.set_xticks(x)
        ax.set_xticklabels(job_labels, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=8)

    fig.suptitle(f"Framework Tax Break (μs) | {title_suffix}")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved per-kernel taxbreak plot to {output_path}")


def get_derived_aten_baseline(final_summary_data: List[Dict[str, Any]]) -> float:
    """
    Derive baseline ATen dispatch cost (T_aten_base) from non-GEMM kernels.
    Non-GEMM kernels (elementwise, etc.) represent pure dispatch without library overhead.
    We use the median T_aten_xlat of these kernels as the baseline.
    """
    non_gemm_vals = []
    for row in final_summary_data:
        # Filter for non-GEMM kernels that have valid timing data
        if (not row.get("is_gemm", False) and 
            row.get("T_aten_xlat") is not None and 
            row.get("T_aten_xlat") > 0.1):
            non_gemm_vals.append(row["T_aten_xlat"])
    
    if not non_gemm_vals:
        # Fallback if no non-GEMM kernels found (unlikely in real models)
        return 2.0 
        
    return statistics.median(non_gemm_vals)


def summarize(model: str, precision: str, include_all_kernels: bool = False):
    """
    Main reporting function.
    Uses derived baseline subtraction for T_cuda calculation (no baremetal dependency).
    """
    # Generate framework summary only (baremetal summary is skipped/optional)
    generate_framework_summary(show_table=True, include_all_kernels=include_all_kernels)
    
    # We still generate the final summary data structure from framework traces
    final_summary_data = generate_final_summary(show_table=True, include_all_kernels=include_all_kernels)

    # Get null kernel T_sys as baseline (hardcoded)
    t_sys_null = get_null_kernel_sys_tax()
    gpu_arch = get_gpu_architecture()
    
    # --- Step 1: Derive Baseline ATen Dispatch (T_aten_base) ---
    t_aten_base = get_derived_aten_baseline(final_summary_data)
    
    print(f"GPU Architecture: {gpu_arch}")
    print(f"Null kernel T_sys baseline: {t_sys_null:.3f} μs (hardcoded)")
    print(f"Derived T_aten_base:        {t_aten_base:.3f} μs (median of non-GEMM T_aten_xlat)")

    # --- Step 2: Calculate Total Structural Overhead with Breakdown ---
    total_structural_overhead_us = 0.0
    total_FT_us = 0.0      # Framework Translation (T_py)
    total_CT_us = 0.0      # Compute Translation (T_aten_base or T_aten_xlat)
    total_KT_us = 0.0      # Kernel Launch Tax (T_sys)
    total_CudaT_us = 0.0   # CUDA Translation Tax (Calculated)
    total_LibT_us = 0.0    # Library overhead (placeholder)
    total_invocations = 0
    gemm_invocations = 0
    non_gemm_invocations = 0
    
    for row in final_summary_data:
        freq = row.get("freq")
        if freq is None:
            continue
            
        total_invocations += freq
        
        # Get components (default to 0.0)
        t_fo = row.get("T_fo") or 0.0
        t_py = row.get("T_py") or 0.0
        t_sys = row.get("T_sys") or row.get("T_sys_fw") or 0.0
        t_aten_xlat = row.get("T_aten_xlat") or 0.0
        
        is_gemm = row.get("is_gemm", False)
        
        # --- Step 3: Apply Subtraction Formula ---
        if is_gemm:
            # GEMM Kernel: Calculate T_cuda
            # T_cuda = T_aten_xlat - T_aten_base
            raw_cuda = t_aten_xlat - t_aten_base
            t_cuda = max(0.0, raw_cuda)
            t_ct = t_aten_base
            gemm_invocations += freq
        else:
            # Non-GEMM: Pure dispatch, no library overhead
            t_cuda = 0.0
            t_ct = t_aten_xlat
            non_gemm_invocations += freq
        
        # Store calculated T_cuda back to row
        row["T_cuda"] = t_cuda
        
        # Aggregate with frequency
        total_structural_overhead_us += t_fo * freq
        total_FT_us += t_py * freq
        total_CT_us += t_ct * freq
        total_KT_us += t_sys * freq
        total_CudaT_us += t_cuda * freq
        # LibT is 0.0 in this calculation method (absorbed into CudaT or ignored)
    
    # Convert to ms for display
    total_structural_overhead_ms = total_structural_overhead_us / 1000.0
    total_FT_ms = total_FT_us / 1000.0
    total_CT_ms = total_CT_us / 1000.0
    total_KT_ms = total_KT_us / 1000.0
    total_CudaT_ms = total_CudaT_us / 1000.0
    
    # Print structural overhead summary
    print()
    print("=== Structural (Orchestrator) Overhead ===")
    print(f"T_structural_total = Σ (T_fo × freq) = {total_structural_overhead_ms:.3f} ms")
    print(f"  ├─ ΔFT   (Python Translation)   = Σ (T_py × freq)        = {total_FT_ms:.3f} ms")
    print(f"  ├─ ΔCT   (Compute Translation)  = Σ (T_ct × freq)        = {total_CT_ms:.3f} ms")
    print(f"  ├─ ΔCudaT (CUDA Translation)    = Σ (T_cuda × freq)      = {total_CudaT_ms:.3f} ms")
    print(f"  └─ ΔKT   (Kernel Launch)        = Σ (T_sys × freq)       = {total_KT_ms:.3f} ms")
    print(f"  (Aggregated over {total_invocations} kernel invocations)")
    print()
    print("Calculation Method (No Baremetal):")
    print(f"  1. T_aten_base = Median(T_aten_xlat of non-GEMM kernels) = {t_aten_base:.3f} μs")
    print(f"  2. GEMM:     T_cuda = max(0, T_aten_xlat - T_aten_base)")
    print(f"  3. Non-GEMM: T_cuda = 0.0")
    print()
    print("Kernel Breakdown:")
    print(f"  * GEMM invocations:     {gemm_invocations}")
    print(f"  * Non-GEMM invocations: {non_gemm_invocations}")
    
    # Save to JSON
    summary_path = utils.get_path("TAX_BREAK_SUMMARY")
    summary_data = {
        "gpu_architecture": gpu_arch,
        "t_aten_base_us": t_aten_base,
        "T_structural_total_ms": total_structural_overhead_ms,
        "breakdown": {
            "FT_python_ms": total_FT_ms,
            "CT_aten_ms": total_CT_ms,
            "CudaT_cuda_runtime_ms": total_CudaT_ms,
            "KT_kernel_launch_ms": total_KT_ms,
        },
        "formula": {
            "T_cuda_gemm": "max(0, T_aten_xlat - T_aten_base)",
            "T_cuda_non_gemm": "0.0",
            "T_aten_base": "Median(Non-GEMM T_aten_xlat)"
        },
        "invocations": {
            "total": total_invocations,
            "gemm": gemm_invocations,
            "non_gemm": non_gemm_invocations,
        },
        "per_kernel": final_summary_data,
    }
    utils.save_json(summary_path, summary_data)
    print(f"\nSaved taxbreak summary to {summary_path}")
    
    # Plot
    plot_per_kernel_taxbreak(final_summary_data, f"{model} | {precision}")
    print("=== TaxBreak Report ===")



def summarize_from_trace_directly(
    unique_sequences: List[Dict[str, Any]],
    model: str,
    precision: str,
    num_runs: int = 1
):
    """
    Compute T_structural directly from trace-derived sequences.

    Each sequence represents a unique kernel invocation (no deduplication).
    Aggregates timing across the entire timeline to capture real-world variations.

    Uses hardcoded null kernel T_sys for launch tax (excludes queue time).

    When num_runs > 1, the trace contains multiple profiled inferences.
    Totals are divided by num_runs to report mean per-inference metrics.

    Args:
        unique_sequences: List of kernel sequences (each invocation is unique)
        model: Model name for reporting
        precision: Precision setting for reporting
        num_runs: Number of profiled inference runs in the trace (for averaging)
    """
    if not unique_sequences:
        print("No kernel sequences provided.")
        return
    
    # Get hardcoded baselines
    t_sys_null = get_null_kernel_sys_tax()  # Hardcoded: 4.51 μs for H100
    gpu_arch = get_gpu_architecture()
    
    # Build summary data from sequences
    summary_data = []
    for idx, seq in enumerate(unique_sequences):
        job_id = f"{idx+1:04d}"
        
        # Extract timing - handle both raw values and dict format
        launch_tax = seq.get("launch_tax", 0)
        aten_xlat_tax = seq.get("aten_xlat_tax", 0)
        py_tax = seq.get("py_tax", 0)
        
        # Handle dict format (from aggregated sequences)
        t_py = py_tax.get("avg", py_tax) if isinstance(py_tax, dict) else py_tax
        t_aten_xlat = aten_xlat_tax.get("avg", aten_xlat_tax) if isinstance(aten_xlat_tax, dict) else aten_xlat_tax
        t_sys_measured = launch_tax.get("avg", launch_tax) if isinstance(launch_tax, dict) else launch_tax
        
        # Ensure numeric and non-negative
        t_py = max(0.0, float(t_py or 0))
        t_aten_xlat = max(0.0, float(t_aten_xlat or 0))
        t_sys_measured = max(0.0, float(t_sys_measured or 0))
        
        # USE HARDCODED T_sys for structural calculation (no queue time)
        t_sys = t_sys_null
        
        freq = seq.get("freq", 1)
        is_gemm = seq.get("is_gemm", False)
        
        # Handle nested dict for kernel/aten_op names
        kernel = seq.get("kernel", {})
        aten_op = seq.get("aten_op", {})
        kernel_name = kernel.get("name", "unknown") if isinstance(kernel, dict) else str(kernel)
        aten_op_name = aten_op.get("name", "unknown") if isinstance(aten_op, dict) else str(aten_op)
        
        # T_fo uses hardcoded T_sys (structural overhead only)
        t_fo = t_py + t_aten_xlat + t_sys
        
        summary_data.append({
            "id": job_id,
            "aten_op": aten_op_name,
            "kernel": kernel_name,
            "kernel_type": "GEMM" if is_gemm else "other",
            "is_gemm": is_gemm,
            "T_fo": t_fo,
            "T_py": t_py,
            "T_aten_xlat": t_aten_xlat,
            "T_sys": t_sys,              # Hardcoded null kernel value
            "T_sys_measured": t_sys_measured,  # Keep measured value for reference
            "T_sys_fw": t_sys,
            "freq": freq,
        })
    
    # Derive T_aten_base from non-GEMM kernels
    non_gemm_vals = [
        row["T_aten_xlat"] for row in summary_data 
        if not row.get("is_gemm", False) and row.get("T_aten_xlat", 0) > 0.1
    ]
    t_aten_base = statistics.median(non_gemm_vals) if non_gemm_vals else 2.0
    
    print(f"GPU Architecture: {gpu_arch}")
    print(f"Null kernel T_sys baseline: {t_sys_null:.3f} μs (hardcoded, used for T_fo)")
    print(f"Derived T_aten_base:        {t_aten_base:.3f} μs (median non-GEMM T_aten_xlat from trace)")
    
    # Print table (show first 100 rows max to avoid spam)
    def fmt_val(v):
        return None if v is None else f"{v:.2f}"
    
    table_data = []
    for row in summary_data[:100]:  # Limit table output
        kernel_display = row["kernel"][:40] + "..." if len(row["kernel"]) > 40 else row["kernel"]
        table_data.append([
            row["id"],
            row["aten_op"],
            kernel_display,
            row["kernel_type"],
            fmt_val(row["T_fo"]),
            fmt_val(row["T_py"]),
            fmt_val(row["T_aten_xlat"]),
            fmt_val(row["T_sys"]),  # Now shows hardcoded value
            row["freq"],
        ])
    
    if len(summary_data) > 100:
        print(f"(Showing first 100 of {len(summary_data)} kernel invocations)")
    
    print_utils.comp_table(
        title="Framework Tax Break (μs) - Direct from Trace (Per Invocation)",
        headers=["ID", "ATen op", "Kernel", "Type", "T_fo", "T_py", "T_aten_xlat", "T_sys", "freq"],
        data=table_data,
    )
    
    # Aggregate per-kernel overheads across entire timeline
    # Also track variance for std/CI computation
    total_structural_overhead_us = 0.0
    total_FT_us = 0.0
    total_CT_us = 0.0
    total_KT_us = 0.0
    total_CudaT_us = 0.0
    total_invocations = 0
    gemm_invocations = 0
    non_gemm_invocations = 0

    # Collect per-run samples for variance calculation
    # Each unique kernel contributes ~1 invocation per run, so we can estimate per-run variance
    per_kernel_fo_samples = []  # T_fo values for each kernel occurrence

    for row in summary_data:
        freq = row.get("freq", 1)
        t_py = row.get("T_py", 0.0)
        t_aten_xlat = row.get("T_aten_xlat", 0.0)
        t_sys = row.get("T_sys", 0.0)  # Hardcoded value
        t_fo = row.get("T_fo", 0.0)
        is_gemm = row.get("is_gemm", False)

        # Calculate T_cuda per kernel
        if is_gemm:
            t_cuda = max(0.0, t_aten_xlat - t_aten_base)
            t_ct = t_aten_base
            gemm_invocations += freq
        else:
            t_cuda = 0.0
            t_ct = t_aten_xlat
            non_gemm_invocations += freq

        # Store T_cuda in row
        row["T_cuda"] = t_cuda

        # Aggregate (freq=1 for each unique invocation)
        total_structural_overhead_us += t_fo * freq
        total_FT_us += t_py * freq
        total_CT_us += t_ct * freq
        total_KT_us += t_sys * freq  # Uses hardcoded T_sys
        total_CudaT_us += t_cuda * freq
        total_invocations += freq

        # Collect samples for variance (replicate t_fo for each occurrence)
        per_kernel_fo_samples.extend([t_fo] * freq)
    
    # Convert to ms and normalize by num_runs for mean per-inference values
    total_structural_overhead_ms = total_structural_overhead_us / 1000.0 / num_runs
    total_FT_ms = total_FT_us / 1000.0 / num_runs
    total_CT_ms = total_CT_us / 1000.0 / num_runs
    total_KT_ms = total_KT_us / 1000.0 / num_runs
    total_CudaT_ms = total_CudaT_us / 1000.0 / num_runs
    invocations_per_run = total_invocations // num_runs if num_runs > 0 else total_invocations

    # Compute standard deviation and 95% CI for T_structural
    # We estimate per-inference variance from the per-kernel variance
    std_structural_ms = 0.0
    ci_lower_ms = total_structural_overhead_ms
    ci_upper_ms = total_structural_overhead_ms
    sem_ms = 0.0  # Standard error of the mean

    if num_runs > 1 and len(per_kernel_fo_samples) > 1:
        # Compute std of per-kernel T_fo values (in μs)
        mean_fo = sum(per_kernel_fo_samples) / len(per_kernel_fo_samples)
        variance_fo = sum((x - mean_fo) ** 2 for x in per_kernel_fo_samples) / (len(per_kernel_fo_samples) - 1)
        std_fo_us = math.sqrt(variance_fo)

        # Per-inference std: approximate as (std_per_kernel * sqrt(kernels_per_run)) / num_runs
        # This is based on: var(sum) = n * var(each) for i.i.d., std(mean) = std/sqrt(n)
        kernels_per_run = invocations_per_run if invocations_per_run > 0 else 1
        # Total std per inference ≈ std_per_kernel * sqrt(kernels_per_run)
        std_structural_us = std_fo_us * math.sqrt(kernels_per_run)
        std_structural_ms = std_structural_us / 1000.0

        # Standard error of the mean (SEM) for confidence interval
        sem_ms = std_structural_ms / math.sqrt(num_runs)

        # 95% CI: mean ± 1.96 * SEM
        ci_lower_ms = total_structural_overhead_ms - 1.96 * sem_ms
        ci_upper_ms = total_structural_overhead_ms + 1.96 * sem_ms

    # Print summary
    print()
    if num_runs > 1:
        print(f"=== Structural (Orchestrator) Overhead (Mean of {num_runs} runs) ===")
    else:
        print("=== Structural (Orchestrator) Overhead (Direct from Trace) ===")

    if num_runs > 1 and std_structural_ms > 0:
        print(f"T_structural (mean ± std) = {total_structural_overhead_ms:.3f} ± {std_structural_ms:.3f} ms")
        print(f"  95% CI: [{ci_lower_ms:.3f}, {ci_upper_ms:.3f}] ms (n={num_runs})")
    else:
        print(f"T_structural (mean) = {total_structural_overhead_ms:.3f} ms")

    print(f"  ├─ ΔFT   (Python Translation)   = {total_FT_ms:.3f} ms")
    print(f"  ├─ ΔCT   (Compute Translation)  = {total_CT_ms:.3f} ms")
    print(f"  ├─ ΔCudaT (CUDA Translation)    = {total_CudaT_ms:.3f} ms")
    print(f"  └─ ΔKT   (Kernel Launch)        = {total_KT_ms:.3f} ms")
    if num_runs > 1:
        print(f"  (Mean over {num_runs} profiled runs, ~{invocations_per_run} kernels/run)")
    else:
        print(f"  (Aggregated over {total_invocations} kernel invocations)")
    print()
    print("Calculation Method (Direct Trace, Per-Invocation):")
    print(f"  1. Profiled runs: {num_runs}")
    print(f"  2. T_aten_base = Median(T_aten_xlat of non-GEMM) = {t_aten_base:.3f} μs")
    print(f"  3. T_sys = {t_sys_null:.3f} μs (hardcoded null kernel baseline)")
    print(f"  4. GEMM:     T_cuda = max(0, T_aten_xlat - T_aten_base)")
    print(f"  5. Non-GEMM: T_cuda = 0.0")
    print()
    print("Invocation Breakdown (total across all runs):")
    print(f"  * GEMM invocations:     {gemm_invocations}")
    print(f"  * Non-GEMM invocations: {non_gemm_invocations}")
    print(f"  * Total invocations:    {total_invocations}")
    
    # Save results
    summary_path = utils.get_path("TAX_BREAK_SUMMARY")
    output_data = {
        "method": "direct_trace_per_invocation",
        "num_profiled_runs": num_runs,
        "gpu_architecture": gpu_arch,
        "t_aten_base_us": t_aten_base,
        "t_sys_null_us": t_sys_null,
        "T_structural_mean_ms": total_structural_overhead_ms,
        "T_structural_std_ms": std_structural_ms,
        "T_structural_95ci_lower_ms": ci_lower_ms,
        "T_structural_95ci_upper_ms": ci_upper_ms,
        "breakdown_mean": {
            "FT_python_ms": total_FT_ms,
            "CT_aten_ms": total_CT_ms,
            "CudaT_cuda_runtime_ms": total_CudaT_ms,
            "KT_kernel_launch_ms": total_KT_ms,
        },
        "statistics": {
            "std_ms": std_structural_ms,
            "sem_ms": sem_ms,
            "ci_95_lower_ms": ci_lower_ms,
            "ci_95_upper_ms": ci_upper_ms,
        },
        "formula": {
            "T_cuda_gemm": "max(0, T_aten_xlat - T_aten_base)",
            "T_cuda_non_gemm": "0.0",
            "T_aten_base": "Median(Non-GEMM T_aten_xlat)",
            "T_sys": f"{t_sys_null:.3f} μs (hardcoded null kernel)"
        },
        "invocations": {
            "total": total_invocations,
            "per_run": invocations_per_run,
            "gemm": gemm_invocations,
            "non_gemm": non_gemm_invocations,
        },
        "per_kernel": summary_data,
    }
    utils.save_json(summary_path, output_data)
    print(f"\nSaved direct trace summary to {summary_path}")
    
    # Plot per-kernel from trace
    plot_per_kernel_taxbreak(summary_data, f"{model} | {precision} (Direct Trace)")
    print("=== TaxBreak Report (Direct Trace) ===")

class TaxBreakAnalyzer:
    def __init__(self, tracer, args):
        self.tracer = tracer
        self.args = args

    def run_direct_from_trace(self) -> None:
        """
        Simplified pipeline: compute T_structural directly from the original trace.
        
        No replay needed - uses the trace data already collected by ModelTracer.
        Each kernel invocation is treated uniquely (no deduplication by config).
        """
        section = "Direct Trace Analysis"
        print_utils.section_start(section)
        
        # Get sequences from tracer (already collected during tracing)
        sequences = list(self.tracer.sequences)
        
        print(f"Raw sequences from tracer: {len(sequences)}")
        
        # Calculate metrics on all sequences (each invocation is unique)
        sequences_with_metrics = utils.calculate_sequence_metrics(
            sequences, 
            metrics=["launch_tax", "aten_xlat_tax", "py_tax"]
        )
        
        # Filter for valid kernel sequences and mark is_gemm
        kernel_sequences = utils.filter_kernel_sequences(sequences_with_metrics)
        
        gemm_count = sum(1 for s in kernel_sequences if s.get("is_gemm", False))
        non_gemm_count = len(kernel_sequences) - gemm_count
        
        print(f"Found {len(kernel_sequences)} kernel invocations in trace:")
        print(f"  - {gemm_count} GEMM invocations")
        print(f"  - {non_gemm_count} non-GEMM invocations")
        
        # NO DEDUPLICATION - each invocation is unique
        # Just add freq=1 to each sequence for compatibility with report
        for seq in kernel_sequences:
            seq["freq"] = 1
        
        # Save all sequences (not deduplicated)
        all_sequences_file = utils.get_path("UNIQUE_ALL_SEQUENCES")
        all_data = {
            "summary": {
                "total_invocations": len(kernel_sequences),
                "gemm_invocations": gemm_count,
                "non_gemm_invocations": non_gemm_count,
                "note": "Each kernel invocation is unique (no deduplication)"
            },
            "sequences": kernel_sequences
        }
        utils.save_json(all_sequences_file, all_data)
        
        print_utils.section_end(section)
        
        # Generate report directly from trace-derived sequences
        section = "TaxBreak Report (Direct Trace)"
        print_utils.section_start(section)
        
        summarize_from_trace_directly(
            unique_sequences=kernel_sequences,  # All invocations, not deduplicated
            model=self.args.model,
            precision=self.args.precision
        )
        
        print_utils.section_end(section)