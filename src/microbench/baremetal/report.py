#!/usr/bin/env python3
"""
Compare PyTorch and baremetal GEMM launch tax.

Joins results from framework/pytorch/output/unique_gemm_sequences.json and
baremetal/output/baremetal_gemm_runs.json, verifies kernel matching,
computes per-kernel launch tax deltas and percentages, and emits
baremetal/output/bm_vs_framework_report.json.
"""

import json
import os
import sys
import statistics
from collections import defaultdict
from typing import Any, Dict, List

import matplotlib.pyplot as plt

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
        import torch
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


def get_null_kernel_sys_tax() -> float:
    """
    Get the null kernel's system launch tax (hardcoded by GPU architecture).
    
    Returns:
        T_sys for null kernel in microseconds
    """
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
        kernel_display = kernel_name[:40] + "..." if len(kernel_name) > 40 else kernel_name

        kernel_display = kernel_name[:40] + "..." if len(kernel_name) > 40 else kernel_name
        
        # Truncate ATen op name slightly if too long, but keep it readable
        aten_op = job_result["aten_op"]
        aten_display = aten_op[:25] + "..." if len(aten_op) > 25 else aten_op 

        framework_summary_table.append([
            job_id,
            job_result["aten_op"],
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

        kernel_display = kernel_name[:15] + "..." if kernel_name and len(kernel_name) > 15 else kernel_name
        
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
            # Formula: T_cuda = T_fo - T_py - T_aten_base - T_sys
            # (Note: T_fo ≈ T_py + T_aten_xlat + T_sys, so this is equivalent to:
            #  T_cuda = T_aten_xlat - T_aten_base)
            
            # We use the component-based formula for clarity:
            # T_cuda = T_aten_xlat - T_aten_base
            # (Since T_aten_xlat includes both pure dispatch + library overhead)
            
            raw_cuda = t_aten_xlat - t_aten_base
            t_cuda = max(0.0, raw_cuda)
            
            # For GEMM, "Compute Translation" is the baseline dispatch cost
            t_ct = t_aten_base
            
            gemm_invocations += freq
        else:
            # Non-GEMM: Pure dispatch, no library overhead
            t_cuda = 0.0
            # For non-GEMM, "Compute Translation" is the full measured cost
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