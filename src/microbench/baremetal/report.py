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
from collections import defaultdict
from typing import Any, Dict, List

import matplotlib.pyplot as plt

from soda.common import utils, print_utils

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
        kernel_display = kernel_name[:20] + "..." if len(kernel_name) > 20 else kernel_name

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

    # Load baremetal results (GEMM only)
    baremetal_file = utils.get_path("BAREMETAL_GEMM_KERNELS")
    baremetal_results = {}
    if os.path.exists(baremetal_file):
        baremetal_results = load_baremetal_results(baremetal_file)

    print(f"Loaded {len(pytorch_results)} PyTorch sequences")
    print(f"Loaded {len(baremetal_results)} baremetal sequences")

    final_summary_table = []
    final_summary_data = []
    all_job_ids = sorted(set(pytorch_results.keys()) | set(baremetal_results.keys()))

    for job_id in all_job_ids:
        fw = pytorch_results.get(job_id)
        bm = baremetal_results.get(job_id)

        py = fw["py_tax"] if fw else None
        aten_xlat = fw["aten_xlat_tax"] if fw else None
        sys_fw = fw["launch_tax"] if fw else None
        is_gemm = fw.get("is_gemm", False) if fw else False

        culib_xlat = bm["culib_xlat_tax"] if bm else None
        shim = bm["shim_tax"] if bm else None
        sys_bm = bm["launch_tax"] if bm else None

        setup = bm["culib_setup"] if bm else None
        heur = bm["culib_heur"] if bm else None

        sys = sys_bm if sys_bm is not None else sys_fw

        kernel_name = fw["kernel"] if fw else (bm["kernel"] if bm else None)
        aten_op_name = fw["aten_op"] if fw else None
        freq = fw["freq"] if fw else None

        # Calculate T_aten (framework ATen overhead minus cuBLASLt translation)
        aten = None
        if aten_xlat is not None and culib_xlat is not None:
            aten = aten_xlat - culib_xlat

        # Framework overhead calculations
        fo_bm = None
        if all(x is not None for x in (py, aten, setup, heur, shim, sys)):
            fo_bm = py + aten + setup + heur + shim + sys

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
            "T_fo": fo_bm if fo_bm is not None else fo_fw,
            "T_fo_bm": fo_bm,
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
        
        # Use T_aten if available (GEMM with baremetal), else T_aten_xlat
        aten_display = aten if aten is not None else aten_xlat

        final_summary_table.append([
            job_id,
            aten_op_name,
            kernel_display,
            kernel_type,
            fmt_val(fo_bm if fo_bm is not None else fo_fw),
            fmt_val(py),
            fmt_val(aten_display),
            fmt_val(setup),
            fmt_val(heur),
            fmt_val(shim),
            fmt_val(sys),
            freq,
        ])

    if show_table:
        print_utils.comp_table(
            title="Framework Tax Break (μs) - All Kernels",
            headers=[
                "ID", "ATen op", "Kernel", "Type",
                "T_fo", "T_py", "T_aten*",
                "T_lib_setup", "T_lib_heur", "T_lib_shim",
                "T_sys", "freq"
            ],
            data=final_summary_table,
        )
        print("Notes:")
        print(" * Type: GEMM (full breakdown) or other (partial - no baremetal comparison)")
        print(" * T_fo: framework overhead")
        print(" * T_py: python xlat tax; torch_op -> aten_op")
        print(" * T_aten*: For GEMM: T_aten_xlat - T_lib_xlat; For others: T_aten_xlat (no breakdown)")
        print(" * T_lib_*: cuBLASLt components (GEMM only)")
        print(" * T_sys: kernel launch tax; launch -> kernel")

    return final_summary_data


def plot_per_kernel_taxbreak(final_summary_data: List[Dict], title_suffix: str):
    """Plot stacked bar chart for all kernels."""
    output_path = utils.get_path("TAX_BREAK_PLOT")

    # Full components (GEMM with baremetal)
    full_components = [
        ("T_py", "#4e79a7"),
        ("T_aten", "#59a14f"),
        ("T_lib_setup", "#f28e2c"),
        ("T_lib_heur", "#e15759"),
        ("T_lib_shim", "#edc949"),
        ("T_sys", "#af7aa1"),
    ]
    
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
        
        if is_gemm:
            vals = [row.get(name) for name, _ in full_components]
            if all(v is not None for v in vals):
                gemm_entries.append((row["id"], row["kernel"], vals, full_components))
            else:
                vals = [row.get(name) for name, _ in pytorch_components]
                if all(v is not None for v in vals):
                    gemm_entries.append((row["id"], row["kernel"], vals, pytorch_components))
        else:
            vals = [row.get(name) for name, _ in pytorch_components]
            if all(v is not None for v in vals):
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

def summarize(model: str, precision: str, include_all_kernels: bool = False):
    """
    Main reporting function.
    
    Args:
        model: Model name for plot title
        precision: Precision for plot title  
        include_all_kernels: If True, include non-GEMM kernels in the report
    """
    generate_framework_summary(show_table=True, include_all_kernels=include_all_kernels)
    generate_baremetal_summary(show_table=True)
    final_summary_data = generate_final_summary(show_table=True, include_all_kernels=include_all_kernels)

    summary_path = utils.get_path("TAX_BREAK_SUMMARY")
    utils.save_json(
        summary_path,
        {
            "summary": {
                "count": len(final_summary_data),
                "gemm_count": sum(1 for d in final_summary_data if d.get("is_gemm", False)),
                "non_gemm_count": sum(1 for d in final_summary_data if not d.get("is_gemm", False)),
            },
            "data": final_summary_data,
        },
    )
    print(f"Saved taxbreak summary to {summary_path}")

    model = model or "<unknown_model>"
    precision = precision or "<unknown_precision>"
    title_suffix = f"{model} [{precision}]"
    plot_per_kernel_taxbreak(final_summary_data, title_suffix=title_suffix)
