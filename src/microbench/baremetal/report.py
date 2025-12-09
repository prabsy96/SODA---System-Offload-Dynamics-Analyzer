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
from pathlib import Path

import matplotlib.pyplot as plt

from soda.common import utils, print_utils

def load_pytorch_results(pytorch_file):
    """
    Load PyTorch event sequences and extract per-kernel launch statistics.
    
    Returns: dict mapping job_id -> {kernel, op_signature, stats, xlat_stats, py_stats}
    """
    data = utils.load_json(pytorch_file)
    
    sequences = data["sequences"]
    results = {}
    
    for idx, sequence in enumerate(sequences):
        if sequence is None:
            continue
        job_id = f"{idx+1:04d}"
        
        results[job_id] = {
            "freq": sequence["freq"],
            "kernel": sequence["kernel"]["name"],
            "aten_op": sequence["aten_op"]["name"],
            "launch_tax": sequence["launch_tax"]["avg"],
            "aten_xlat_tax": sequence["aten_xlat_tax"]["avg"],
            "py_tax": sequence["py_tax"]["avg"],
        }
    
    return results

def load_baremetal_results(baremetal_file):
    """
    Load baremetal results 
    
    Returns: dict mapping job_id -> {kernel, launch_tax, shim_tax, culib_xlat_tax}
    """
    data = utils.load_json(baremetal_file)

    sequences = data["sequences"]
    results = {}
    
    for sequence in sequences:
        # Skip None entries (e.g., skipped batched GEMM jobs)
        if sequence is None:
            continue

        job_id = sequence["job_id"]
        # Null kernel jobs have no cuBLASLt translation phases.
        # Shim tax is still valid, however, its defined as shim_tax = launch - run 
        # Were run is the time __null_kernel__ is called.
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


def get_framework_summary(show_table: bool = True):
    """Per GEMM Framework Overhead (μs) @ Pytorch Scope"""
    def fmt_val(v):
        return None if v is None else f"{v:.2f}"

    pytorch_file = utils.get_path("PYTORCH_GEMM_SEQUENCES")
    utils.ensure_file(pytorch_file)
    pytorch_results = load_pytorch_results(pytorch_file)
    print(f"Loaded {len(pytorch_results)} PyTorch sequences")

    framework_summary_data = []
    for job_id, job_result in sorted(pytorch_results.items()):

        py_tax = job_result["py_tax"]
        aten_xlat_tax = job_result["aten_xlat_tax"]
        launch_tax = job_result["launch_tax"]
        # Framework overhead is everything from torch_op to kernel 
        # torch_op -> aten_op -> launch -> kernel
        # where torch_op is start of torch.mm() call 
        # and kernel is start of kernel execution
        fo = py_tax + aten_xlat_tax + launch_tax

        framework_summary_data.append([
            job_id,
            job_result["kernel"],
            fmt_val(fo),
            fmt_val(py_tax),
            fmt_val(aten_xlat_tax),
            fmt_val(launch_tax),
            job_result["freq"],
        ])

    if show_table:
        print_utils.comp_table(
            title=f"Per GEMM Framework Overhead (μs) @ Pytorch Scope",
            headers=["ID", "Kernel", "T_fo", "T_py", "T_aten+lib", "T_sys", "freq"],
            data=framework_summary_data,
        )
        print("Notes:")
        print(" * Kernel: CUDA kernel signature")
        print(" * T_fo: framework overhead; T_py + T_aten+lib + T_sys")
        print(" * T_py: python xlat tax; torch_op -> aten_op")
        print(" * T_aten+culib: ATen+cuBLASLt xlat tax; aten_op -> launch")
        print(" * T_sys: kernel launch tax; launch -> kernel")
        print(" * freq: times this aten_op->kernel sequence appears in one forward pass (unique key=name+grid+block+smem+input_dims)")
    return framework_summary_data, pytorch_results


def get_baremetal_summary(show_table: bool = True):
    """Per GEMM Baremetal Overhead (μs)"""
    def fmt_val(v):
        return None if v is None else f"{v:.2f}"

    baremetal_file = utils.get_path("BAREMETAL_GEMM_KERNELS")
    utils.ensure_file(baremetal_file)
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
        print("Notes:")
        print(" * Kernel: CUDA kernel signature")
        print(" * T_culib_xlat: cuBLASLt translation (setup+heur+shim)")
        print(" * T_setup: cuBLASLt matmul descriptor/config setup")
        print(" * T_heur: cuBLASLt heuristic search for algorithm")
        print(" * T_shim: launch - matMul (null kernel uses run == __null_kernel__)")
        print(" * T_sys: kernel launch tax; launch -> kernel")
    return baremetal_summary_data, baremetal_results


def get_final_summary(pytorch_results, baremetal_results, show_table: bool = True):
    """
    Combined summary across framework and baremetal.
    Columns: ID, Kernel, aten_op, T_fo, T_py, T_aten, T_lib_setup, T_lib_heur, T_lib_shim, T_sys, freq.
    """
    def fmt_val(v):
        return None if v is None else f"{v:.2f}"

    final_summary_table = []
    final_summary_data = []
    all_job_ids = sorted(set(pytorch_results.keys()) | set(baremetal_results.keys()))

    for job_id in all_job_ids:
        # Get framework and baremetal results for this job
        fw = pytorch_results.get(job_id)
        bm = baremetal_results.get(job_id)

        py = fw["py_tax"] if fw else None
        aten_xlat = fw["aten_xlat_tax"] if fw else None
        sys_fw = fw["launch_tax"] if fw else None

        culib_xlat = bm["culib_xlat_tax"] if bm else None
        shim = bm["shim_tax"] if bm else None
        sys_bm = bm["launch_tax"] if bm else None

        setup = None
        heur = None
        if bm:
            culib_setup = bm["culib_setup"]
            culib_heur = bm["culib_heur"]
            setup = culib_setup if culib_setup is not None else None
            heur = culib_heur if culib_heur is not None else None

        # Prefer baremetal T_sys; fall back to framework if needed.
        # TODO: Which sys should we pick? Using bm for now.
        sys = sys_bm if sys_bm is not None else sys_fw

        kernel_name = None
        if fw:
            kernel_name = fw["kernel"]
        elif bm:
            kernel_name = bm["kernel"]

        aten_op_name = fw["aten_op"] if fw and fw.get("aten_op") else None

        freq = fw["freq"] if fw and fw.get("freq") is not None else None

        aten = None
        if aten_xlat is not None and culib_xlat is not None:
            aten = aten_xlat - culib_xlat

        fo_bm = None
        if all(x is not None for x in (py, aten, setup, heur, shim, sys)):
            fo_bm = py + aten + setup + heur + shim + sys

        fo_fw = None
        if all(x is not None for x in (py, aten, sys)):
            fo_fw = py + aten + sys

        final_summary_data.append({
            "id": job_id,
            "aten_op": aten_op_name,
            "kernel": kernel_name,
            "T_fo": fo_bm,
            "T_fo_bm": fo_bm,
            "T_fo_fw": fo_fw,
            "T_py": py,
            "T_aten": aten,
            "T_lib_setup": setup,
            "T_lib_heur": heur,
            "T_lib_shim": shim,
            "T_sys_bm": sys_bm,
            "T_sys_fw": sys_fw,
            "T_sys": sys,
            "freq": freq,
        })

        final_summary_table.append([
            job_id,
            aten_op_name,
            kernel_name,
            fmt_val(fo_bm),
            fmt_val(py),
            fmt_val(aten),
            fmt_val(setup),
            fmt_val(heur),
            fmt_val(shim),
            fmt_val(sys),
            freq,
        ])

    if show_table:
        print_utils.comp_table(
            title="Framework Tax Break (us)",
            headers=[
                "ID", "ATen op", "Kernel",
                "T_fo", "T_py", "T_aten",
                "T_lib_setup", "T_lib_heur", "T_lib_shim",
                "T_sys", "freq"
            ],
            data=final_summary_table,
        )
        print("Notes:")
        print(" * T_fo (fw): framework overhead; T_py + T_aten+lib + T_sys (fm) (but we're using the bm flavor)")
        print(" * T_fo (bm): framework overhead; T_py + T_aten + T_lib_setup + T_lib_heur + T_lib_shim + T_sys (bm)")
        print(" * T_py: python xlat tax; torch_op -> aten_op")
        print(" * T_aten: framework aten+lib minus cuBLASLt translation (T_aten+lib - T_lib_xlat)")
        print(" * T_lib_xlat: cuBLASLt translation (setup+heur+shim)")
        print(" * T_lib_setup: cuBLASLt matmul descriptor/config setup")
        print(" * T_lib_heur: cuBLASLt heuristic search for algorithm")
        print(" * T_lib_shim: launch - matMul (null kernel uses run == __null_kernel__)")
        print(" * freq: times this aten_op->kernel sequence appears")

    return final_summary_table, final_summary_data


def compute_weighted_averages(final_summary_data):
    """
    Weighted averages over entries where all components are present.
    Weights: freq (defaults to 1 when None).
    """
    keys = ["T_py", "T_aten", "T_lib_setup", "T_lib_heur", "T_lib_shim", "T_sys", "T_fo"]
    totals = {k: 0.0 for k in keys}
    weight_sum = 0.0

    for row in final_summary_data:
        if not all(row.get(k) is not None for k in keys):
            continue
        w = row["freq"] if row.get("freq") is not None else 1.0
        weight_sum += w
        for k in keys:
            totals[k] += row[k] * w

    if weight_sum == 0.0:
        return {k: None for k in keys}

    return {k: totals[k] / weight_sum for k in keys}


def plot_weighted_average_stacked(averages, output_path: Path):
    """
    Plot a stacked bar of weighted averages.
    """
    components = [
        ("T_py", "#4e79a7"),
        ("T_aten", "#59a14f"),
        ("T_lib_setup", "#f28e2c"),
        ("T_lib_heur", "#e15759"),
        ("T_lib_shim", "#edc949"),
        ("T_sys", "#af7aa1"),
    ]
    vals = [averages.get(name) for name, _ in components]
    if any(v is None for v in vals):
        return None

    bottoms = []
    acc = 0.0
    for v in vals:
        bottoms.append(acc)
        acc += v

    fig, ax = plt.subplots(figsize=(6, 4))
    x = [0]
    for (name, color), v, b in zip(components, vals, bottoms):
        ax.bar(x, [v], bottom=[b], label=name, color=color)

    ax.set_ylabel("μs")
    ax.set_xticks([])
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_per_kernel_taxbreak(final_summary_data, output_path: Path, title_suffix: str = ""):
    """
    Plot one stacked bar per kernel (job_id) for entries with complete data.
    """
    components = [
        ("T_py", "#4e79a7"),
        ("T_aten", "#59a14f"),
        ("T_lib_setup", "#f28e2c"),
        ("T_lib_heur", "#e15759"),
        ("T_lib_shim", "#edc949"),
        ("T_sys", "#af7aa1"),
    ]

    filtered = []
    for row in final_summary_data:
        vals = [row.get(name) for name, _ in components]
        if any(v is None for v in vals):
            continue
        filtered.append((row["id"], vals))

    if not filtered:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = list(range(len(filtered)))
    job_labels = [jid for jid, _ in filtered]

    bottoms = [0.0] * len(filtered)
    for (name, color), idx_component in zip(components, range(len(components))):
        heights = [vals[idx_component] for _, vals in filtered]
        ax.bar(x, heights, bottom=bottoms, label=name, color=color)
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    ax.set_ylabel("μs")
    title = "Framework Tax Break (us)"
    if title_suffix:
        title = f"{title} | {title_suffix}"
    ax.set_title(title)
    ax.set_xlabel("Job ID")
    ax.set_xticks(x)
    ax.set_xticklabels(job_labels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path

def summarize(model_name: str = None, dtype: str = None, output_path: Path = None):
    """
    Main reporting function.
    """
    print("\n")
    framework_summary, pytorch_results = get_framework_summary(show_table=True)
    print("\n")
    baremetal_summary, baremetal_results = get_baremetal_summary(show_table=True)
    print("\n")
    final_summary_table, final_summary_data = get_final_summary(pytorch_results, baremetal_results, show_table=True)

    if output_path is None:
        base_dir = utils.get_path("BAREMETAL_OUTPUT_DIR").parent
        output_path = base_dir / "taxbreak.png"
    title_suffix = ""
    if model_name:
        title_suffix += f"{model_name}"
    if dtype:
        title_suffix += f" [{dtype}]"
    plot_path = plot_per_kernel_taxbreak(final_summary_data, output_path, title_suffix=title_suffix)
    if plot_path:
        print(f"Saved per-kernel taxbreak plot to {plot_path}")
    else:
        print("No plot saved (no entries with complete data).")

    return {
        "framework_summary": framework_summary,
        "baremetal_summary": baremetal_summary,
        "final_summary_table": final_summary_table,
        "final_summary_data": final_summary_data,
        "plot_path": str(plot_path) if plot_path else None,
    }
