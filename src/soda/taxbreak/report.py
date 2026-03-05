"""
Enhanced TaxBreak report generation.

Merges kernel-database metadata, dynamic system-floor measurements,
per-kernel isolation-replay launch taxes and ATen dispatch taxes (nsys),
and optional ncu cache/memory metrics into a single
``enhanced_taxbreak.json`` report.

CT (ATen dispatch tax) and KT (launch tax) are preferentially sourced
from isolation replay when available, falling back to full-trace
subtraction-based values.
"""

import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from soda.common import print_utils

# Map raw ncu metric names → friendly report keys.
_NCU_FRIENDLY = {
    "l1tex__t_sector_hit_rate.pct": "l1_hit_rate_pct",
    "lts__t_sector_hit_rate.pct": "l2_hit_rate_pct",
    "l1tex__t_bytes.sum.per_second": "l1_throughput_bytes_per_sec",
    "lts__t_bytes.sum.per_second": "l2_throughput_bytes_per_sec",
    "dram__bytes.sum.per_second": "dram_throughput_bytes_per_sec",
    "dram__bytes_read.sum": "dram_bytes_read",
    "dram__bytes_write.sum": "dram_bytes_write",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "compute_throughput_pct",
}


def _remap_ncu_metrics(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Translate raw ncu metric names to friendly report names."""
    friendly = {}
    for raw_key, value in raw.items():
        name = _NCU_FRIENDLY.get(raw_key, raw_key)
        friendly[name] = value
    return friendly


def _derive_dispatch_base(
    kernels: List[Dict[str, Any]],
) -> float:
    """Derive baseline dispatch cost (T_dispatch_base) from I_lib=0 kernels.

    Returns the median ``avg_T_dispatch_us`` across framework-native (I_lib=0)
    kernels per paper Eq.5:

        T_dispatch_base = median({T_dispatch : k ∈ I_lib=0})

    This correctly includes framework-native GEMM kernels (nvjet/wgmma,
    is_gemm=True but i_lib=0) that the old ``not is_gemm`` filter excluded.
    """
    vals = [
        entry["taxes"]["avg_T_dispatch_us"]
        for entry in kernels
        if entry["classification"].get("i_lib", 0) == 0
        and entry["taxes"].get("avg_T_dispatch_us", 0) > 0.1
    ]
    if not vals:
        print(
            "Warning: _derive_dispatch_base found no suitable I_lib=0 (framework-native) "
            "kernels to estimate T_dispatch_base. Falling back to 0.0 µs (no δCT split "
            "will be applied). Re-run with a model that has I_lib=0 kernels, or check "
            "that the kernel DB was captured with sufficient coverage.",
            file=__import__('sys').stderr,
        )
        return 0.0
    return statistics.median(vals)


def _derive_dispatch_base_replay(
    nsys_results: Dict[str, Dict[str, Any]],
    kernels: List[Dict[str, Any]],
) -> Optional[float]:
    """Derive dispatch baseline (T_dispatch_base) from replay ``t_dispatch`` for I_lib=0 kernels.

    Uses the median replay-measured ATen dispatch time (NVTX → cudaLaunch gap)
    across framework-native (I_lib=0) kernels per paper Eq.5.  This is the
    replay-based analogue of ``_derive_dispatch_base()`` and is preferred when
    isolation replay data with NVTX instrumentation is available.

    Correctly includes framework-native GEMM kernels (nvjet/wgmma: is_gemm=True
    but i_lib=0) that the old ``is_gemm`` gate excluded.

    Returns:
        Median T_dispatch_base in microseconds, or ``None`` if no replay data exists.
    """
    vals = []
    for entry in kernels:
        kid = entry["id"]
        if entry["classification"].get("i_lib", 0) != 0:
            continue
        nsys = nsys_results.get(kid)
        if nsys and "t_dispatch" in nsys:
            avg = nsys["t_dispatch"]["avg_us"]
            if avg > 0.1:
                vals.append(avg)
    return statistics.median(vals) if vals else None


def _make_verbose_args(verbose: bool):
    """Minimal args-like namespace so render_taxbreak_analysis can read args.verbose."""
    import types
    ns = types.SimpleNamespace()
    ns.verbose = verbose
    return ns


def generate_enhanced_report(
    kernel_db: Dict[str, Any],
    floor: Dict[str, Any],
    nsys_results: Dict[str, Dict[str, Any]],
    ncu_results: Dict[str, Dict[str, Any]],
    output_dir: Path,
    verbose: bool = False,
) -> Path:
    """Build and write the enhanced TaxBreak report.

    Args:
        kernel_db: Full kernel database dict (from ``kernel_database.json``).
        floor: Dynamic system-floor measurement dict.
        nsys_results: ``{kernel_id: nsys_result_dict}`` from isolation replay.
        ncu_results: ``{kernel_id: ncu_result_dict}`` from ncu profiling.
        output_dir: Directory to write the report into.
        verbose: If True, print full expert tables in addition to the
                 layperson summary (default False).

    Returns:
        Path to the written ``enhanced_taxbreak.json`` file.
    """
    metadata = kernel_db.get("metadata", {})
    kernels = kernel_db.get("kernels", [])

    # Derive baseline dispatch cost (T_dispatch_base) for delta_CT calculation.
    # Prefer replay-based baseline when NVTX data is available.
    t_dispatch_base_replay = _derive_dispatch_base_replay(nsys_results, kernels)
    t_dispatch_base_trace = _derive_dispatch_base(kernels)
    t_dispatch_base = t_dispatch_base_replay if t_dispatch_base_replay is not None else t_dispatch_base_trace
    t_dispatch_base_source = "replay" if t_dispatch_base_replay is not None else "trace"

    # System floor: the irreducible hardware launch latency (null-kernel baseline).
    # KT_framework = max(0, KT_measured - T_sys) isolates the framework's contribution.
    floor_avg = floor.get("avg_us", 0.0)

    # --- per-kernel entries ---
    per_kernel: List[Dict[str, Any]] = []
    total_structural_us = 0.0
    total_FT_us = 0.0
    total_FT_dispatch_us = 0.0   # baseline dispatch cost (T_dispatch_base × freq) — FT component
    total_delta_ct_us = 0.0      # vendor library overhead (delta_CT × freq) — i_lib=1 only
    total_KT_us = 0.0         # raw (includes T_sys hardware floor)
    total_KT_adj_us = 0.0     # framework-only: max(0, KT - T_sys) per kernel
    total_T_sys_us = 0.0      # hardware floor contribution: T_sys × freq
    total_invocations = 0
    gemm_invocations = 0          # i_lib=1: vendor-library GEMM (cuBLAS/cutlass)
    framework_gemm_invocations = 0 # is_gemm but i_lib=0: framework-native (nvjet/wgmma)
    non_gemm_invocations = 0      # neither GEMM class

    for entry in kernels:
        kid = entry["id"]
        freq = entry["statistics"]["frequency"]
        is_gemm = entry["classification"]["is_gemm"]
        # i_lib resolution: prefer runtime detection from isolation replay (ground truth)
        # over the Phase 1 heuristic (kernel name matching).
        # Backward compat: old DBs without "i_lib" fall back to is_gemm (conservative).
        nsys = nsys_results.get(kid)
        i_lib_db = entry["classification"].get("i_lib", int(is_gemm))
        if nsys and "i_lib_detected" in nsys:
            i_lib = int(nsys["i_lib_detected"])
            i_lib_source = "runtime"
        else:
            i_lib = i_lib_db
            i_lib_source = "heuristic"
        total_invocations += freq

        # Taxes from kernel DB (full-trace measurements, paper notation)
        taxes = entry.get("taxes", {})
        py_tax = taxes.get("avg_T_Py_us", 0.0)
        aten_xlat = taxes.get("avg_T_dispatch_us", 0.0)

        # Launch tax: prefer isolation replay if available
        if nsys:
            launch_tax_avg = nsys["launch_tax"]["avg_us"]
            launch_tax_entry = nsys["launch_tax"]
            replay_method = nsys["replay_method"]
        else:
            launch_tax_avg = taxes.get("avg_T_launch_us", 0.0)
            launch_tax_entry = {
                "avg_us": launch_tax_avg,
                "min_us": launch_tax_avg,
                "max_us": launch_tax_avg,
                "std_us": 0.0,
            }
            replay_method = "trace"

        # T_dispatch: prefer replay-based t_dispatch (NVTX → cudaLaunchKernel gap) if available
        if nsys and "t_dispatch" in nsys:
            dispatch_time = nsys["t_dispatch"]["avg_us"]
            dispatch_entry = nsys["t_dispatch"]
            dispatch_source = "replay"
        else:
            dispatch_time = aten_xlat
            dispatch_entry = {"avg_us": aten_xlat}
            dispatch_source = "trace"

        # delta_CT: CUDA library translation overhead — only for vendor-library kernels (i_lib=1).
        # nvjet/wgmma/s884gemm have i_lib=0; they have no CUDA library front-end phase.
        # Uses replay-based dispatch_time when available, trace-based aten_xlat otherwise.
        if i_lib == 1:
            delta_ct = max(0.0, dispatch_time - t_dispatch_base)
            ft_dispatch = t_dispatch_base
            gemm_invocations += freq          # cuBLAS/cutlass path
        elif is_gemm:
            delta_ct = 0.0
            ft_dispatch = dispatch_time
            framework_gemm_invocations += freq  # nvjet/wgmma path
        else:
            delta_ct = 0.0
            ft_dispatch = dispatch_time
            non_gemm_invocations += freq      # elementwise, norm, etc.

        # KT decomposition: T_sys (hardware floor) + KT_framework (software overhead)
        kt_framework = max(0.0, launch_tax_avg - floor_avg)

        # Aggregate
        t_fo = py_tax + dispatch_time + launch_tax_avg
        total_structural_us += t_fo * freq
        total_FT_us += py_tax * freq
        total_FT_dispatch_us += ft_dispatch * freq
        total_delta_ct_us += delta_ct * freq
        total_KT_us += launch_tax_avg * freq
        total_KT_adj_us += kt_framework * freq
        total_T_sys_us += floor_avg * freq

        # ncu metrics (optional) — remap to friendly names
        ncu_entry = ncu_results.get(kid)
        ncu_data = _remap_ncu_metrics(ncu_entry["metrics"]) if ncu_entry else None

        per_kernel.append({
            "id": kid,
            "rank": entry["rank"],
            "kernel_name": entry["kernel"]["name"],
            "aten_op": entry["aten_op"].get("name", ""),
            "classification": entry["classification"],
            "frequency": freq,
            "replay_method": replay_method,
            "taxes": {
                "launch_tax_us": launch_tax_entry,
                "kt_framework_us": {"avg_us": round(kt_framework, 4)},
                "aten_xlat_tax_us": {"avg_us": aten_xlat},
                "t_dispatch_us": dispatch_entry,
                "py_tax_us": {"avg_us": py_tax},
                "delta_ct_us": {"avg_us": round(delta_ct, 4)},
                "dispatch_source": dispatch_source,
            },
            "kernel_duration_us": entry["statistics"]["avg_duration_us"],
            "i_lib_runtime": i_lib,
            "i_lib_source": i_lib_source,
            "ncu": ncu_data,
            # Temp field for roofline FLOPs derivation (stripped before JSON)
            "_aten_op_full": entry.get("aten_op", {}),
        })

    # --- roofline analysis (only when ncu data is available) ---
    roofline_section = None
    roofline_data = []
    if ncu_results:
        try:
            from soda.roofline import (
                compute_roofline_data,
                generate_roofline_plot,
                get_gpu_specs,
            )
            gpu_name = metadata.get("gpu_name", "")
            gpu_specs = get_gpu_specs(gpu_name) if gpu_name else None
            if gpu_specs is None:
                print(f"  [roofline] Warning: unknown GPU '{gpu_name}', skipping roofline analysis")
            else:
                roofline_data = compute_roofline_data(per_kernel, gpu_specs)
                if roofline_data:
                    # Generate plot
                    plot_path = output_dir / "roofline.png"
                    generate_roofline_plot(roofline_data, gpu_specs, str(plot_path),
                                          model_name=metadata.get("model", ""))
                    print(f"  [roofline] Saved roofline plot to {plot_path}")

                    # Index by kernel ID for table display
                    roofline_by_id = {r["id"]: r for r in roofline_data}
                    for pk in per_kernel:
                        rf = roofline_by_id.get(pk["id"])
                        if rf:
                            pk["_roofline"] = rf

                    roofline_section = {
                        "gpu_specs": {
                            "gpu_key": gpu_specs["gpu_key"],
                            "peak_tflops_fp16": gpu_specs["peak_tflops_fp16"],
                            "peak_bw_tb_s": gpu_specs["peak_bw_tb_s"],
                            "ridge_point": gpu_specs["ridge_point"],
                        },
                        "kernels": roofline_data,
                        "plot_path": str(plot_path),
                    }
        except ImportError:
            print("  [roofline] Warning: could not import roofline module, skipping")
        except Exception as e:
            print(f"  [roofline] Warning: roofline analysis failed: {e}")

    # --- strip temp fields before serialization ---
    for pk in per_kernel:
        pk.pop("_aten_op_full", None)
        pk.pop("_roofline", None)

    # --- aggregate ---
    total_structural_ms = total_structural_us / 1000.0
    total_FT_ms = total_FT_us / 1000.0
    total_FT_dispatch_ms = total_FT_dispatch_us / 1000.0
    total_delta_ct_ms = total_delta_ct_us / 1000.0
    total_KT_ms = total_KT_us / 1000.0
    total_KT_adj_ms = total_KT_adj_us / 1000.0
    total_T_sys_ms = total_T_sys_us / 1000.0
    # T_structural_adj excludes the hardware floor from KT
    total_structural_adj_ms = total_structural_ms - total_T_sys_ms

    # --- HDBI using dynamic T_sys (requires TaxBreak floor measurement) ---
    # total_kernel_exec_us: weighted sum of avg kernel duration × invocation count
    total_kernel_exec_us = sum(
        e["statistics"]["avg_duration_us"] * e["statistics"]["frequency"]
        for e in kernels
    )
    # T_Orchestrate = FT_python + FT_dispatch + delta_CT + delta_KT(T_sys).
    # FT_dispatch and delta_CT come from replay-based dispatch_time when available,
    # falling back to trace-based aten_xlat.  In both cases:
    #   i_lib=1: FT_dispatch = t_dispatch_base, delta_CT = dispatch - t_dispatch_base
    #   i_lib=0: FT_dispatch = dispatch, delta_CT = 0
    total_xlat_for_hdbi_ms = (total_FT_us + total_FT_dispatch_us + total_delta_ct_us) / 1000.0
    try:
        from soda.common import utils as _utils
        hdbi_metrics = _utils.calculate_hdbi(
            total_kernel_exec_time_ms=total_kernel_exec_us / 1000.0,
            t_orchestrate_excl_kt_ms=total_xlat_for_hdbi_ms,
            num_total_kernels=total_invocations,
            t_sys_us=floor_avg,
        )
    except Exception:
        hdbi_metrics = None

    # T_Orchestrate (paper Eq.2): ΔFT + I_lib·ΔCT + ΔKT(T_sys)
    # ΔKT = total_invocations × T_sys
    _delta_KT_ms = total_invocations * (floor_avg / 1000.0)
    _T_Orchestrate_ms = total_xlat_for_hdbi_ms + _delta_KT_ms

    report = {
        "version": "2.0",
        "timestamp": datetime.now().isoformat(),
        "system_floor": floor,
        "model_metadata": metadata,
        "summary": {
            "total_unique_kernels": len(kernels),
            "kernels_with_nsys": len(nsys_results),
            "kernels_with_ncu": len(ncu_results),
            "total_invocations": total_invocations,
            # Vendor-library GEMM (i_lib=1): cuBLAS/cuBLASLt/cutlass kernels
            "library_gemm_invocations": gemm_invocations,
            # Framework-native GEMM (is_gemm but i_lib=0): nvjet/wgmma/s884gemm
            "framework_gemm_invocations": framework_gemm_invocations,
            # Non-GEMM: elementwise, normalization, copy, etc.
            "non_gemm_invocations": non_gemm_invocations,
        },
        "derived_baselines": {
            "t_dispatch_base_us": round(t_dispatch_base, 4),
            "t_dispatch_base_source": t_dispatch_base_source,
            "system_floor_avg_us": floor["avg_us"],
        },
        "aggregate": {
            # T_host_observed_ms: total raw host overhead per inference pass (T_launch includes T_sys floor)
            "T_host_observed_ms": round(total_structural_ms, 4),
            # T_host_framework_ms: framework-attributable overhead only (delta_KT_framework replaces raw T_launch)
            "T_host_framework_ms": round(total_structural_adj_ms, 4),
            # T_Orchestrate_ms: paper Eq.2 — Σ(ΔFT + I_lib·ΔCT + ΔKT(T_sys))
            "T_Orchestrate_ms": round(_T_Orchestrate_ms, 4),
            # T_sys floor contribution across all invocations in this run
            "T_sys_floor_ms": round(total_T_sys_ms, 4),
            "breakdown_mean": {
                "delta_FT_py_ms": round(total_FT_ms, 4),
                "delta_FT_dispatch_ms": round(total_FT_dispatch_ms, 4),
                "delta_CT_ms": round(total_delta_ct_ms, 4),
                # T_launch_raw_ms: raw measured (= delta_KT_framework + T_sys floor per kernel)
                "T_launch_raw_ms": round(total_KT_ms, 4),
                # delta_KT_framework_ms: framework-attributable launch overhead only
                # = max(0, T_launch_measured - T_sys) per kernel × frequency
                "delta_KT_framework_ms": round(total_KT_adj_ms, 4),
            },
        },
        "per_kernel": per_kernel,
    }

    if hdbi_metrics is not None:
        report["hdbi"] = hdbi_metrics

    if roofline_section:
        report["roofline"] = roofline_section

    # --- write JSON ---
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "enhanced_taxbreak.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # --- layperson summary (always shown) ---
    from soda.common.summary_report import render_taxbreak_analysis
    render_taxbreak_analysis(report, _make_verbose_args(verbose), output_dir)

    # --- expert output (only with --verbose) ---
    if verbose:
        _print_summary(report, floor, nsys_results, ncu_results, roofline_data)
        _print_per_kernel_table(per_kernel, ncu_results, roofline_data)

    print(f"\nSaved enhanced TaxBreak report to {report_path}")
    return report_path


# ------------------------------------------------------------------
# Console output helpers
# ------------------------------------------------------------------

def _print_summary(
    report: Dict[str, Any],
    floor: Dict[str, Any],
    nsys_results: Dict[str, Any],
    ncu_results: Dict[str, Any],
    roofline_data: List[Dict[str, Any]] = None,
) -> None:
    """Print the aggregate structural overhead summary."""
    agg = report["aggregate"]
    bk = agg["breakdown_mean"]
    meta = report.get("model_metadata", {})
    baselines = report.get("derived_baselines", {})
    summary = report.get("summary", {})

    model = meta.get("model", "unknown")
    gpu = meta.get("gpu_name", "unknown")

    print()
    print(f"=== Enhanced TaxBreak Report: {model} on {gpu} ===")
    print(f"  System floor (dynamic) : {floor['avg_us']:.2f} us "
          f"(min={floor['min_us']:.2f}, max={floor['max_us']:.2f})")
    print(f"  T_dispatch_base ({baselines.get('t_dispatch_base_source', 'trace')})  : {baselines.get('t_dispatch_base_us', 0):.2f} us")
    print()
    print(f"  T_host_observed (total/inference)     : {agg['T_host_observed_ms']:.3f} ms  [raw, T_launch includes T_sys]")
    print(f"  T_host_framework (framework overhead)  : {agg.get('T_host_framework_ms', 0):.3f} ms  [delta_KT_framework only]")
    print(f"  T_Orchestrate    (paper Eq.2)          : {agg.get('T_Orchestrate_ms', 0):.3f} ms  [ΔFT+I_lib·ΔCT+ΔKT(T_sys)]")
    print(f"  T_sys floor  (hardware latency)    : {agg.get('T_sys_floor_ms', 0):.3f} ms  [unavoidable]")
    print()
    print(f"  Breakdown (raw, per inference):")
    print(f"    ΔFT_py   (T_Py, Python layer)         : {bk['delta_FT_py_ms']:.3f} ms")
    print(f"    ΔFT_disp (ATen dispatch baseline)     : {bk['delta_FT_dispatch_ms']:.3f} ms")
    print(f"    ΔCT      (CUDA lib, I_lib=1 only)     : {bk['delta_CT_ms']:.3f} ms")
    print(f"    T_launch (kernel launch, raw)         : {bk['T_launch_raw_ms']:.3f} ms")
    print(f"    ΔKT_fw   (kernel launch, fwk only)    : {bk.get('delta_KT_framework_ms', 0):.3f} ms")

    # HDBI (requires dynamic T_sys — computed only in TaxBreak pipeline)
    hdbi = report.get("hdbi")
    if hdbi:
        print()
        print(f"  HDBI (dynamic T_sys={floor['avg_us']:.2f} µs):")
        print(f"    HDBI value           : {hdbi['hdbi_value']:.3f}")
        print(f"    T_DeviceActive       : {hdbi['T_DeviceActive_ms']:.3f} ms")
        print(f"    T_Orchestrate        : {hdbi['T_Orchestrate_ms']:.3f} ms")
        print(f"      ΔKT (T_sys×N)      : {hdbi['delta_KT_ms']:.3f} ms")
    print()
    print(f"  Kernels: {summary.get('total_unique_kernels', 0)} unique, "
          f"{summary.get('total_invocations', 0)} invocations "
          f"({summary.get('library_gemm_invocations', 0)} lib-GEMM, "
          f"{summary.get('framework_gemm_invocations', 0)} fw-GEMM, "
          f"{summary.get('non_gemm_invocations', 0)} non-GEMM)")
    print(f"  Profiled: {len(nsys_results)} nsys, {len(ncu_results)} ncu")

    # Roofline summary
    if roofline_data:
        roofline_info = report.get("roofline", {})
        specs = roofline_info.get("gpu_specs", {})
        compute_bound = sum(1 for r in roofline_data if r["bound"] == "compute")
        memory_bound = sum(1 for r in roofline_data if r["bound"] == "memory")
        print()
        print(f"  Roofline: {len(roofline_data)} kernels plotted "
              f"({compute_bound} compute-bound, {memory_bound} memory-bound)")
        if specs.get("ridge_point"):
            print(f"  Ridge point: {specs['ridge_point']:.2f} FLOP/byte")


def _fmt(v, precision=2):
    """Format a numeric value or return '-'."""
    if v is None:
        return "-"
    try:
        return f"{float(v):.{precision}f}"
    except (TypeError, ValueError):
        return str(v)


def _print_per_kernel_table(
    per_kernel: List[Dict[str, Any]],
    ncu_results: Dict[str, Any],
    roofline_data: List[Dict[str, Any]] = None,
) -> None:
    """Print a per-kernel summary table."""
    headers = [
        "ID", "ATen Op", "Kernel", "Type", "Freq",
        "T_py", "T_disp", "delta_CT", "T_launch", "src",
    ]

    has_ncu = any(entry.get("ncu") for entry in per_kernel)
    if has_ncu:
        headers += ["L1%", "L2%", "Compute%"]

    # Roofline columns
    roofline_by_id = {}
    has_roofline = bool(roofline_data)
    if has_roofline:
        roofline_by_id = {r["id"]: r for r in roofline_data}
        headers += ["AI", "GFLOP/s", "Bound"]

    rows = []
    for entry in per_kernel:
        is_gemm = entry["classification"]["is_gemm"]
        # Prefer runtime-detected i_lib (matches tax computation) over DB heuristic
        i_lib = entry.get("i_lib_runtime", entry["classification"].get("i_lib", int(is_gemm)))
        kernel_class = entry["classification"].get("kernel_class", "gemm" if is_gemm else "non_gemm")
        if i_lib == 1:
            ktype = "Library-mediated (I_lib=1)"
        elif kernel_class == "gemm":
            ktype = "FW-native (I_lib=0)"
        elif kernel_class == "unknown":
            ktype = "unknown"
        else:
            ktype = "FW-native (I_lib=0)"
        kname = entry["kernel_name"]
        kname_display = kname[:35] + "..." if len(kname) > 35 else kname

        row = [
            entry["id"],
            entry["aten_op"],
            kname_display,
            ktype,
            entry["frequency"],
            _fmt(entry["taxes"]["py_tax_us"]["avg_us"]),
            _fmt(entry["taxes"]["t_dispatch_us"]["avg_us"]),
            _fmt(entry["taxes"]["delta_ct_us"]["avg_us"]),
            _fmt(entry["taxes"]["launch_tax_us"]["avg_us"]),
            entry["taxes"].get("dispatch_source", "trace")[:1],  # "r" or "t"
        ]

        if has_ncu:
            ncu = entry.get("ncu") or {}
            row += [
                _fmt(ncu.get("l1_hit_rate_pct")),
                _fmt(ncu.get("l2_hit_rate_pct")),
                _fmt(ncu.get("compute_throughput_pct")),
            ]

        if has_roofline:
            rf = roofline_by_id.get(entry["id"])
            if rf:
                row += [_fmt(rf["ai"]), _fmt(rf["achieved_gflops"], 0), rf["bound"]]
            else:
                row += ["-", "-", "-"]

        rows.append(row)

    print_utils.comp_table(
        title="Enhanced TaxBreak Per-Kernel Summary (us)",
        headers=headers,
        data=rows,
    )
