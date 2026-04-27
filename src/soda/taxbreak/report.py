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
from typing import Any, Dict, List, Optional, Tuple

from soda.common import print_utils

# Map raw ncu metric names → friendly report keys.
_NCU_FRIENDLY = {
    "l1tex__t_sector_hit_rate.pct": "l1_hit_rate_pct",
    "lts__t_sector_hit_rate.pct": "l2_hit_rate_pct",
    "l1tex__t_bytes.sum.per_second": "l1_throughput_bytes_per_sec",
    "lts__t_bytes.sum.per_second": "l2_throughput_bytes_per_sec",
    "dram__bytes.sum.per_second": "dram_throughput_bytes_per_sec",
    "dram__bytes_read.sum": "dram_bytes_read",          # pre-Blackwell
    "dram__bytes_write.sum": "dram_bytes_write",         # pre-Blackwell
    "dram__bytes_op_read.sum": "dram_bytes_read",        # Blackwell CC 12.x+
    "dram__bytes_op_write.sum": "dram_bytes_write",      # Blackwell CC 12.x+
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "compute_throughput_pct",
}

# Ops with embedded CPU-GPU synchronization between kernel steps.
# Their isolation-replay launch_tax captures algorithmic sync wait, not framework
# overhead. Substitute floor_avg and flag as "sync_floor" to exclude from KT_fw.
_SYNC_OPS = frozenset({
    "aten::nonzero",
    "aten::unique",
    "aten::unique_consecutive",
})


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

    This correctly includes framework-native kernels (nvjet/wgmma,
    is_library_mediated=True at op level but i_lib=0) that need to be
    in the baseline pool.
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

    Correctly includes framework-native kernels (nvjet/wgmma: is_library_mediated=True
    at op level but i_lib=0) that the old ``is_gemm`` gate excluded.

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


def _should_sanitize_replay(
    nsys_result: Dict[str, Any],
    floor_avg: float,
    is_framework_native: bool,
) -> List[str]:
    """Return replay contamination reasons for obviously anomalous measurements."""
    if not nsys_result or not is_framework_native:
        return []

    launch = nsys_result.get("launch_tax", {}).get("avg_us", 0.0)
    launch_std = nsys_result.get("launch_tax", {}).get("std_us", 0.0)
    kernel_dur = nsys_result.get("kernel_duration", {}).get("avg_us", 0.0)
    multi_candidate_iterations = nsys_result.get("multi_candidate_iterations", 0)
    measured_iterations = max(1, nsys_result.get("measured_iterations", nsys_result.get("samples", 0)))
    matched_iterations = nsys_result.get("matched_iterations", nsys_result.get("samples", 0))
    match_ratio = matched_iterations / measured_iterations
    variant_match = nsys_result.get("kernel_variant_match", True)

    reasons = []
    if launch > floor_avg * 20 and multi_candidate_iterations > 0:
        reasons.append("ambiguous_multi_kernel_replay")
    if launch > floor_avg * 20 and not variant_match:
        reasons.append("variant_mismatch_replay")
    if launch > floor_avg * 50 and launch_std > floor_avg * 10:
        reasons.append("unstable_launch_distribution")
    if kernel_dur > 0 and launch > max(floor_avg * 100, kernel_dur * 0.75):
        reasons.append("launch_exceeds_kernel_scale")
    if match_ratio < 0.8 and launch > floor_avg * 10:
        reasons.append("incomplete_iteration_match")
    return reasons


def _clamp_dispatch_time(dispatch_time: float, t_dispatch_base: float, floor_avg: float) -> float:
    """Clamp contaminated replay dispatch to a conservative but nonzero bound."""
    max_dispatch = max(t_dispatch_base * 4.0, floor_avg * 10.0)
    return min(dispatch_time, max_dispatch)


def generate_enhanced_report(
    kernel_db: Dict[str, Any],
    floor: Dict[str, Any],
    nsys_results: Dict[str, Dict[str, Any]],
    ncu_results: Dict[str, Dict[str, Any]],
    output_dir: Path,
    verbose: bool = False,
    power_replay_results: Optional[Dict[str, Dict[str, Any]]] = None,
    power_idle_baseline_w: float = 0.0,
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
    library_mediated_invocations = 0     # i_lib=1: vendor-library kernels (cuBLAS/cutlass)
    fw_native_lib_op_invocations = 0     # ATen op suggests library but i_lib=0 (nvjet/wgmma)
    framework_native_invocations = 0     # framework-native: no vendor library involvement

    for entry in kernels:
        kid = entry["id"]
        freq = entry["statistics"]["frequency"]
        is_lib_mediated = entry["classification"].get(
            "is_library_mediated", entry["classification"].get("is_gemm", False)
        )
        # i_lib resolution: prefer runtime detection from isolation replay (ground truth)
        # over the Phase 1 heuristic (kernel name matching).
        # Backward compat: old DBs without "i_lib" fall back to is_library_mediated.
        nsys = nsys_results.get(kid)
        i_lib_db = entry["classification"].get("i_lib", int(is_lib_mediated))
        # Only override the Phase-1 DB heuristic when vendor tracing was active
        # (vendor_tracing_available=True).  If the flag is absent (old cache entries)
        # or False, the detection was inconclusive and we fall back to the DB value.
        if nsys and "i_lib_detected" in nsys and nsys.get("vendor_tracing_available", False):
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
        replay_anomaly_reasons: List[str] = []
        if nsys:
            launch_tax_avg = nsys["launch_tax"]["avg_us"]
            launch_tax_entry = nsys["launch_tax"]
            replay_method = nsys["replay_method"]
            kt_source = "replay"
        else:
            # Replay failed (CPU-only op or no kernel match): use T_sys floor as the
            # only hardware-verified lower bound. Trace-based avg_T_launch_us carries
            # GPU-queue artifacts and must not be used for KT decomposition.
            launch_tax_avg = floor_avg
            launch_tax_entry = {
                "avg_us": floor_avg,
                "min_us": floor_avg,
                "max_us": floor_avg,
                "std_us": 0.0,
            }
            replay_method = "floor_only"
            kt_source = "floor_only"

        replay_anomaly_reasons = _should_sanitize_replay(
            nsys_result=nsys or {},
            floor_avg=floor_avg,
            is_framework_native=(i_lib == 0),
        )
        if replay_anomaly_reasons:
            launch_tax_avg = floor_avg
            launch_tax_entry = {
                "avg_us": floor_avg,
                "min_us": floor_avg,
                "max_us": floor_avg,
                "std_us": 0.0,
                "raw_avg_us": nsys["launch_tax"]["avg_us"],
            }
            replay_method = "anomaly_floor"
            kt_source = "anomaly_floor"

        # Sync-op override: if the op embeds a CPU-GPU sync barrier, the replay
        # launch_tax captures algorithmic wait time, not framework overhead.
        # Only override when replay produced a suspiciously large value (>10x floor).
        op_name_str = entry["aten_op"].get("name", "")
        if kt_source == "replay" and op_name_str in _SYNC_OPS and launch_tax_avg > floor_avg * 10:
            launch_tax_avg = floor_avg
            launch_tax_entry = {
                "avg_us": floor_avg,
                "min_us": floor_avg,
                "max_us": floor_avg,
                "std_us": 0.0,
            }
            replay_method = "sync_floor"
            kt_source = "sync_floor"

        # T_dispatch: prefer replay-based t_dispatch (NVTX → cudaLaunchKernel gap) if available
        if nsys and "t_dispatch" in nsys:
            dispatch_time = nsys["t_dispatch"]["avg_us"]
            dispatch_entry = nsys["t_dispatch"]
            dispatch_source = "replay"
        else:
            dispatch_time = aten_xlat
            dispatch_entry = {"avg_us": aten_xlat}
            dispatch_source = "trace"

        if replay_anomaly_reasons:
            dispatch_time = _clamp_dispatch_time(dispatch_time, t_dispatch_base, floor_avg)
            dispatch_entry = dict(dispatch_entry)
            dispatch_entry["avg_us"] = round(dispatch_time, 4)
            dispatch_entry["raw_avg_us"] = round(nsys.get("t_dispatch", {}).get("avg_us", aten_xlat), 4) if nsys else round(aten_xlat, 4)
            dispatch_source = "anomaly_clamped"

        # delta_CT: CUDA library translation overhead — only for vendor-library kernels (i_lib=1).
        # nvjet/wgmma/s884gemm have i_lib=0; they have no CUDA library front-end phase.
        # Uses replay-based dispatch_time when available, trace-based aten_xlat otherwise.
        if i_lib == 1:
            delta_ct = max(0.0, dispatch_time - t_dispatch_base)
            ft_dispatch = t_dispatch_base
            library_mediated_invocations += freq     # cuBLAS/cutlass path
        elif is_lib_mediated:
            delta_ct = 0.0
            ft_dispatch = dispatch_time
            fw_native_lib_op_invocations += freq     # nvjet/wgmma path
        else:
            delta_ct = 0.0
            ft_dispatch = dispatch_time
            framework_native_invocations += freq     # elementwise, norm, etc.

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

        # power replay (optional)
        power_data = (power_replay_results or {}).get(kid)

        per_kernel.append({
            "id": kid,
            "rank": entry["rank"],
            "kernel_name": entry["kernel"]["name"],
            "aten_op": entry["aten_op"].get("name", ""),
            "classification": entry["classification"],
            "frequency": freq,
            "replay_method": replay_method,
            "kt_source": kt_source,
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
            "power_replay": power_data,
            "replay_anomaly_reasons": replay_anomaly_reasons,
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
            "kernels_with_power_replay": len(power_replay_results) if power_replay_results else 0,
            "total_measured_energy_uj": round(
                sum(r.get("energy_uj", 0.0) for r in (power_replay_results or {}).values()), 2
            ),
            "idle_baseline_w": round(power_idle_baseline_w, 2),
            "total_invocations": total_invocations,
            # Library-mediated (I_lib=1): cuBLAS/cuBLASLt/cutlass kernels
            "library_mediated_invocations": library_mediated_invocations,
            # Framework-native but ATen op suggests library (I_lib=0): nvjet/wgmma/s884gemm
            "fw_native_lib_op_invocations": fw_native_lib_op_invocations,
            # Framework-native: elementwise, normalization, copy, etc.
            "framework_native_invocations": framework_native_invocations,
            # Backward-compatible aliases (deprecated)
            "library_gemm_invocations": library_mediated_invocations,
            "framework_gemm_invocations": fw_native_lib_op_invocations,
            "non_gemm_invocations": framework_native_invocations,
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

    # --- standalone power report (when power replay was run) ---
    if power_replay_results:
        # Ground truth inference power from Stage 1 energy counter measurement
        ground_truth = metadata.get("inference_energy", {})
        stage1_power_profile = metadata.get("power_profile", {})

        # Wall-clock inference time from Stage 1. Provides T_total for energy_balance
        # time decomposition even when NVML energy counter is unavailable.
        inference_time_ms = metadata.get("inference_time_ms", 0.0)

        # P_active_idle from null-kernel tight-loop measurement (Step 4).
        # Stored in floor dict by pipeline.py when --power-replay is active.
        active_idle_power_w = floor.get("active_idle_power_w", 0.0)

        _, inference_power = _write_power_report(
            power_replay_results=power_replay_results,
            idle_baseline_w=power_idle_baseline_w,
            per_kernel=per_kernel,
            metadata=metadata,
            output_dir=output_dir,
            ground_truth=ground_truth,
            inference_time_ms=inference_time_ms,
            active_idle_power_w=active_idle_power_w,
            stage1_power_profile=stage1_power_profile,
        )
        # Surface reconstructed inference power into the main report so that
        # render_taxbreak_analysis can display it without reading a second file.
        report["summary"]["inference_power"] = inference_power

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
# Power report
# ------------------------------------------------------------------

def _write_power_report(
    power_replay_results: Dict[str, Dict[str, Any]],
    idle_baseline_w: float,
    per_kernel: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    output_dir: Path,
    ground_truth: Optional[Dict[str, Any]] = None,
    inference_time_ms: float = 0.0,
    active_idle_power_w: float = 0.0,
    stage1_power_profile: Optional[Dict[str, Any]] = None,
    stage1_trace_file: Optional[Path] = None,
) -> "tuple[Path, Dict[str, Any]]":
    """Write power_report.json and print a console power summary.

    Includes per-kernel isolated power measurements, a reconstructed
    full-inference power estimate, and an energy balance decomposition.

    The kernel-only reconstruction (P_kernel_active_w) assigns idle_baseline_w
    to all non-kernel time — a significant underestimate for host-bound workloads.
    The energy_balance section derives the true overhead power from ground truth:

      E_total = E_kernel + E_overhead          (energy conservation)
      P_overhead = E_overhead / T_overhead     (derived, exact)

    If active_idle_power_w is provided (from null-kernel tight-loop measurement),
    a duty-cycle corrected prediction is also computed:

      P_predicted = kernel_fraction × P_kernel + (1 - kernel_fraction) × P_active_idle

    Returns the path to the written file.
    """

    def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not intervals:
            return []
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged: List[Tuple[float, float]] = []
        cur_s, cur_e = sorted_intervals[0]
        for s, e in sorted_intervals[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        return merged

    def _interval_overlap(s0: float, e0: float, s1: float, e1: float) -> float:
        return max(0.0, min(e0, e1) - max(s0, s1))

    def _build_observed_inter_kernel_gap(
        power_profile: Dict[str, Any],
        trace_file: Path,
        num_gpus: int,
    ) -> Optional[Dict[str, Any]]:
        """Estimate inter-kernel gap power from Stage-1 power samples + trace."""
        if num_gpus != 1:
            return {
                "available": False,
                "reason": "multi_gpu_not_supported",
            }

        samples = power_profile.get("samples_ms")
        if not isinstance(samples, dict) or not samples:
            return {
                "available": False,
                "reason": "missing_stage1_samples_ms",
            }

        gpu0_samples = samples.get("0")
        if not isinstance(gpu0_samples, list) or len(gpu0_samples) < 2:
            return {
                "available": False,
                "reason": "insufficient_stage1_samples",
            }

        profile_window = power_profile.get("profile_window_s")
        sampling_window = power_profile.get("sampling_window_s")
        if not isinstance(profile_window, dict) or not isinstance(sampling_window, dict):
            return {
                "available": False,
                "reason": "missing_stage1_timing_windows",
            }

        try:
            p_start = float(profile_window["start"])
            p_end = float(profile_window["end"])
            s_start = float(sampling_window["start"])
            s_end = float(sampling_window["end"])
        except Exception:
            return {
                "available": False,
                "reason": "invalid_stage1_timing_windows",
            }

        if p_end <= p_start or s_end <= s_start:
            return {
                "available": False,
                "reason": "non_positive_stage1_window",
            }

        if not trace_file.exists():
            return {
                "available": False,
                "reason": "trace_file_missing",
            }

        try:
            trace = json.loads(trace_file.read_text())
        except Exception:
            return {
                "available": False,
                "reason": "trace_read_failed",
            }

        from soda.common import utils as _utils

        events = _utils.collect_events(trace)
        kernels = events.get("gpu", {}).get("kernels", [])
        if not kernels:
            return {
                "available": False,
                "reason": "no_kernel_events_in_trace",
            }

        k_ts_min = min(float(k.get("ts", 0.0)) for k in kernels)
        k_ts_max = max(float(k.get("ts", 0.0)) + float(k.get("dur", 0.0)) for k in kernels)
        if k_ts_max <= k_ts_min:
            return {
                "available": False,
                "reason": "degenerate_kernel_trace_span",
            }

        trace_span_us = k_ts_max - k_ts_min
        profile_span_s = p_end - p_start

        # Map kernel-active trace intervals into the Stage-1 monotonic clock domain.
        active_intervals_s: List[Tuple[float, float]] = []
        for k in kernels:
            ts_us = float(k.get("ts", 0.0))
            te_us = ts_us + float(k.get("dur", 0.0))
            rs = (ts_us - k_ts_min) / trace_span_us
            re = (te_us - k_ts_min) / trace_span_us
            ks = p_start + rs * profile_span_s
            ke = p_start + re * profile_span_s
            cs = max(ks, s_start)
            ce = min(ke, s_end)
            if ce > cs:
                active_intervals_s.append((cs, ce))

        merged_active = _merge_intervals(active_intervals_s)
        if not merged_active:
            return {
                "available": False,
                "reason": "no_overlap_with_sampling_window",
            }

        samples_s: List[Tuple[float, float]] = []
        for pair in gpu0_samples:
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            try:
                t_s = float(pair[0]) / 1000.0
                w = float(pair[1])
            except Exception:
                continue
            if s_start <= t_s <= s_end:
                samples_s.append((t_s, w))
        samples_s.sort(key=lambda x: x[0])

        if len(samples_s) < 2:
            return {
                "available": False,
                "reason": "insufficient_samples_in_sampling_window",
            }

        active_energy_j = 0.0
        gap_energy_j = 0.0
        active_time_s = 0.0
        gap_time_s = 0.0

        # Piecewise-constant integration: interval [t_i, t_{i+1}) uses sample_i power.
        for idx in range(len(samples_s) - 1):
            t0, watts = samples_s[idx]
            t1, _ = samples_s[idx + 1]
            if t1 <= t0:
                continue

            seg_s = max(t0, s_start)
            seg_e = min(t1, s_end)
            if seg_e <= seg_s:
                continue

            active_dt = 0.0
            for a0, a1 in merged_active:
                if a0 >= seg_e:
                    break
                if a1 <= seg_s:
                    continue
                active_dt += _interval_overlap(seg_s, seg_e, a0, a1)

            seg_dt = seg_e - seg_s
            active_dt = min(active_dt, seg_dt)
            gap_dt = max(0.0, seg_dt - active_dt)

            active_energy_j += watts * active_dt
            gap_energy_j += watts * gap_dt
            active_time_s += active_dt
            gap_time_s += gap_dt

        if gap_time_s <= 0.0:
            return {
                "available": False,
                "reason": "no_gap_time_after_intersection",
            }

        out: Dict[str, Any] = {
            "available": True,
            "P_inter_kernel_observed_w": gap_energy_j / gap_time_s,
            "gap_time_s": gap_time_s,
            "active_time_s": active_time_s,
            "sample_intervals_used": len(samples_s) - 1,
            "kernel_active_coverage_pct": (
                (active_time_s / (active_time_s + gap_time_s)) * 100.0
                if (active_time_s + gap_time_s) > 0
                else 0.0
            ),
        }
        if active_time_s > 0.0:
            out["P_kernel_observed_w"] = active_energy_j / active_time_s
        return out

    # Build a lookup from kernel_id → per_kernel entry for name/freq/duration
    pk_by_id = {k["id"]: k for k in per_kernel}

    # Build per-kernel power rows (only kernels that have power data)
    rows: List[Dict[str, Any]] = []
    total_energy_uj = 0.0          # Σ energy_uj × freq (net, kernel-active only)
    reliable_count = 0
    energy_counter_count = 0
    dual_consensus_count = 0
    unknown_method_count = 0
    kernels_retried = 0
    kernels_with_ncu = 0
    kernels_with_dram = 0
    kernels_with_l2 = 0
    kernels_with_compute = 0
    known_methods = {"nvml_polling", "energy_counter", "dual_consensus"}

    # Accumulators for reconstructed inference power
    # raw = total GPU power per kernel (includes its own idle baseline);
    # net = raw minus per-kernel idle (above-idle contribution only).
    # Use raw for energy totals: avoids the per-kernel idle variation problem
    # where a single global idle_baseline_w ≠ per-kernel idle_power_w values.
    weighted_raw_power_sum = 0.0   # Σ raw_w × dur_us × freq  (W·µs = µJ)
    weighted_net_power_sum = 0.0   # Σ net_w × dur_us × freq  (W·µs = µJ, above-idle only)
    total_kernel_time_us = 0.0     # Σ dur_us × freq           (µs)
    total_launches = 0             # Σ freq — for per-launch overhead attribution (Fix B)

    for kid, pr in power_replay_results.items():
        pk = pk_by_id.get(kid, {})
        freq = pk.get("frequency", 1)
        dur_us = pk.get("kernel_duration_us", 0.0)
        net_w = pr.get("net_power_w", 0.0)
        energy_uj = pr.get("energy_uj", 0.0)
        is_reliable = pr.get("is_reliable", False)
        method = pr.get("measurement_method", "unknown")
        if method not in known_methods:
            unknown_method_count += 1
            method = "unknown"
        ncu_data = pk.get("ncu") or {}
        retry_attempts = int(pr.get("retry_attempts", 0) or 0)

        raw_w = pr.get("raw_power_w", 0.0)
        total_energy_uj += energy_uj * freq
        weighted_raw_power_sum += raw_w * dur_us * freq
        weighted_net_power_sum += net_w * dur_us * freq
        total_kernel_time_us += dur_us * freq
        total_launches += freq

        if is_reliable:
            reliable_count += 1
        if method in ("energy_counter", "dual_consensus"):
            energy_counter_count += 1
        if method == "dual_consensus":
            dual_consensus_count += 1
        if retry_attempts > 0:
            kernels_retried += 1
        if ncu_data:
            kernels_with_ncu += 1
            if ncu_data.get("dram_throughput_bytes_per_sec") is not None or (
                ncu_data.get("dram_bytes_read") is not None
                and ncu_data.get("dram_bytes_write") is not None
            ):
                kernels_with_dram += 1
            if ncu_data.get("l2_throughput_bytes_per_sec") is not None or ncu_data.get("l2_hit_rate_pct") is not None:
                kernels_with_l2 += 1
            if ncu_data.get("compute_throughput_pct") is not None:
                kernels_with_compute += 1

        rows.append({
            "kernel_id": kid,
            "kernel_name": pk.get("kernel_name", ""),
            "aten_op": pk.get("aten_op", ""),
            "frequency": freq,
            "avg_duration_us": round(dur_us, 3),
            "raw_power_w": round(pr.get("raw_power_w", 0.0), 3),
            "idle_power_w": round(pr.get("idle_power_w", idle_baseline_w), 3),
            "net_power_w": round(net_w, 3),
            "std_power_w": round(pr.get("std_power_w", 0.0), 3),
            "thermal_variance_pct": round(pr.get("thermal_variance_pct", 0.0), 2),
            "energy_uj": round(energy_uj, 2),
            "net_energy_uj_total": round(energy_uj * freq, 2),
            "raw_energy_uj_total": round(raw_w * dur_us * freq, 2),
            "is_reliable": is_reliable,
            "measurement_method": method,
            "num_windows": pr.get("num_windows", 0),
            "initial_windows": pr.get("initial_windows"),
            "final_windows": pr.get("final_windows"),
            "retry_attempts": retry_attempts,
            "consensus_windows_checked": pr.get("consensus_windows_checked"),
            "consensus_windows_agree": pr.get("consensus_windows_agree"),
            "consensus_windows_excluded": pr.get("consensus_windows_excluded"),
            "consensus_tolerance_pct": pr.get("consensus_tolerance_pct"),
            "ncu": ncu_data if ncu_data else None,
        })

    rows.sort(key=lambda r: r["net_power_w"], reverse=True)
    n_kernels = len(rows)

    # Kernel-active energy = Σ(raw_power_w × dur_us × freq) across all kernels.
    # raw_power_w is the total GPU power measured per kernel (includes each kernel's
    # own idle baseline, which varies due to thermal drift during the replay session).
    # Using raw directly avoids the error from applying a single global idle_baseline_w
    # to all kernels when per-kernel idle values differ.
    kernel_active_energy_uj = weighted_raw_power_sum          # W × µs = µJ (total, incl. idle)
    kernel_net_energy_uj = weighted_net_power_sum             # W × µs = µJ (above-idle only)

    # Duration-weighted average power during kernel-active time (from raw measurements)
    if total_kernel_time_us > 0:
        reconstructed_inference_power_w = weighted_raw_power_sum / total_kernel_time_us
        net_inference_power_w = weighted_net_power_sum / total_kernel_time_us
    else:
        reconstructed_inference_power_w = 0.0
        net_inference_power_w = 0.0

    # total_inference_energy_uj = E_kernel (raw, measured) — what flows into energy_balance.
    # This is the correct quantity to subtract from E_NVML to derive E_overhead.
    total_inference_energy_uj = kernel_active_energy_uj

    # --- Energy balance decomposition ---
    # Two tiers, in order of preference:
    #   Tier 1 (energy_counter): NVML hardware counter → E_overhead exact via conservation
    #   Tier 2 (time_decomposition_only): inference_time_ms from Stage 1 → timing only,
    #                                     E_overhead absent (no energy ground truth)
    # No estimation, no fallbacks. Each tier is clearly labeled by "method".
    energy_balance: Optional[Dict[str, Any]] = None

    if ground_truth and ground_truth.get("energy_mj", 0) > 0 and ground_truth.get("duration_s"):
        # Tier 1: full ground truth — NVML energy counter available
        # E_total = E_kernel + E_overhead  (energy conservation, exact)
        # P_overhead = E_overhead / T_overhead  (derived, not an estimate)
        E_total_uj = ground_truth["energy_mj"] * 1_000.0       # mJ → µJ
        E_overhead_uj = E_total_uj - total_inference_energy_uj
        T_total_us = ground_truth["duration_s"] * 1_000_000.0
        T_overhead_us = max(0.0, T_total_us - total_kernel_time_us)
        # µJ / µs = W (units cancel: µJ/µs = (1e-6 J)/(1e-6 s) = J/s = W)
        P_overhead_derived_w = E_overhead_uj / T_overhead_us if T_overhead_us > 0 else 0.0
        kernel_fraction = total_kernel_time_us / T_total_us if T_total_us > 0 else 0.0

        energy_balance = {
            "method": "energy_counter",
            "E_total_uj": round(E_total_uj, 2),
            "E_kernel_uj": round(total_inference_energy_uj, 2),
            "E_overhead_uj": round(E_overhead_uj, 2),
            "T_total_us": round(T_total_us, 1),
            "T_kernel_active_us": round(total_kernel_time_us, 1),
            "T_overhead_us": round(T_overhead_us, 1),
            "P_kernel_active_w": round(reconstructed_inference_power_w, 3),
            "P_overhead_derived_w": round(P_overhead_derived_w, 3),
            "kernel_fraction": round(kernel_fraction, 6),
        }

        # Stage-1 macro measured system power from NVML energy counter.
        # This is a measurement identity, not a predictive reconstruction.
        P_measured_macro_w = (E_total_uj / T_total_us) if T_total_us > 0 else 0.0
        energy_balance["P_measured_macro_w"] = round(P_measured_macro_w, 3)
        energy_balance["workload_regime"] = (
            "compute_bound" if kernel_fraction >= 0.80 else "host_bound"
        )
        energy_balance["system_reconstruction_available"] = False

        gap_power_candidate = None
        gap_power_source = None

        trace_file = stage1_trace_file or (output_dir.parent / "trace.json")
        observed_gap = _build_observed_inter_kernel_gap(
            power_profile=stage1_power_profile or {},
            trace_file=trace_file,
            num_gpus=int(metadata.get("num_gpus", 1) or 1),
        )
        if observed_gap and observed_gap.get("available"):
            gap_power_candidate = float(observed_gap["P_inter_kernel_observed_w"])
            gap_power_source = "observed_from_stage1_power_samples"
            energy_balance["P_inter_kernel_observed_w"] = round(gap_power_candidate, 3)
            if "P_kernel_observed_w" in observed_gap:
                energy_balance["P_kernel_observed_w"] = round(
                    float(observed_gap["P_kernel_observed_w"]), 3
                )
            energy_balance["inter_kernel_observed_meta"] = {
                "gap_time_s": round(float(observed_gap.get("gap_time_s", 0.0)), 6),
                "active_time_s": round(float(observed_gap.get("active_time_s", 0.0)), 6),
                "sample_intervals_used": int(observed_gap.get("sample_intervals_used", 0)),
                "kernel_active_coverage_pct": round(
                    float(observed_gap.get("kernel_active_coverage_pct", 0.0)), 2
                ),
            }
        else:
            energy_balance["inter_kernel_observed_unavailable_reason"] = (
                (observed_gap or {}).get("reason", "stage1_power_profile_unavailable")
            )

        # Duty-cycle corrected prediction using independently measured P_active_idle.
        # Note: substituting P_overhead_derived here would tautologically yield P_measured.
        # P_active_idle (null-kernel tight-loop) is an independent calibration point.
        if gap_power_candidate is None and active_idle_power_w > 0:
            gap_power_candidate = active_idle_power_w
            gap_power_source = "null_kernel_active_idle"
            energy_balance["P_active_idle_measured_w"] = round(active_idle_power_w, 3)

        if gap_power_candidate is not None:
            P_duty_cycle_w = (
                kernel_fraction * reconstructed_inference_power_w
                + (1.0 - kernel_fraction) * gap_power_candidate
            )
            energy_balance["P_inter_kernel_gap_w"] = round(gap_power_candidate, 3)
            energy_balance["inter_kernel_gap_source"] = gap_power_source
            energy_balance["P_duty_cycle_predicted_w"] = round(P_duty_cycle_w, 3)

            # --- Predictive energy estimator (Fix B) ---
            # E_predicted = kernel_active_energy + P_gap × T_overhead
            #             ≡ P_duty_cycle × T_total  (algebraically identical)
            # Per-launch attribution: overhead energy uniformly distributed across
            # total kernel launches (one dispatch gap per launch on avg).
            E_gap_predicted_uj = gap_power_candidate * T_overhead_us  # W × µs = µJ
            E_predicted_total_uj = total_inference_energy_uj + E_gap_predicted_uj
            overhead_energy_per_launch_uj = (
                E_gap_predicted_uj / total_launches if total_launches > 0 else 0.0
            )
            overhead_time_per_launch_us = (
                T_overhead_us / total_launches if total_launches > 0 else 0.0
            )
            energy_balance["E_gap_predicted_uj"] = round(E_gap_predicted_uj, 2)
            energy_balance["E_predicted_total_uj"] = round(E_predicted_total_uj, 2)
            energy_balance["overhead_energy_per_launch_uj"] = round(overhead_energy_per_launch_uj, 4)
            energy_balance["overhead_time_per_launch_us"] = round(overhead_time_per_launch_us, 4)
            energy_balance["total_launches"] = total_launches
            if E_total_uj > 0:
                energy_balance["predicted_energy_error_pct"] = round(
                    (E_predicted_total_uj - E_total_uj) / E_total_uj * 100.0, 2
                )

            # Always surface predictive reconstruction for sanity checking,
            # even in host-bound workloads where macro measured power remains authoritative.
            P_system_recon_w = P_duty_cycle_w
            recon_method = "duty_cycle_prediction"
            energy_balance["P_system_reconstructed_w"] = round(P_system_recon_w, 3)
            energy_balance["system_reconstructed_method"] = recon_method
            energy_balance["system_reconstruction_available"] = True

            if kernel_fraction < 0.80:
                energy_balance["P_host_bound_measured_w"] = round(P_measured_macro_w, 3)
                energy_balance["host_bound_power_method"] = "macro_energy_balance"

            if ground_truth.get("power_w") and ground_truth["power_w"] > 0:
                energy_balance["duty_cycle_prediction_error_pct"] = round(
                    (P_duty_cycle_w - ground_truth["power_w"])
                    / ground_truth["power_w"] * 100.0, 2
                )
                energy_balance["system_reconstruction_error_pct"] = round(
                    (P_system_recon_w - ground_truth["power_w"])
                    / ground_truth["power_w"] * 100.0, 2
                )
        else:
            energy_balance["system_reconstruction_unavailable_reason"] = (
                "missing_inter_kernel_gap_power"
            )

    elif inference_time_ms > 0:
        # Tier 2: timing ground truth only — inference_time_ms from Stage 1 kernel DB.
        # E_overhead cannot be computed without NVML energy counter.
        T_total_us = inference_time_ms * 1000.0
        T_overhead_us = max(0.0, T_total_us - total_kernel_time_us)
        kernel_fraction = total_kernel_time_us / T_total_us if T_total_us > 0 else 0.0
        energy_balance = {
            "method": "time_decomposition_only",
            "note": (
                "NVML energy counter unavailable — E_overhead cannot be computed. "
                "Re-run Stage 1 with --power-sample to capture energy_mj ground truth."
            ),
            "T_total_us": round(T_total_us, 1),
            "T_kernel_active_us": round(total_kernel_time_us, 1),
            "T_overhead_us": round(T_overhead_us, 1),
            "kernel_fraction": round(kernel_fraction, 6),
            "E_kernel_uj": round(total_inference_energy_uj, 2),
        }

    # --- Validation: compare reconstructed power against ground truth ---
    validation: Optional[Dict[str, Any]] = None
    measured_power_w: Optional[float] = None
    if ground_truth and ground_truth.get("power_w") and ground_truth["power_w"] > 0:
        measured_power_w = ground_truth["power_w"]
        error_w = reconstructed_inference_power_w - measured_power_w
        error_pct = (error_w / measured_power_w * 100.0) if measured_power_w > 0 else 0.0
        validation = {
            "measured_inference_power_w": round(measured_power_w, 3),
            "measured_energy_mj": ground_truth.get("energy_mj"),
            "measured_duration_s": ground_truth.get("duration_s"),
            "measured_method": ground_truth.get("method", "unknown"),
            "reconstructed_inference_power_w": round(reconstructed_inference_power_w, 3),
            "error_w": round(error_w, 3),
            "error_pct": round(error_pct, 2),
        }

    # --- Build ATen op → kernels dispatch map ---
    # Groups all kernels by their parent ATen op, with per-kernel power/energy data.
    # One ATen op may dispatch multiple kernel variants (different grid/block configs).
    #
    # Energy accounting uses raw_power_w × dur_us × freq (W·µs = µJ) — the same formula
    # as kernel_active_energy_uj in inference_power — so that:
    #   Σ(op.total_raw_energy_uj across all ops) == kernel_active_energy_uj
    # energy_uj (net, above-idle) is also stored per-kernel for informational purposes.
    op_kernel_map: Dict[str, Any] = {}
    for pk in per_kernel:
        op_name = pk.get("aten_op", "") or "unknown"
        kid = pk.get("id", "")
        dur_us = pk.get("kernel_duration_us", 0.0)
        freq = pk.get("frequency", 1)
        # Prefer per-kernel embedded replay data; fall back to the primary replay map
        # so op-level totals remain consistent even if per_kernel lacks power_replay.
        pr = pk.get("power_replay") or power_replay_results.get(kid) or {}
        # raw energy is consistent with kernel_active_energy_uj in inference_power
        raw_energy_uj_k = pr.get("raw_power_w", 0.0) * dur_us * freq if pr else 0.0
        net_energy_uj_k = pr.get("energy_uj", 0.0) * freq if pr else 0.0

        if op_name not in op_kernel_map:
            op_kernel_map[op_name] = {
                "aten_op": op_name,
                "total_raw_energy_uj": 0.0,   # Σ(raw_power_w × dur × freq) — matches kernel_active_energy_uj
                "total_net_energy_uj": 0.0,   # Σ(net_power_w × dur × freq) — above-idle only
                "total_duration_us": 0.0,
                "kernels": [],
            }

        op_kernel_map[op_name]["total_raw_energy_uj"] += raw_energy_uj_k
        op_kernel_map[op_name]["total_net_energy_uj"] += net_energy_uj_k
        op_kernel_map[op_name]["total_duration_us"] += dur_us * freq
        op_kernel_map[op_name]["kernels"].append({
            "kernel_id": kid,
            "kernel_name": pk.get("kernel_name", ""),
            "frequency": freq,
            "avg_duration_us": round(dur_us, 3),
            "raw_power_w": round(pr.get("raw_power_w", 0.0), 3) if pr else None,
            "net_power_w": round(pr.get("net_power_w", 0.0), 3) if pr else None,
            # energy_uj: net, per single invocation (net_power_w × avg_dur_us)
            "energy_uj": round(pr.get("energy_uj", 0.0), 2) if pr else None,
            # raw_energy_uj_total: raw × dur × freq — sums consistently with kernel_active_energy_uj
            "raw_energy_uj_total": round(raw_energy_uj_k, 2),
            "is_reliable": pr.get("is_reliable") if pr else None,
        })

    # Sort ops by total raw energy descending; round accumulated totals
    op_kernel_list = sorted(
        op_kernel_map.values(),
        key=lambda x: x["total_raw_energy_uj"],
        reverse=True,
    )
    for entry in op_kernel_list:
        entry["total_raw_energy_uj"] = round(entry["total_raw_energy_uj"], 2)
        entry["total_net_energy_uj"] = round(entry["total_net_energy_uj"], 2)
        entry["total_duration_us"] = round(entry["total_duration_us"], 1)
        # Sort kernels within each op by raw energy descending
        entry["kernels"].sort(key=lambda k: k["raw_energy_uj_total"], reverse=True)

    # --- Fix B: per-launch overhead attribution ---
    # When the predictive estimator is available (gap power + NVML ground truth),
    # distribute E_overhead across all kernel launches uniformly (one dispatch gap
    # per launch on average). This makes per-kernel energy aggregates sum to
    # E_predicted_total_uj (≈ E_NVML_total) rather than kernel-active-only.
    oepl = None
    otpl = None
    if isinstance(energy_balance, dict):
        oepl = energy_balance.get("overhead_energy_per_launch_uj")
        otpl = energy_balance.get("overhead_time_per_launch_us")
    total_attributed_energy_uj = 0.0
    if oepl is not None and otpl is not None:
        for r in rows:
            freq = r["frequency"]
            dur_us = r["avg_duration_us"]
            attributed_overhead = oepl * freq
            r["attributed_overhead_energy_uj"] = round(attributed_overhead, 2)
            r["attributed_energy_uj_total"] = round(
                r["raw_energy_uj_total"] + attributed_overhead, 2
            )
            r["attributed_energy_per_inv_uj"] = round(
                (r["raw_energy_uj_total"] + attributed_overhead) / freq if freq > 0 else 0.0,
                4,
            )
            total_time_us = (dur_us + otpl) * freq
            r["effective_power_w"] = round(
                r["attributed_energy_uj_total"] / total_time_us, 3
            ) if total_time_us > 0 else 0.0
            total_attributed_energy_uj += r["attributed_energy_uj_total"]

        # Propagate into op_kernel_map: per-kernel attribution + op rollups
        for entry in op_kernel_list:
            op_total_attrib = 0.0
            op_total_overhead = 0.0
            for k in entry["kernels"]:
                freq = k["frequency"]
                dur_us = k["avg_duration_us"]
                attrib_oh = oepl * freq
                attrib_total = k["raw_energy_uj_total"] + attrib_oh
                k["attributed_overhead_energy_uj"] = round(attrib_oh, 2)
                k["attributed_energy_uj_total"] = round(attrib_total, 2)
                total_time_us = (dur_us + otpl) * freq
                k["effective_power_w"] = round(
                    attrib_total / total_time_us, 3
                ) if total_time_us > 0 else 0.0
                op_total_attrib += attrib_total
                op_total_overhead += attrib_oh
            entry["total_attributed_energy_uj"] = round(op_total_attrib, 2)
            entry["total_attributed_overhead_energy_uj"] = round(op_total_overhead, 2)

        energy_balance["total_attributed_energy_uj"] = round(total_attributed_energy_uj, 2)
        # Reconciliation: Σ attributed (rows) ≡ E_predicted_total_uj by construction.
        # Report residual for sanity (should be ~0 modulo rounding).
        E_pred = energy_balance.get("E_predicted_total_uj", 0.0)
        energy_balance["attribution_reconciliation_residual_uj"] = round(
            total_attributed_energy_uj - E_pred, 4
        )

    power_report = {
        "version": "1.1",
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "model": metadata.get("model", "unknown"),
            "gpu_name": metadata.get("gpu_name", "unknown"),
        },
        "inference_power": {
            # Duration-weighted average raw GPU power during kernel-active time.
            # = Σ(raw_power_w × dur × freq) / Σ(dur × freq) — uses per-kernel raw
            # (not global idle_baseline_w) to avoid thermal-drift artifacts.
            "reconstructed_inference_power_w": round(reconstructed_inference_power_w, 3),
            "net_kernel_power_w": round(net_inference_power_w, 3),
            "idle_baseline_w": round(idle_baseline_w, 3),
            # Energy per inference (kernel-active portion only, excludes host-overhead gaps)
            # E_kernel = Σ(raw_power_w × dur_us × freq) — correct quantity for energy balance.
            "kernel_active_time_us": round(total_kernel_time_us, 1),
            "kernel_active_energy_uj": round(kernel_active_energy_uj, 2),
            "kernel_net_energy_uj": round(kernel_net_energy_uj, 2),
            "total_inference_energy_uj": round(total_inference_energy_uj, 2),
            # Predictive estimator output (present only when gap power + NVML
            # ground truth available). Mirrors energy_balance fields for consumers
            # that read inference_power directly.
            "predicted_inference_energy_uj": (
                energy_balance.get("E_predicted_total_uj")
                if isinstance(energy_balance, dict) else None
            ),
            "predicted_inference_power_w": (
                energy_balance.get("P_duty_cycle_predicted_w")
                if isinstance(energy_balance, dict) else None
            ),
            "predicted_energy_error_pct": (
                energy_balance.get("predicted_energy_error_pct")
                if isinstance(energy_balance, dict) else None
            ),
            "note": (
                "kernel_active_energy_uj = Σ(raw_power_w × dur_us × freq) — total GPU "
                "energy during kernel-active time only (excludes host-overhead). "
                "E_overhead = E_NVML_total − kernel_active_energy_uj (see energy_balance). "
                "predicted_inference_energy_uj = kernel_active_energy_uj + P_gap × T_overhead "
                "(predictive estimator; closes the loop to E_NVML via P_gap calibration)."
            ),
        },
        "energy_balance": energy_balance,
        "validation": validation,
        "per_kernel_summary": {
            "kernels_profiled": n_kernels,
            "kernels_reliable": reliable_count,
            "kernels_energy_counter": energy_counter_count,
            "kernels_dual_consensus": dual_consensus_count,
            "kernels_unknown_method": unknown_method_count,
            "kernels_retried": kernels_retried,
            "kernels_with_ncu": kernels_with_ncu,
            "kernels_with_dram_metrics": kernels_with_dram,
            "kernels_with_l2_metrics": kernels_with_l2,
            "kernels_with_compute_metrics": kernels_with_compute,
            "total_kernel_energy_uj": round(total_energy_uj, 2),
        },
        "kernels": rows,
        "op_kernel_map": op_kernel_list,
    }

    report_path = output_dir / "power_report.json"
    with open(report_path, "w") as f:
        json.dump(power_report, f, indent=2)

    # Console summary
    print()
    print("=== Per-Kernel Power Replay ===")
    print(f"  Idle baseline              : {idle_baseline_w:.2f} W")
    print(f"  Avg raw kernel power       : {reconstructed_inference_power_w:.2f} W  "
          f"[Σ(raw_power × dur × freq) / Σ(dur × freq)]")
    print(f"  Avg net kernel power       : {net_inference_power_w:.2f} W  "
          f"[above-idle only]")
    if validation:
        print(f"  Measured infer power (GT)  : {measured_power_w:.2f} W  "
              f"[energy counter during live inference]")
        err_sign = "+" if validation["error_w"] >= 0 else ""
        print(f"  Validation error           : {err_sign}{validation['error_w']:.2f} W  "
              f"({err_sign}{validation['error_pct']:.1f}%)")
    else:
        print(f"  Measured infer power (GT)  : N/A  [re-run Stage 1 to capture ground truth]")
    print(f"  Kernel active time         : {total_kernel_time_us:.1f} µs  "
          f"[Σ avg_dur × freq across {n_kernels} kernels]")
    print(f"  Kernel energy (raw total)  : {kernel_active_energy_uj / 1e3:.4f} mJ  "
          f"[Σ(raw_power × dur × freq) — kernel-active only, excludes host-overhead]")
    if energy_balance:
        eb = energy_balance
        method = eb.get("method", "")
        T_tot = eb.get("T_total_us", 0.0)
        T_k = eb.get("T_kernel_active_us", total_kernel_time_us)
        T_oh = eb.get("T_overhead_us", 0.0)
        kf = eb.get("kernel_fraction", 0.0)
        print()
        print("=== Energy Balance / Time Decomposition ===")
        print(f"  Inference wall time        : {T_tot:.1f} µs  [Stage 1 ground truth]")
        print(f"  Kernel active time         : {T_k:.1f} µs  ({kf*100:.2f}% duty cycle)")
        print(f"  Host-overhead time         : {T_oh:.1f} µs  ({(1-kf)*100:.2f}% — GPU idle / CPU dispatch)")
        if method == "energy_counter":
            E_tot_mj = eb["E_total_uj"] / 1_000.0
            E_k_mj = eb["E_kernel_uj"] / 1_000.0
            E_oh_mj = eb["E_overhead_uj"] / 1_000.0
            E_k_pct = eb["E_kernel_uj"] / eb["E_total_uj"] * 100.0 if eb["E_total_uj"] > 0 else 0.0
            print(f"  E_total (NVML counter)     : {E_tot_mj:.2f} mJ  [hardware ground truth]")
            print(f"  E_kernel (from replay)     : {E_k_mj:.3f} mJ  ({E_k_pct:.2f}% of total)")
            print(f"  E_overhead (derived)       : {E_oh_mj:.2f} mJ  ({100-E_k_pct:.2f}%)")
            print(f"  P_overhead (derived)       : {eb['P_overhead_derived_w']:.1f} W  "
                  f"[GPU power during host-overhead phases]")
            print(f"  Regime classification      : {eb.get('workload_regime', 'unknown')}")
            print(f"  P_macro_measured (E/T)     : {eb.get('P_measured_macro_w', 0):.1f} W  "
                f"[macro energy balance identity]")
            if "P_inter_kernel_gap_w" in eb:
                print(
                    f"  P_inter_kernel_gap         : {eb.get('P_inter_kernel_gap_w', 0):.1f} W  "
                    f"[{eb.get('inter_kernel_gap_source', 'unknown')}]"
                )
            if "P_inter_kernel_observed_w" in eb:
                print(
                    f"  P_inter_kernel_observed    : {eb.get('P_inter_kernel_observed_w', 0):.1f} W  "
                    f"[from Stage 1 samples + trace intersection]"
                )
            if "P_active_idle_measured_w" in eb:
                print(f"  P_active_idle (measured)   : {eb.get('P_active_idle_measured_w', 0):.1f} W  "
                    f"[null-kernel tight loop]")
                print(f"  P_duty_cycle_predicted     : {eb.get('P_duty_cycle_predicted_w', 0):.1f} W  "
                    f"[kernel_frac×P_kernel + overhead_frac×P_active_idle]")
            elif "P_duty_cycle_predicted_w" in eb:
                print(f"  P_duty_cycle_predicted     : {eb.get('P_duty_cycle_predicted_w', 0):.1f} W  "
                    f"[kernel_frac×P_kernel + overhead_frac×P_inter_kernel_gap]")
                if "duty_cycle_prediction_error_pct" in eb:
                  err = eb["duty_cycle_prediction_error_pct"]
                  err_sign = "+" if err >= 0 else ""
                  print(f"  Duty-cycle error vs GT     : {err_sign}{err:.1f}%")
            if "P_system_reconstructed_w" in eb:
                recon_method = eb.get("system_reconstructed_method", "unknown")
                print(f"  P_system_reconstructed     : {eb['P_system_reconstructed_w']:.1f} W  "
                    f"[{recon_method}]")
                if "system_reconstruction_error_pct" in eb:
                  err = eb["system_reconstruction_error_pct"]
                  err_sign = "+" if err >= 0 else ""
                  print(f"  System recon error vs GT   : {err_sign}{err:.1f}%")
            elif "P_host_bound_measured_w" in eb:
                print(f"  P_host_bound_measured      : {eb['P_host_bound_measured_w']:.1f} W  "
                    f"[{eb.get('host_bound_power_method', 'macro_energy_balance')}]")
            if "E_predicted_total_uj" in eb:
                E_pred_mj = eb["E_predicted_total_uj"] / 1_000.0
                E_gap_pred_mj = eb.get("E_gap_predicted_uj", 0.0) / 1_000.0
                print(f"  E_predicted (estimator)    : {E_pred_mj:.2f} mJ  "
                      f"[kernel + P_gap × T_gap, gap={E_gap_pred_mj:.2f} mJ]")
                if "predicted_energy_error_pct" in eb:
                    err = eb["predicted_energy_error_pct"]
                    err_sign = "+" if err >= 0 else ""
                    print(f"  Predicted energy error     : {err_sign}{err:.1f}%  "
                          f"[vs E_NVML ground truth]")
                if "overhead_energy_per_launch_uj" in eb:
                    print(f"  Overhead per launch        : "
                          f"{eb['overhead_energy_per_launch_uj']:.3f} µJ  "
                          f"[attributed to each of {eb.get('total_launches', 0)} kernel launches]")
        else:
            print("  E_overhead                 : N/A  [NVML counter unavailable — re-run Stage 1]")
        print(f"  Kernels profiled           : {n_kernels}  ({reliable_count} reliable, "
              f"{energy_counter_count} via energy counter/consensus, "
              f"{dual_consensus_count} dual-consensus, {kernels_retried} retried)")
        if unknown_method_count > 0:
            print(f"  Unknown method rows        : {unknown_method_count}  [schema/version mismatch]")
        if kernels_with_ncu > 0:
            print(f"  NCU channel coverage       : {kernels_with_ncu}/{n_kernels} kernels "
                  f"(DRAM {kernels_with_dram}, L2 {kernels_with_l2}, compute {kernels_with_compute})")
    print()
    top_n = min(10, len(rows))
    if top_n:
        print(f"  Top {top_n} kernels by net power draw:")
        fmt = "    {:>4}  {:>8.2f} W  {:>7.1f} µJ  {:>5}  {:<30}  {}"
        print(f"    {'ID':>4}  {'Net(W)':>8}  {'Energy':>7}  {'Freq':>5}  {'ATen Op':<30}  Kernel")
        for r in rows[:top_n]:
            print(fmt.format(
                r["kernel_id"],
                r["net_power_w"],
                r["energy_uj"],
                r["frequency"],
                (r["aten_op"] or "")[:30],
                (r["kernel_name"] or "")[:40],
            ))
    # Op → kernels dispatch summary (top ops by total energy)
    top_ops = min(10, len(op_kernel_list))
    if top_ops:
        print()
        print(f"  ATen op → kernel dispatch map (top {top_ops} ops by energy):")
        print(f"    {'ATen Op':<35}  {'#Kernels':>8}  {'TotalDur':>10}  {'TotalEnergy':>12}")
        for op_entry in op_kernel_list[:top_ops]:
            n_k = len(op_entry["kernels"])
            print(f"    {op_entry['aten_op']:<35}  {n_k:>8}  "
                  f"{op_entry['total_duration_us']:>8.1f}µs  "
                  f"{op_entry['total_raw_energy_uj']:>10.1f}µJ")
            for k in op_entry["kernels"]:
                net_str = f"{k['net_power_w']:.2f}W" if k["net_power_w"] is not None else "  N/A "
                print(f"      {k['kernel_id']:>4}  ×{k['frequency']:<5}  {net_str:>7}  "
                      f"{k['avg_duration_us']:>7.1f}µs  {k['kernel_name'][:50]}")
    print(f"\nSaved power report to {report_path}")
    return report_path, power_report["inference_power"]


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
          f"({summary.get('library_mediated_invocations', summary.get('library_gemm_invocations', 0))} library-mediated, "
          f"{summary.get('fw_native_lib_op_invocations', summary.get('framework_gemm_invocations', 0))} fw-native-lib-op, "
          f"{summary.get('framework_native_invocations', summary.get('non_gemm_invocations', 0))} framework-native)")
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
        is_lib_mediated = entry["classification"].get(
            "is_library_mediated", entry["classification"].get("is_gemm", False)
        )
        # Prefer runtime-detected i_lib (matches tax computation) over DB heuristic
        i_lib = entry.get("i_lib_runtime", entry["classification"].get("i_lib", int(is_lib_mediated)))
        kernel_class = entry["classification"].get("kernel_class", "library_mediated" if is_lib_mediated else "framework_native")
        if i_lib == 1:
            ktype = "Library-mediated (I_lib=1)"
        elif kernel_class in ("library_mediated", "gemm"):
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
