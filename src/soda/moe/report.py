"""MoE per-expert-type memory profiling report generator.

Aggregates NCU isolation (absolute HBM bytes, L1/L2 self-reuse) and
NVBit in-context (in-model-run L1/L2 cache state, cross-expert reuse)
results into moe_profile.json.

No T_launch, HDBI, or overhead decomposition — memory metrics only.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


def generate_moe_report(
    classified_kernels: List[Dict],
    ncu_results: Dict[str, Dict],
    nvbit_results: Optional[Dict],
    output_dir: Path,
    args,
) -> Path:
    """Generate moe_profile.json from NCU + NVBit results.

    Args:
        classified_kernels: Entries annotated with expert_type from classify_kernel_entries().
        ncu_results: kernel_id -> NCU metric dict (from _run_ncu_pass).
        nvbit_results: Parsed NVBit reuse log (from parse_reuse_log), or None.
        output_dir: Directory to write moe_profile.json.
        args: CLI args namespace (for moe_shared_dim, moe_routed_dim overrides).

    Returns:
        Path to the written moe_profile.json.
    """
    output_dir = Path(output_dir)

    # Build classification summary and per-expert ncu aggregates
    classification_summary = _build_classification_summary(classified_kernels)
    moe_config = _extract_moe_config(classified_kernels)
    per_expert_type = _aggregate_per_expert(classified_kernels, ncu_results, nvbit_results)
    data_reuse = _extract_data_reuse(nvbit_results)

    report = {
        "moe_config": moe_config,
        "classification_summary": classification_summary,
        "per_expert_type": per_expert_type,
        "data_reuse": data_reuse,
    }

    output_path = output_dir / "moe_profile.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _print_console_summary(report)
    return output_path


# ---------------------------------------------------------------------------
# Config / summary helpers
# ---------------------------------------------------------------------------

def _extract_moe_config(classified_kernels: List[Dict]) -> Dict:
    """Re-derive the MoE config that was used for classification."""
    from soda.moe.detect import detect_moe_config
    kernels_raw = [
        {k: v for k, v in e.items() if k != "expert_type"}
        for e in classified_kernels
    ]
    config = detect_moe_config(kernels_raw)
    return {
        "detection_method": config.get("detection_method", "cardinality"),
        "num_experts": config.get("num_experts"),
        "shared_dim": config.get("shared_dim"),
        "routed_dim": config.get("routed_dim"),
        "hidden_dim": config.get("hidden_dim"),
    }


def _build_classification_summary(classified_kernels: List[Dict]) -> Dict:
    """Count entries per expert_type."""
    counts = Counter(e.get("expert_type", "other") for e in classified_kernels)
    summary = {}
    for et in ["shared_expert", "routed_expert", "gate", "attention", "other"]:
        n = counts.get(et, 0)
        if n > 0:
            summary[et] = {"entry_count": n}
    return summary


# ---------------------------------------------------------------------------
# Per-expert-type aggregation
# ---------------------------------------------------------------------------

def _aggregate_per_expert(
    classified_kernels: List[Dict],
    ncu_results: Dict[str, Dict],
    nvbit_results: Optional[Dict],
) -> Dict:
    """Build per-expert-type report section."""
    per_type: Dict[str, Dict] = {}

    expert_types = ["shared_expert", "routed_expert", "gate", "attention"]
    for et in expert_types:
        entries = [e for e in classified_kernels if e.get("expert_type") == et]
        if not entries:
            continue

        section: Dict = {}

        # NCU isolation results for this expert type
        ncu_section = _aggregate_ncu_for_type(entries, ncu_results, et)
        if ncu_section:
            section["ncu_isolation"] = ncu_section

        # NVBit in-context results for this expert type
        if nvbit_results is not None:
            nvbit_section = _extract_nvbit_for_type(nvbit_results, et)
            if nvbit_section:
                section["nvbit_in_context"] = nvbit_section

        if section:
            per_type[et] = section

    return per_type


def _aggregate_ncu_for_type(
    entries: List[Dict],
    ncu_results: Dict[str, Dict],
    expert_type: str,
) -> Optional[Dict]:
    """Aggregate NCU metrics for all profiled entries of one expert type."""
    # Collect NCU results that match this expert type
    relevant = [
        v for v in ncu_results.values()
        if v.get("expert_type") == expert_type
    ]
    if not relevant:
        return None

    def _avg(key: str) -> Optional[float]:
        vals = [r[key] for r in relevant if key in r and r[key] is not None]
        return round(statistics.mean(vals), 2) if vals else None

    hbm_read = _avg("hbm_read_bytes")
    hbm_write = _avg("hbm_write_bytes")
    l1_hit = _avg("l1_hit_rate_pct")
    l2_hit = _avg("l2_hit_rate_pct")
    compute_util = _avg("compute_util_pct")

    # HBM bandwidth (TB/s): bytes / duration_us * 1e-6
    bw_vals = []
    for r in relevant:
        dur_us = r.get("kernel_duration_us")
        hbm_r = r.get("hbm_read_bytes", 0) or 0
        hbm_w = r.get("hbm_write_bytes", 0) or 0
        if dur_us and dur_us > 0:
            bw_tbs = (hbm_r + hbm_w) / dur_us / 1e6
            bw_vals.append(bw_tbs)
    avg_bw = round(statistics.mean(bw_vals), 4) if bw_vals else None

    result: Dict = {
        "profiled_count": len(relevant),
        "note": "isolation replay — L1/L2 reflects self-reuse only; HBM is accurate",
    }
    if hbm_read is not None:
        result["avg_hbm_read_bytes"] = hbm_read
    if hbm_write is not None:
        result["avg_hbm_write_bytes"] = hbm_write
    if avg_bw is not None:
        result["avg_hbm_bandwidth_tb_s"] = avg_bw
    if l1_hit is not None:
        result["avg_l1_hit_rate_pct"] = l1_hit
    if l2_hit is not None:
        result["avg_l2_hit_rate_pct"] = l2_hit
    if compute_util is not None:
        result["avg_compute_util_pct"] = compute_util

    return result


def _extract_nvbit_for_type(nvbit_results: Dict, expert_type: str) -> Optional[Dict]:
    """Extract NVBit in-context stats for one expert type."""
    per_type = nvbit_results.get("per_expert_type", {})
    stats = per_type.get(expert_type)
    if not stats:
        return None

    result = {
        "note": "in-context profiling — L1/L2 reflects actual model-run cache state",
        "total_invocations": stats.get("total_invocations"),
        "total_global_loads": stats.get("total_global_loads"),
        "avg_global_loads_per_invocation": stats.get("avg_global_loads_per_invocation"),
        "hbm_bytes_per_invocation": stats.get("avg_hbm_bytes_per_invocation"),
        "total_cachelines_tracked": stats.get("total_cachelines_tracked"),
    }
    # Remove None values
    return {k: v for k, v in result.items() if v is not None}


# ---------------------------------------------------------------------------
# Data reuse section
# ---------------------------------------------------------------------------

def _extract_data_reuse(nvbit_results: Optional[Dict]) -> Dict:
    """Build the data_reuse section from NVBit cross-expert reuse metrics."""
    if nvbit_results is None:
        return {
            "available": False,
            "note": "NVBit pass not run — provide --nvbit-lib for in-context reuse analysis",
        }

    cross = nvbit_results.get("cross_expert_reuse", {})
    result: Dict = {"available": True}

    # Cross-expert: shared → routed
    shared_to_routed = cross.get("shared_to_routed")
    if shared_to_routed:
        result["cross_expert_shared_to_routed"] = {
            "reuse_fraction": shared_to_routed.get("reuse_fraction"),
            "overlap_cachelines": shared_to_routed.get("overlap_cachelines"),
            "method": shared_to_routed.get("method", "exact_set_intersection"),
            "interpretation": shared_to_routed.get("note", ""),
        }

    # Within-type inter-layer reuse
    within: Dict = {}
    for key, stats in cross.items():
        if key.endswith("_within_type"):
            et = key.replace("_within_type", "")
            within[et] = {
                "reuse_fraction": stats.get("reuse_fraction"),
                "overlap_cachelines": stats.get("overlap_cachelines"),
                "method": stats.get("method", "exact_set_intersection"),
                "interpretation": stats.get("note", ""),
            }
    if within:
        result["within_type_inter_layer_reuse"] = within

    return result


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _print_console_summary(report: Dict) -> None:
    """Print a human-readable summary of the MoE profile report."""
    cfg = report.get("moe_config", {})
    print("\n" + "=" * 60)
    print("MoE Memory Profile Summary")
    print("=" * 60)

    # Config
    print(f"\nDetection method : {cfg.get('detection_method', '?')}")
    if cfg.get("num_experts"):
        print(f"Num experts      : {cfg['num_experts']}")
    if cfg.get("shared_dim"):
        print(f"Shared dim       : {cfg['shared_dim']}")
    if cfg.get("routed_dim"):
        print(f"Routed dim       : {cfg['routed_dim']}")
    if cfg.get("hidden_dim"):
        print(f"Hidden dim       : {cfg['hidden_dim']}")

    # Classification
    print("\nClassification:")
    for et, s in report.get("classification_summary", {}).items():
        print(f"  {et:<18} {s['entry_count']:>5} entries")

    # Per-expert type metrics
    per_type = report.get("per_expert_type", {})
    if per_type:
        print("\nPer-Expert Memory Metrics:")
        for et, section in per_type.items():
            print(f"\n  [{et}]")
            ncu = section.get("ncu_isolation", {})
            if ncu:
                print(f"    NCU isolation ({ncu.get('profiled_count', '?')} kernels profiled):")
                if "avg_hbm_read_bytes" in ncu:
                    read_mb = ncu["avg_hbm_read_bytes"] / 1e6
                    write_mb = (ncu.get("avg_hbm_write_bytes") or 0) / 1e6
                    print(f"      HBM read  : {read_mb:.1f} MB avg")
                    print(f"      HBM write : {write_mb:.1f} MB avg")
                if "avg_hbm_bandwidth_tb_s" in ncu:
                    print(f"      HBM bw    : {ncu['avg_hbm_bandwidth_tb_s']:.3f} TB/s avg")
                if "avg_l1_hit_rate_pct" in ncu:
                    print(f"      L1 hit    : {ncu['avg_l1_hit_rate_pct']:.1f}% (isolation)")
                if "avg_l2_hit_rate_pct" in ncu:
                    print(f"      L2 hit    : {ncu['avg_l2_hit_rate_pct']:.1f}% (isolation)")
                if "avg_compute_util_pct" in ncu:
                    print(f"      Compute   : {ncu['avg_compute_util_pct']:.1f}%")

            nvbit = section.get("nvbit_in_context", {})
            if nvbit:
                print(f"    NVBit in-context ({nvbit.get('total_invocations', '?')} invocations):")
                if "hbm_bytes_per_invocation" in nvbit:
                    hbm_mb = nvbit["hbm_bytes_per_invocation"] / 1e6
                    print(f"      HBM/invoc : {hbm_mb:.2f} MB")
                if "avg_global_loads_per_invocation" in nvbit:
                    print(f"      GL/invoc  : {nvbit['avg_global_loads_per_invocation']:.0f}")

    # Data reuse
    dr = report.get("data_reuse", {})
    if dr.get("available"):
        print("\nData Reuse (NVBit in-context):")
        s2r = dr.get("cross_expert_shared_to_routed")
        if s2r:
            pct = (s2r.get("reuse_fraction") or 0) * 100
            print(f"  Shared→Routed L2 reuse : {pct:.1f}%")
            if s2r.get("interpretation"):
                print(f"    {s2r['interpretation']}")
        within = dr.get("within_type_inter_layer_reuse", {})
        for et, stats in within.items():
            pct = (stats.get("reuse_fraction") or 0) * 100
            print(f"  {et} inter-layer : {pct:.1f}%")
    else:
        if dr.get("note"):
            print(f"\nData Reuse: {dr['note']}")

    print()
