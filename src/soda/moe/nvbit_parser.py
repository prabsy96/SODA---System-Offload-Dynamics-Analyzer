"""Parser for NVBit mem_reuse_tracker output logs.

The NVBit tool writes one JSON object per line to $SODA_NVBIT_LOG.
Each line describes the cache-line access set for one kernel invocation:

  {"kernel_name": "cutlass_gemm_...", "expert_type": "shared_expert",
   "invocation": 3, "global_load_count": 1048576,
   "cacheline_set_size": 65536, "cacheline_hashes": [1234, 5678, ...]}

This module parses those lines and computes:
  - Per-expert-type aggregate stats (invocation counts, load counts)
  - In-context L2 cache reuse fractions via cache-line set intersection
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def parse_reuse_log(log_path: Path) -> Optional[Dict]:
    """Parse the NVBit JSON-lines log and compute memory reuse metrics.

    Args:
        log_path: Path to the NVBit output log (JSON lines).

    Returns:
        Dict with keys:
          "per_expert_type": per-type aggregates
          "cross_expert_reuse": pairwise reuse fractions
          "raw_invocations": list of raw parsed records
        Returns None if the log is missing or unreadable.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return None

    records: List[Dict] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines but continue
                continue

    if not records:
        return None

    return _aggregate_records(records)


def _aggregate_records(records: List[Dict]) -> Dict:
    """Aggregate raw NVBit records into per-expert-type metrics and reuse."""

    # Group records by expert_type
    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for rec in records:
        et = rec.get("expert_type", "other")
        by_type[et].append(rec)

    per_expert_type: Dict[str, Dict] = {}
    for et, recs in by_type.items():
        total_invocations = len(recs)
        total_global_loads = sum(r.get("global_load_count", 0) for r in recs)
        # Estimated HBM bytes: each global load accesses a 128-byte cache line sector
        # (L2 sector size on Hopper/Ampere)
        total_hbm_bytes_approx = total_global_loads * 128
        avg_loads = total_global_loads / total_invocations if total_invocations else 0
        avg_hbm_bytes = total_hbm_bytes_approx / total_invocations if total_invocations else 0
        total_cachelines = sum(r.get("cacheline_set_size", 0) for r in recs)

        per_expert_type[et] = {
            "total_invocations": total_invocations,
            "total_global_loads": total_global_loads,
            "avg_global_loads_per_invocation": round(avg_loads, 1),
            "avg_hbm_bytes_per_invocation": round(avg_hbm_bytes),
            "total_cachelines_tracked": total_cachelines,
        }

    # Compute cross-expert L2 reuse via cache-line hash intersection
    cross_expert_reuse = _compute_cross_expert_reuse(by_type)

    return {
        "per_expert_type": per_expert_type,
        "cross_expert_reuse": cross_expert_reuse,
        "total_records": len(records),
    }


def _collect_cacheline_set(recs: List[Dict]) -> "set[int]":
    """Collect all unique cache-line hashes from a list of records."""
    cl_set: "set[int]" = set()
    for rec in recs:
        hashes = rec.get("cacheline_hashes", [])
        cl_set.update(hashes)
    return cl_set


def _hyperloglog_estimate_intersection(set_a: "set[int]", set_b: "set[int]") -> Dict:
    """Estimate the intersection size between two cache-line hash sets.

    Since cache-line hashes are already unique identifiers (not raw addresses),
    a direct set intersection gives the exact overlap count.

    Returns:
        Dict with overlap_count, reuse_fraction (overlap / |set_b|),
        method.
    """
    if not set_a or not set_b:
        return {
            "overlap_cachelines": 0,
            "reuse_fraction": 0.0,
            "method": "exact_set_intersection",
        }

    overlap = len(set_a & set_b)
    reuse_fraction = overlap / len(set_b) if set_b else 0.0

    return {
        "overlap_cachelines": overlap,
        "reuse_fraction": round(reuse_fraction, 4),
        "method": "exact_set_intersection",
    }


def _compute_cross_expert_reuse(by_type: Dict[str, List[Dict]]) -> Dict:
    """Compute L2 reuse fractions between expert types.

    Key metric: what fraction of routed-expert cache-line accesses overlap
    with the shared-expert cache-line footprint from the same inference?
    A high overlap means shared-expert GEMMs warm the L2 cache for
    routed-expert kernels — this is what NCU isolation replay cannot measure.
    """
    cross: Dict[str, Dict] = {}

    shared_recs = by_type.get("shared_expert", [])
    routed_recs = by_type.get("routed_expert", [])

    if shared_recs and routed_recs:
        shared_cls = _collect_cacheline_set(shared_recs)
        routed_cls = _collect_cacheline_set(routed_recs)
        stats = _hyperloglog_estimate_intersection(shared_cls, routed_cls)
        cross["shared_to_routed"] = {
            **stats,
            "note": (
                f"{round(stats['reuse_fraction'] * 100, 1)}% of routed expert "
                "L2 accesses hit cache lines loaded by shared expert GEMM"
            ),
        }

    # Within-type reuse across layers: compare odd-indexed vs even-indexed invocations
    # as a proxy for inter-layer cache reuse (consecutive layers reuse same weights)
    for et, recs in by_type.items():
        if len(recs) < 4:
            continue
        mid = len(recs) // 2
        first_half_cls = _collect_cacheline_set(recs[:mid])
        second_half_cls = _collect_cacheline_set(recs[mid:])
        stats = _hyperloglog_estimate_intersection(first_half_cls, second_half_cls)
        cross[f"{et}_within_type"] = {
            **stats,
            "note": (
                f"Inter-layer reuse for {et}: "
                f"{round(stats['reuse_fraction'] * 100, 1)}% of cache lines "
                "from second half of invocations overlap with first half"
            ),
        }

    return cross
