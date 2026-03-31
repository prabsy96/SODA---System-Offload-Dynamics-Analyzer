"""NCU bridge: per-unique-(op, shape, dtype) HBM profiling for the MoE pipeline.

When ``torch.profiler`` CUPTI counters are unavailable or return all-zero
(``hbm_source == "cupti_zero"`` / ``"cupti_unavailable"``), this module
deduplicates profiler events by ``(aten_op, input_shapes, input_types)`` and
runs ``ncu_profile_kernel()`` on each unique combination to get ground-truth
DRAM read/write bytes from hardware counters.

The deduplication key matches the Phase 2 kernel identity heuristic used in
``kerneldb.py``: ``σ(k) = (aten_op, input_dims, input_type)``.  All unique
shapes across all prompts are profiled — including all routed-expert variants.

Workflow::

    profiler events (all prompts merged)
         │
         ▼
    deduplicate → unique (op, shapes, dtypes) entries
         │
         ▼
    ncu_profile_kernel() for each unique entry
         │
         ▼
    {(op, shapes_key, types_key): {"hbm_read_bytes", "hbm_write_bytes", ...}}
         │
         ▼
    merge_ncu_into_events() — stamps each event with measured bytes
         │
         hbm_source = "ncu"
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Metrics needed for HBM + L2 byte counts (totals, not rates).
# Phase 2 NCU_METRICS only has lts__t_bytes.sum.per_second (a rate);
# the bridge needs the absolute-byte variants so hbm/l2 accounting is correct.
_BRIDGE_NCU_METRICS = [
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_bytes.sum",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shapes_key(input_shapes: List) -> Tuple:
    """Deterministic hashable key for a list of input shapes."""
    return tuple(
        tuple(int(d) for d in s) if isinstance(s, (list, tuple)) else ()
        for s in (input_shapes or [])
    )


def _types_key(input_types: List[str]) -> Tuple:
    """Deterministic hashable key for a list of input dtypes."""
    return tuple(input_types or [])


def _make_kernel_entry(
    op_name: str,
    input_shapes: List,
    model_dtype: str,
    entry_id: str,
) -> Dict[str, Any]:
    """Build a kernel_entry dict compatible with ncu_profile_kernel().

    ``ncu_profile_kernel`` needs:
      - ``id``
      - ``aten_op.name``, ``aten_op.input_dims``, ``aten_op.input_type``
      - ``kernel.name``, ``kernel.raw_name``

    Empty shapes (``[]``) are filtered out — they represent ``None`` tensor
    arguments (e.g. the absent bias in ``aten::linear(input, weight, bias=None)``
    where the profiler records ``input_shapes = [[M,K], [N,K], []]``).
    Passing an empty shape to ``create_input_tensors()`` creates a 0-dim scalar
    tensor, which causes ``F.linear`` to raise "bias must be 1-dimensional".
    """
    filtered = [s for s in input_shapes if s]
    return {
        "id": entry_id,
        "aten_op": {
            "name": op_name,
            "input_dims": filtered,
            "input_type": [model_dtype] * len(filtered),
            "input_strides": [],
            "concrete_inputs": [],
        },
        "kernel": {"name": "", "raw_name": ""},
    }


# ---------------------------------------------------------------------------
# Op filter
# ---------------------------------------------------------------------------

# Only GEMM-class and elementwise-compute ops are NCU-replayable without
# special input constraints.  Ops like aten::embedding require valid integer
# indices bounded by the vocab/table size, which create_input_tensors() cannot
# derive from shapes alone → replay crashes with out-of-bounds assertions.
_NCU_ELIGIBLE_OPS = frozenset({
    "aten::mm",
    "aten::bmm",
    "aten::addmm",
    "aten::matmul",
    "aten::linear",
    "aten::_scaled_mm",
    "aten::baddbmm",
    "aten::conv2d",
    "aten::layer_norm",
    "aten::group_norm",
    "aten::rms_norm",
    "aten::softmax",
    "aten::log_softmax",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_ncu_on_profiler_events(
    events: List[Dict],
    ncu_output_dir: Path,
    model_dtype: str = "bfloat16",
) -> Dict[Tuple, Dict]:
    """Run NCU on each unique (op_name, input_shapes, input_types) seen in profiler events.

    Deduplication key matches the Phase 2 kernel identity heuristic:
    ``(aten_op, input_dims, input_type)``.  All unique shapes are profiled —
    including all routed-expert variants across all prompts.

    Args:
        events: Per-event records from ``profile_single_prompt()`` across all
            prompts (each has ``aten_op``, ``input_shapes``, ``expert_type``).
        ncu_output_dir: Directory for ncu CSV output files.
        model_dtype: Data type string (e.g. ``"bfloat16"``), used as
            ``input_type`` when constructing the replay script.

    Returns:
        Dict mapping ``(op_name, shapes_key, types_key)`` →
        ``{"hbm_read_bytes", "hbm_write_bytes", "l2_bytes", "hbm_bytes"}``.
        Only entries for which NCU succeeded are present.
    """
    from soda.ncu import ncu_profile_kernel

    ncu_output_dir = Path(ncu_output_dir)
    ncu_output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: collect unique (op, shapes, dtypes) across all events ----
    # Key: (op_name, shapes_key, types_key) → input_shapes
    seen: Dict[Tuple, List] = {}

    for evt in events:
        op_name = evt.get("aten_op", "")
        if op_name not in _NCU_ELIGIBLE_OPS:
            continue
        input_shapes = evt.get("input_shapes") or []
        input_types = [model_dtype] * len(input_shapes)
        key = (op_name, _shapes_key(input_shapes), _types_key(input_types))
        if key not in seen:
            seen[key] = input_shapes

    if not seen:
        warnings.warn(
            "ncu_bridge: no NCU-eligible ops found in events "
            f"(eligible: {sorted(_NCU_ELIGIBLE_OPS)}).",
            stacklevel=2,
        )
        return {}

    print(
        f"[ncu_bridge] Profiling {len(seen)} unique (op, shape, dtype) combinations."
    )

    # ---- Step 2: run ncu for each unique entry ----
    results: Dict[Tuple, Dict] = {}

    for idx, (key, input_shapes) in enumerate(seen.items()):
        op_name = key[0]
        entry_id = f"NCU_{idx:04d}_{op_name.replace('aten::', '').replace(':', '_')}"
        kernel_entry = _make_kernel_entry(op_name, input_shapes, model_dtype, entry_id)

        try:
            ncu_result = ncu_profile_kernel(
                kernel_entry, ncu_output_dir,
                metrics=_BRIDGE_NCU_METRICS,
                warmup=2,
                pick_best_kernel=True,
            )
        except Exception as exc:
            warnings.warn(
                f"ncu_bridge: NCU failed for {entry_id} ({op_name}): {exc}",
                stacklevel=2,
            )
            continue

        if ncu_result is None:
            continue

        metrics = ncu_result.get("metrics", {})
        hbm_read = float(metrics.get("dram__bytes_read.sum", 0.0) or 0.0)
        hbm_write = float(metrics.get("dram__bytes_write.sum", 0.0) or 0.0)
        l2_bytes = float(metrics.get("lts__t_bytes.sum", 0.0) or 0.0)

        results[key] = {
            "hbm_read_bytes": hbm_read,
            "hbm_write_bytes": hbm_write,
            "hbm_bytes": hbm_read + hbm_write,
            "l2_bytes": l2_bytes,
            "ncu_entry_id": entry_id,
        }

    print(
        f"[ncu_bridge] NCU completed: {len(results)}/{len(seen)} entries succeeded."
    )
    return results


def merge_ncu_into_events(
    events: List[Dict],
    ncu_results: Dict[Tuple, Dict],
    model_dtype: str = "bfloat16",
) -> None:
    """Stamp NCU-measured HBM/L2 bytes into profiler events in-place.

    Events whose ``(aten_op, shapes_key, types_key)`` is in ``ncu_results``
    get:
    - ``hbm_bytes``, ``hbm_read_bytes``, ``hbm_write_bytes``, ``l2_bytes``
      overwritten with measured values.
    - ``hbm_source`` set to ``"ncu"``.

    Events with no NCU match retain their existing values unchanged.

    Args:
        model_dtype: Must match the dtype used in ``run_ncu_on_profiler_events``
            so the lookup key is computed identically.
    """
    if not ncu_results:
        return

    hit = 0
    for evt in events:
        op_name = evt.get("aten_op", "")
        input_shapes = evt.get("input_shapes") or []
        input_types = [model_dtype] * len(input_shapes)
        key = (op_name, _shapes_key(input_shapes), _types_key(input_types))
        ncu = ncu_results.get(key)
        if ncu is None:
            continue
        evt["hbm_bytes"] = ncu["hbm_bytes"]
        evt["hbm_read_bytes"] = ncu["hbm_read_bytes"]
        evt["hbm_write_bytes"] = ncu["hbm_write_bytes"]
        evt["l2_bytes"] = ncu["l2_bytes"]
        evt["hbm_source"] = "ncu"
        hit += 1

    print(f"[ncu_bridge] Merged NCU results into {hit}/{len(events)} events.")
