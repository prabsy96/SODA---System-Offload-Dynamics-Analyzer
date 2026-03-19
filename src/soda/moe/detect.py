"""Expert type classification from the SODA kernel database.

Classifies each kernel DB entry into one of:
  "shared_expert"  — fixed full-batch GEMM (always activated, large intermediate)
  "routed_expert"  — variable-batch GEMM (token-routing, many unique activation shapes)
  "gate"           — small routing linear (weight[0] == num_experts)
  "attention"      — attention projection (3D input or square weight)
  "other"          — anything else

Primary detection signal: entry cardinality per weight shape.

The kernel identity key includes input_dims, so each unique activation shape
(token count N) creates its own DB entry.  Routed experts have high cardinality
(variable N -> many entries per weight shape) while shared/gate/attention have
low cardinality (fixed activation shape -> 1-3 entries per weight shape).
"""
from __future__ import annotations

import copy
import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ATen op names that represent GEMM-type operations.
GEMM_OPS: frozenset = frozenset({
    "aten::linear",
    "aten::mm",
    "aten::bmm",
    "aten::addmm",
    "aten::matmul",
    "aten::_scaled_mm",
})

# Primary classification thresholds.
CARDINALITY_THRESHOLD = 5  # >= N unique activation shapes per weight -> routed_expert
CV_THRESHOLD = 0.30        # coefficient of variation on act first-dim -> routed_expert


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_weight_shape(entry: Dict) -> Optional[Tuple[int, ...]]:
    """Return the weight tensor shape or None.

    Accounts for op-specific input ordering:
      aten::addmm(bias, input, weight, ...) → weight at input_dims[2]
      all others                             → weight at input_dims[1]
    """
    op_name = entry.get("aten_op", {}).get("name", "")
    input_dims = entry.get("aten_op", {}).get("input_dims", [])
    if op_name == "aten::addmm":
        if len(input_dims) >= 3 and input_dims[2]:
            return tuple(int(d) for d in input_dims[2])
        return None
    if len(input_dims) >= 2 and input_dims[1]:
        return tuple(int(d) for d in input_dims[1])
    return None


def _get_activation_shape(entry: Dict) -> Optional[Tuple[int, ...]]:
    """Return the activation tensor shape or None.

    Accounts for op-specific input ordering:
      aten::addmm(bias, input, weight, ...) → activation at input_dims[1]
      all others                             → activation at input_dims[0]
    """
    op_name = entry.get("aten_op", {}).get("name", "")
    input_dims = entry.get("aten_op", {}).get("input_dims", [])
    if op_name == "aten::addmm":
        if len(input_dims) >= 2 and input_dims[1]:
            return tuple(int(d) for d in input_dims[1])
        return None
    if input_dims and input_dims[0]:
        return tuple(int(d) for d in input_dims[0])
    return None


def _get_gemm_dims(weight_shape: Tuple[int, ...], op_name: str) -> Tuple[int, int]:
    """Return (output_dim, input_dim) based on ATen op weight storage convention.

    aten::linear → weight = (N, K): output_dim = w[0], input_dim = w[1]
    aten::mm/addmm/matmul/bmm/etc. → weight = (..., K, N): output = w[-1], input = w[-2]
    """
    if not weight_shape or len(weight_shape) < 2:
        w0 = weight_shape[0] if weight_shape else 0
        return (w0, 0)
    if op_name == "aten::linear":
        return (int(weight_shape[0]), int(weight_shape[1]))
    # mm, addmm, matmul, _scaled_mm, bmm: (..., K, N)
    return (int(weight_shape[-1]), int(weight_shape[-2]))


def _is_gemm_entry(entry: Dict) -> bool:
    """Return True if the entry's ATen op is a GEMM-type operation."""
    return entry.get("aten_op", {}).get("name", "") in GEMM_OPS


def _compute_group_signals(entries: List[Dict]) -> Dict:
    """Compute cardinality and variance signals for a group of entries sharing a weight shape."""
    act_shapes = set()
    act_first_dims = []
    is_3d = False

    for e in entries:
        act = _get_activation_shape(e)
        if act is not None:
            act_shapes.add(act)
            if len(act) > 0:
                act_first_dims.append(act[0])
            if len(act) >= 3:
                is_3d = True

    n_unique = len(act_shapes)

    if len(act_first_dims) > 1:
        mean_dim = statistics.mean(act_first_dims)
        std_dim = statistics.stdev(act_first_dims)
        cv = std_dim / mean_dim if mean_dim > 0 else 0.0
    else:
        cv = 0.0

    mean_freq = statistics.mean(
        e.get("statistics", {}).get("frequency", 1) for e in entries
    )

    return {
        "n_unique_act": n_unique,
        "cv": cv,
        "mean_freq": mean_freq,
        "is_3d": is_3d,
        "act_first_dims": act_first_dims,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_moe_config(
    kernels: List[Dict],
    shared_dim_override: Optional[int] = None,
    routed_dim_override: Optional[int] = None,
) -> Dict:
    """Auto-detect MoE configuration from structural signals in the kernel database.

    Args:
        kernels: List of entries from kernel_database.json["kernels"].
        shared_dim_override: If provided, treat this as shared expert weight[0].
        routed_dim_override: If provided, treat this as routed expert weight[0].

    Returns:
        Dict with keys:
          num_experts (int|None), shared_dim (int|None), routed_dim (int|None),
          hidden_dim (int|None), detection_method (str),
          cardinality_per_group (Dict[str, int])
    """
    if shared_dim_override is not None and routed_dim_override is not None:
        return {
            "num_experts": None,
            "shared_dim": shared_dim_override,
            "routed_dim": routed_dim_override,
            "hidden_dim": None,
            "detection_method": "override",
            "cardinality_per_group": {},
        }

    # Group GEMM entries by weight shape
    groups: Dict[Tuple[int, ...], List[Dict]] = defaultdict(list)
    for entry in kernels:
        if not _is_gemm_entry(entry):
            continue
        weight_shape = _get_weight_shape(entry)
        if weight_shape and len(weight_shape) >= 1:
            groups[weight_shape].append(entry)

    if not groups:
        return {
            "num_experts": None,
            "shared_dim": shared_dim_override,
            "routed_dim": routed_dim_override,
            "hidden_dim": None,
            "detection_method": "none",
            "cardinality_per_group": {},
        }

    # Compute signals per group
    cardinality_per_group: Dict[str, int] = {}
    group_stats = []
    for weight_shape, entries in groups.items():
        signals = _compute_group_signals(entries)
        cardinality_per_group[str(list(weight_shape))] = signals["n_unique_act"]
        # Derive GEMM output/input dims based on dominant ATen op type
        dominant_op = entries[0].get("aten_op", {}).get("name", "")
        output_dim, input_dim = _get_gemm_dims(weight_shape, dominant_op)
        group_stats.append({
            "weight_shape": weight_shape,
            "entries": entries,
            "w0": weight_shape[0],
            "w1": weight_shape[1] if len(weight_shape) > 1 else 0,
            "output_dim": output_dim,
            "input_dim": input_dim,
            **signals,
        })

    # Infer hidden_dim: most common input_dim (contraction dimension K).
    # For aten::linear weight=(N,K), input_dim=K.  For aten::mm weight=(K,N),
    # input_dim=K.  In both cases K is the hidden dimension.
    input_dim_values = [g["input_dim"] for g in group_stats if g["input_dim"] > 0]
    hidden_dim = max(set(input_dim_values), key=input_dim_values.count) if input_dim_values else None

    # Split into routed (high cardinality) vs fixed (low cardinality) groups
    routed_groups = [
        g for g in group_stats
        if g["n_unique_act"] >= CARDINALITY_THRESHOLD or g["cv"] > CV_THRESHOLD
    ]
    fixed_groups = [g for g in group_stats if g not in routed_groups]

    # Identify gate: smallest output_dim significantly below hidden (routing vector).
    # Exclude 3D groups (bmm attention heads) and scalar projections (output_dim <= 1).
    gate_dim = None
    if hidden_dim and fixed_groups:
        small_fixed = [
            g for g in fixed_groups
            if not g["is_3d"]
            and g["output_dim"] > 1
            and g["output_dim"] <= hidden_dim // 4
        ]
        if small_fixed:
            gate_dim = min(g["output_dim"] for g in small_fixed)

    # Identify shared expert: largest output_dim among non-3D, non-gate fixed groups
    shared_dim = None
    non_3d_fixed = [
        g for g in fixed_groups
        if not g["is_3d"] and g["output_dim"] != gate_dim
    ]
    if non_3d_fixed:
        best = max(non_3d_fixed, key=lambda g: g["output_dim"])
        shared_dim = best["output_dim"]

    # Identify routed expert: most common routed group by entry count.
    # Prefer groups whose output_dim != hidden_dim to avoid conflating
    # the routed intermediate dimension with the down-projection dimension.
    routed_dim = None
    if routed_groups:
        non_hidden_routed = [
            g for g in routed_groups if g["output_dim"] != hidden_dim
        ]
        candidates = non_hidden_routed if non_hidden_routed else routed_groups
        most_common = max(candidates, key=lambda g: len(g["entries"]))
        routed_dim = most_common["output_dim"]

    return {
        "num_experts": gate_dim,
        "shared_dim": shared_dim,
        "routed_dim": routed_dim,
        "hidden_dim": hidden_dim,
        "detection_method": "cardinality",
        "cardinality_per_group": cardinality_per_group,
    }


def classify_kernel_entries(
    kernels: List[Dict],
    model_config: Optional[Dict] = None,
    shared_dim_override: Optional[int] = None,
    routed_dim_override: Optional[int] = None,
) -> List[Dict]:
    """Annotate each kernel DB entry with an expert_type field.

    Detection priority:
      1. CLI overrides (shared_dim_override / routed_dim_override)
      2. HuggingFace model_config (moe_intermediate_size,
         shared_expert_intermediate_size)
      3. Cardinality heuristic (primary):
           n_unique_act >= CARDINALITY_THRESHOLD or CV > CV_THRESHOLD
           -> "routed_expert"
      4. Tiebreaker on low-cardinality groups by weight shape characteristics

    Args:
        kernels: List of entries from kernel_database.json["kernels"].
        model_config: Optional HuggingFace model config dict.
        shared_dim_override: Override for shared expert intermediate dim.
        routed_dim_override: Override for routed expert intermediate dim.

    Returns:
        New list of entries, each annotated with "expert_type" key.
    """
    # Apply HuggingFace model_config if provided
    if model_config is not None:
        if shared_dim_override is None:
            shared_dim_override = model_config.get("shared_expert_intermediate_size")
        if routed_dim_override is None:
            routed_dim_override = model_config.get("moe_intermediate_size")

    config = detect_moe_config(
        kernels,
        shared_dim_override=shared_dim_override,
        routed_dim_override=routed_dim_override,
    )

    shared_dim = config.get("shared_dim")
    routed_dim = config.get("routed_dim")
    num_experts = config.get("num_experts")
    hidden_dim = config.get("hidden_dim") or 1
    detection_method = config.get("detection_method", "cardinality")

    # Group GEMM entries by weight shape to batch-assign types
    groups: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for idx, entry in enumerate(kernels):
        if not _is_gemm_entry(entry):
            continue
        weight_shape = _get_weight_shape(entry)
        if weight_shape is not None:
            groups[weight_shape].append(idx)

    # Assign expert_type per weight shape group
    group_type: Dict[Tuple[int, ...], str] = {}
    for weight_shape, indices in groups.items():
        entries = [kernels[i] for i in indices]
        # Derive semantic output/input dims from the op type
        dominant_op = entries[0].get("aten_op", {}).get("name", "")
        output_dim, input_dim = _get_gemm_dims(weight_shape, dominant_op)

        if detection_method == "override":
            if shared_dim is not None and output_dim == shared_dim:
                group_type[weight_shape] = "shared_expert"
            elif routed_dim is not None and output_dim == routed_dim:
                group_type[weight_shape] = "routed_expert"
            # Down-projection: output_dim = hidden, input_dim = intermediate
            elif shared_dim is not None and input_dim == shared_dim:
                group_type[weight_shape] = "shared_expert"
            elif routed_dim is not None and input_dim == routed_dim:
                group_type[weight_shape] = "routed_expert"
            else:
                group_type[weight_shape] = "other"
            continue

        signals = _compute_group_signals(entries)

        # Primary signal: cardinality / variance
        if (signals["n_unique_act"] >= CARDINALITY_THRESHOLD
                or signals["cv"] > CV_THRESHOLD):
            group_type[weight_shape] = "routed_expert"
            continue

        # Known dims from auto-detection
        if shared_dim is not None and output_dim == shared_dim:
            group_type[weight_shape] = "shared_expert"
        elif routed_dim is not None and output_dim == routed_dim:
            group_type[weight_shape] = "routed_expert"
        elif num_experts is not None and output_dim == num_experts:
            group_type[weight_shape] = "gate"
        # Tiebreakers
        elif signals["is_3d"]:
            group_type[weight_shape] = "attention"
        elif output_dim <= 1:
            group_type[weight_shape] = "other"
        elif output_dim == hidden_dim:
            # Down-projection: check input_dim against known intermediates
            if shared_dim is not None and input_dim == shared_dim:
                group_type[weight_shape] = "shared_expert"
            elif routed_dim is not None and input_dim == routed_dim:
                group_type[weight_shape] = "routed_expert"
            else:
                group_type[weight_shape] = "attention"
        elif output_dim > 1 and output_dim <= hidden_dim // 4:
            group_type[weight_shape] = "gate"
        else:
            group_type[weight_shape] = "shared_expert"

    # Build annotated output (copy each entry, add expert_type)
    result = []
    for entry in kernels:
        annotated = copy.copy(entry)
        if _is_gemm_entry(entry):
            weight_shape = _get_weight_shape(entry)
            if weight_shape is not None and weight_shape in group_type:
                annotated["expert_type"] = group_type[weight_shape]
            else:
                annotated["expert_type"] = "other"
        else:
            annotated["expert_type"] = "other"
        result.append(annotated)

    return result


def get_entries_by_type(classified: List[Dict], expert_type: str) -> List[Dict]:
    """Return only entries matching the given expert_type."""
    return [e for e in classified if e.get("expert_type") == expert_type]


def sample_routed_entries(
    entries: List[Dict],
    n_samples: int = 10,
) -> List[Dict]:
    """Select a representative sample of routed_expert entries.

    Bins the activation first-dimension range into n_samples equal-width
    buckets and picks one entry per bucket, covering small-N, mid-N, and
    large-N token-count regimes without profiling all entries.

    Args:
        entries: List of routed_expert kernel DB entries.
        n_samples: Maximum number of representative samples to return.

    Returns:
        Sampled list of at most n_samples entries.
    """
    if len(entries) <= n_samples:
        return list(entries)

    # Build (act_first_dim, entry) pairs
    dim_entries = []
    for e in entries:
        act = _get_activation_shape(e)
        if act and len(act) > 0:
            dim_entries.append((act[0], e))

    if not dim_entries:
        return entries[:n_samples]

    dim_entries.sort(key=lambda x: x[0])
    min_dim = dim_entries[0][0]
    max_dim = dim_entries[-1][0]

    if min_dim == max_dim:
        return [dim_entries[0][1]]

    bucket_size = (max_dim - min_dim) / n_samples
    sampled: Dict[int, Dict] = {}
    for dim, entry in dim_entries:
        bucket = min(int((dim - min_dim) / bucket_size), n_samples - 1)
        if bucket not in sampled:
            sampled[bucket] = entry
        if len(sampled) >= n_samples:
            break

    return list(sampled.values())
