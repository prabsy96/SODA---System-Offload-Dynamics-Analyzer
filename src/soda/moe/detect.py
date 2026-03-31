"""Expert type classification from the SODA kernel database.

Classifies each kernel DB entry into one of:
  "shared_expert"  — fixed full-batch GEMM (always activated, large intermediate)
  "routed_expert"  — variable-batch GEMM (token-routing, many unique activation shapes)
  "gate"           — small routing linear (weight[0] == num_experts)
  "attention"      — attention projection (3D input or square weight)
  "other"          — anything else

Primary detection: HuggingFace model config (deterministic dimension matching).

The model config provides ``hidden_size``, ``intermediate_size``,
``moe_intermediate_size``, ``shared_expert_intermediate_size``, and
``num_experts`` / ``num_local_experts`` / ``n_routed_experts``.  Weight
shapes are matched against these known dimensions to classify each op.

Fallback: entry cardinality per weight shape (empirical, when config is
unavailable).  The kernel identity key includes input_dims, so each unique
activation shape creates its own DB entry.  Routed experts have high
cardinality while shared/gate/attention have low cardinality.
"""
from __future__ import annotations

import copy
import statistics
import warnings
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
# HuggingFace config normalisation
# ---------------------------------------------------------------------------

def _normalize_moe_hf_config(model_config: Dict) -> Dict:
    """Normalise HuggingFace model config fields across MoE architectures.

    Different architectures use different field names for the same concepts:

    +---------------------------------+------------+---------+---------------+-------+
    | Field                           | Qwen2-MoE  | Mixtral | DeepSeek-V2   | OLMoE |
    +---------------------------------+------------+---------+---------------+-------+
    | routed intermediate             | moe_inter… | inter…  | moe_inter…    | inter…|
    | shared intermediate             | shared_e…  | —       | inter…        | —     |
    | num routed experts              | num_exp…   | num_lo… | n_routed_exp… | num_… |
    | has shared experts              | yes (1)    | no      | yes (2)       | no    |
    +---------------------------------+------------+---------+---------------+-------+

    Returns:
        Dict with normalised keys:
          hidden_dim, routed_dim, shared_dim (or None), num_experts,
          has_shared_experts, top_k, first_dense_layers, detection_method
    """
    hidden = model_config.get("hidden_size")

    # --- num_experts ---
    num_experts = (
        model_config.get("num_experts")
        or model_config.get("num_local_experts")
        or model_config.get("n_routed_experts")
    )

    # --- has_shared_experts ---
    n_shared = model_config.get("n_shared_experts", 0)
    has_shared = (
        model_config.get("shared_expert_intermediate_size") is not None
        or (isinstance(n_shared, int) and n_shared > 0)
    )

    # --- routed_dim ---
    # moe_intermediate_size is specific to routed experts when present.
    # Mixtral / OLMoE use intermediate_size for their (only) expert type.
    routed_dim = model_config.get("moe_intermediate_size")
    if routed_dim is None:
        routed_dim = model_config.get("intermediate_size")

    # --- shared_dim ---
    # shared_expert_intermediate_size is explicit in Qwen2-MoE.
    # DeepSeek-V2 uses intermediate_size for its shared experts (moe_intermediate_size
    # is the smaller routed dim).
    # Models without shared experts: shared_dim = None.
    shared_dim: Optional[int] = None
    if has_shared:
        shared_dim = model_config.get("shared_expert_intermediate_size")
        if shared_dim is None:
            shared_dim = model_config.get("intermediate_size")

    # --- top_k ---
    top_k = model_config.get("num_experts_per_tok")

    # --- dense layers (DeepSeek first_k_dense_replace) ---
    first_dense = model_config.get("first_k_dense_replace", 0)

    return {
        "hidden_dim": hidden,
        "routed_dim": routed_dim,
        "shared_dim": shared_dim,
        "num_experts": num_experts,
        "has_shared_experts": has_shared,
        "top_k": top_k,
        "first_dense_layers": first_dense,
        "detection_method": "model_config",
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

    # Identify routed expert FIRST (needed to bound shared_dim candidates).
    # Most common routed group by entry count.  Prefer groups whose
    # output_dim != hidden_dim to avoid conflating the routed intermediate
    # dimension with the down-projection dimension.
    routed_dim = None
    if routed_groups:
        non_hidden_routed = [
            g for g in routed_groups if g["output_dim"] != hidden_dim
        ]
        candidates = non_hidden_routed if non_hidden_routed else routed_groups
        most_common = max(candidates, key=lambda g: len(g["entries"]))
        routed_dim = most_common["output_dim"]

    # Identify shared expert: largest output_dim among non-3D, non-gate
    # fixed groups that falls within a structurally plausible MLP range.
    # Without bounds, the LM head (vocab-sized output_dim, low cardinality)
    # gets selected as shared_dim — see job_90128 Qwen1.5-MoE-A2.7B bug.
    shared_dim = None
    non_3d_fixed = [
        g for g in fixed_groups
        if not g["is_3d"]
        and g["output_dim"] != gate_dim
        and g["output_dim"] > 1
    ]
    if non_3d_fixed:
        # Bound candidates: shared intermediate must be in a plausible range.
        # Use both routed_dim and hidden_dim as anchors when available to
        # exclude vocab-sized projections (LM head).  The tighter of the
        # two bounds wins: e.g. Mixtral has routed_dim=14336 but
        # hidden_dim*6=24576 correctly excludes vocab=32000.
        if routed_dim is not None and hidden_dim is not None:
            lo = routed_dim
            hi = min(routed_dim * 8, hidden_dim * 6)
        elif routed_dim is not None:
            lo, hi = routed_dim, routed_dim * 8
        elif hidden_dim is not None:
            lo, hi = hidden_dim, hidden_dim * 6
        else:
            lo, hi = 0, float("inf")
        bounded = [
            g for g in non_3d_fixed
            if lo <= g["output_dim"] <= hi
            and g["output_dim"] != hidden_dim  # exclude attention-like dims
        ]
        if bounded:
            best = max(bounded, key=lambda g: g["output_dim"])
            shared_dim = best["output_dim"]
        elif non_3d_fixed:
            # No candidate in range — pick largest that isn't hidden_dim,
            # excluding obvious outliers (> 6x hidden_dim = likely vocab/embed).
            if routed_dim is not None and hidden_dim is not None:
                max_dim = min(routed_dim * 8, hidden_dim * 6)
            elif hidden_dim is not None:
                max_dim = hidden_dim * 6
            else:
                max_dim = float("inf")
            fallback = [
                g for g in non_3d_fixed
                if g["output_dim"] != hidden_dim
                and g["output_dim"] <= max_dim
            ]
            if fallback:
                best = max(fallback, key=lambda g: g["output_dim"])
                shared_dim = best["output_dim"]

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
      2. HuggingFace model_config — normalised via
         :func:`_normalize_moe_hf_config` to handle Qwen, Mixtral,
         DeepSeek, OLMoE field name differences.  Deterministic
         dimension matching (no heuristics).
      3. Cardinality heuristic (last resort, when no config available):
           n_unique_act >= CARDINALITY_THRESHOLD or CV > CV_THRESHOLD
           -> "routed_expert"

    Weight-shape matching rules (when config dimensions are known):

      +-------------------+------------------+-------------------+
      | output_dim        | input_dim        | expert_type       |
      +-------------------+------------------+-------------------+
      | == routed_dim     | == hidden_dim    | routed_expert     |
      | == hidden_dim     | == routed_dim    | routed_expert     |
      | == shared_dim     | == hidden_dim    | shared_expert     |
      | == hidden_dim     | == shared_dim    | shared_expert     |
      | == num_experts    | == hidden_dim    | gate              |
      | 3D input tensor   | —                | attention         |
      | == hidden_dim     | == hidden_dim    | attention         |
      +-------------------+------------------+-------------------+

    Projection sub-type (``projection_type``):
      - ``output_dim == hidden_dim`` → ``"down_proj"``
      - else → ``"gate_up_proj"`` (split later by trace or frequency)

    Args:
        kernels: List of entries from kernel_database.json["kernels"].
        model_config: Optional HuggingFace model config dict.
        shared_dim_override: Override for shared expert intermediate dim.
        routed_dim_override: Override for routed expert intermediate dim.

    Returns:
        New list of entries, each annotated with "expert_type" key.
    """
    # ------------------------------------------------------------------
    # Resolve configuration: CLI overrides > model_config > cardinality
    # ------------------------------------------------------------------
    hf_cfg: Optional[Dict] = None
    if model_config is not None:
        hf_cfg = _normalize_moe_hf_config(model_config)

    # CLI overrides take highest precedence
    if shared_dim_override is not None and hf_cfg is not None:
        hf_cfg["shared_dim"] = shared_dim_override
    if routed_dim_override is not None and hf_cfg is not None:
        hf_cfg["routed_dim"] = routed_dim_override

    # If we have CLI overrides but no model_config, build a minimal config
    if hf_cfg is None and (shared_dim_override or routed_dim_override):
        hf_cfg = {
            "hidden_dim": None,
            "routed_dim": routed_dim_override,
            "shared_dim": shared_dim_override,
            "num_experts": None,
            "has_shared_experts": shared_dim_override is not None,
            "top_k": None,
            "first_dense_layers": 0,
            "detection_method": "override",
        }

    # Use config-based path if we have at least one dimension.
    use_config = (
        hf_cfg is not None
        and (hf_cfg.get("routed_dim") or hf_cfg.get("shared_dim"))
    )

    if use_config:
        return _classify_from_config(kernels, hf_cfg)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Fallback: cardinality heuristic (no model config available)
    # ------------------------------------------------------------------
    return _classify_from_cardinality(
        kernels,
        shared_dim_override=shared_dim_override,
        routed_dim_override=routed_dim_override,
    )


def _classify_from_config(
    kernels: List[Dict],
    cfg: Dict,
) -> List[Dict]:
    """Config-based classification — deterministic dimension matching.

    Uses known dimensions from the HuggingFace model config (normalised)
    to classify each GEMM weight-shape group.  No cardinality heuristics.
    """
    hidden_dim = cfg.get("hidden_dim")
    routed_dim = cfg.get("routed_dim")
    shared_dim = cfg.get("shared_dim")
    num_experts = cfg.get("num_experts")
    has_shared = cfg.get("has_shared_experts", False)
    detection_method = cfg.get("detection_method", "model_config")

    # Group GEMM entries by weight shape
    groups: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for idx, entry in enumerate(kernels):
        if not _is_gemm_entry(entry):
            continue
        ws = _get_weight_shape(entry)
        if ws is not None:
            groups[ws].append(idx)

    # If hidden_dim is unknown, try to infer from kernel data.  Requires
    # >= 3 weight-shape groups to be reliable (with fewer groups, the most
    # common input_dim may be intermediate, not hidden).
    if hidden_dim is None and len(groups) >= 3:
        input_dims: List[int] = []
        for ws, indices in groups.items():
            dom_op = kernels[indices[0]].get("aten_op", {}).get("name", "")
            _, inp = _get_gemm_dims(ws, dom_op)
            if inp > 0:
                input_dims.append(inp)
        if input_dims:
            hidden_dim = max(set(input_dims), key=input_dims.count)

    def _dim_match(dim_a: Optional[int], dim_b: Optional[int]) -> bool:
        """True when both dims are known and equal."""
        return dim_a is not None and dim_b is not None and dim_a == dim_b

    # Classify each weight-shape group
    group_type: Dict[Tuple[int, ...], str] = {}
    for ws, indices in groups.items():
        dominant_op = kernels[indices[0]].get("aten_op", {}).get("name", "")
        output_dim, input_dim = _get_gemm_dims(ws, dominant_op)

        # Check 3D input → attention (e.g., aten::bmm)
        entries = [kernels[i] for i in indices]
        is_3d = any(
            len((_get_activation_shape(e) or ())) >= 3 for e in entries
        )
        if is_3d:
            group_type[ws] = "attention"
            continue

        # --- Deterministic dimension matching ---
        classified = False

        # Gate: output_dim == num_experts (small routing linear)
        if not classified and _dim_match(output_dim, num_experts):
            group_type[ws] = "gate"
            classified = True

        # Shared expert: expanding (H→shared) or contracting (shared→H)
        if not classified and has_shared and shared_dim is not None:
            is_expand = _dim_match(output_dim, shared_dim) and _dim_match(input_dim, hidden_dim)
            is_contract = _dim_match(output_dim, hidden_dim) and _dim_match(input_dim, shared_dim)
            # Relaxed: if hidden_dim unknown, match on shared_dim alone
            if hidden_dim is None:
                is_expand = output_dim == shared_dim
                is_contract = input_dim == shared_dim
            if is_expand or is_contract:
                group_type[ws] = "shared_expert"
                classified = True

        # Routed expert: expanding (H→routed) or contracting (routed→H)
        if not classified and routed_dim is not None:
            is_expand = _dim_match(output_dim, routed_dim) and _dim_match(input_dim, hidden_dim)
            is_contract = _dim_match(output_dim, hidden_dim) and _dim_match(input_dim, routed_dim)
            if hidden_dim is None:
                is_expand = output_dim == routed_dim
                is_contract = input_dim == routed_dim
            if is_expand or is_contract:
                group_type[ws] = "routed_expert"
                classified = True

        # When shared_dim == routed_dim AND has_shared, both match.
        # Use activation shape variance to distinguish.
        if not classified and has_shared and shared_dim == routed_dim:
            signals = _compute_group_signals(entries)
            if (signals["n_unique_act"] >= CARDINALITY_THRESHOLD
                    or signals["cv"] > CV_THRESHOLD):
                group_type[ws] = "routed_expert"
            else:
                group_type[ws] = "shared_expert"
            classified = True

        # Attention: square weight (hidden→hidden, Q/K/V/O projections)
        if not classified and hidden_dim and output_dim == hidden_dim and input_dim == hidden_dim:
            group_type[ws] = "attention"
            classified = True

        # Anything remaining
        if not classified:
            group_type[ws] = "other"

    return _build_annotated_entries(kernels, group_type, hidden_dim, detection_method)


def _classify_from_cardinality(
    kernels: List[Dict],
    shared_dim_override: Optional[int] = None,
    routed_dim_override: Optional[int] = None,
) -> List[Dict]:
    """Cardinality-based classification — last-resort fallback.

    Used only when no HuggingFace model config is available.
    """
    config = detect_moe_config(
        kernels,
        shared_dim_override=shared_dim_override,
        routed_dim_override=routed_dim_override,
    )

    shared_dim = config.get("shared_dim")
    routed_dim = config.get("routed_dim")
    num_experts = config.get("num_experts")
    hidden_dim_raw = config.get("hidden_dim")
    hidden_dim = hidden_dim_raw or 1
    detection_method = config.get("detection_method", "cardinality")

    groups: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for idx, entry in enumerate(kernels):
        if not _is_gemm_entry(entry):
            continue
        ws = _get_weight_shape(entry)
        if ws is not None:
            groups[ws].append(idx)

    group_type: Dict[Tuple[int, ...], str] = {}
    for ws, indices in groups.items():
        entries = [kernels[i] for i in indices]
        dominant_op = entries[0].get("aten_op", {}).get("name", "")
        output_dim, input_dim = _get_gemm_dims(ws, dominant_op)

        if detection_method == "override":
            if shared_dim is not None and output_dim == shared_dim:
                group_type[ws] = "shared_expert"
            elif routed_dim is not None and output_dim == routed_dim:
                group_type[ws] = "routed_expert"
            elif shared_dim is not None and input_dim == shared_dim:
                group_type[ws] = "shared_expert"
            elif routed_dim is not None and input_dim == routed_dim:
                group_type[ws] = "routed_expert"
            else:
                group_type[ws] = "other"
            continue

        signals = _compute_group_signals(entries)

        if (signals["n_unique_act"] >= CARDINALITY_THRESHOLD
                or signals["cv"] > CV_THRESHOLD):
            group_type[ws] = "routed_expert"
            continue

        if shared_dim is not None and output_dim == shared_dim:
            group_type[ws] = "shared_expert"
        elif routed_dim is not None and output_dim == routed_dim:
            group_type[ws] = "routed_expert"
        elif num_experts is not None and output_dim == num_experts:
            group_type[ws] = "gate"
        elif signals["is_3d"]:
            group_type[ws] = "attention"
        elif output_dim <= 1:
            group_type[ws] = "other"
        elif output_dim == hidden_dim:
            if shared_dim is not None and input_dim == shared_dim:
                group_type[ws] = "shared_expert"
            elif routed_dim is not None and input_dim == routed_dim:
                group_type[ws] = "routed_expert"
            else:
                group_type[ws] = "attention"
        elif output_dim > 1 and output_dim <= hidden_dim // 4:
            group_type[ws] = "gate"
        elif (shared_dim is not None and input_dim == shared_dim):
            # Contracting shared expert projection (shared_dim → hidden_dim
            # variant with non-standard hidden_dim, e.g. after layer norm)
            group_type[ws] = "shared_expert"
        elif (routed_dim is not None and input_dim == routed_dim):
            group_type[ws] = "routed_expert"
        else:
            # Unmatched fixed-cardinality GEMM (LM head, embeddings, etc.)
            group_type[ws] = "other"

    _has_experts = any(
        t in ("shared_expert", "routed_expert") for t in group_type.values()
    )
    if _has_experts and hidden_dim_raw is None:
        warnings.warn(
            "MoE classification: hidden_dim could not be auto-detected. "
            "Projection sub-types (gate/up/down) will be missing. "
            "Provide --moe-shared-dim and --moe-routed-dim to override.",
            stacklevel=2,
        )

    return _build_annotated_entries(kernels, group_type, hidden_dim_raw, detection_method)


def _build_annotated_entries(
    kernels: List[Dict],
    group_type: Dict[Tuple[int, ...], str],
    hidden_dim: Optional[int],
    detection_method: str,
) -> List[Dict]:
    """Build annotated output list with expert_type + projection_type."""
    result = []
    for entry in kernels:
        annotated = copy.copy(entry)
        if _is_gemm_entry(entry):
            ws = _get_weight_shape(entry)
            if ws is not None and ws in group_type:
                annotated["expert_type"] = group_type[ws]
                et = annotated["expert_type"]
                if et in ("shared_expert", "routed_expert") and hidden_dim is not None and len(ws) >= 2:
                    op_name = entry.get("aten_op", {}).get("name", "")
                    output_dim, _ = _get_gemm_dims(ws, op_name)
                    annotated["projection_type"] = (
                        "down_proj" if output_dim == hidden_dim else "gate_up_proj"
                    )
                annotated["detection_method"] = detection_method
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


def apply_trace_projection_tags(
    classified: List[Dict],
    sequences: List[Dict],
) -> List[Dict]:
    """Enrich classified entries with gate/up distinction from trace sequences.

    Uses :func:`tag_moe_projections_from_sequences` to detect SwiGLU
    gate/up pairs in the trace via temporal ATen dispatch order, then maps
    the results back to classified entries via weight-shape matching.

    Only modifies entries with ``projection_type == "gate_up_proj"``.
    Entries without trace evidence are split via the frequency-ratio
    fallback in :func:`refine_gate_up_projections`.

    Args:
        classified: Entries from ``classify_kernel_entries()``.
        sequences: Output of ``link_sequences()`` from the same model's
            ``trace.json``.

    Returns:
        New list with ``gate_up_proj`` entries split into ``gate_proj`` +
        ``up_proj`` where the trace confirms the SwiGLU pattern.
    """
    tagged_seqs = tag_moe_projections_from_sequences(sequences)

    # Count projection tags per weight shape from tagged sequences.
    shape_tags: Dict[Tuple[int, ...], Dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for seq in tagged_seqs:
        tag = seq.get("moe_projection")
        if tag is None:
            continue
        ws = _get_weight_shape(seq)
        if ws is not None:
            shape_tags[ws][tag] += 1

    # Walk classified entries; split gate_up_proj where trace confirms.
    result: List[Dict] = []
    unresolved: List[Dict] = []

    for entry in classified:
        pt = entry.get("projection_type")
        if pt != "gate_up_proj":
            result.append(entry)
            continue

        ws = _get_weight_shape(entry)
        tags = shape_tags.get(ws, {}) if ws is not None else {}

        if "gate_proj" in tags and "up_proj" in tags:
            freq = entry.get("statistics", {}).get("frequency", 0)
            half = freq // 2

            gate = copy.deepcopy(entry)
            gate["projection_type"] = "gate_proj"
            gate["statistics"]["frequency"] = half
            gate["statistics"]["total_duration_us"] = round(
                gate["statistics"].get("total_duration_us", 0) / 2, 4
            )
            gate["trace_confirmed"] = True

            up = copy.deepcopy(entry)
            up["projection_type"] = "up_proj"
            up["statistics"]["frequency"] = freq - half
            up["statistics"]["total_duration_us"] = round(
                up["statistics"].get("total_duration_us", 0) / 2, 4
            )
            up["trace_confirmed"] = True

            result.append(gate)
            result.append(up)
        else:
            # No trace evidence — collect for frequency-ratio fallback.
            unresolved.append(entry)

    # Apply frequency-ratio fallback to any unresolved gate_up_proj entries.
    # Include down_proj entries for context (refine needs them for ratio check).
    if unresolved:
        down_context = [
            e for e in classified if e.get("projection_type") == "down_proj"
        ]
        fallback = refine_gate_up_projections(unresolved + down_context)
        # Only keep entries that originated from unresolved (drop duplicate down_proj).
        unresolved_ids = {id(e) for e in unresolved}
        for fb in fallback:
            # refine_gate_up_projections deep-copies splits, so identity check
            # won't match — but down_proj entries pass through as-is.
            if fb.get("projection_type") == "down_proj":
                continue  # already in result from the first loop
            result.append(fb)

    return result


def tag_moe_projections_from_sequences(
    sequences: List[Dict],
) -> List[Dict]:
    """Tag each sequence with ``moe_projection`` using ATen op dispatch order.

    SODA's ``link_sequences()`` produces one dict per ATen op → kernel
    dispatch, preserving the CPU-side temporal ordering.  In SwiGLU MLP,
    gate and up are **separate** ATen ops that dispatch identical kernels
    (same weight shape).  They appear as consecutive GEMM dispatches:

        GEMM(W_shape)  →  silu/gelu  →  GEMM(W_shape)  →  mul  →  GEMM(W')

    Detection rule — consecutive GEMM pair with **identical weight shape**:
        first  = ``gate_proj``
        second = ``up_proj``
    Remaining GEMMs = ``down_proj`` (or attention/other — untagged).

    No kernel DB needed.  Works directly from ``trace.json`` →
    ``collect_events()`` → ``link_sequences()``.

    Args:
        sequences: Output of ``link_sequences()`` — list of dicts, each
            with ``aten_op``, ``kernel``, ``cuda_launch`` keys.

    Returns:
        Same list with ``moe_projection`` field added to tagged sequences
        (``"gate_proj"``, ``"up_proj"``).  Untagged sequences get no field.
    """
    # Sort by ATen op CPU dispatch timestamp — this is the dispatch order.
    sorted_seqs = sorted(sequences, key=lambda s: s.get("aten_op", {}).get("ts", 0))

    # Extract weight shape per sequence for GEMM ops.
    def _weight_key(seq: Dict) -> Optional[Tuple[int, ...]]:
        aten = seq.get("aten_op", {})
        name = aten.get("name", "")
        if name not in GEMM_OPS:
            return None
        dims = aten.get("input_dims", [])
        # addmm: weight at [2]; others: weight at [1]
        if name == "aten::addmm":
            raw = dims[2] if len(dims) > 2 else None
        else:
            raw = dims[1] if len(dims) > 1 else None
        if raw and isinstance(raw, (list, tuple)):
            return tuple(int(d) for d in raw)
        return None

    # Walk sorted sequences and detect consecutive same-shape GEMM pairs.
    i = 0
    n = len(sorted_seqs)
    while i < n - 1:
        wk_i = _weight_key(sorted_seqs[i])
        if wk_i is None:
            i += 1
            continue

        # Look ahead for the next GEMM (skip non-GEMM ops between them).
        j = i + 1
        while j < n and _weight_key(sorted_seqs[j]) is None:
            j += 1

        if j >= n:
            break

        wk_j = _weight_key(sorted_seqs[j])
        if wk_i == wk_j:
            # Consecutive GEMM pair with same weight shape → gate + up.
            sorted_seqs[i]["moe_projection"] = "gate_proj"
            sorted_seqs[j]["moe_projection"] = "up_proj"
            i = j + 1  # skip past the pair
        else:
            i += 1

    return sorted_seqs


def refine_gate_up_projections(classified: List[Dict]) -> List[Dict]:
    """Split gate_up_proj entries using trace-based tagging or frequency ratio.

    Delegates to ``tag_moe_projections_from_sequences()`` when a trace is
    available (called from the pipeline with tagged sequences).  This
    function applies the results to classified kernel DB entries: for each
    ``gate_up_proj`` entry whose frequency is even, clones it into
    ``gate_proj`` + ``up_proj`` with halved frequency.

    Falls back to frequency-ratio verification when trace data is not
    available: ``gate_up_proj.frequency == 2 × down_proj.frequency``.

    Args:
        classified: Entries from ``classify_kernel_entries()`` with
            ``expert_type`` and ``projection_type`` fields.

    Returns:
        New list with gate_up_proj entries replaced by two entries
        (gate_proj + up_proj) where possible.
    """
    # Collect down_proj frequencies per expert_type for ratio verification.
    down_freqs: Dict[str, List[int]] = defaultdict(list)
    for entry in classified:
        if entry.get("projection_type") == "down_proj":
            et = entry.get("expert_type", "other")
            freq = entry.get("statistics", {}).get("frequency", 0)
            if freq > 0:
                down_freqs[et].append(freq)

    result = []
    for entry in classified:
        pt = entry.get("projection_type")
        et = entry.get("expert_type", "other")

        if pt != "gate_up_proj":
            result.append(entry)
            continue

        freq = entry.get("statistics", {}).get("frequency", 0)
        half_freq = freq // 2

        # Verify: even frequency AND at least one down_proj for this type.
        can_split = half_freq > 0 and freq % 2 == 0 and down_freqs.get(et)

        if not can_split:
            if freq > 0:
                warnings.warn(
                    f"MoE detect: cannot verify gate/up split for "
                    f"expert_type='{et}', frequency={freq}. "
                    f"Keeping merged gate_up_proj label.",
                    stacklevel=2,
                )
            result.append(entry)
            continue

        gate_entry = copy.deepcopy(entry)
        gate_entry["projection_type"] = "gate_proj"
        gate_entry["statistics"]["frequency"] = half_freq
        gate_entry["statistics"]["total_duration_us"] = round(
            gate_entry["statistics"].get("total_duration_us", 0) / 2, 4
        )

        up_entry = copy.deepcopy(entry)
        up_entry["projection_type"] = "up_proj"
        up_entry["statistics"]["frequency"] = freq - half_freq
        up_entry["statistics"]["total_duration_us"] = round(
            up_entry["statistics"].get("total_duration_us", 0) / 2, 4
        )

        result.append(gate_entry)
        result.append(up_entry)

    return result
