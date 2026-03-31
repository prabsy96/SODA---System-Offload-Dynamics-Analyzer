"""Classify profiler events by MoE expert type using nn.Module hierarchy.

When ``torch.profiler.profile(with_modules=True)`` is used, each profiler
event carries a module scope string (e.g.,
``model.layers.0.mlp.shared_expert.gate_proj``).  This module resolves the
expert type, layer ID, and projection type deterministically — no
cardinality heuristics or shape matching required.

Supports:
  - **Qwen2-MoE**: ``layers.N.mlp.shared_expert.{gate,up,down}_proj``,
    ``layers.N.mlp.experts.M.{gate,up,down}_proj``,
    ``layers.N.mlp.gate``, ``layers.N.mlp.shared_expert_gate``
  - **Mixtral**: ``layers.N.block_sparse_moe.experts.M.{w1,w2,w3}``,
    ``layers.N.block_sparse_moe.gate``
  - **DeepSeek-V2**: ``layers.N.mlp.experts.M.{gate,up,down}_proj``,
    ``layers.N.mlp.shared_experts.{gate,up,down}_proj``
  - Generic attention: ``layers.N.self_attn.{q,k,v,o}_proj``
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Module-path regex patterns
# ---------------------------------------------------------------------------
# Each pattern captures layer_id via ``layers\.(\d+)`` and optionally
# captures the projection suffix.

# Shared expert projections (Qwen: shared_expert, DeepSeek: shared_experts)
_RE_SHARED_EXPERT_PROJ = re.compile(
    r"layers\.(\d+)\..*shared_expert[s]?\.(gate_proj|up_proj|down_proj)"
)

# Shared expert gate (Qwen2-MoE specific: shared_expert_gate is a
# separate Linear that gates the shared expert output)
_RE_SHARED_EXPERT_GATE = re.compile(
    r"layers\.(\d+)\..*shared_expert_gate"
)

# Routed expert projections — Qwen/DeepSeek naming
_RE_ROUTED_EXPERT_PROJ = re.compile(
    r"layers\.(\d+)\..*experts\.(\d+)\.(gate_proj|up_proj|down_proj)"
)

# Routed expert projections — Mixtral naming (w1=gate, w2=down, w3=up)
_RE_ROUTED_EXPERT_W = re.compile(
    r"layers\.(\d+)\..*experts\.(\d+)\.(w1|w2|w3)"
)

# MoE gate (router): ``mlp.gate`` or ``block_sparse_moe.gate``
_RE_MOE_GATE = re.compile(
    r"layers\.(\d+)\.(?:mlp|block_sparse_moe)\.gate(?:\b|$)"
)

# Attention projections
_RE_ATTENTION_PROJ = re.compile(
    r"layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
)

# Generic layer match (fallback for layer_id extraction)
_RE_LAYER = re.compile(r"layers\.(\d+)\.")

# Mixtral w-name to standard projection mapping
_MIXTRAL_W_MAP: Dict[str, str] = {
    "w1": "gate_proj",
    "w2": "down_proj",
    "w3": "up_proj",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_profiler_event(
    module_path: str,
    aten_op: str = "",
) -> Tuple[str, int, Optional[str]]:
    """Classify a single profiler event by its nn.Module scope.

    Args:
        module_path: The module hierarchy string from the profiler event
            (e.g., ``model.layers.5.mlp.shared_expert.gate_proj``).
        aten_op: ATen operator name (e.g., ``aten::linear``).  Currently
            unused but reserved for future disambiguation.

    Returns:
        Tuple of ``(expert_type, layer_id, projection_type)``:
        - ``expert_type``: one of ``"shared_expert"``, ``"routed_expert"``,
          ``"gate"``, ``"attention"``, ``"other"``.
        - ``layer_id``: integer layer index extracted from
          ``layers.(\\d+)``, or ``-1`` if not in a numbered layer.
        - ``projection_type``: ``"gate_proj"``, ``"up_proj"``,
          ``"down_proj"``, or ``None`` for non-projection ops.
    """
    if not module_path:
        return ("other", -1, None)

    # --- Shared expert projections ---
    m = _RE_SHARED_EXPERT_PROJ.search(module_path)
    if m:
        return ("shared_expert", int(m.group(1)), m.group(2))

    # --- Shared expert gate (Qwen2-MoE: gates the shared expert output) ---
    m = _RE_SHARED_EXPERT_GATE.search(module_path)
    if m:
        return ("shared_expert", int(m.group(1)), None)

    # --- Routed expert projections (Qwen/DeepSeek naming) ---
    m = _RE_ROUTED_EXPERT_PROJ.search(module_path)
    if m:
        return ("routed_expert", int(m.group(1)), m.group(3))

    # --- Routed expert projections (Mixtral w1/w2/w3 naming) ---
    m = _RE_ROUTED_EXPERT_W.search(module_path)
    if m:
        proj = _MIXTRAL_W_MAP.get(m.group(3))
        return ("routed_expert", int(m.group(1)), proj)

    # --- MoE gate (router) ---
    m = _RE_MOE_GATE.search(module_path)
    if m:
        return ("gate", int(m.group(1)), None)

    # --- Attention projections ---
    m = _RE_ATTENTION_PROJ.search(module_path)
    if m:
        return ("attention", int(m.group(1)), m.group(2))

    # --- Fallback: extract layer_id if possible ---
    m = _RE_LAYER.search(module_path)
    layer_id = int(m.group(1)) if m else -1
    return ("other", layer_id, None)


def classify_events(
    events: List[Dict],
) -> List[Dict]:
    """Classify a list of profiler event dicts in-place.

    Each event dict is expected to have at least:
    - ``"module_path"`` (str): nn.Module hierarchy from profiler.
    - ``"aten_op"`` (str): ATen operator name.

    Adds three keys to each dict:
    - ``"expert_type"`` (str)
    - ``"layer_id"`` (int)
    - ``"projection_type"`` (str or None)

    Returns:
        The same list, mutated with classification fields.
    """
    for evt in events:
        et, lid, pt = classify_profiler_event(
            evt.get("module_path", ""),
            evt.get("aten_op", ""),
        )
        evt["expert_type"] = et
        evt["layer_id"] = lid
        evt["projection_type"] = pt
    return events
