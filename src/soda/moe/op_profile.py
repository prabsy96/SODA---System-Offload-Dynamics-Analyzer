"""Per-layer per-op kernel profile generator.

Generates op_profile.json: one record per unique kernel invocation type per layer
across the full model, with is_shared_expert flag for downstream filtering.

Data source: classified kernel DB entries (from classify_kernel_entries()).
No GPU required — derived from kernel_database.json + MoE classification.
"""
from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ATen ops that represent GEMM-type operations (matches detect.py GEMM_OPS).
_GEMM_OPS = frozenset({
    "aten::linear",
    "aten::mm",
    "aten::bmm",
    "aten::addmm",
    "aten::matmul",
    "aten::_scaled_mm",
})


# ---------------------------------------------------------------------------
# Public helpers (exported for tests)
# ---------------------------------------------------------------------------

def _dtype_bytes(precision: str) -> int:
    """Return bytes per element for the given precision string."""
    _MAP = {
        "bfloat16": 2,
        "float16": 2,
        "fp16": 2,
        "float32": 4,
        "fp32": 4,
        "int8": 1,
        "int4": 1,
    }
    return _MAP.get(precision.lower(), 2)


def _compute_hbm_fields(
    aten_op_name: str,
    input_dims: List,
    dtype_bytes: int,
) -> Dict:
    """Compute weight_bytes, activation_bytes, hbm_bytes, kv_bytes, flops.

    For GEMM ops, derives M, K, N from input shapes and returns shape-based
    estimates.  For non-GEMM, returns activation bytes from first input only.

    Args:
        aten_op_name: ATen op name (e.g., "aten::linear").
        input_dims:   input_dims list from kernel DB entry's aten_op dict.
        dtype_bytes:  Bytes per element for the run precision.

    Returns:
        Dict with keys: flops, weight_bytes, activation_bytes, hbm_bytes, kv_bytes.
    """
    _zero = {"flops": 0, "weight_bytes": 0.0, "activation_bytes": 0.0,
             "hbm_bytes": 0.0, "kv_bytes": 0.0}

    if not input_dims:
        return _zero

    # addmm(bias, input, weight, ...) → activation at [1], weight at [2]
    if aten_op_name == "aten::addmm":
        act_shape = _normalize_shape(input_dims[1]) if len(input_dims) > 1 else []
        weight_shape = _normalize_shape(input_dims[2]) if len(input_dims) > 2 else []
    else:
        act_shape = _normalize_shape(input_dims[0]) if len(input_dims) > 0 else []
        weight_shape = _normalize_shape(input_dims[1]) if len(input_dims) > 1 else []

    if aten_op_name not in _GEMM_OPS:
        # Non-GEMM: report activation bytes from first input.
        # Some ops (aten::cat, aten::stack) receive a *list of tensors* as
        # input_dims[0], e.g. [[1,16,1024,64],[1,16,1024,64]].  Detect this
        # (list-of-lists) and sum each tensor's bytes individually instead of
        # blindly flattening into one huge product.
        raw_first = input_dims[0] if input_dims else []
        if raw_first and isinstance(raw_first, (list, tuple)) and raw_first and isinstance(raw_first[0], (list, tuple)):
            # List of tensor shapes: sum each tensor's bytes.
            act_bytes = float(sum(_product(t) for t in raw_first) * dtype_bytes)
        else:
            act_bytes = float(_product(act_shape) * dtype_bytes)
        return {**_zero, "activation_bytes": act_bytes, "hbm_bytes": act_bytes}

    if not act_shape or not weight_shape:
        return _zero

    kv_bytes = 0.0

    if aten_op_name == "aten::linear":
        # weight stored as (N, K); input as (..., K).
        N = int(weight_shape[0]) if len(weight_shape) > 0 else 1
        K = int(weight_shape[1]) if len(weight_shape) > 1 else 1
        # Flatten all batch dims of the activation.
        M = _product(act_shape[:-1]) if len(act_shape) > 1 else int(act_shape[0])
        flops = 2 * M * K * N
        weight_bytes = float(N * K * dtype_bytes)
        activation_bytes = float((M * K + M * N) * dtype_bytes)

    elif aten_op_name in ("aten::mm", "aten::addmm"):
        # mm:    (M, K) × (K, N)  → weight_shape = (K, N)
        # addmm: bias, (M, K), (K, N) → weight_shape = (K, N) (extracted above)
        K = int(weight_shape[0]) if len(weight_shape) > 0 else 1
        N = int(weight_shape[1]) if len(weight_shape) > 1 else 1
        M = _product(act_shape[:-1]) if len(act_shape) > 1 else int(act_shape[0])
        flops = 2 * M * K * N
        weight_bytes = float(K * N * dtype_bytes)
        activation_bytes = float((M * K + M * N) * dtype_bytes)

    elif aten_op_name == "aten::bmm":
        # Standard 3D: (B, M, K) × (B, K, N)
        # Attention 4D: (B, H, M, K) × (B, H, K, N)
        is_4d = len(act_shape) == 4 or len(weight_shape) == 4
        if is_4d:
            if len(weight_shape) == 4:
                B = int(weight_shape[0])
                H = int(weight_shape[1])
                K = int(weight_shape[2])
                N = int(weight_shape[3])
            else:
                B, H, K, N = 1, 1, 1, 1
            M = int(act_shape[2]) if len(act_shape) > 2 else 1
            flops = 2 * B * H * M * K * N
            weight_bytes = 0.0  # both tensors are activations in attention
            activation_bytes = float(
                (_product(act_shape) + _product(weight_shape) + B * H * M * N)
                * dtype_bytes
            )
            kv_bytes = float(_product(weight_shape) * dtype_bytes)
        else:
            # 3D bmm
            if len(weight_shape) >= 3:
                B = int(weight_shape[0])
                K = int(weight_shape[1])
                N = int(weight_shape[2])
            else:
                B, K, N = 1, 1, 1
            M = int(act_shape[1]) if len(act_shape) > 1 else 1
            flops = 2 * B * M * K * N
            weight_bytes = 0.0
            activation_bytes = float(
                (_product(act_shape) + _product(weight_shape) + B * M * N)
                * dtype_bytes
            )

    elif aten_op_name in ("aten::matmul", "aten::_scaled_mm"):
        if len(act_shape) >= 2 and len(weight_shape) >= 2:
            M = _product(act_shape[:-1])
            K = int(act_shape[-1])
            N = int(weight_shape[-1])
            flops = 2 * M * K * N
            weight_bytes = float(_product(weight_shape) * dtype_bytes)
            activation_bytes = float((_product(act_shape) + M * N) * dtype_bytes)
        else:
            return _zero

    else:
        return _zero

    hbm_bytes = weight_bytes + activation_bytes
    return {
        "flops": flops,
        "weight_bytes": weight_bytes,
        "activation_bytes": activation_bytes,
        "hbm_bytes": hbm_bytes,
        "kv_bytes": kv_bytes,
    }


def _infer_op_name(
    aten_op_name: str,
    expert_type: str,
    input_dims: List,
    position: int,
    projection_type: Optional[str] = None,
) -> str:
    """Map (aten_op, expert_type, input_dims, position) to a human-readable op_name.

    Args:
        projection_type: "gate_up_proj" or "down_proj" annotation from
            classify_kernel_entries(). When present, overrides the shape
            heuristic for accurate gate/up vs down classification.
    """
    # Non-GEMM special cases.
    if aten_op_name in ("aten::rms_norm",):
        return "rmsnorm"
    if aten_op_name in ("aten::native_layer_norm", "aten::layer_norm"):
        return "layernorm"
    if aten_op_name in ("aten::softmax", "aten::_softmax"):
        return "softmax"
    if aten_op_name in ("aten::silu", "aten::gelu", "aten::relu"):
        return "activation"
    if aten_op_name in ("aten::mul", "aten::add", "aten::div", "aten::sub"):
        return "elementwise"

    # bmm: check 4D vs 3D.
    if aten_op_name == "aten::bmm":
        act_shape = _normalize_shape(input_dims[0]) if input_dims else []
        weight_shape = _normalize_shape(input_dims[1]) if len(input_dims) > 1 else []
        if len(act_shape) == 4 or len(weight_shape) == 4:
            return "attn_bmm_kv"
        return "bmm"

    # GEMM ops: classify by expert_type and projection direction.
    if aten_op_name in _GEMM_OPS:
        if projection_type is not None:
            # Dimension-aware classification from detect.py annotation.
            if expert_type == "shared_expert":
                if projection_type == "gate_proj":
                    return "shared_expert_gate_proj"
                if projection_type == "up_proj":
                    return "shared_expert_up_proj"
                if projection_type == "gate_up_proj":
                    return "shared_expert_gate_proj" if position == 0 else "shared_expert_up_proj"
                return "shared_expert_down_proj"
            if expert_type == "routed_expert":
                if projection_type == "gate_proj":
                    return "routed_expert_gate_proj"
                if projection_type == "up_proj":
                    return "routed_expert_up_proj"
                if projection_type == "gate_up_proj":
                    return "routed_expert_gate_up_proj"
                return "routed_expert_down_proj"
            if expert_type == "gate":
                return "moe_gate_proj"
            if expert_type == "attention":
                return "attn_proj"
            return "linear"

        # No annotation — for expert types, warn user and fall back to
        # shape heuristic.  Non-expert types don't need projection_type.
        if expert_type in ("shared_expert", "routed_expert"):
            import warnings
            warnings.warn(
                "MoE op_profile: entry missing projection_type annotation for "
                "expert_type='{}'. Classification may be wrong for models where "
                "intermediate_size < hidden_dim.".format(expert_type),
                stacklevel=2,
            )
        # Fix addmm weight index (weight at input_dims[2], not [1]).
        if aten_op_name == "aten::addmm":
            weight_shape = _normalize_shape(input_dims[2]) if len(input_dims) > 2 else []
        else:
            weight_shape = _normalize_shape(input_dims[1]) if len(input_dims) > 1 else []
        expanding = _is_expanding(aten_op_name, weight_shape)

        if expert_type == "shared_expert":
            if expanding:
                return "shared_expert_gate_proj" if position == 0 else "shared_expert_up_proj"
            return "shared_expert_down_proj"
        if expert_type == "routed_expert":
            return "routed_expert_gate_up_proj" if expanding else "routed_expert_down_proj"
        if expert_type == "gate":
            return "moe_gate_proj"
        if expert_type == "attention":
            return "attn_proj"
        return "linear"

    # Fallback: strip the aten:: prefix.
    return aten_op_name.split("::")[-1] if "::" in aten_op_name else aten_op_name


def _detect_num_layers(classified_kernels: List[Dict]) -> int:
    """Detect number of transformer layers from shared expert entry frequencies.

    Uses GCD of all shared expert entry frequencies as the proxy for num_layers.
    Returns 1 if no shared expert entries are found (safe default).
    """
    shared_freqs = [
        e["statistics"]["frequency"]
        for e in classified_kernels
        if e.get("expert_type") == "shared_expert"
        and e.get("statistics", {}).get("frequency", 0) > 0
    ]
    if not shared_freqs:
        return 1
    result = shared_freqs[0]
    for f in shared_freqs[1:]:
        result = math.gcd(result, f)
    return max(1, result)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_shape(shape) -> List[int]:
    """Normalize possibly nested shape containers into a flat int list."""
    if shape is None:
        return []
    if isinstance(shape, (int, float)):
        return [int(shape)]

    normalized: List[int] = []
    stack = [shape]
    while stack:
        current = stack.pop()
        if isinstance(current, (list, tuple)):
            for item in reversed(current):
                stack.append(item)
            continue
        if isinstance(current, bool):
            normalized.append(int(current))
            continue
        if isinstance(current, str):
            text = current.strip()
            if text:
                normalized.append(int(float(text)))
            continue
        if isinstance(current, Real):
            normalized.append(int(current))
            continue
        raise TypeError(f"Unsupported shape element type: {type(current)}")

    return normalized

def _product(shape) -> int:
    """Return the integer product of a shape list; 0 for empty."""
    dims = _normalize_shape(shape)
    if not dims:
        return 0
    result = 1
    for d in dims:
        result *= int(d)
    return result


def _is_expanding(aten_op_name: str, weight_shape: List) -> bool:
    """Return True if this GEMM op expands the feature dimension (N > K).

    aten::linear stores weight as (N, K) → expanding iff weight_shape[0] > weight_shape[1].
    aten::mm / aten::addmm store the second matrix as (K, N) → expanding iff weight_shape[1] > weight_shape[0].
    """
    if not weight_shape or len(weight_shape) < 2:
        return False
    if aten_op_name == "aten::linear":
        return int(weight_shape[0]) > int(weight_shape[1])
    # mm, addmm, matmul, _scaled_mm: second matrix is (K, N).
    return int(weight_shape[-1]) > int(weight_shape[-2])


def _ops_per_layer(freq: int, num_layers: int) -> int:
    """Return ops per layer; 0 if freq is not evenly divisible by num_layers."""
    if num_layers <= 0 or freq <= 0:
        return 0
    if freq % num_layers == 0:
        return freq // num_layers
    return 0


def _make_record(
    layer_id: int,
    op_name: str,
    hbm_fields: Dict,
    cta_count: int,
    latency_us: float,
    is_shared: bool,
    expert_type: str,
    l2_bytes: float = 0.0,
) -> Dict:
    """Build a single op_profile record dict."""
    shared_expert_bytes = hbm_fields["hbm_bytes"] if is_shared else 0.0
    return {
        "layer_id": layer_id,
        "op_name": op_name,
        "flops": hbm_fields["flops"],
        "hbm_bytes": hbm_fields["hbm_bytes"],
        "weight_bytes": hbm_fields["weight_bytes"],
        "activation_bytes": hbm_fields["activation_bytes"],
        "kv_bytes": hbm_fields["kv_bytes"],
        "shared_expert_bytes": shared_expert_bytes,
        "l2_bytes": l2_bytes,
        "cta_count": cta_count,
        "latency_us": latency_us,
        "is_shared_expert": is_shared,
        "expert_type": expert_type,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_op_profile(
    classified_kernels: List[Dict],
    num_layers: int,
    precision: str = "bfloat16",
    ncu_results: Optional[Dict[str, Dict]] = None,
    output_path: Optional[Path] = None,
) -> List[Dict]:
    """Generate per-layer per-op records for all kernels.

    Each kernel DB entry that fires an integer multiple of num_layers times
    is expanded into num_layers records (layer_id = 0 … num_layers-1).
    Entries with frequencies that are not divisible by num_layers, or that
    belong to the routed_expert type, get a single record with layer_id = -1.

    Shared expert kernels are flagged with is_shared_expert=True and
    shared_expert_bytes populated.

    Args:
        classified_kernels: Entries annotated with expert_type from
            classify_kernel_entries().
        num_layers: Number of transformer layers.
        precision:  Data type precision string (e.g., "bfloat16").
        ncu_results: Optional kernel_id -> NCU result dict; overrides
            shape-estimated hbm_bytes with actual dram_read + dram_write.
        output_path: If provided, writes op_profile.json to this path.

    Returns:
        List of record dicts, sorted by layer_id ASC (layer_id=-1 at end),
        then by op_name.
    """
    if not classified_kernels:
        records: List[Dict] = []
        if output_path is not None:
            Path(output_path).write_text(json.dumps(records, indent=2))
        return records

    dtype_bytes = _dtype_bytes(precision)
    ncu_results = ncu_results or {}
    num_layers = max(1, int(num_layers))

    # Track position per (weight_shape_tuple, expert_type) to distinguish
    # gate_proj (pos=0) from up_proj (pos=1) for same-shape shared expert GEMMs.
    shape_position: Dict[Tuple, int] = {}
    records = []

    for entry in classified_kernels:
        expert_type = entry.get("expert_type", "other")
        projection_type = entry.get("projection_type")  # from classify_kernel_entries
        aten_op = entry.get("aten_op", {})
        aten_op_name = aten_op.get("name", "")
        input_dims = aten_op.get("input_dims", [])
        kernel = entry.get("kernel", {})
        stats = entry.get("statistics", {})
        entry_id = entry.get("id", "")

        freq = int(stats.get("frequency", 1))
        latency_us = float(stats.get("avg_duration_us", 0.0) or 0.0)

        # CTA count from kernel grid dims.
        grid = kernel.get("grid") or [1, 1, 1]
        cta_count = 1
        for dim in grid:
            cta_count *= int(dim) if dim else 1

        # Shape-based HBM field estimates.
        hbm_fields = _compute_hbm_fields(aten_op_name, input_dims, dtype_bytes)

        # NCU override: replace hbm_bytes with actual DRAM counters if available.
        if entry_id in ncu_results:
            ncu = ncu_results[entry_id]
            ncu_hbm = float(
                (ncu.get("hbm_read_bytes") or 0) + (ncu.get("hbm_write_bytes") or 0)
            )
            if ncu_hbm > 0:
                hbm_fields = dict(hbm_fields)
                hbm_fields["hbm_bytes"] = ncu_hbm

        is_shared = expert_type == "shared_expert"

        # Determine layer expansion: routed experts and non-divisible frequencies → layer_id=-1.
        ops_count = _ops_per_layer(freq, num_layers) if expert_type != "routed_expert" else 0
        is_layer_local = ops_count > 0

        # Position tracking: distinguishes gate_proj (pos=0) from up_proj (pos=1).
        w_shape = tuple(_normalize_shape(input_dims[1])) if len(input_dims) > 1 and input_dims[1] else ()
        shape_key = (w_shape, expert_type)
        pos = shape_position.get(shape_key, 0)
        shape_position[shape_key] = pos + (ops_count if is_layer_local else 1)

        if is_layer_local:
            for layer_id in range(num_layers):
                for sub_pos in range(ops_count):
                    op_name_n = _infer_op_name(aten_op_name, expert_type, input_dims, pos + sub_pos, projection_type)
                    records.append(_make_record(
                        layer_id=layer_id,
                        op_name=op_name_n,
                        hbm_fields=hbm_fields,
                        cta_count=cta_count,
                        latency_us=latency_us,
                        is_shared=is_shared,
                        expert_type=expert_type,
                    ))
        else:
            op_name = _infer_op_name(aten_op_name, expert_type, input_dims, pos, projection_type)
            records.append(_make_record(
                layer_id=-1,
                op_name=op_name,
                hbm_fields=hbm_fields,
                cta_count=cta_count,
                latency_us=latency_us,
                is_shared=is_shared,
                expert_type=expert_type,
            ))

    # Sort: layer_id ASC with -1 at the end, then op_name for stability.
    records.sort(key=lambda r: (
        r["layer_id"] if r["layer_id"] >= 0 else 10 ** 9,
        r["op_name"],
    ))

    if output_path is not None:
        Path(output_path).write_text(json.dumps(records, indent=2))

    return records


# ---------------------------------------------------------------------------
# CUPTI-based op_profile builder
# ---------------------------------------------------------------------------

def _infer_op_name_from_module(
    aten_op: str,
    expert_type: str,
    projection_type: Optional[str],
) -> str:
    """Map profiler event fields to a human-readable op_name.

    Similar to ``_infer_op_name()`` but uses the projection_type from the
    module classifier instead of shape heuristics.
    """
    # Non-GEMM special cases by aten op name.
    suffix = aten_op.split("::")[-1] if "::" in aten_op else aten_op
    if suffix in ("rms_norm",):
        return "rmsnorm"
    if suffix in ("native_layer_norm", "layer_norm"):
        return "layernorm"
    if suffix in ("softmax", "_softmax"):
        return "softmax"
    if suffix in ("silu", "gelu", "relu"):
        return "activation"
    if suffix in ("mul", "add", "div", "sub"):
        return "elementwise"
    if suffix == "bmm":
        return "bmm"

    # GEMM ops with known expert_type + projection_type
    if aten_op in _GEMM_OPS or suffix == "linear":
        if expert_type == "shared_expert":
            if projection_type == "gate_proj":
                return "shared_expert_gate_proj"
            if projection_type == "up_proj":
                return "shared_expert_up_proj"
            if projection_type == "down_proj":
                return "shared_expert_down_proj"
            return "shared_expert_gate"
        if expert_type == "routed_expert":
            if projection_type == "gate_proj":
                return "routed_expert_gate_proj"
            if projection_type == "up_proj":
                return "routed_expert_up_proj"
            if projection_type == "down_proj":
                return "routed_expert_down_proj"
            return "routed_expert_proj"
        if expert_type == "gate":
            return "moe_gate_proj"
        if expert_type == "attention":
            return "attn_proj"
        return "linear"

    # Fallback: strip aten:: prefix.
    return suffix if suffix else aten_op


def generate_op_profile_from_cupti(
    cupti_events: List[Dict],
    output_path: Optional[Path] = None,
) -> List[Dict]:
    """Build op_profile records from classified CUPTI profiler events.

    Each event dict is expected to contain the fields produced by
    :func:`cupti_profiler.profile_single_prompt`:
    ``aten_op``, ``expert_type``, ``layer_id``, ``projection_type``,
    ``hbm_bytes``, ``l2_bytes``, ``flops``, ``weight_bytes``,
    ``activation_bytes``, ``cuda_time_us``, ``num_kernels``.

    Args:
        cupti_events: Classified event dicts from the CUPTI profiler.
        output_path: If provided, writes ``op_profile.json`` to this path.

    Returns:
        List of record dicts in the standard op_profile schema, sorted by
        ``layer_id`` ASC (``-1`` at end), then ``op_name``.
    """
    records: List[Dict] = []

    for evt in cupti_events:
        expert_type = evt.get("expert_type", "other")
        projection_type = evt.get("projection_type")
        aten_op = evt.get("aten_op", "")
        layer_id = evt.get("layer_id", -1)

        # Skip events with no CUDA activity and no meaningful classification
        if evt.get("num_kernels", 0) == 0 and evt.get("cuda_time_us", 0) == 0:
            continue

        op_name = _infer_op_name_from_module(aten_op, expert_type, projection_type)
        is_shared = expert_type == "shared_expert"

        hbm_fields = {
            "flops": evt.get("flops", 0),
            "hbm_bytes": evt.get("hbm_bytes", 0.0),
            "weight_bytes": evt.get("weight_bytes", 0.0),
            "activation_bytes": evt.get("activation_bytes", 0.0),
            "kv_bytes": 0.0,
        }

        record = _make_record(
            layer_id=layer_id,
            op_name=op_name,
            hbm_fields=hbm_fields,
            cta_count=evt.get("num_kernels", 0),
            latency_us=evt.get("cuda_time_us", 0.0),
            is_shared=is_shared,
            expert_type=expert_type,
            l2_bytes=evt.get("l2_bytes", 0.0),
        )
        record["hbm_source"] = evt.get("hbm_source", "unknown")
        records.append(record)

    # Sort: layer_id ASC with -1 at end, then op_name.
    records.sort(key=lambda r: (
        r["layer_id"] if r["layer_id"] >= 0 else 10 ** 9,
        r["op_name"],
    ))

    if output_path is not None:
        Path(output_path).write_text(json.dumps(records, indent=2))

    return records
