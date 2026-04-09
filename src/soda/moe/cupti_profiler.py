"""Per-operator HBM and L2 traffic measurement via PyTorch Profiler CUPTI.

Instruments ``model.generate()`` under ``torch.profiler`` with hardware
counters enabled through ``_ExperimentalConfig``.  Collects:

- **HBM bytes**: ``dram__bytes_read.sum + dram__bytes_write.sum``
- **L2 bytes**: ``lts__t_bytes.sum``

per CPU-side ATen operator by walking child CUDA kernel events.

HBM and L2 bytes are strictly from CUPTI counters (no shape-based fallback).
FLOPs and weight/activation bytes are derived analytically from input shapes
since CUPTI does not report FLOPs directly.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import torch

from soda.moe.module_classifier import classify_profiler_event
from soda.moe.op_profile import _compute_hbm_fields, _dtype_bytes

# CUPTI metric names requested from the profiler via _ExperimentalConfig.
# Both pre-Blackwell and Blackwell names are requested — on each generation
# the unsupported names are silently ignored by the profiler.
#   Pre-Blackwell (Ampere/Hopper):  dram__bytes_read.sum, dram__bytes_write.sum
#   Blackwell (CC 12.x):            dram__bytes_op_read.sum, dram__bytes_op_write.sum
_CUPTI_METRICS = [
    "dram__bytes_read.sum",        # pre-Blackwell (Ampere/Hopper)
    "dram__bytes_write.sum",       # pre-Blackwell (Ampere/Hopper)
    "dram__bytes_op_read.sum",     # Blackwell CC 12.x+
    "dram__bytes_op_write.sum",    # Blackwell CC 12.x+
    "lts__t_bytes.sum",
]


def _is_cuda_device_type(device_type: Any) -> bool:
    """Return True if profiler device_type denotes CUDA across API variants."""
    if device_type is None:
        return False

    cuda_enum = getattr(torch.autograd.DeviceType, "CUDA", None)
    if cuda_enum is not None and device_type == cuda_enum:
        return True

    try:
        if cuda_enum is not None and isinstance(device_type, int) and int(cuda_enum) == device_type:
            return True
    except Exception:
        pass

    name = getattr(device_type, "name", None)
    if isinstance(name, str) and name.upper() == "CUDA":
        return True

    text = str(device_type).upper()
    return "CUDA" in text


def _is_cpu_device_type(device_type: Any) -> bool:
    """Return True if profiler device_type denotes CPU across API variants."""
    if device_type is None:
        return False

    cpu_enum = getattr(torch.autograd.DeviceType, "CPU", None)
    if cpu_enum is not None and device_type == cpu_enum:
        return True

    try:
        if cpu_enum is not None and isinstance(device_type, int) and int(cpu_enum) == device_type:
            return True
    except Exception:
        pass

    name = getattr(device_type, "name", None)
    if isinstance(name, str) and name.upper() == "CPU":
        return True

    text = str(device_type).upper()
    return "CPU" in text


def _coerce_metric_dict(metrics: Any) -> Dict[str, float]:
    """Best-effort conversion of metric containers to a plain dict."""
    if metrics is None:
        return {}

    # Native dict / dict-like
    if isinstance(metrics, dict):
        out = {}
        for k, v in metrics.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        return out

    if hasattr(metrics, "items"):
        out = {}
        try:
            for k, v in metrics.items():
                try:
                    out[str(k)] = float(v)
                except Exception:
                    continue
            if out:
                return out
        except Exception:
            pass

    # Sequence of (k, v) pairs
    if isinstance(metrics, (list, tuple)):
        out = {}
        for item in metrics:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            k, v = item
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        return out

    # Generic object with attributes
    out = {}
    for attr in dir(metrics):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(metrics, attr)
        except Exception:
            continue
        if callable(val):
            continue
        try:
            out[str(attr)] = float(val)
        except Exception:
            continue
    return out


def _find_metric_value(metrics: Dict[str, float], kind: str) -> float:
    """Find metric value by exact and fuzzy key matching."""
    if not metrics:
        return 0.0

    # Exact preferred keys — Blackwell names first (CC 12.x+), then pre-Blackwell.
    exact = {
        "hbm_read":  ["dram__bytes_op_read.sum",  "dram__bytes_read.sum",  "dram__bytes_read"],
        "hbm_write": ["dram__bytes_op_write.sum", "dram__bytes_write.sum", "dram__bytes_write"],
        "l2_bytes":  ["lts__t_bytes.sum", "lts__t_bytes"],
    }
    for k in exact.get(kind, []):
        if k in metrics:
            return float(metrics[k])

    # Normalized key fallback
    for k, v in metrics.items():
        lk = str(k).lower()
        if kind == "hbm_read" and ("dram" in lk and "read" in lk and "bytes" in lk):
            return float(v)
        if kind == "hbm_write" and ("dram" in lk and "write" in lk and "bytes" in lk):
            return float(v)
        if kind == "l2_bytes" and (("lts" in lk and "bytes" in lk) or ("l2" in lk and "bytes" in lk)):
            return float(v)

    return 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _try_build_experimental_config():
    """Return an ``_ExperimentalConfig`` with CUPTI metrics, or None."""
    try:
        from torch.profiler import _ExperimentalConfig
        return _ExperimentalConfig(
            profiler_metrics=_CUPTI_METRICS,
            profiler_measure_per_kernel=True,
        )
    except Exception as exc:
        warnings.warn(
            f"CUPTI _ExperimentalConfig unavailable ({exc}). "
            "MoE CUPTI profiling requires these metrics and will abort.",
            stacklevel=2,
        )
        return None


def _extract_cupti_from_kernel(kernel_evt) -> Dict[str, float]:
    """Extract CUPTI counter values from a CUDA kernel FunctionEvent.

    Tries multiple access patterns since the PyTorch profiler API is
    experimental and attribute names vary across versions.

    Returns dict with ``hbm_read``, ``hbm_write``, ``l2_bytes`` (all floats,
    0.0 if unavailable).
    """
    result = {"hbm_read": 0.0, "hbm_write": 0.0, "l2_bytes": 0.0}

    # Pattern 1/2+: metric containers on known attributes.
    for attr_name in ("cuda_metrics", "metrics", "extra_fields"):
        container = getattr(kernel_evt, attr_name, None)
        metrics = _coerce_metric_dict(container)
        if metrics:
            result["hbm_read"] = _find_metric_value(metrics, "hbm_read")
            result["hbm_write"] = _find_metric_value(metrics, "hbm_write")
            result["l2_bytes"] = _find_metric_value(metrics, "l2_bytes")
            if (result["hbm_read"] > 0) or (result["hbm_write"] > 0) or (result["l2_bytes"] > 0):
                return result

    # Pattern 3: direct attributes
    for attr, key in [
        ("dram__bytes_read.sum", "hbm_read"),
        ("dram__bytes_write.sum", "hbm_write"),
        ("lts__t_bytes.sum", "l2_bytes"),
    ]:
        val = getattr(kernel_evt, attr.replace(".", "_"), None)
        if val is not None:
            result[key] = float(val)

    if (result["hbm_read"] > 0) or (result["hbm_write"] > 0) or (result["l2_bytes"] > 0):
        return result

    # Pattern 4: scan all numeric attributes for likely metric keys.
    # Suppress FutureWarnings from deprecated cuda_time* / self_cuda_* aliases.
    import warnings as _w
    attrs = {}
    for attr in dir(kernel_evt):
        if attr.startswith("_"):
            continue
        try:
            with _w.catch_warnings():
                _w.simplefilter("ignore", FutureWarning)
                val = getattr(kernel_evt, attr)
        except Exception:
            continue
        if callable(val):
            continue
        try:
            attrs[attr] = float(val)
        except Exception:
            continue

    if attrs:
        result["hbm_read"] = _find_metric_value(attrs, "hbm_read")
        result["hbm_write"] = _find_metric_value(attrs, "hbm_write")
        result["l2_bytes"] = _find_metric_value(attrs, "l2_bytes")

    return result


def _shapes_to_lists(shapes) -> List[List[int]]:
    """Convert profiler input_shapes (list of tuples/lists) to List[List[int]]."""
    if not shapes:
        return []
    out = []
    for s in shapes:
        if isinstance(s, (list, tuple)):
            out.append([int(d) for d in s])
        else:
            out.append([])
    return out


# Prefix used by our explicit record_function hooks (PyTorch 2.9+).
# In PyTorch ≤2.8, with_modules=True emitted "nn.Module: ClassName.path"
# wrapper events.  In PyTorch 2.9 those events disappeared from prof.events().
# We work around this by monkey-patching relevant module forward() methods
# to emit record_function("soda_module:<path>") events, which DO appear as
# cpu_parent of aten ops in all PyTorch versions.
_SODA_MODULE_PREFIX = "soda_module:"

# Legacy prefix kept for backward-compat when running on older PyTorch.
_NN_MODULE_PREFIX = "nn.Module: "

import re as _re

# Module name patterns worth instrumenting for MoE classification.
_INSTRUMENT_PATTERNS = [
    _re.compile(r"layers\.\d+\..*shared_expert[s]?\.(gate_proj|up_proj|down_proj)"),
    _re.compile(r"layers\.\d+\..*shared_expert_gate$"),
    _re.compile(r"layers\.\d+\..*experts\.\d+\.(gate_proj|up_proj|down_proj|w1|w2|w3)"),
    _re.compile(r"layers\.\d+\.(?:mlp|block_sparse_moe)\.gate$"),
    _re.compile(r"layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"),
]


def _should_instrument(module_name: str) -> bool:
    return any(p.search(module_name) for p in _INSTRUMENT_PATTERNS)


def _install_profiler_module_hooks(model) -> list:
    """Monkey-patch forward() on MoE-relevant modules to emit record_function events.

    In PyTorch 2.9+, ``with_modules=True`` no longer produces ``nn.Module:``
    wrapper events in ``prof.events()``.  This function instruments each
    relevant module's ``forward`` with
    ``torch.profiler.record_function("soda_module:<name>")``, which creates
    profiler events that DO appear as ``cpu_parent`` of the aten ops inside.

    Returns a list of ``(module, original_forward)`` tuples for cleanup.
    """
    patched = []
    for name, module in model.named_modules():
        if not _should_instrument(name):
            continue
        original_forward = module.forward
        soda_path = f"{_SODA_MODULE_PREFIX}{name}"

        def _make_wrapped(orig, path):
            def _wrapped(*args, **kwargs):
                with torch.profiler.record_function(path):
                    return orig(*args, **kwargs)
            return _wrapped

        module.forward = _make_wrapped(original_forward, soda_path)
        patched.append((module, original_forward))

    return patched


def _uninstall_profiler_module_hooks(patched: list) -> None:
    """Restore original forward() methods after profiling."""
    for module, original_forward in patched:
        module.forward = original_forward


def _find_module_path(evt) -> str:
    """Walk the cpu_parent chain to find the nearest soda_module: ancestor.

    Works for both PyTorch 2.9+ (where we install explicit record_function
    hooks that emit ``soda_module:<path>`` events) and older PyTorch (where
    ``with_modules=True`` emitted ``nn.Module: ClassName.path`` events).

    Returns the dotted module path (e.g.
    ``layers.5.mlp.shared_expert.gate_proj``) or ``""`` if not found.
    """
    parent = getattr(evt, "cpu_parent", None)
    while parent is not None:
        name = getattr(parent, "name", "") or ""
        if name.startswith(_SODA_MODULE_PREFIX):
            return name[len(_SODA_MODULE_PREFIX):]
        if name.startswith(_NN_MODULE_PREFIX):
            # Legacy PyTorch ≤2.8 fallback.
            # Format: "nn.Module: Qwen2MoeModel.layers.5.mlp.shared_expert.gate_proj"
            raw = name[len(_NN_MODULE_PREFIX):]
            dot = raw.find(".")
            if dot >= 0:
                return "model" + raw[dot:]
            return raw
        parent = getattr(parent, "cpu_parent", None)
    return ""


def _iter_cuda_kernel_events(evt):
    """Yield CUDA kernel-like child events for a CPU ATen event.

    PyTorch profiler trees can place CUDA kernels at different depths, e.g.
    ``aten::op -> cudaLaunchKernel (CPU) -> kernel (CUDA)``. This helper
    walks descendants (not just direct children) and also falls back to
    ``evt.kernels`` when available.
    """
    seen = set()

    # Depth-first walk through cpu_children descendants.
    stack = list(getattr(evt, "cpu_children", None) or [])
    while stack:
        child = stack.pop()
        obj_id = id(child)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        child_device = getattr(child, "device_type", None)
        if _is_cuda_device_type(child_device):
            yield child

        grand_children = getattr(child, "cpu_children", None) or []
        if grand_children:
            stack.extend(grand_children)

    # Some profiler versions expose CUDA kernels directly via evt.kernels.
    kernels_obj = getattr(evt, "kernels", None)
    if callable(kernels_obj):
        try:
            kernels = kernels_obj()
        except Exception:
            kernels = []
    else:
        kernels = kernels_obj or []

    for k in kernels:
        obj_id = id(k)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        yield k


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover_available_cupti_metrics() -> Dict[str, Any]:
    """Probe torch.profiler CUPTI integration to find valid metric names.

    Run on a GPU compute node to diagnose why ``hbm_bytes``/``l2_bytes`` are
    zero.  The function:

    1. Checks ``_ExperimentalConfig`` availability.
    2. Runs ``torch.mm(128×128 float16)`` under torch.profiler with several
       candidate metric-name sets (current NCU names, kineto names, etc.).
    3. For each run, scans Python ``FunctionEvent`` attributes on CUDA kernel
       events and parses the Chrome-trace JSON GPU event "args".
    4. Returns a structured report identifying which metric names and attribute
       access paths yield nonzero data.

    Returns:
        Dict with keys:
            - ``experimental_config_available`` (bool)
            - ``cuda_available`` (bool)
            - ``pytorch_version`` (str)
            - ``baseline_cuda_event_attrs`` — all attrs on CUDA events with no
              profiler_metrics set
            - ``chrome_trace_gpu_args`` — arg keys/values in the Chrome trace
              GPU events from the baseline run
            - ``candidate_sets`` — dict[name -> results] for each metric set
              tried; each result has ``nonzero_cuda_attrs``, ``cupti_like_nonzero``,
              ``chrome_gpu_args``
            - ``recommended_metrics`` — metric name list that worked (or [])
            - ``recommended_access_path`` — str describing where the data lives
    """
    import json
    import tempfile
    from pathlib import Path

    report: Dict[str, Any] = {
        "experimental_config_available": False,
        "cuda_available": torch.cuda.is_available(),
        "pytorch_version": torch.__version__,
        "baseline_cuda_event_attrs": {},
        "chrome_trace_gpu_args": {},
        "candidate_sets": {},
        "recommended_metrics": [],
        "recommended_access_path": "none",
    }

    try:
        from torch.profiler import _ExperimentalConfig, ProfilerActivity
        report["experimental_config_available"] = True
    except ImportError as exc:
        report["error"] = f"_ExperimentalConfig not importable: {exc}"
        return report

    if not torch.cuda.is_available():
        report["error"] = "No CUDA device available — run on a compute node"
        return report

    _a = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    _b = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()

    def _run_prof(metrics: List[str]) -> Optional[Any]:
        kw: Dict[str, Any] = dict(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        )
        try:
            if metrics:
                kw["experimental_config"] = _ExperimentalConfig(
                    profiler_metrics=metrics,
                    profiler_measure_per_kernel=True,
                )
        except Exception:
            pass
        try:
            with torch.profiler.profile(**kw) as p:
                torch.mm(_a, _b)
                torch.cuda.synchronize()
            return p
        except Exception:
            return None

    def _scan_cuda_events(prof: Any) -> Dict[str, Any]:
        nonzero: Dict[str, float] = {}
        attr_sample: Dict[str, str] = {}
        for evt in prof.events():
            if not _is_cuda_device_type(getattr(evt, "device_type", None)):
                continue
            for attr in dir(evt):
                if attr.startswith("_"):
                    continue
                try:
                    val = getattr(evt, attr)
                except Exception:
                    continue
                if callable(val):
                    continue
                attr_sample[attr] = repr(val)[:60]
                try:
                    fval = float(val)
                    if fval > 0:
                        nonzero[attr] = fval
                except Exception:
                    pass
        return {"nonzero": nonzero, "attr_sample": attr_sample}

    def _scan_chrome_trace(prof: Any) -> Dict[str, Any]:
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".json", delete=False, mode="w"
            ) as tf:
                tmp = tf.name
            prof.export_chrome_trace(tmp)
            with open(tmp) as f:
                data = json.load(f)
            Path(tmp).unlink(missing_ok=True)
            gpu_evts = [
                e for e in data.get("traceEvents", [])
                if e.get("ph") == "X"
                and e.get("cat") in ("gpu_user", "kernel", "gpu", "cuda")
                and isinstance(e.get("args"), dict)
            ]
            merged: Dict[str, Any] = {}
            for e in gpu_evts[:20]:
                for k, v in e.get("args", {}).items():
                    if k not in merged:
                        merged[k] = v
            return {
                "gpu_event_count": len(gpu_evts),
                "all_event_cats": list(
                    {e.get("cat") for e in data.get("traceEvents", []) if e.get("ph") == "X"}
                ),
                "gpu_args_keys": merged,
            }
        except Exception as exc:
            return {"error": str(exc)}

    # Baseline: no profiler_metrics — see natural CUDA event attrs
    base_prof = _run_prof([])
    if base_prof is not None:
        base_scan = _scan_cuda_events(base_prof)
        report["baseline_cuda_event_attrs"] = base_scan
        report["chrome_trace_gpu_args"] = _scan_chrome_trace(base_prof)

    # Candidate metric-name sets to try
    _CANDIDATES: Dict[str, List[str]] = {
        "ncu_chip": list(_CUPTI_METRICS),
        "kineto_dram": ["dram_read_bytes", "dram_write_bytes", "l2_read_bytes", "l2_write_bytes"],
        "dram_sectors": ["dram__sectors_read.sum", "dram__sectors_write.sum"],
        "l2_sector": ["lts__t_sectors_op_read.sum", "lts__t_sectors_op_write.sum"],
        "sm_bytes": ["sm__bytes_read.sum", "sm__bytes_write.sum"],
        "kineto_gpu": [
            "kineto__cuda_core_fp16", "kineto__cuda_core_bf16", "kineto__tensor_core_insts"
        ],
    }

    best_metrics: Optional[List[str]] = None
    best_path = "none"

    for cname, mlist in _CANDIDATES.items():
        p = _run_prof(mlist)
        if p is None:
            report["candidate_sets"][cname] = {"error": "profiler run failed"}
            continue

        ev_info = _scan_cuda_events(p)
        cr_info = _scan_chrome_trace(p)

        nonzero = ev_info["nonzero"]
        _mem_kws = ("dram", "hbm", "mem", "l2", "lts", "sector", "byte")
        cupti_like = {k: v for k, v in nonzero.items()
                      if any(kw in k.lower() for kw in _mem_kws)}
        chrome_args = cr_info.get("gpu_args_keys", {})
        chrome_metric = {k: v for k, v in chrome_args.items()
                         if any(kw in k.lower() for kw in _mem_kws)}

        report["candidate_sets"][cname] = {
            "metric_names": mlist,
            "nonzero_cuda_attrs": nonzero,
            "cupti_like_nonzero": cupti_like,
            "chrome_gpu_args": chrome_args,
            "chrome_metric_keys": chrome_metric,
        }

        if cupti_like and best_metrics is None:
            best_metrics = mlist
            best_path = f"event_attr: {sorted(cupti_like.keys())}"
        if chrome_metric and best_metrics is None:
            best_metrics = mlist
            best_path = f"chrome_trace_args: {sorted(chrome_metric.keys())}"

    report["recommended_metrics"] = best_metrics or []
    report["recommended_access_path"] = best_path
    return report


def profile_single_prompt(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    max_seq_len: int = 4096,
    max_new_tokens: int = 1,
    device: Optional[torch.device] = None,
    precision: str = "bfloat16",
) -> List[Dict]:
    """Profile a single prompt and return per-operator memory measurements.

    The prompt is tokenized at its **natural length** (no padding).  Each
    curated prompt from ``soda.moe.prompts`` has a deliberate token count;
    padding would distort the operator shapes that CUPTI measures.
    ``max_seq_len`` serves only as a safety truncation cap.

    Args:
        model: A loaded HuggingFace model (``model.eval()`` expected).
        tokenizer: Corresponding tokenizer.
        prompt_text: The text prompt to profile.
        max_seq_len: Safety truncation limit.  Prompts longer than this
            are truncated; shorter prompts keep their natural length.
        max_new_tokens: Tokens to generate beyond the prompt.
        device: Target device.  Defaults to model's device.
        precision: Precision string for shape-based fallback (e.g. "bfloat16").

    Returns:
        List of dicts, one per CPU ATen operator event, each containing:
        ``aten_op``, ``module_path``, ``input_shapes``, ``cuda_time_us``,
        ``hbm_bytes``, ``l2_bytes``, ``hbm_read_bytes``, ``hbm_write_bytes``,
        ``flops``, ``expert_type``, ``layer_id``, ``projection_type``,
        ``cupti_available`` (bool).
    """
    from torch.profiler import ProfilerActivity

    if device is None:
        device = next(model.parameters()).device

    # Tokenize at natural length — no padding.  Truncate only as a safety cap.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        max_length=max_seq_len,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Build profiler config
    exp_config = _try_build_experimental_config()
    cupti_available = exp_config is not None

    if not cupti_available:
        warnings.warn(
            "CUPTI _ExperimentalConfig unavailable — profiling without hardware "
            "memory counters. hbm_bytes and l2_bytes will be zero. "
            "Run discover_available_cupti_metrics() to diagnose.",
            UserWarning,
            stacklevel=2,
        )

    profiler_kwargs = dict(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_modules=True,  # kept for PyTorch ≤2.8 compat; no-op in 2.9+
    )
    if exp_config is not None:
        profiler_kwargs["experimental_config"] = exp_config

    # Install explicit record_function hooks on MoE-relevant modules.
    # Needed in PyTorch 2.9+ where with_modules=True no longer emits
    # nn.Module: wrapper events in prof.events().
    _patched = _install_profiler_module_hooks(model)

    # Profile
    try:
        with torch.no_grad():
            with torch.profiler.profile(**profiler_kwargs) as prof:
                model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
    finally:
        _uninstall_profiler_module_hooks(_patched)

    # Collect results — only aten:: operators, not profiler infrastructure
    dtype_bytes = _dtype_bytes(precision)
    records: List[Dict] = []

    for evt in prof.events():
        # Only process CPU-side aten:: operator events.
        # with_modules=True also emits "nn.Module:" wrapper events — skip those
        # as well as profiler infrastructure ("Activity Buffer Request", etc.).
        if not _is_cpu_device_type(getattr(evt, "device_type", None)):
            continue

        aten_op = evt.name or ""
        if not aten_op.startswith("aten::"):
            continue

        input_shapes = _shapes_to_lists(getattr(evt, "input_shapes", None))
        # device_time_total is the PyTorch ≥2.9 name; cuda_time_total is the
        # legacy alias (still works but emits FutureWarning in ≥2.9).
        _cuda_t = getattr(evt, "device_time_total", None)
        if _cuda_t is None:
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore", FutureWarning)
                _cuda_t = getattr(evt, "cuda_time_total", None)
        cuda_time_us = float(_cuda_t or 0.0)

        # Walk parent chain to find nn.Module: ancestor for classification
        module_path = _find_module_path(evt)

        # Accumulate CUPTI metrics from CUDA kernel events
        hbm_read = 0.0
        hbm_write = 0.0
        l2_bytes = 0.0
        num_kernels = 0

        for kernel_evt in _iter_cuda_kernel_events(evt):
            cupti = _extract_cupti_from_kernel(kernel_evt)
            hbm_read += cupti["hbm_read"]
            hbm_write += cupti["hbm_write"]
            l2_bytes += cupti["l2_bytes"]
            num_kernels += 1

        hbm_bytes = hbm_read + hbm_write

        # FLOPs from input shapes (analytic, not a fallback — CUPTI does not
        # report FLOPs directly).
        hbm_fields = _compute_hbm_fields(aten_op, input_shapes, dtype_bytes)
        flops = hbm_fields.get("flops", 0)
        weight_bytes = hbm_fields.get("weight_bytes", 0.0)
        activation_bytes = hbm_fields.get("activation_bytes", 0.0)

        # hbm_bytes is purely from CUPTI counters — no shape-based fallback.
        # If CUPTI returned zero, the output reflects that ground truth.

        # Classify via module hierarchy
        expert_type, layer_id, projection_type = classify_profiler_event(
            module_path, aten_op
        )

        if not cupti_available:
            hbm_source = "cupti_unavailable"
        elif hbm_bytes > 0 or l2_bytes > 0:
            hbm_source = "cupti"
        else:
            hbm_source = "cupti_zero"

        records.append({
            "aten_op": aten_op,
            "module_path": module_path,
            "input_shapes": input_shapes,
            "cuda_time_us": cuda_time_us,
            "hbm_bytes": hbm_bytes,
            "hbm_read_bytes": hbm_read,
            "hbm_write_bytes": hbm_write,
            "l2_bytes": l2_bytes,
            "flops": flops,
            "weight_bytes": weight_bytes,
            "activation_bytes": activation_bytes,
            "num_kernels": num_kernels,
            "expert_type": expert_type,
            "layer_id": layer_id,
            "projection_type": projection_type,
            "cupti_available": cupti_available,
            "hbm_source": hbm_source,
        })

    # Soft validation: warn if CUDA was active but no memory data was collected.
    # The caller (pipeline.py) may follow up with NCU profiling if needed.
    cuda_active = [r for r in records if (r.get("cuda_time_us", 0.0) or 0.0) > 0.0]
    if cuda_active:
        nonzero_mem = [
            r for r in cuda_active
            if ((r.get("hbm_bytes", 0.0) or 0.0) > 0.0) or ((r.get("l2_bytes", 0.0) or 0.0) > 0.0)
        ]
        if not nonzero_mem:
            warnings.warn(
                "PyTorch profiler recorded CUDA activity but CUPTI memory counters "
                "(dram__bytes_*, lts__t_bytes) are all zero. "
                "Records have hbm_source='cupti_zero'. "
                "Run discover_available_cupti_metrics() to find correct metric names.",
                UserWarning,
                stacklevel=2,
            )

    return records
