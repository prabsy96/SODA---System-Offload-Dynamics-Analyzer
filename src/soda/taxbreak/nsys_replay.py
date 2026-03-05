"""
nsys-based isolation replay for PyTorch kernels.

Generates a minimal Python replay script for a single ATen operation,
profiles it under ``nsys``, and extracts per-kernel launch tax from the
resulting SQLite trace.  This extends nsys profiling from baremetal-only
(cuBLAS GEMMs) to cover ALL kernel types dispatched through PyTorch.
"""

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from soda.microbench.baremetal.utils import (
    detect_vendor_library_events,
    extract_culib_markers_sql,
    extract_kernels_sql,
    extract_launches_sql,
    nsys_profile,
)


def _generate_replay_script(
    aten_op: Dict[str, Any],
    warmup: int,
    runs: int,
    script_path: Path,
    inferred_size: Optional[int] = None,
) -> None:
    """Write a standalone Python script that replays a single ATen operation.

    The generated script imports ``create_input_tensors`` and
    ``execute_operation`` from the existing PyTorch profiling module,
    recreates inputs from the serialised ATen-op metadata, and runs
    *warmup* + *runs* iterations with ``torch.cuda.synchronize()``
    after each invocation to serialise GPU work.

    Args:
        aten_op: ATen operation metadata dict.
        warmup: Number of warmup iterations (not measured by nsys).
        runs: Number of measured iterations.
        script_path: Where to write the script.
        inferred_size: When all input_dims are empty (shapes not captured in
            trace), use this as the first-dimension size for tensor inputs.
            Derived from ``kernel.grid[0] * kernel.block[0]`` so the replay
            launches the same kernel variant as the original profiled run.
    """
    aten_op_json = json.dumps(aten_op)

    # Build size-inference block: when input_dims are all empty we can't know
    # the original tensor shape, so we estimate it from the kernel's grid×block.
    # This helps PyTorch dispatch the same elementwise kernel variant (e.g.
    # unrolled vs vectorized) as the original run.
    size_hint_block = ""
    if inferred_size and inferred_size > 1:
        size_hint_block = f'''\

# Size inference: input shapes were not captured in the trace.
# Resize any 0/1-element tensor inputs to the inferred size ({inferred_size})
# derived from the kernel's grid×block so PyTorch dispatches the same
# kernel variant as the original profiled run.
_inferred_size = {inferred_size}
_resized = []
for _inp in inputs:
    import torch as _torch
    if isinstance(_inp, _torch.Tensor) and _inp.numel() <= 1:
        _dtype = _inp.dtype
        if _dtype in (_torch.int64, _torch.int32, _torch.int16, _torch.int8,
                      _torch.uint8, _torch.bool):
            _resized.append(_torch.randint(0, 256, (_inferred_size,),
                                           dtype=_dtype, device="cuda"))
        else:
            _resized.append(_torch.randn(_inferred_size, dtype=_dtype, device="cuda"))
    else:
        _resized.append(_inp)
inputs = _resized
'''

    script = f'''\
#!/usr/bin/env python3
"""Auto-generated replay script for nsys profiling."""
import json
import sys
import torch
import torch.cuda.nvtx as nvtx
from soda.microbench.framework.pytorch.profile import (
    create_input_tensors,
    execute_operation,
)

aten_op = json.loads({aten_op_json!r})
op_name = aten_op["name"]

# Create inputs from ATen op metadata
try:
    inputs = create_input_tensors(aten_op)
except Exception as exc:
    print(f"Failed to create inputs for {{op_name}}: {{exc}}", file=sys.stderr)
    sys.exit(1)
{size_hint_block}
# Warmup (not measured — NVTX markers included for consistent count)
with torch.no_grad():
    for _ in range({warmup}):
        nvtx.range_push("aten_dispatch")
        try:
            execute_operation(op_name, inputs)
        except Exception:
            pass
        nvtx.range_pop()
        torch.cuda.synchronize()

# Measured runs
with torch.no_grad():
    for _ in range({runs}):
        nvtx.range_push("aten_dispatch")
        try:
            execute_operation(op_name, inputs)
        except Exception:
            pass
        nvtx.range_pop()
        torch.cuda.synchronize()
'''

    script_path.parent.mkdir(parents=True, exist_ok=True)
    with open(script_path, "w") as f:
        f.write(script)


def _compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute summary statistics over a list of float values."""
    avg = sum(values) / len(values)
    min_val = min(values)
    max_val = max(values)
    if len(values) > 1:
        variance = sum((x - avg) ** 2 for x in values) / (len(values) - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
    return {
        "avg_us": round(avg, 4),
        "min_us": round(min_val, 4),
        "max_us": round(max_val, 4),
        "std_us": round(std, 4),
    }


def nsys_profile_pytorch_kernel(
    kernel_entry: Dict[str, Any],
    warmup: int = 50,
    runs: int = 200,
    extra_env: Optional[dict] = None,
) -> Optional[Dict[str, Any]]:
    """Profile a single kernel via PyTorch replay under nsys.

    Generates a temporary replay script for the ATen operation that
    dispatches *kernel_entry*, profiles it with ``nsys``, and computes
    per-kernel launch-tax statistics from the isolation trace.

    Args:
        kernel_entry: A single entry from ``kernel_database.json`` with
            ``kernel``, ``aten_op``, and ``id`` keys.
        warmup: Warmup iterations (not measured).
        runs: Measured iterations.

    Returns:
        Dict with ``launch_tax``, ``kernel_duration``, ``samples``,
        ``replay_method``, and optionally ``t_dispatch`` (T_dispatch
        measured via NVTX→cudaLaunch gap) keys, or ``None`` if
        profiling fails.
    """
    kernel_id = kernel_entry["id"]
    aten_op = kernel_entry["aten_op"]
    target_kernel_name = kernel_entry["kernel"]["name"]
    op_name = aten_op.get("name", "unknown")

    # Compute size inference hint: when all input_dims are empty (the trace
    # did not capture shapes), we estimate tensor size from the kernel's
    # grid×block.  This improves kernel-variant matching for elementwise ops.
    input_dims = aten_op.get("input_dims", [])
    all_dims_empty = all(not d for d in input_dims)
    inferred_size: Optional[int] = None
    if all_dims_empty:
        k_info = kernel_entry.get("kernel", {})
        grid = k_info.get("grid", [1, 1, 1])
        block = k_info.get("block", [1, 1, 1])
        total_threads = (grid[0] or 1) * (block[0] or 1)
        if total_threads > 1:
            inferred_size = total_threads

    # 1. Generate replay script in a temporary file
    tmp_dir = Path(tempfile.mkdtemp(prefix="soda_replay_"))
    try:
        script_path = tmp_dir / f"replay_{kernel_id}.py"
        _generate_replay_script(aten_op, warmup, runs, script_path,
                                inferred_size=inferred_size)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    # 2. Profile under nsys
    trace_name = f"pytorch_replay_{kernel_id}"
    success, trace_sql, msg = nsys_profile(
        trace_file_name=trace_name,
        args=["python", str(script_path)],
        timeout=300,
        extra_env=extra_env,
        trace_apis="cuda,osrt,nvtx,cublas,cudnn",
    )

    shutil.rmtree(tmp_dir, ignore_errors=True)

    if not success:
        print(f"nsys replay failed for {kernel_id} ({op_name}): {msg}")
        return None

    # 3. Extract kernels and launches from the SQLite trace
    kernels = extract_kernels_sql(trace_sql, filter_gemm_only=False)
    launches = extract_launches_sql(trace_sql)

    if not kernels:
        # Common cause: CPU-only op (e.g., aten::arange) that dispatches no CUDA kernels.
        print(f"  {kernel_id} ({op_name}): no GPU kernels in trace (CPU-only op or unsupported)")
        return None

    # 4. Filter to the target kernel by cleaned name
    kernel_variant_match = True
    matched = [k for k in kernels if k.name == target_kernel_name]
    if not matched:
        # Fallback 1: partial substring match (name may differ by template args).
        # Guard k.name with truthiness so empty-named anonymous kernels (k.name='')
        # never match — '' is trivially a substring of any string.
        matched = [
            k for k in kernels
            if k.name and (target_kernel_name in k.name or k.name in target_kernel_name)
        ]
    if not matched:
        # Fallback 2: accept the most-frequent kernel dispatched by this op.
        # This handles cases where the replay dispatches a different variant of
        # the same kernel family (e.g. unrolled_elementwise vs vectorized_elementwise)
        # because the original input shapes weren't captured in the trace.
        from collections import Counter
        name_counts = Counter(k.name for k in kernels)
        if name_counts:
            most_common_name, _ = name_counts.most_common(1)[0]
            matched = [k for k in kernels if k.name == most_common_name]
            kernel_variant_match = False
            print(
                f"  Warning: expected '{target_kernel_name}', got "
                f"'{most_common_name}' (variant mismatch — input dims not "
                f"captured in trace; used inferred_size={inferred_size})"
            )
    if not matched:
        print(
            f"Target kernel '{target_kernel_name}' not found in trace for "
            f"{kernel_id}. Found: {[k.name for k in kernels[:5]]}"
        )
        return None

    # 5. Match kernels to launches and compute per-invocation metrics
    launch_taxes: List[float] = []
    durations: List[float] = []
    dispatch_taxes: List[float] = []

    # Extract NVTX "aten_dispatch" ranges for T_dispatch measurement (NVTX start → cudaLaunchKernel)
    nvtx_ranges = extract_culib_markers_sql(trace_sql)
    aten_dispatch_ranges = sorted(
        [r for r in nvtx_ranges if r["name"] == "aten_dispatch"],
        key=lambda r: r["ts"],
    )

    for kernel in matched:
        corr_id = kernel.correlation
        if corr_id in launches:
            launch = launches[corr_id]
            tax = kernel.ts - launch["ts"]
            launch_taxes.append(tax)
            durations.append(kernel.dur)

            # CT: find the NVTX "aten_dispatch" range containing this launch
            launch_ts = launch["ts"]
            for nvtx_r in aten_dispatch_ranges:
                if nvtx_r["ts"] <= launch_ts <= nvtx_r["ts"] + nvtx_r["dur"]:
                    dispatch_taxes.append(launch_ts - nvtx_r["ts"])
                    break

    if not launch_taxes:
        print(f"No launch-kernel matches for {kernel_id} ({op_name})")
        return None

    # 6. Keep only the last ``runs`` samples (skip warmup)
    if len(launch_taxes) > runs:
        launch_taxes = launch_taxes[-runs:]
        durations = durations[-runs:]
    if len(dispatch_taxes) > runs:
        dispatch_taxes = dispatch_taxes[-runs:]

    actual_kernel_name = matched[0].name if matched else target_kernel_name

    # Runtime vendor-library detection: check if the isolation replay produced
    # cuBLAS/cuDNN API events (requires --trace=cublas,cudnn in nsys).
    i_lib_detected = detect_vendor_library_events(trace_sql) if trace_sql else False

    result = {
        "kernel_id": kernel_id,
        "kernel_name": actual_kernel_name,
        "aten_op": op_name,
        "launch_tax": _compute_stats(launch_taxes),
        "kernel_duration": _compute_stats(durations),
        "samples": len(launch_taxes),
        "replay_method": "pytorch",
        "kernel_variant_match": kernel_variant_match,
        "i_lib_detected": i_lib_detected,
    }

    # Add t_dispatch (T_dispatch = NVTX→cudaLaunchKernel gap) if NVTX matching succeeded
    if dispatch_taxes:
        result["t_dispatch"] = _compute_stats(dispatch_taxes)

    variant_tag = "" if kernel_variant_match else " [variant fallback]"
    ct_info = ""
    if "t_dispatch" in result:
        ct_info = f", T_dispatch avg={result['t_dispatch']['avg_us']:.2f} us"
    print(
        f"  {kernel_id} ({op_name} -> {actual_kernel_name}){variant_tag}: "
        f"launch_tax avg={result['launch_tax']['avg_us']:.2f} us, "
        f"dur avg={result['kernel_duration']['avg_us']:.2f} us{ct_info} "
        f"({result['samples']} samples)"
    )

    return result
