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
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _kernel_name_match_kind(target_name: str, actual_name: str) -> Optional[str]:
    """Classify how well a replayed kernel name matches the target name."""
    if actual_name == target_name:
        return "exact"
    if actual_name and (target_name in actual_name or actual_name in target_name):
        return "substring"
    return None


def _is_exact_signature_match(
    kernel: Any,
    target_grid: Tuple[int, ...],
    target_block: Tuple[int, ...],
    target_smem: int,
    target_regs: Optional[int],
) -> bool:
    """Return True when a replayed kernel matches the DB launch signature."""
    return (
        tuple(kernel.grid) == target_grid
        and tuple(kernel.block) == target_block
        and kernel.shared_memory == target_smem
        and (
            target_regs is None
            or kernel.registers_per_thread is None
            or kernel.registers_per_thread == target_regs
        )
    )


def _select_replay_samples(
    matched_kernels: List[Any],
    launches: Dict[int, Dict[str, float]],
    dispatch_ranges: List[Dict[str, float]],
    target_kernel_name: str,
    target_duration_us: float,
    target_grid: Tuple[int, ...],
    target_block: Tuple[int, ...],
    target_smem: int,
    target_regs: Optional[int],
    runs: int,
) -> Tuple[List[Tuple[float, float, Optional[float]]], Dict[str, Any]]:
    """Select at most one replay sample per measured dispatch range.

    This avoids contaminating a kernel's replay stats with later matching kernels
    from the same ATen op invocation.
    """
    if not matched_kernels:
        return [], {
            "selection_strategy": "none",
            "matched_iterations": 0,
            "measured_iterations": 0,
            "multi_candidate_iterations": 0,
            "kernel_variant_match": False,
            "selected_kernel_names": [],
        }

    selected: List[Tuple[float, float, Optional[float]]] = []
    selected_names: List[str] = []
    multi_candidate_iterations = 0
    kernel_variant_match = True

    if dispatch_ranges:
        measured_ranges = dispatch_ranges[-runs:] if len(dispatch_ranges) > runs else list(dispatch_ranges)
        for nvtx_r in measured_ranges:
            range_start = nvtx_r["ts"]
            range_end = nvtx_r["ts"] + nvtx_r["dur"]
            candidates = []
            for kernel in matched_kernels:
                launch = launches.get(kernel.correlation)
                if not launch:
                    continue
                launch_ts = launch["ts"]
                if not (range_start <= launch_ts <= range_end):
                    continue
                name_kind = _kernel_name_match_kind(target_kernel_name, kernel.name)
                name_penalty = {"exact": 0, "substring": 1}.get(name_kind, 2)
                sig_penalty = 0 if _is_exact_signature_match(
                    kernel,
                    target_grid,
                    target_block,
                    target_smem,
                    target_regs,
                ) else 1
                duration_abs_err = abs(kernel.dur - target_duration_us)
                duration_rel_err = duration_abs_err / max(target_duration_us, 1.0)
                launch_tax = kernel.ts - launch_ts
                dispatch_tax = launch_ts - range_start
                candidates.append({
                    "kernel": kernel,
                    "launch_tax": launch_tax,
                    "duration": kernel.dur,
                    "dispatch_tax": dispatch_tax,
                    "score": (
                        name_penalty,
                        sig_penalty,
                        round(duration_rel_err, 6),
                        round(duration_abs_err, 4),
                        round(dispatch_tax, 4),
                    ),
                })

            if not candidates:
                continue

            if len(candidates) > 1:
                multi_candidate_iterations += 1

            best = min(candidates, key=lambda item: item["score"])
            best_kernel = best["kernel"]
            selected.append((best["launch_tax"], best["duration"], best["dispatch_tax"]))
            selected_names.append(best_kernel.name)
            if best_kernel.name != target_kernel_name or best["score"][1] > 0:
                kernel_variant_match = False

        return selected, {
            "selection_strategy": "per_dispatch_range",
            "matched_iterations": len(selected),
            "measured_iterations": len(measured_ranges),
            "multi_candidate_iterations": multi_candidate_iterations,
            "kernel_variant_match": kernel_variant_match,
            "selected_kernel_names": selected_names,
        }

    tax_triples: List[Tuple[float, float, Optional[float]]] = []
    for kernel in matched_kernels:
        launch = launches.get(kernel.correlation)
        if not launch:
            continue
        name_kind = _kernel_name_match_kind(target_kernel_name, kernel.name)
        if name_kind != "exact":
            kernel_variant_match = False
        tax_triples.append((kernel.ts - launch["ts"], kernel.dur, None))
        selected_names.append(kernel.name)

    if len(tax_triples) > runs:
        tax_triples = tax_triples[-runs:]
        selected_names = selected_names[-runs:]

    return tax_triples, {
        "selection_strategy": "global_tail_trim",
        "matched_iterations": len(tax_triples),
        "measured_iterations": len(tax_triples),
        "multi_candidate_iterations": 0,
        "kernel_variant_match": kernel_variant_match,
        "selected_kernel_names": selected_names,
    }


def nsys_profile_pytorch_kernel(
    kernel_entry: Dict[str, Any],
    warmup: int = 20,
    runs: int = 50,
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

    # Ops with a mid-kernel CPU–GPU sync (e.g. scan→compact that must read back
    # the element count to allocate a variable-size output buffer).  Under nsys
    # profiling the sync blocks indefinitely; skip the replay and let report.py
    # assign T_sys floor (KT_framework = 0) for these entries.
    _SKIP_REPLAY_OPS = {
        "aten::nonzero",        # DeviceSelectSweepKernel scan+compact; CPU readback hangs nsys
        "aten::nonzero_numpy",  # alias
        "aten::unique",         # also variable-size output
        "aten::unique_consecutive",
        "aten::_unique2",
        "aten::index_add_",     # indexFuncLargeIndex; slow for large tensors (>300 s at bs=4/sl=4096)
        "aten::index_add",
    }
    if op_name in _SKIP_REPLAY_OPS:
        print(f"  {kernel_id} ({op_name}): skipped (sync-op, would hang under nsys)")
        return None

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
        if msg and msg.startswith("TIMEOUT:"):
            print(
                f"  WARNING: {kernel_id} ({op_name}): nsys replay timed out (>300s) — "
                f"large tensor replay; assigning T_sys floor (KT_framework=0)"
            )
        else:
            print(f"nsys replay failed for {kernel_id} ({op_name}): {msg}")
        return None

    # 3. Extract kernels and launches from the SQLite trace
    kernels = extract_kernels_sql(trace_sql, filter_gemm_only=False)
    launches = extract_launches_sql(trace_sql)

    if not kernels:
        # Common cause: CPU-only op (e.g., aten::arange) that dispatches no CUDA kernels.
        print(f"  {kernel_id} ({op_name}): no GPU kernels in trace (CPU-only op or unsupported)")
        return None

    # 4. Filter to the target kernel — four-tier fallback
    kernel_variant_match = True
    matched = []

    # Tier 0: full launch-config signature match — (grid, block, shared_memory,
    #         registers_per_thread). These four signals form a complete kernel binary
    #         fingerprint: grid/block = tile config; smem = tile memory footprint;
    #         registers = compiled variant. Handles nvjet → cuBLAS dispatch in subprocess
    #         where the name changes but the tile regime stays the same.
    k_meta = kernel_entry.get("kernel", {})
    target_grid = tuple(k_meta.get("grid", []))
    target_block = tuple(k_meta.get("block", []))
    target_smem = k_meta.get("shared_memory", 0)      # always int in DB; 0 is valid
    target_regs = k_meta.get("registers_per_thread")  # None if not captured in Chrome trace
    # Guard: (0,0,0) is truthy but is the DB fallback for missing data — skip Tier 0 for it.
    if all(g > 0 for g in target_grid) and all(b > 0 for b in target_block):
        sig_matched = [
            k for k in kernels
            if k.grid == target_grid
            and k.block == target_block
            and k.shared_memory == target_smem
            # registers_per_thread: only constrain when both sides have it
            and (target_regs is None or k.registers_per_thread is None
                 or k.registers_per_thread == target_regs)
        ]
        if sig_matched:
            # Prefer exact name within signature candidates; otherwise keep all
            named = [k for k in sig_matched if k.name == target_kernel_name]
            matched = named if named else sig_matched
            if not named:
                kernel_variant_match = False
                print(
                    f"  Tier0 sig match: expected '{target_kernel_name}', "
                    f"got '{matched[0].name}' "
                    f"(grid={list(target_grid)} block={list(target_block)} "
                    f"smem={target_smem} regs={target_regs})"
                )

    # Tier 1: exact name match
    if not matched:
        matched = [k for k in kernels if k.name == target_kernel_name]

    # Tier 2: substring match (name may differ by template args).
    # Guard k.name so empty-named anonymous kernels never match via '' in any_string.
    if not matched:
        matched = [
            k for k in kernels
            if k.name and (target_kernel_name in k.name or k.name in target_kernel_name)
        ]
        if matched:
            kernel_variant_match = False

    # Tier 3: counter fallback — most-frequent kernel dispatched by this op.
    if not matched:
        name_counts = Counter(k.name for k in kernels)
        if name_counts:
            most_common_name, _ = name_counts.most_common(1)[0]
            matched = [k for k in kernels if k.name == most_common_name]
            kernel_variant_match = False
            print(
                f"  Warning: expected '{target_kernel_name}', got "
                f"'{most_common_name}' (variant mismatch — used inferred_size={inferred_size})"
            )

    if not matched:
        print(
            f"Target kernel '{target_kernel_name}' not found in trace for "
            f"{kernel_id}. Found: {[k.name for k in kernels[:5]]}"
        )
        return None

    # 5. Match kernels to launches and compute per-invocation metrics.
    # Use exactly one selected kernel per measured NVTX dispatch range when possible.
    nvtx_ranges = extract_culib_markers_sql(trace_sql)
    aten_dispatch_ranges = sorted(
        [r for r in nvtx_ranges if r["name"] == "aten_dispatch"],
        key=lambda r: r["ts"],
    )

    target_duration_us = kernel_entry.get("statistics", {}).get("avg_duration_us", 0.0)
    tax_triples, selection_meta = _select_replay_samples(
        matched_kernels=matched,
        launches=launches,
        dispatch_ranges=aten_dispatch_ranges,
        target_kernel_name=target_kernel_name,
        target_duration_us=target_duration_us,
        target_grid=target_grid,
        target_block=target_block,
        target_smem=target_smem,
        target_regs=target_regs,
        runs=runs,
    )
    kernel_variant_match = kernel_variant_match and selection_meta["kernel_variant_match"]

    if not tax_triples:
        print(f"No launch-kernel matches for {kernel_id} ({op_name})")
        return None

    launch_taxes = [t[0] for t in tax_triples]
    durations = [t[1] for t in tax_triples]
    dispatch_taxes = [t[2] for t in tax_triples if t[2] is not None]

    selected_kernel_names = selection_meta.get("selected_kernel_names", [])
    actual_kernel_name = Counter(selected_kernel_names).most_common(1)[0][0] if selected_kernel_names else (matched[0].name if matched else target_kernel_name)

    # Runtime vendor-library detection: check if the isolation replay produced
    # cuBLAS/cuDNN API events (requires --trace=cublas,cudnn in nsys).
    # detect_vendor_library_events returns True/False/None:
    #   True/False → vendor tables were present (tracing was active) → result is authoritative
    #   None       → vendor tables absent (tracing inactive or file unreadable) → inconclusive
    _vendor_detection = detect_vendor_library_events(trace_sql) if trace_sql else None
    # vendor_tracing_available: True only when the trace actually contained vendor tables
    # (i.e., nsys was invoked with --trace=cublas,cudnn and the tables were queryable).
    vendor_tracing_available = _vendor_detection is not None
    i_lib_detected = _vendor_detection if vendor_tracing_available else False

    result = {
        "kernel_id": kernel_id,
        "kernel_name": actual_kernel_name,
        "aten_op": op_name,
        "launch_tax": _compute_stats(launch_taxes),
        "kernel_duration": _compute_stats(durations),
        "samples": len(launch_taxes),
        "replay_method": "pytorch",
        "kernel_variant_match": kernel_variant_match,
        "selection_strategy": selection_meta["selection_strategy"],
        "matched_iterations": selection_meta["matched_iterations"],
        "measured_iterations": selection_meta["measured_iterations"],
        "multi_candidate_iterations": selection_meta["multi_candidate_iterations"],
        "i_lib_detected": i_lib_detected,
        "vendor_tracing_available": vendor_tracing_available,
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
