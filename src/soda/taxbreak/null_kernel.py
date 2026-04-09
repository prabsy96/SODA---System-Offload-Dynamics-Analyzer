"""
Dynamic system floor measurement via null kernel profiling.

Replaces the hardcoded NULL_KERNEL_SYS_TAX baselines with a runtime
measurement of the minimum achievable launch tax (T_sys floor).
"""

import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from soda.common import utils
from soda.microbench.baremetal.utils import (
    build_binary,
    extract_kernels_sql,
    extract_launches_sql,
    nsys_check_available,
    nsys_profile,
)


def _measure_floor_on_device(
    gpu_id: int,
    warmup: int,
    runs: int,
) -> Optional[Dict[str, float]]:
    """
    Measure the system launch-tax floor on a single GPU device.

    Args:
        gpu_id: CUDA device index to target.
        warmup: Number of warmup iterations (not measured).
        runs: Number of measured iterations.

    Returns:
        Dict with floor statistics, or None on failure.
    """
    binary_path = utils.get_path("BAREMETAL_BINARY")
    binary_args = [
        str(binary_path),
        "--null_kernel",
        "--warmup", str(warmup),
        "--runs", str(runs),
    ]

    trace_name = f"null_kernel_floor_gpu{gpu_id}"
    env_override = dict(os.environ)
    env_override["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    success, trace_sql, msg = nsys_profile(
        trace_file_name=trace_name,
        args=binary_args,
        timeout=120,
        extra_env=env_override,
    )
    if not success:
        print(f"Warning: nsys null-kernel profiling failed for GPU {gpu_id}: {msg}")
        return None

    kernels = extract_kernels_sql(trace_sql, filter_gemm_only=False)
    launches = extract_launches_sql(trace_sql)

    if not kernels:
        print(f"Warning: No null kernels found in nsys trace for GPU {gpu_id}")
        return None

    launch_taxes = []
    for kernel in kernels:
        corr_id = kernel.correlation
        if corr_id in launches:
            launch = launches[corr_id]
            tax = kernel.ts - launch["ts"]
            launch_taxes.append(tax)

    if not launch_taxes:
        print(f"Warning: Could not match any null kernel for GPU {gpu_id}")
        return None

    if len(launch_taxes) > runs:
        launch_taxes = launch_taxes[-runs:]

    avg = sum(launch_taxes) / len(launch_taxes)
    min_val = min(launch_taxes)
    max_val = max(launch_taxes)

    if len(launch_taxes) > 1:
        variance = sum((x - avg) ** 2 for x in launch_taxes) / (len(launch_taxes) - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0

    return {
        "avg_us": round(avg, 4),
        "min_us": round(min_val, 4),
        "max_us": round(max_val, 4),
        "std_us": round(std, 4),
        "samples": len(launch_taxes),
    }


def measure_system_floor(
    warmup: int = 20,
    runs: int = 50,
    num_gpus: int = 1,
) -> Dict[str, float]:
    """
    Measure the system launch-tax floor by profiling a null (empty) kernel.

    When ``num_gpus > 1``, the measurement is run on each GPU and the
    **minimum** average T_sys is returned (conservative hardware floor).

    Args:
        warmup: Number of warmup iterations (not measured).
        runs: Number of measured iterations.
        num_gpus: Number of GPUs in use. Measurements run on each device.

    Returns:
        Dict with keys ``avg_us``, ``min_us``, ``max_us``, ``std_us``,
        ``method`` (always ``"dynamic"``), and optionally
        ``per_gpu_avg_us`` (list) and ``num_gpus`` when > 1.
    """
    # 0. Verify nsys is available before doing any work
    if not nsys_check_available():
        raise RuntimeError(
            "nsys not found in PATH. "
            "Load the Nsight module (e.g. 'module load cuda12.8/nsight/12.8.1') "
            "or install NVIDIA Nsight Systems."
        )

    # 1. Ensure binary exists
    build_binary()

    if num_gpus <= 1:
        # Single-GPU path — delegate to the shared helper (gpu_id=0).
        r = _measure_floor_on_device(0, warmup, runs)
        if r is None:
            raise RuntimeError(
                "nsys profiling of null kernel failed on GPU 0. "
                "Check nsys availability and baremetal binary path."
            )
        result: Dict[str, object] = {**r, "method": "dynamic"}
        print(f"System floor (null kernel): avg={result['avg_us']:.2f} us, "
              f"min={result['min_us']:.2f} us, max={result['max_us']:.2f} us, "
              f"std={result['std_us']:.2f} us ({result['samples']} samples)")
        return result

    # Multi-GPU path: measure each device, return minimum avg (conservative floor)
    per_gpu_results = []
    for gpu_id in range(num_gpus):
        print(f"Measuring system floor on GPU {gpu_id}...")
        r = _measure_floor_on_device(gpu_id, warmup, runs)
        if r is not None:
            per_gpu_results.append(r)

    if not per_gpu_results:
        print(
            f"Warning: system floor measurement failed on all {num_gpus} GPU(s). "
            "Using 0.0 µs floor — KT_framework values may be inflated.",
            file=sys.stderr,
        )
        return {"avg_us": 0.0, "std_us": 0.0, "num_gpus": num_gpus, "method": "dynamic"}

    # Use the minimum average as the conservative hardware floor
    best = min(per_gpu_results, key=lambda r: r["avg_us"])
    result = {
        "avg_us": best["avg_us"],
        "min_us": best["min_us"],
        "max_us": best["max_us"],
        "std_us": best["std_us"],
        "samples": best["samples"],
        "method": "dynamic",
        "num_gpus": num_gpus,
        "per_gpu_avg_us": [r["avg_us"] for r in per_gpu_results],
    }

    print(f"System floor (null kernel, {num_gpus} GPUs): "
          f"avg={result['avg_us']:.2f} us (min across GPUs), "
          f"per-GPU: {result['per_gpu_avg_us']}")

    return result


# ---------------------------------------------------------------------------
# Active-idle power measurement
# ---------------------------------------------------------------------------

_ACTIVE_IDLE_SCRIPT = '''\
#!/usr/bin/env python3
"""Auto-generated active-idle power measurement script.

Runs a single-element CUDA op in a tight loop to represent the GPU's power
state during maximum-rate kernel dispatch with zero compute throughput.
This is the upper bound on overhead power when inter-kernel gaps are shorter
than the SM clock-gating time constant (~10-100 us on H200).
"""
import sys
import time
import torch

torch.cuda.init()
# Single-element tensor: negligible compute, maximum dispatch rate
_x = torch.zeros(1, device="cuda")
_y = torch.zeros(1, device="cuda")

SYNC_BATCH = 1000
WARMUP_ITERS = {warmup_iters}
MEAS_ITERS   = {meas_iters}
NUM_WINDOWS  = {num_windows}

# Phase 1: warmup (bring GPU to steady-state power/temperature)
with torch.no_grad():
    for _i in range(WARMUP_ITERS):
        _x.add_(_y)
        if (_i + 1) % SYNC_BATCH == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()

# Phase 2: measurement windows
with torch.no_grad():
    for _w in range(NUM_WINDOWS):
        sys.stdout.write(f"WINDOW_START {{time.monotonic():.6f}}\\n")
        sys.stdout.flush()
        for _i in range(MEAS_ITERS):
            _x.add_(_y)
            if (_i + 1) % SYNC_BATCH == 0:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        sys.stdout.write(f"WINDOW_END {{time.monotonic():.6f}}\\n")
        sys.stdout.flush()
'''


def measure_active_idle_power(
    warmup_ms: int = 1000,
    num_windows: int = 3,
    window_ms: int = 500,
    interval_ms: int = 50,
) -> Optional[Dict[str, Any]]:
    """Measure GPU package power during maximum-rate null-dispatch (active-idle state).

    Runs a single-element CUDA op in a tight loop as a subprocess — zero compute
    throughput but full dispatch overhead — and reads GPU package power via NVML
    energy counter from the parent process.

    This measures ``P_active_idle``: the GPU power floor during inter-kernel gaps
    when SM clocks have not yet gated.  It is the correct overhead power estimate
    for workloads with inter-kernel gaps shorter than the clock-gating time constant
    (~10–100 µs on H200 NVL).

    Args:
        warmup_ms:   Warmup phase duration (ms) to stabilise GPU temperature.
        num_windows: Number of measurement windows.
        window_ms:   Target measurement duration per window (ms).
        interval_ms: NVML polling interval (ms) for the fallback path.

    Returns:
        Dict with keys ``active_idle_power_w``, ``std_w``, ``measurement_method``,
        or ``None`` if NVML is unavailable or the subprocess fails.
    """
    from soda.power_sampler import make_power_sampler, _NoOpSampler

    sampler = make_power_sampler(gpu_ids=[0], interval_ms=interval_ms, enabled=True)
    if isinstance(sampler, _NoOpSampler):
        print("  measure_active_idle_power: NVML unavailable, skipping.", file=sys.stderr)
        return None

    sync_batch = 1000
    warmup_iters = max(sync_batch, int(warmup_ms * 1_000 / 1.0 / sync_batch) * sync_batch)
    meas_iters = max(sync_batch * 4, int(window_ms * 1_000 / 1.0 / sync_batch) * sync_batch)

    script_src = _ACTIVE_IDLE_SCRIPT.format(
        warmup_iters=warmup_iters,
        meas_iters=meas_iters,
        num_windows=num_windows,
    )

    # Write script to a temp file in the system temp dir
    import tempfile
    tmp_dir = Path(tempfile.gettempdir())
    script_path = tmp_dir / "soda_active_idle_power.py"
    script_path.write_text(script_src)

    python_exe = sys.executable
    expected_s = (warmup_ms + num_windows * window_ms) / 1_000.0
    timeout_s = 30.0 + expected_s * 3.0

    windows: List = []
    t_start: Optional[float] = None
    energy_counter_windows: List[float] = []
    e_start: Optional[float] = None
    parent_t_start: Optional[float] = None
    stderr_text: str = ""
    proc_returncode: Optional[int] = None

    _use_energy_counter = (
        hasattr(sampler, "get_energy_counter_mj")
        and sampler.get_energy_counter_mj(0) is not None
    )

    sampler.start()
    try:
        proc = subprocess.Popen(
            [python_exe, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=dict(os.environ),
        )
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.strip()
            if line.startswith("WINDOW_START"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        t_start = float(parts[1])
                        if _use_energy_counter:
                            e_start = sampler.get_energy_counter_mj(0)
                            parent_t_start = time.monotonic()
                    except ValueError:
                        pass
            elif line.startswith("WINDOW_END") and t_start is not None:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        t_end = float(parts[1])
                        windows.append((t_start, t_end))
                        if _use_energy_counter and e_start is not None and parent_t_start is not None:
                            e_end = sampler.get_energy_counter_mj(0)
                            parent_t_end = time.monotonic()
                            if e_end is not None:
                                delta_mj = e_end - e_start
                                dur_s = parent_t_end - parent_t_start
                                if delta_mj >= 0.01 and dur_s > 0.001:
                                    energy_counter_windows.append(
                                        (delta_mj * 1e-3) / dur_s  # W
                                    )
                    except ValueError:
                        pass
                    t_start = None
                    e_start = None
                    parent_t_start = None
        proc.stdout.close()  # type: ignore[union-attr]
        if proc.stderr:
            try:
                stderr_text = proc.stderr.read()
            except Exception:
                pass
            proc.stderr.close()
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        proc_returncode = proc.returncode
    except Exception as exc:
        print(f"  measure_active_idle_power subprocess error: {exc}", file=sys.stderr)
        proc_returncode = -1
        stderr_text = str(exc)
    finally:
        sampler.stop()

    if proc_returncode != 0:
        msg = (stderr_text or "")[:200].strip()
        print(
            f"  measure_active_idle_power: subprocess failed (rc={proc_returncode})"
            + (f": {msg}" if msg else ""),
            file=sys.stderr,
        )
        return None

    if not windows:
        print("  measure_active_idle_power: no measurement windows received.", file=sys.stderr)
        return None

    if len(energy_counter_windows) >= max(1, len(windows) // 2):
        per_window_means = energy_counter_windows
        method = "energy_counter"
    else:
        from soda.taxbreak.kernel_power_replay import _slice_samples_by_windows
        per_window_means = _slice_samples_by_windows(sampler._samples, windows)
        method = "nvml_polling"

    if not per_window_means:
        print("  measure_active_idle_power: no data within measurement windows.", file=sys.stderr)
        return None

    active_idle_power_w = statistics.mean(per_window_means)
    std_w = statistics.stdev(per_window_means) if len(per_window_means) > 1 else 0.0

    print(f"  Active-idle power (null-kernel loop): {active_idle_power_w:.1f} W  "
          f"(std={std_w:.1f} W, method={method}, windows={len(per_window_means)})")

    return {
        "active_idle_power_w": round(active_idle_power_w, 3),
        "std_w": round(std_w, 3),
        "measurement_method": method,
        "num_windows": len(per_window_means),
    }
