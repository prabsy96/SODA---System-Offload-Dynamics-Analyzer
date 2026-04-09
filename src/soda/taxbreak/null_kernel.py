"""
Dynamic system floor measurement via null kernel profiling.

Replaces the hardcoded NULL_KERNEL_SYS_TAX baselines with a runtime
measurement of the minimum achievable launch tax (T_sys floor).
"""

import math
import os
import sys
from typing import Dict, List, Optional

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
