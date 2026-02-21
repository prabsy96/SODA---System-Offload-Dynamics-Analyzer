"""
Dynamic system floor measurement via null kernel profiling.

Replaces the hardcoded NULL_KERNEL_SYS_TAX baselines with a runtime
measurement of the minimum achievable launch tax (T_sys floor).
"""

import math
from typing import Dict

from soda.common import utils
from soda.microbench.baremetal.utils import (
    build_binary,
    extract_kernels_sql,
    extract_launches_sql,
    nsys_check_available,
    nsys_profile,
)


def measure_system_floor(
    warmup: int = 50,
    runs: int = 200,
) -> Dict[str, float]:
    """
    Measure the system launch-tax floor by profiling a null (empty) kernel.

    Builds the baremetal binary (if needed), runs it under nsys with
    ``--null_kernel``, then computes launch_tax = kernel.ts - launch.ts
    for each measured run.

    Args:
        warmup: Number of warmup iterations (not measured).
        runs: Number of measured iterations.

    Returns:
        Dict with keys ``avg_us``, ``min_us``, ``max_us``, ``std_us``,
        and ``method`` (always ``"dynamic"``).
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

    # 2. Construct binary args
    binary_path = utils.get_path("BAREMETAL_BINARY")
    binary_args = [
        str(binary_path),
        "--null_kernel",
        "--warmup", str(warmup),
        "--runs", str(runs),
    ]

    # 3. Profile under nsys
    trace_name = "null_kernel_floor"
    success, trace_sql, msg = nsys_profile(
        trace_file_name=trace_name,
        args=binary_args,
        timeout=120,
    )
    if not success:
        raise RuntimeError(f"nsys profiling of null kernel failed: {msg}")

    # 4. Extract kernels and launches from the SQLite trace
    kernels = extract_kernels_sql(trace_sql, filter_gemm_only=False)
    launches = extract_launches_sql(trace_sql)

    if not kernels:
        raise RuntimeError("No null kernels found in nsys trace")

    # 5. Match kernels to launches by correlation ID and compute launch tax
    launch_taxes = []
    for kernel in kernels:
        corr_id = kernel.correlation
        if corr_id in launches:
            launch = launches[corr_id]
            # launch_tax = time between CPU launch call and GPU kernel start
            tax = kernel.ts - launch["ts"]
            launch_taxes.append(tax)

    if not launch_taxes:
        raise RuntimeError(
            "Could not match any null kernel to its launch event"
        )

    # The trace contains warmup + measured runs. The measured runs are the
    # last ``runs`` entries (warmup runs are "cold", measured are "hot").
    # Filter by taking only the last ``runs`` samples if we have more.
    if len(launch_taxes) > runs:
        launch_taxes = launch_taxes[-runs:]

    # 6. Compute statistics
    avg = sum(launch_taxes) / len(launch_taxes)
    min_val = min(launch_taxes)
    max_val = max(launch_taxes)

    if len(launch_taxes) > 1:
        variance = sum((x - avg) ** 2 for x in launch_taxes) / (len(launch_taxes) - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0

    result = {
        "avg_us": round(avg, 4),
        "min_us": round(min_val, 4),
        "max_us": round(max_val, 4),
        "std_us": round(std, 4),
        "samples": len(launch_taxes),
        "method": "dynamic",
    }

    print(f"System floor (null kernel): avg={result['avg_us']:.2f} us, "
          f"min={result['min_us']:.2f} us, max={result['max_us']:.2f} us, "
          f"std={result['std_us']:.2f} us ({result['samples']} samples)")

    return result
