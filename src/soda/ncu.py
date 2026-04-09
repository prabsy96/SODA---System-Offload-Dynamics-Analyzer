"""
NVIDIA Compute Profiler (ncu) integration.

Provides wrappers for running ``ncu`` on individual kernels (via PyTorch
replay or baremetal binary) and parsing the resulting CSV into structured
per-kernel cache / memory metrics.
"""

import csv
import io
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from soda.common import utils

def _extract_kernel_function_name(raw_name: str) -> str:
    """Extract a regex-safe base function name from a full demangled CUDA kernel signature.

    ncu ``--kernel-name`` treats its argument as a regex.  Passing the full
    C++ template signature (e.g. ``void (anonymous namespace)::softmax_warp_forward<…>``
    ) breaks the regex because the ``(`` in ``(anonymous namespace)`` is
    interpreted as a regex group.  This helper returns just the bare function
    name (e.g. ``softmax_warp_forward``) which is regex-safe and unique enough
    to filter the target kernel in a single-op replay script.

    Examples::

        "void at::native::vectorized_elementwise_kernel<8, ...>(...)"
            → "vectorized_elementwise_kernel"
        "void (anonymous namespace)::softmax_warp_forward<...>(...)"
            → "softmax_warp_forward"
        "void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<...>"
            → "vectorized_layer_norm_kernel"
        "void gemv2T_kernel_val<...>(...)"
            → "gemv2T_kernel_val"
    """
    if not raw_name:
        return ""
    name = raw_name
    if name.startswith("void "):
        name = name[5:]
    # Strip C++ template args — everything from the first '<' onward
    name = name.split("<")[0]
    # Find the last ::identifier segment (word chars only) using regex.
    # This correctly skips "(anonymous namespace)" qualifiers that contain '('
    # and would break if we simply split on '::' or '('.
    parts = re.findall(r"::([A-Za-z_]\w*)", name)
    if parts:
        return re.escape(parts[-1])
    # No '::' namespace — strip any remaining '(' and return the base name.
    base = name.split("(")[0].strip()
    return re.escape(base) if base else ""


# Default ncu metric set — cache hierarchy + compute utilisation.
# DRAM byte counter names changed in Blackwell (CC 12.x):
#   Pre-Blackwell (Ampere/Hopper): dram__bytes_read.sum, dram__bytes_write.sum
#   Blackwell CC 12.x+:            dram__bytes_op_read.sum, dram__bytes_op_write.sum
# Both sets are requested so the same binary works on all generations.
NCU_METRICS = [
    "l1tex__t_sector_hit_rate.pct",                         # L1 hit rate
    "lts__t_sector_hit_rate.pct",                            # L2 hit rate
    "l1tex__t_bytes.sum.per_second",                         # L1 throughput
    "lts__t_bytes.sum.per_second",                           # L2 throughput
    "dram__bytes.sum.per_second",                            # HBM throughput
    "dram__bytes_read.sum",                                  # DRAM reads (pre-Blackwell)
    "dram__bytes_write.sum",                                 # DRAM writes (pre-Blackwell)
    "dram__bytes_op_read.sum",                               # DRAM reads (Blackwell CC 12.x+)
    "dram__bytes_op_write.sum",                              # DRAM writes (Blackwell CC 12.x+)
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",      # Compute utilization
]


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def ncu_check_available() -> bool:
    """Verify that ``ncu`` is in PATH and can run.

    Returns True if ncu is found and executable, False otherwise.
    Prints a diagnostic message on failure.
    """
    if shutil.which("ncu") is None:
        print("ncu not found in PATH. Install NVIDIA Nsight Compute.")
        return False

    try:
        result = subprocess.run(
            ["ncu", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            print(f"ncu --version failed: {result.stderr.strip()}")
            return False
    except Exception as exc:
        print(f"ncu availability check failed: {exc}")
        return False

    return True


# ---------------------------------------------------------------------------
# Low-level profiling
# ---------------------------------------------------------------------------

def ncu_profile(
    command_args: List[str],
    metrics: List[str],
    output_csv: Path,
    kernel_name: Optional[str] = None,
    launch_skip: int = 5,
    launch_count: int = 1,
    timeout: int = 300,
    extra_env: Optional[dict] = None,
) -> Tuple[bool, str]:
    """Run ``ncu`` with the given metrics on *command_args*.

    Args:
        command_args: The command to profile (e.g. ``["python", "replay.py"]``).
        metrics: List of ncu metric strings.
        output_csv: Path to write CSV output.
        kernel_name: If set, only profile kernels whose demangled name
            contains this substring (``--kernel-name``).
        launch_skip: Number of kernel launches to skip (warmup).
        launch_count: Number of kernel launches to measure.
        timeout: Seconds before killing the process.
        extra_env: Optional environment dict forwarded to the subprocess.

    Returns:
        ``(success, message)`` tuple.
    """
    ncu_args = [
        "ncu",
        "--metrics", ",".join(metrics),
        "--target-processes", "all",
        "--csv",
        "--log-file", str(output_csv),
        "--launch-skip", str(launch_skip),
        "--launch-count", str(launch_count),
    ]

    if kernel_name:
        ncu_args += ["--kernel-name", kernel_name]

    ncu_args += list(command_args)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            ncu_args,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=extra_env,
        )
    except subprocess.TimeoutExpired:
        return False, f"ncu timed out after {timeout}s"

    if result.returncode != 0:
        return False, result.stderr or result.stdout

    if not output_csv.exists():
        return False, "ncu completed but CSV output not found"

    return True, ""


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def parse_ncu_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Parse an ncu ``--csv`` output file into structured dicts.

    ncu CSV rows contain one metric per row.  This function groups them
    by kernel launch and returns a list of dicts (one per launch), each
    mapping metric names to their float values.

    Returns:
        List of ``{"kernel_name": str, "metrics": {metric: value}}`` dicts,
        one per profiled kernel launch.
    """
    text = csv_path.read_text()

    # Detect the "no kernels profiled" warning that ncu emits when --kernel-name
    # doesn't match any kernel in the workload.  In this case the file contains
    # only diagnostic text, not CSV data, so return empty immediately.
    if "No kernels were profiled" in text:
        return []

    # ncu prepends informational lines starting with "==" before the CSV.
    csv_lines = [line for line in text.splitlines() if not line.startswith("==")]
    if not csv_lines:
        return []

    reader = csv.DictReader(io.StringIO("\n".join(csv_lines)))

    # Group by (ID, Kernel Name) — one group per profiled launch.
    launches: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in reader:
        launch_id = row.get("ID", "0")
        kernel = row.get("Kernel Name", "unknown")
        key = (launch_id, kernel)

        if key not in launches:
            launches[key] = {"kernel_name": kernel, "metrics": {}, "grid_size": [1, 1, 1]}

        # Parse Grid Size once per launch (same for all metric rows of a launch).
        if launches[key]["grid_size"] == [1, 1, 1]:
            grid_str = row.get("Grid Size", "")
            if grid_str:
                nums = [int(x) for x in re.findall(r"\d+", grid_str)]
                if nums:
                    while len(nums) < 3:
                        nums.append(1)
                    launches[key]["grid_size"] = nums[:3]

        metric_name = row.get("Metric Name", "")
        metric_value = row.get("Metric Value", "")

        try:
            value = float(metric_value.replace(",", ""))
        except (ValueError, AttributeError):
            value = metric_value  # keep as string if not numeric

        launches[key]["metrics"][metric_name] = value

    return list(launches.values())


# ---------------------------------------------------------------------------
# Per-kernel profiling
# ---------------------------------------------------------------------------

def _generate_ncu_replay_script(
    aten_op: Dict[str, Any],
    warmup: int,
    runs: int,
    script_path: Path,
) -> None:
    """Write a replay script suitable for ncu profiling.

    Similar to the nsys replay script but with more iterations
    (warmup + runs) so that ncu's ``--launch-skip`` / ``--launch-count``
    can select the right invocations.
    """
    aten_op_json = json.dumps(aten_op)

    script = f'''\
#!/usr/bin/env python3
"""Auto-generated replay script for ncu profiling."""
import json
import sys
import torch
from soda.microbench.framework.pytorch.profile import (
    create_input_tensors,
    execute_operation,
)

aten_op = json.loads({aten_op_json!r})
op_name = aten_op["name"]

try:
    inputs = create_input_tensors(aten_op)
except Exception as exc:
    print(f"Failed to create inputs for {{op_name}}: {{exc}}", file=sys.stderr)
    sys.exit(1)

# Flush L2 cache before replay: read a buffer larger than L2 capacity
# so that input tensors are evicted.  Without this, create_input_tensors()
# warms L2 (torch.randn writes to DRAM, data stays in L2), causing the
# first GEMM launch to read weights from L2 instead of DRAM — producing
# near-zero dram__bytes_read.sum.
_flush_size = 128 * 1024 * 1024  # 128 MB > Blackwell L2 (96 MB); H200 L2 is 50 MB
_flush_buf = torch.empty(_flush_size, dtype=torch.uint8, device="cuda")
torch.sum(_flush_buf)
torch.cuda.synchronize()
del _flush_buf

# Run warmup + measured iterations.
# ncu --launch-skip/--launch-count handles the split externally.
total = {warmup + runs}
_fail_count = 0
with torch.no_grad():
    for _i in range(total):
        try:
            execute_operation(op_name, inputs)
        except Exception as _exc:
            _fail_count += 1
            if _fail_count <= 2:
                print(f"execute_operation failed iter {{_i}}: {{_exc}}", file=sys.stderr)
        torch.cuda.synchronize()

if _fail_count == total:
    print(f"ALL {{total}} iterations of {{op_name}} failed", file=sys.stderr)
    sys.exit(2)
'''

    script_path.parent.mkdir(parents=True, exist_ok=True)
    with open(script_path, "w") as f:
        f.write(script)


def _pick_best_launch(launches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the compute kernel launch with the highest DRAM read bytes.

    Used by ``ncu_profile_kernel(pick_best_kernel=True)`` to select the
    actual compute kernel from a set of captures that may include
    cuBLASLt workspace/init kernels, bias-add epilogues, L2 flush
    kernels, or other overhead kernels.

    Overhead kernels that are explicitly excluded:
    - ``distribution_elementwise_grid_stride_kernel``: random tensor init
    - ``unrolled_elementwise_kernel``: elementwise copies (e.g. L2 flush)
    - ``reduce_kernel``: reductions (e.g. torch.sum for L2 flush)

    Falls back to all launches if filtering leaves nothing, then to the
    first launch if no candidate has non-zero DRAM reads.
    """
    _OVERHEAD_KERNEL_SUBSTRINGS = (
        "distribution_elementwise_grid_stride_kernel",
        "unrolled_elementwise_kernel",
        "reduce_kernel",
    )

    def _is_overhead(launch: Dict[str, Any]) -> bool:
        name = launch.get("kernel_name", "")
        return any(s in name for s in _OVERHEAD_KERNEL_SUBSTRINGS)

    def _dram_reads(launch: Dict[str, Any]) -> float:
        m = launch.get("metrics", {})
        # Try Blackwell name first (CC 12.x+), then pre-Blackwell name.
        val = m.get("dram__bytes_op_read.sum") or m.get("dram__bytes_read.sum", 0)
        try:
            return float(val or 0)
        except (TypeError, ValueError):
            return 0.0

    # Prefer compute kernels; fall back to all launches if none found.
    compute_launches = [l for l in launches if not _is_overhead(l)]
    candidates = compute_launches if compute_launches else launches

    best = max(candidates, key=_dram_reads)
    return best if _dram_reads(best) > 0 else candidates[0]


def ncu_profile_kernel(
    kernel_entry: Dict[str, Any],
    output_dir: Path,
    warmup: int = 5,
    runs: int = 1,
    metrics: Optional[List[str]] = None,
    timeout: int = 300,
    extra_env: Optional[dict] = None,
    pick_best_kernel: bool = False,
) -> Optional[Dict[str, Any]]:
    """Profile a single kernel with ncu and return cache/memory metrics.

    Generates a PyTorch replay script for the kernel's ATen operation,
    runs it under ``ncu`` with the requested metrics, and returns the
    parsed results.

    Args:
        kernel_entry: Entry from ``kernel_database.json``.
        output_dir: Directory for replay scripts and CSV output.
        warmup: Iterations to skip (``--launch-skip``).  Ignored when
            ``pick_best_kernel=True`` (all launches are captured instead).
        runs: Iterations to measure (``--launch-count``).
        metrics: Metric list (defaults to ``NCU_METRICS``).
        timeout: Seconds before killing ncu.
        pick_best_kernel: When True, skip=0 and capture all kernel
            launches from the replay, then return the one with the
            highest ``dram__bytes_read.sum``.  Avoids the cuBLASLt
            helper-kernel problem where workspace/init launches consume
            the ``--launch-skip`` budget and the actual GEMM is missed.

    Returns:
        Dict with ``kernel_name``, ``aten_op``, ``metrics``, and
        ``replay_method`` keys, or ``None`` on failure.
    """
    if metrics is None:
        metrics = NCU_METRICS

    kernel_id = kernel_entry["id"]
    aten_op = kernel_entry["aten_op"]
    raw_kernel_name = kernel_entry["kernel"].get("raw_name", "")
    cleaned_name = kernel_entry["kernel"]["name"]
    op_name = aten_op.get("name", "unknown")

    # Extract a regex-safe function name for --kernel-name.
    # Using the full demangled template (e.g. "void (anonymous namespace)::foo<...>")
    # breaks ncu's regex because of the parentheses.  The bare function name is
    # sufficient and unambiguous within a single-op replay script.
    kernel_filter = _extract_kernel_function_name(raw_kernel_name) or None

    # 1. Generate replay script
    tmp_dir = Path(tempfile.mkdtemp(prefix="soda_ncu_"))
    try:
        script_path = tmp_dir / f"ncu_replay_{kernel_id}.py"
        _generate_ncu_replay_script(aten_op, warmup, runs, script_path)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    # 2. Run ncu
    csv_path = output_dir / f"ncu_{kernel_id}.csv"

    if pick_best_kernel:
        # Capture all kernel launches from the full replay (skip=0, large count).
        # cuBLASLt ops dispatch helper/workspace kernels that would consume
        # the skip budget in the standard path, causing the actual GEMM to be
        # missed.  Capturing everything and selecting by highest DRAM reads is
        # robust to any number of helper kernels.
        ncu_launch_skip = 0
        ncu_launch_count = (warmup + runs) * 8  # generous upper bound
    else:
        ncu_launch_skip = warmup
        ncu_launch_count = runs

    # Build a subprocess env that guarantees the soda package is importable.
    # The NCU replay script does `from soda.microbench... import ...`.  On SLURM
    # nodes the conda env's sys.path may not be reflected in the inherited
    # PYTHONPATH, causing silent ImportError → hbm_bytes stays 0 for all ops.
    import os
    import sys as _sys
    _env = dict(extra_env) if extra_env else dict(os.environ)
    _src_paths = [p for p in _sys.path if p and "soda" in p.lower()]
    _all_paths = _src_paths + [p for p in _sys.path if p and p not in _src_paths]
    _existing = _env.get("PYTHONPATH", "")
    _env["PYTHONPATH"] = ":".join(
        [p for p in _all_paths if p] + ([_existing] if _existing else [])
    )

    try:
        success, msg = ncu_profile(
            command_args=["python", str(script_path)],
            metrics=metrics,
            output_csv=csv_path,
            kernel_name=kernel_filter,
            launch_skip=ncu_launch_skip,
            launch_count=ncu_launch_count,
            timeout=timeout,
            extra_env=_env,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not success:
        print(f"ncu profiling failed for {kernel_id} ({op_name}): {msg}")
        return None

    # 3. Parse CSV
    launches = parse_ncu_csv(csv_path)
    if not launches:
        print(f"No ncu results for {kernel_id} ({op_name})")
        return None

    # Select the best launch — either by DRAM reads (pick_best_kernel) or first.
    if pick_best_kernel and len(launches) > 1:
        best_launch = _pick_best_launch(launches)
    else:
        best_launch = launches[0]

    ncu_metrics = best_launch["metrics"]
    ncu_grid_size = best_launch.get("grid_size", [1, 1, 1])

    result = {
        "kernel_id": kernel_id,
        "kernel_name": cleaned_name,
        "aten_op": op_name,
        "metrics": ncu_metrics,
        "replay_method": "pytorch",
        "grid_size": ncu_grid_size,
    }

    # Pretty-print key metrics
    l1_hit = ncu_metrics.get("l1tex__t_sector_hit_rate.pct", "N/A")
    l2_hit = ncu_metrics.get("lts__t_sector_hit_rate.pct", "N/A")
    compute = ncu_metrics.get(
        "sm__throughput.avg.pct_of_peak_sustained_elapsed", "N/A"
    )
    print(
        f"  {kernel_id} ({op_name} -> {cleaned_name}): "
        f"L1={l1_hit}%, L2={l2_hit}%, compute={compute}%"
    )

    return result
