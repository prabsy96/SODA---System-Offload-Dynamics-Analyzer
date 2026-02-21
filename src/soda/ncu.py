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
NCU_METRICS = [
    "l1tex__t_sector_hit_rate.pct",                         # L1 hit rate
    "lts__t_sector_hit_rate.pct",                            # L2 hit rate
    "l1tex__t_bytes.sum.per_second",                         # L1 throughput
    "lts__t_bytes.sum.per_second",                           # L2 throughput
    "dram__bytes.sum.per_second",                            # HBM throughput
    "dram__bytes_read.sum",                                  # DRAM reads
    "dram__bytes_write.sum",                                 # DRAM writes
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
            launches[key] = {"kernel_name": kernel, "metrics": {}}

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

# Run warmup + measured iterations.
# ncu --launch-skip/--launch-count handles the split externally.
total = {warmup + runs}
with torch.no_grad():
    for _ in range(total):
        try:
            execute_operation(op_name, inputs)
        except Exception:
            pass
        torch.cuda.synchronize()
'''

    script_path.parent.mkdir(parents=True, exist_ok=True)
    with open(script_path, "w") as f:
        f.write(script)


def ncu_profile_kernel(
    kernel_entry: Dict[str, Any],
    output_dir: Path,
    warmup: int = 5,
    runs: int = 1,
    metrics: Optional[List[str]] = None,
    timeout: int = 300,
) -> Optional[Dict[str, Any]]:
    """Profile a single kernel with ncu and return cache/memory metrics.

    Generates a PyTorch replay script for the kernel's ATen operation,
    runs it under ``ncu`` with the requested metrics, and returns the
    parsed results.

    Args:
        kernel_entry: Entry from ``kernel_database.json``.
        output_dir: Directory for replay scripts and CSV output.
        warmup: Iterations to skip (``--launch-skip``).
        runs: Iterations to measure (``--launch-count``).
        metrics: Metric list (defaults to ``NCU_METRICS``).
        timeout: Seconds before killing ncu.

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
    script_path = tmp_dir / f"ncu_replay_{kernel_id}.py"
    _generate_ncu_replay_script(aten_op, warmup, runs, script_path)

    # 2. Run ncu
    csv_path = output_dir / f"ncu_{kernel_id}.csv"

    success, msg = ncu_profile(
        command_args=["python", str(script_path)],
        metrics=metrics,
        output_csv=csv_path,
        kernel_name=kernel_filter,
        launch_skip=warmup,
        launch_count=runs,
        timeout=timeout,
    )

    if not success:
        print(f"ncu profiling failed for {kernel_id} ({op_name}): {msg}")
        return None

    # 3. Parse CSV
    launches = parse_ncu_csv(csv_path)
    if not launches:
        print(f"No ncu results for {kernel_id} ({op_name})")
        return None

    # Take the first (or only) profiled launch
    ncu_metrics = launches[0]["metrics"]

    result = {
        "kernel_id": kernel_id,
        "kernel_name": cleaned_name,
        "aten_op": op_name,
        "metrics": ncu_metrics,
        "replay_method": "pytorch",
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
