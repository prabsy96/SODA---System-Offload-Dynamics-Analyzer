"""
Per-kernel power measurement via isolated tight-loop replay + NVML.

For every unique kernel in the Stage 2 kernel database, this module:
  1. Generates a standalone Python replay script that runs the kernel in
     a tight loop (no NCU instrumentation, no L2 flush between iters).
  2. Executes the script as a child process in a single CUDA context so
     L2 warms up once (warmup phase) and stays warm across all measurement
     windows.
  3. Samples GPU package power via NVMLPowerSampler in the parent, slicing
     samples by window timestamps emitted to stdout by the script.
  4. Subtracts a one-time idle baseline (CUDA context alive, no kernels)
     so that reported net_power_w reflects the kernel's incremental draw.

Design choices (accuracy rationale):
  - Single subprocess per kernel: a new process would start with a cold
    CUDA context, discarding L2 warmup from a previous warmup subprocess.
  - Batched sync: torch.cuda.synchronize() every SYNC_BATCH iters (not
    every iter) so that short kernels keep ~99.9 % GPU duty cycle.
    Per-iter sync for a 10 µs kernel adds ≥15 µs overhead → 40 % idle,
    pulling NVML readings toward the idle baseline.
  - Stdout timestamp protocol: script prints WINDOW_START / WINDOW_END
    with time.monotonic() values; parent slices NVML samples by those
    timestamps (same CLOCK_MONOTONIC domain on Linux).
  - Idle baseline subtraction: idle H200 CUDA context draws 20–50 W
    (HBM refresh, clock gating). Subtracting removes this constant floor.

Remaining fundamental limit: NVML reports GPU package power (die + HBM +
NVLink), not per-SM power. This is the highest-fidelity measurement
available without hardware power rails and is the correct metric for
"how much power does the GPU draw while running this kernel at 100 %
duty cycle."
"""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from soda.power_sampler import make_power_sampler, _NoOpSampler


# ---------------------------------------------------------------------------
# Iteration-count formula
# ---------------------------------------------------------------------------

def _compute_replay_iters(
    avg_dur_us: float,
    warmup_ms: int = 500,
    meas_ms: int = 300,
) -> Tuple[int, int, int]:
    """Compute iteration counts and sync batch size for the replay script.

    Args:
        avg_dur_us: Average kernel duration in microseconds (from kernel DB).
        warmup_ms:  Target warmup phase duration in milliseconds.
        meas_ms:    Target per-window measurement duration in milliseconds.

    Returns:
        (warmup_iters, meas_iters, sync_batch) where sync_batch is the number
        of kernel iterations between consecutive torch.cuda.synchronize() calls.

    Design:
        - sync_batch is chosen so that GPU duty cycle ≥ 99 % for any kernel
          duration ≥ 1 µs (sync overhead ≈ 15 µs is amortised over the batch).
        - warmup_iters and meas_iters are rounded up to the nearest multiple
          of sync_batch so the loop structure is uniform.
    """
    if avg_dur_us <= 0.0:
        avg_dur_us = 10.0  # safe fallback for unknown/zero-duration entries

    # Batch size: more syncs for long kernels (queue depth OK), fewer for short
    if avg_dur_us >= 500.0:
        sync_batch = 1
    elif avg_dur_us >= 50.0:
        sync_batch = 10
    else:
        sync_batch = 1000

    # Effective duration per sync-batch interval (kernel time + one sync)
    effective_batch_us = avg_dur_us * sync_batch + 15.0

    warmup_batches = max(1, math.ceil(warmup_ms * 1_000 / effective_batch_us))
    meas_batches   = max(1, math.ceil(meas_ms   * 1_000 / effective_batch_us))

    # Ensure minimums before multiplying back to iters
    warmup_iters = max(sync_batch * 2,  warmup_batches * sync_batch)
    meas_iters   = max(sync_batch * 4,  meas_batches   * sync_batch)

    return warmup_iters, meas_iters, sync_batch


# ---------------------------------------------------------------------------
# Replay script generation
# ---------------------------------------------------------------------------

def _generate_power_replay_script(
    entry: Dict[str, Any],
    warmup_iters: int,
    meas_iters: int,
    sync_batch: int,
    num_windows: int,
    output_dir: Path,
) -> Path:
    """Write a standalone replay script for power measurement.

    The script:
      - Imports create_input_tensors / execute_operation (same as NCU replay).
      - Runs warmup_iters iterations (no measurement) to warm L2 and GPU thermal
        state — no stdout output during warmup.
      - Runs num_windows measurement windows of meas_iters each, printing
        ``WINDOW_START <t>`` / ``WINDOW_END <t>`` to stdout so the parent
        can slice NVML samples precisely.

    Unlike the NCU replay script this script does NOT flush L2 between
    iterations (letting L2 reach steady-state cache occupancy, matching
    real inference conditions).

    Args:
        entry:        Kernel DB entry (must contain "aten_op" and "id").
        warmup_iters: Number of warmup iterations.
        meas_iters:   Number of measurement iterations per window.
        sync_batch:   Call torch.cuda.synchronize() every this many iters.
        num_windows:  Number of measurement windows.
        output_dir:   Directory for the generated script.

    Returns:
        Path to the written script.
    """
    kid = entry["id"]
    aten_op = entry.get("aten_op", {})
    aten_op_json = json.dumps(aten_op)

    script = f'''\
#!/usr/bin/env python3
"""Auto-generated power replay script for {kid}."""
import json
import sys
import time
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

SYNC_BATCH = {sync_batch}
WARMUP_ITERS = {warmup_iters}
MEAS_ITERS = {meas_iters}
NUM_WINDOWS = {num_windows}

# ── Phase 1: Thermal warmup ────────────────────────────────────────────────
# Run kernel in a tight loop to warm L2 cache and stabilise GPU temperature.
# No stdout output; NVML sampler is not active during this phase.
_fail = 0
with torch.no_grad():
    for _i in range(WARMUP_ITERS):
        try:
            execute_operation(op_name, inputs)
        except Exception as _exc:
            _fail += 1
            if _fail <= 2:
                print(f"Warmup execute_operation failed iter {{_i}}: {{_exc}}", file=sys.stderr)
        if (_i + 1) % SYNC_BATCH == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()

if _fail == WARMUP_ITERS:
    print(f"ALL {{WARMUP_ITERS}} warmup iterations of {{op_name}} failed", file=sys.stderr)
    sys.exit(2)

# ── Phase 2: Measurement windows ──────────────────────────────────────────
# Each window prints WINDOW_START / WINDOW_END timestamps to stdout so
# the parent can slice NVML samples to the measurement interval only.
with torch.no_grad():
    for _w in range(NUM_WINDOWS):
        sys.stdout.write(f"WINDOW_START {{time.monotonic():.6f}}\\n")
        sys.stdout.flush()
        _wfail = 0
        for _i in range(MEAS_ITERS):
            try:
                execute_operation(op_name, inputs)
            except Exception as _exc:
                _wfail += 1
                if _wfail <= 2:
                    print(f"Window {{_w}} execute_operation failed iter {{_i}}: {{_exc}}", file=sys.stderr)
            if (_i + 1) % SYNC_BATCH == 0:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        sys.stdout.write(f"WINDOW_END {{time.monotonic():.6f}}\\n")
        sys.stdout.flush()
'''

    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / f"power_replay_{kid}.py"
    with open(script_path, "w") as fh:
        fh.write(script)
    return script_path


# ---------------------------------------------------------------------------
# NVML sample slicing
# ---------------------------------------------------------------------------

def _slice_samples_by_windows(
    sampler_samples: Dict[int, List[Tuple[float, float]]],
    windows: List[Tuple[float, float]],
) -> List[float]:
    """Return per-window mean power (watts) from NVML samples.

    Args:
        sampler_samples: ``{gpu_id: [(t_ms, watts), ...]}`` from
            ``NVMLPowerSampler._samples``.  ``t_ms`` is ``time.monotonic()
            * 1000`` recorded inside the sampler's polling thread.
        windows: List of ``(start_s, end_s)`` pairs in ``time.monotonic()``
            seconds as printed by the replay script.

    Returns:
        List of mean watt readings, one per window that had ≥1 NVML sample.
        Windows with no samples are silently skipped.
    """
    # Flatten all GPU readings into a single list (power replay is single-GPU serial)
    all_readings: List[Tuple[float, float]] = []
    for gpu_readings in sampler_samples.values():
        all_readings.extend(gpu_readings)

    per_window_means: List[float] = []
    for (ws, we) in windows:
        ws_ms = ws * 1_000.0
        we_ms = we * 1_000.0
        watts = [w for (t_ms, w) in all_readings if ws_ms <= t_ms <= we_ms]
        if watts:
            per_window_means.append(statistics.mean(watts))

    return per_window_means


# ---------------------------------------------------------------------------
# Single-kernel profiler
# ---------------------------------------------------------------------------

def power_profile_kernel(
    entry: Dict[str, Any],
    output_dir: Path,
    idle_baseline_w: float = 0.0,
    target_warmup_ms: int = 500,
    target_meas_ms: int = 500,
    num_windows: int = 3,
    interval_ms: int = 50,
    extra_env: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Measure GPU power for a single kernel via isolated tight-loop replay.

    Spawns a child process that runs the kernel in a tight loop.  The child
    signals measurement-window boundaries via stdout; this function records
    hardware-integrated energy counters (preferred) or slices NVML polling
    samples (fallback) for each window.

    Measurement method priority:
      1. Energy counter (``nvmlDeviceGetTotalEnergyConsumption``): hardware-
         integrated, no sampling uncertainty.  Used when ≥ half the windows
         yield a non-zero counter increment.
      2. NVML polling (``nvmlDeviceGetPowerUsage``): background thread at
         ``interval_ms``.  Used when energy counter is unavailable or all
         windows yield zero increment (very short windows).

    H200 hardware note: NVML power readings update every ~100 ms (25 ms
    averaging window).  At 50 ms polling, every other poll returns a stale
    value.  The energy counter avoids this entirely.

    Args:
        entry:           Kernel DB entry (must have "aten_op.name").
        output_dir:      Directory for generated replay scripts.
        idle_baseline_w: GPU idle power (W) to subtract from raw readings.
        target_warmup_ms: Warmup phase target duration (ms).
        target_meas_ms:  Per-window measurement target duration (ms).
                         Default 500 ms to cover ≥ 5 NVML hardware cycles.
        num_windows:     Number of measurement windows (≥2 recommended for
                         thermal variance detection).
        interval_ms:     NVML polling interval (ms).
        extra_env:       Environment for the child process (defaults to
                         ``os.environ``).

    Returns:
        Dict with keys:
            kernel_id, raw_power_w, idle_power_w, net_power_w,
            std_power_w, thermal_variance_pct, energy_nj,
            sample_count_per_window, num_windows,
            warmup_iters, meas_iters, sync_batch, backend,
            measurement_method, is_reliable.
        Returns None if: NVML unavailable, op not replayable, subprocess
        fails, or no measurement data falls in any measurement window.
    """
    kid = entry["id"]
    avg_dur_us = entry.get("statistics", {}).get("avg_duration_us", 10.0) or 10.0
    aten_op = entry.get("aten_op", {})

    if not aten_op.get("name"):
        return None  # no ATen op → cannot create inputs

    # Create the sampler once; bail early if NVML unavailable
    sampler = make_power_sampler(gpu_ids=[0], interval_ms=interval_ms, enabled=True)
    if isinstance(sampler, _NoOpSampler):
        return None

    # Probe whether the energy counter is accessible on this GPU
    _use_energy_counter = (
        hasattr(sampler, "get_energy_counter_mj")
        and sampler.get_energy_counter_mj(0) is not None
    )

    warmup_iters, meas_iters, sync_batch = _compute_replay_iters(
        avg_dur_us, target_warmup_ms, target_meas_ms
    )

    script_path = _generate_power_replay_script(
        entry=entry,
        warmup_iters=warmup_iters,
        meas_iters=meas_iters,
        sync_batch=sync_batch,
        num_windows=num_windows,
        output_dir=output_dir,
    )

    env = extra_env if extra_env is not None else dict(os.environ)
    python_exe = sys.executable

    # Conservative timeout: base startup overhead + 5× expected runtime
    expected_s = (target_warmup_ms + num_windows * target_meas_ms) / 1_000.0
    timeout_s = 30.0 + expected_s * 5.0

    windows: List[Tuple[float, float]] = []
    t_start: Optional[float] = None
    # Energy counter tracking (one entry per completed window)
    energy_counter_windows: List[float] = []  # per-window power in W
    e_start: Optional[float] = None
    parent_t_start: Optional[float] = None
    proc_returncode: Optional[int] = None
    stderr_text: str = ""

    sampler.start()
    try:
        proc = subprocess.Popen(
            [python_exe, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        # Read stdout line-by-line for window boundary timestamps
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.strip()
            if line.startswith("WINDOW_START"):
                parts = line.split()
                if len(parts) < 2:
                    print(
                        f"  Warning: malformed WINDOW_START line for {kid}: {line!r}",
                        file=sys.stderr,
                    )
                else:
                    try:
                        t_start = float(parts[1])
                        if _use_energy_counter:
                            e_start = sampler.get_energy_counter_mj(0)
                            parent_t_start = time.monotonic()
                    except ValueError:
                        print(
                            f"  Warning: could not parse WINDOW_START timestamp "
                            f"for {kid}: {parts[1]!r}",
                            file=sys.stderr,
                        )
            elif line.startswith("WINDOW_END") and t_start is not None:
                parts = line.split()
                if len(parts) < 2:
                    print(
                        f"  Warning: malformed WINDOW_END line for {kid}: {line!r}",
                        file=sys.stderr,
                    )
                    t_start = None
                    e_start = None
                    parent_t_start = None
                else:
                    try:
                        t_end = float(parts[1])
                        windows.append((t_start, t_end))
                        # Energy counter: read immediately on WINDOW_END
                        if _use_energy_counter and e_start is not None and parent_t_start is not None:
                            e_end = sampler.get_energy_counter_mj(0)
                            parent_t_end = time.monotonic()
                            if e_end is not None:
                                delta_mj = e_end - e_start
                                dur_s = parent_t_end - parent_t_start
                                if delta_mj >= 0.01 and dur_s > 0.001:
                                    # W = J/s; convert mJ → J with ×1e-3
                                    energy_counter_windows.append(
                                        (delta_mj * 1e-3) / dur_s
                                    )
                                else:
                                    print(
                                        f"  Power replay {kid}: energy counter did not "
                                        f"increment (ΔE={delta_mj:.3f} mJ, "
                                        f"dur={dur_s*1000:.1f} ms) — using polling for window",
                                        file=sys.stderr,
                                    )
                        t_start = None
                        e_start = None
                        parent_t_start = None
                    except ValueError:
                        print(
                            f"  Warning: could not parse WINDOW_END timestamp "
                            f"for {kid}: {parts[1]!r}",
                            file=sys.stderr,
                        )
                        t_start = None
                        e_start = None
                        parent_t_start = None
        proc.stdout.close()  # type: ignore[union-attr]
        # Drain stderr to prevent subprocess blocking on a full pipe buffer
        if proc.stderr:
            try:
                stderr_text = proc.stderr.read()
            except Exception as exc:
                print(
                    f"  Warning: could not read stderr for {kid}: {exc}",
                    file=sys.stderr,
                )
            proc.stderr.close()
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        proc_returncode = proc.returncode
    except Exception as exc:
        print(f"  Power replay error for {kid}: {exc}", file=sys.stderr)
        proc_returncode = -1
        stderr_text = str(exc)
    finally:
        sampler.stop()

    if proc_returncode != 0:
        msg = (stderr_text or "")[:200].strip()
        print(
            f"  Power replay subprocess failed for {kid} "
            f"(rc={proc_returncode})"
            + (f": {msg}" if msg else "")
        )
        return None

    if not windows:
        print(f"  Power replay: no measurement windows received for {kid}")
        return None

    # Select measurement source: prefer energy counter when ≥ half windows valid
    if len(energy_counter_windows) >= max(1, len(windows) // 2):
        per_window_means = energy_counter_windows
        measurement_method = "energy_counter"
    else:
        per_window_means = _slice_samples_by_windows(sampler._samples, windows)
        measurement_method = "nvml_polling"
        if _use_energy_counter and energy_counter_windows:
            print(
                f"  Power replay {kid}: only {len(energy_counter_windows)}/"
                f"{len(windows)} energy counter windows valid — using polling fallback"
            )

    if not per_window_means:
        print(
            f"  Power replay: no measurement data within windows "
            f"for {kid} (interval={interval_ms} ms, windows={len(windows)})"
        )
        return None

    raw_power_w = statistics.mean(per_window_means)
    std_power_w = statistics.stdev(per_window_means) if len(per_window_means) > 1 else 0.0
    thermal_variance_pct = (
        (std_power_w / raw_power_w * 100.0) if raw_power_w > 0.0 else 0.0
    )
    is_reliable = thermal_variance_pct <= 5.0
    net_power_w = max(0.0, raw_power_w - idle_baseline_w)

    if not is_reliable:
        print(
            f"  Warning: {kid} thermal variance {thermal_variance_pct:.1f}% > 5%"
            " — GPU may not be at thermal steady state (is_reliable=False)"
        )

    # energy = net power × avg kernel duration (W × µs × 1e-3 = nJ)
    energy_nj = net_power_w * avg_dur_us * 1e-3

    # Rough sample count: total samples / number of completed windows
    total_samples = sum(len(r) for r in sampler._samples.values())
    sample_count_per_window = (
        total_samples // len(windows) if windows else 0
    )

    return {
        "kernel_id": kid,
        "raw_power_w": round(raw_power_w, 2),
        "idle_power_w": round(idle_baseline_w, 2),
        "net_power_w": round(net_power_w, 2),
        "std_power_w": round(std_power_w, 2),
        "thermal_variance_pct": round(thermal_variance_pct, 2),
        "energy_nj": round(energy_nj, 2),
        "sample_count_per_window": sample_count_per_window,
        "num_windows": len(per_window_means),
        "warmup_iters": warmup_iters,
        "meas_iters": meas_iters,
        "sync_batch": sync_batch,
        "backend": sampler._backend,
        "measurement_method": measurement_method,
        "is_reliable": is_reliable,
    }


# ---------------------------------------------------------------------------
# All-kernels driver
# ---------------------------------------------------------------------------

def power_profile_all_kernels(
    kernel_db_entries: List[Dict[str, Any]],
    output_dir: Path,
    gpu_ids: Optional[List[int]] = None,
    target_warmup_ms: int = 500,
    target_meas_ms: int = 500,
    num_windows: int = 3,
    interval_ms: int = 50,
    max_kernels: Optional[int] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], float]:
    """Profile power for all (or up to max_kernels) unique kernels.

    Measures an idle-power baseline once before iterating kernels.
    Power replay is always serial (single-GPU) — running kernels in parallel
    on separate GPUs would contaminate per-GPU NVML readings.

    Args:
        kernel_db_entries: List of kernel DB entry dicts.
        output_dir:        Directory for replay scripts.
        gpu_ids:           GPU indices to sample (default: [0]).
        target_warmup_ms:  Warmup phase target (ms).
        target_meas_ms:    Per-window measurement target (ms).
        num_windows:       Measurement windows per kernel.
        interval_ms:       NVML polling interval (ms).
        max_kernels:       If set, cap the number of kernels profiled.
        extra_env:         Environment for replay subprocesses.

    Returns:
        Tuple of (results_dict, idle_baseline_w) where results_dict maps
        kernel_id → power result dict, and idle_baseline_w is the measured
        idle GPU power in watts (0.0 if NVML unavailable).
    """
    if gpu_ids is None:
        gpu_ids = [0]

    entries = kernel_db_entries
    if max_kernels is not None:
        entries = entries[:max_kernels]

    # ── Idle baseline measurement ──────────────────────────────────────────
    idle_baseline_w = 0.0
    _idle_sampler = make_power_sampler(gpu_ids=gpu_ids, interval_ms=interval_ms, enabled=True)
    if not isinstance(_idle_sampler, _NoOpSampler):
        print("  Measuring GPU idle power baseline (1 s)...")
        _idle_sampler.start()
        time.sleep(1.0)
        _idle_sampler.stop()
        idle_r = _idle_sampler.get_results()
        idle_baseline_w = idle_r.get("mean_power_w", 0.0)
        print(f"  Idle baseline: {idle_baseline_w:.1f} W")
    else:
        print(
            "  Warning: NVML unavailable — power replay will return no results. "
            "Install pynvml (pip install pynvml) or ensure nvidia-smi is in PATH."
        )

    # ── Per-kernel replay ──────────────────────────────────────────────────
    replay_dir = output_dir / "power_replay_scripts"
    results: Dict[str, Dict[str, Any]] = {}
    total = len(entries)

    for i, entry in enumerate(entries, 1):
        # Periodically re-measure idle baseline to track GPU thermal drift.
        # A drift > 5 W over 10 kernels (~14 s) indicates warm-up effects.
        if i > 1 and (i - 1) % 10 == 0 and not isinstance(
            make_power_sampler(gpu_ids=gpu_ids, interval_ms=interval_ms, enabled=True),
            _NoOpSampler,
        ):
            _re_idle = make_power_sampler(gpu_ids=gpu_ids, interval_ms=interval_ms, enabled=True)
            _re_idle.start()
            time.sleep(0.5)
            _re_idle.stop()
            new_baseline = _re_idle.get_results().get("mean_power_w", idle_baseline_w)
            if abs(new_baseline - idle_baseline_w) > 5.0:
                print(
                    f"  Idle baseline drift detected: {idle_baseline_w:.1f} W → "
                    f"{new_baseline:.1f} W (updating after kernel {i - 1})"
                )
                idle_baseline_w = new_baseline

        kid = entry["id"]
        kname = entry.get("kernel", {}).get("name", "?")
        op = entry.get("aten_op", {}).get("name", "?")
        print(f"  Power replay [{i}/{total}] {kid}: {op} → {kname[:50]}")

        result = power_profile_kernel(
            entry=entry,
            output_dir=replay_dir,
            idle_baseline_w=idle_baseline_w,
            target_warmup_ms=target_warmup_ms,
            target_meas_ms=target_meas_ms,
            num_windows=num_windows,
            interval_ms=interval_ms,
            extra_env=extra_env,
        )
        if result is not None:
            results[kid] = result

    print(
        f"\n  Power replay complete: {len(results)}/{total} kernels profiled"
        f" (idle baseline {idle_baseline_w:.1f} W)"
    )
    return results, idle_baseline_w
