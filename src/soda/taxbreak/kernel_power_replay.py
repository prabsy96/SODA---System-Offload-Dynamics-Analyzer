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
# Duration grouping helpers
# ---------------------------------------------------------------------------

_DURATION_CLASS_ORDER = ("ultra_short", "short", "medium", "long")


def _duration_class(avg_dur_us: float) -> str:
    """Return a duration class label for replay scheduling.

    Classes are intentionally coarse and measurement-oriented:
      - ultra_short: < 1 us
      - short:       1 to < 50 us
      - medium:      50 to < 500 us
      - long:        >= 500 us
    """
    if avg_dur_us <= 0.0:
        avg_dur_us = 10.0

    if avg_dur_us < 1.0:
        return "ultra_short"
    if avg_dur_us < 50.0:
        return "short"
    if avg_dur_us < 500.0:
        return "medium"
    return "long"


def _group_entries_by_duration_class(
    entries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Group entries by duration class, preserving in-class input order."""
    buckets: Dict[str, List[Dict[str, Any]]] = {k: [] for k in _DURATION_CLASS_ORDER}
    for entry in entries:
        avg_dur_us = entry.get("statistics", {}).get("avg_duration_us", 10.0) or 10.0
        buckets[_duration_class(avg_dur_us)].append(entry)

    ordered: List[Dict[str, Any]] = []
    for klass in _DURATION_CLASS_ORDER:
        ordered.extend(buckets[klass])
    return ordered


def _flatten_sampler_watts(sampler: Any) -> List[float]:
    """Return flattened power samples (W) from a sampler's internal buffers."""
    samples = getattr(sampler, "_samples", None)
    if not isinstance(samples, dict):
        return []

    watts: List[float] = []
    for gpu_samples in samples.values():
        if not isinstance(gpu_samples, list):
            continue
        for pair in gpu_samples:
            if not isinstance(pair, tuple) or len(pair) != 2:
                continue
            _, w = pair
            try:
                watts.append(float(w))
            except (TypeError, ValueError):
                continue
    return watts


def _measure_idle_baseline(
    gpu_ids: List[int],
    interval_ms: int,
    min_settle_ms: int = 1500,
    max_settle_ms: int = 6000,
    settle_step_ms: int = 500,
    settle_std_threshold_pct: float = 3.0,
) -> Tuple[float, float, int, bool]:
    """Measure idle baseline with convergence-based thermal settling.

    The sampler runs for at least ``min_settle_ms`` and then keeps extending
    (up to ``max_settle_ms``) until recent idle samples stabilize under the
    relative standard-deviation threshold.

    Returns:
        (idle_mean_w, rel_std_pct, settle_ms, settled)
    """
    sampler = make_power_sampler(gpu_ids=gpu_ids, interval_ms=interval_ms, enabled=True)
    if isinstance(sampler, _NoOpSampler):
        return 0.0, 0.0, 0, False

    step_ms = max(100, int(settle_step_ms))
    min_settle_ms = max(0, int(min_settle_ms))
    max_settle_ms = max(min_settle_ms, int(max_settle_ms))
    settle_std_threshold_pct = max(0.0, float(settle_std_threshold_pct))

    min_steps = max(1, math.ceil(min_settle_ms / step_ms))
    max_steps = max(min_steps, math.ceil(max_settle_ms / step_ms))
    recent_samples_target = max(3, int(math.ceil(1000.0 / max(interval_ms, 1))))

    settle_ms = 0
    rel_std_pct = 0.0
    settled = False

    sampler.start()
    try:
        for step in range(1, max_steps + 1):
            time.sleep(step_ms / 1000.0)
            settle_ms = step * step_ms

            watts = _flatten_sampler_watts(sampler)
            if len(watts) < 2:
                continue

            recent = watts[-recent_samples_target:]
            if len(recent) < 2:
                continue

            mean_w = statistics.mean(recent)
            if mean_w <= 0.0:
                continue

            std_w = statistics.stdev(recent)
            rel_std_pct = (std_w / mean_w) * 100.0

            if step >= min_steps and rel_std_pct <= settle_std_threshold_pct:
                settled = True
                break
    finally:
        sampler.stop()

    idle_r = sampler.get_results()
    idle_mean_w = float(idle_r.get("mean_power_w", 0.0) or 0.0)
    return idle_mean_w, rel_std_pct, settle_ms, settled


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

    # Batch size: amortize synchronize() overhead most aggressively for ultra-short
    # kernels where per-iter sync would dominate runtime and collapse duty cycle.
    if avg_dur_us >= 500.0:
        sync_batch = 1
    elif avg_dur_us >= 50.0:
        sync_batch = 10
    elif avg_dur_us >= 10.0:
        sync_batch = 1000
    elif avg_dur_us >= 1.0:
        sync_batch = 2000
    else:
        sync_batch = 5000

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
    aligned = _slice_samples_by_windows_aligned(sampler_samples, windows)
    return [w for w in aligned if w is not None]


def _slice_samples_by_windows_aligned(
    sampler_samples: Dict[int, List[Tuple[float, float]]],
    windows: List[Tuple[float, float]],
) -> List[Optional[float]]:
    """Return per-window mean power aligned to replay windows.

    The returned list has exactly ``len(windows)`` elements where each element
    is either a mean watt value for that window or ``None`` when no NVML sample
    fell inside the window.
    """
    # Flatten all GPU readings into a single list (power replay is single-GPU serial)
    all_readings: List[Tuple[float, float]] = []
    for gpu_readings in sampler_samples.values():
        all_readings.extend(gpu_readings)

    per_window_means: List[Optional[float]] = []
    for (ws, we) in windows:
        ws_ms = ws * 1_000.0
        we_ms = we * 1_000.0
        watts = [w for (t_ms, w) in all_readings if ws_ms <= t_ms <= we_ms]
        if watts:
            per_window_means.append(statistics.mean(watts))
        else:
            per_window_means.append(None)

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
    reliability_threshold_pct: float = 5.0,
    consensus_tolerance_pct: float = 15.0,
) -> Optional[Dict[str, Any]]:
    """Measure GPU power for a single kernel via isolated tight-loop replay.

    Spawns a child process that runs the kernel in a tight loop.  The child
    signals measurement-window boundaries via stdout; this function records
    hardware-integrated energy counters (preferred) or slices NVML polling
    samples (fallback) for each window.

    Measurement method policy:
        1. Dual-channel consensus (strict): windows where energy counter and
           NVML polling agree within ``consensus_tolerance_pct`` are accepted.
        2. If the energy counter is available but consensus does not pass a
           strict majority threshold, the kernel is rejected (fail-closed).
        3. Polling-only fallback is used only when the energy counter is not
           available on the platform.

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
        reliability_threshold_pct: Thermal-variance threshold used to set
                 ``is_reliable``.
        consensus_tolerance_pct: Max allowed percent difference between
             energy-counter and polling per-window means for high-confidence
             consensus windows.

    Returns:
        Dict with keys:
            kernel_id, raw_power_w, idle_power_w, net_power_w,
            std_power_w, thermal_variance_pct, energy_uj,
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
    energy_counter_windows: List[Optional[float]] = []  # aligned per-window power in W
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
                        window_power_w: Optional[float] = None
                        # Energy counter: read immediately on WINDOW_END
                        if _use_energy_counter and e_start is not None and parent_t_start is not None:
                            e_end = sampler.get_energy_counter_mj(0)
                            parent_t_end = time.monotonic()
                            if e_end is not None:
                                delta_mj = e_end - e_start
                                dur_s = parent_t_end - parent_t_start
                                if delta_mj >= 0.01 and dur_s > 0.001:
                                    # W = J/s; convert mJ → J with ×1e-3
                                    window_power_w = (delta_mj * 1e-3) / dur_s
                                else:
                                    print(
                                        f"  Power replay {kid}: energy counter did not "
                                        f"increment (ΔE={delta_mj:.3f} mJ, "
                                        f"dur={dur_s*1000:.1f} ms) — using polling for window",
                                        file=sys.stderr,
                                    )
                        energy_counter_windows.append(window_power_w)
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

    # Build aligned polling and energy-counter vectors per window.
    polling_windows = _slice_samples_by_windows_aligned(sampler._samples, windows)
    # Defensive alignment in case an older path produced a shorter list.
    if len(energy_counter_windows) < len(windows):
        energy_counter_windows.extend([None] * (len(windows) - len(energy_counter_windows)))

    valid_energy_windows = [w for w in energy_counter_windows if w is not None]
    valid_polling_windows = [w for w in polling_windows if w is not None]

    # Dual-channel consensus windows: both channels present and within tolerance.
    consensus_windows: List[float] = []
    consensus_windows_checked = 0
    consensus_windows_agree = 0
    consensus_windows_excluded = 0
    for ec_w, poll_w in zip(energy_counter_windows, polling_windows):
        if ec_w is None or poll_w is None:
            continue
        consensus_windows_checked += 1
        denom = max((abs(ec_w) + abs(poll_w)) / 2.0, 1e-9)
        diff_pct = abs(ec_w - poll_w) / denom * 100.0
        if diff_pct <= consensus_tolerance_pct:
            consensus_windows_agree += 1
            consensus_windows.append((ec_w + poll_w) / 2.0)
        else:
            consensus_windows_excluded += 1

    # Strict majority of all configured windows (e.g., 2/3, 2/2, 3/4).
    consensus_min_windows = max(1, (len(windows) // 2) + 1)
    # Select measurement source with strong validation policy. No single-channel fallbacks allowed
    # when the hardware supports dual-channel measurement.
    if _use_energy_counter:
        if (
            consensus_windows_checked >= consensus_min_windows
            and consensus_windows_agree >= consensus_min_windows
        ):
            per_window_means = consensus_windows
            measurement_method = "dual_consensus"
        else:
            print(
                f"  Error: {kid} failed dual-consensus validation ({consensus_windows_agree}/{consensus_windows_checked} "
                f"agreed, min={consensus_min_windows}). Single-channel fallbacks are disabled."
            )
            return None
    else:
        # Legacy hardware fallback (only when energy counter is fundamentally unavailable).
        per_window_means = valid_polling_windows
        measurement_method = "nvml_polling"

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
    effective_windows = len(per_window_means)
    # A single effective window cannot establish stability; require >= 2 windows.
    is_reliable = (
        effective_windows >= 2
        and thermal_variance_pct <= reliability_threshold_pct
    )
    net_power_w = max(0.0, raw_power_w - idle_baseline_w)

    if not is_reliable:
        if effective_windows < 2:
            print(
                f"  Warning: {kid} has only {effective_windows} effective window(s) "
                "— cannot assess thermal stability (is_reliable=False)"
            )
        else:
            print(
                f"  Warning: {kid} thermal variance {thermal_variance_pct:.1f}% > "
                f"{reliability_threshold_pct:.1f}%"
                " — GPU may not be at thermal steady state (is_reliable=False)"
            )

    # energy = net power × avg kernel duration; W × µs = µJ (direct, no scaling)
    energy_uj = net_power_w * avg_dur_us

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
        "energy_uj": round(energy_uj, 2),
        "sample_count_per_window": sample_count_per_window,
        "num_windows": len(per_window_means),
        "warmup_iters": warmup_iters,
        "meas_iters": meas_iters,
        "sync_batch": sync_batch,
        "backend": sampler._backend,
        "measurement_method": measurement_method,
        "is_reliable": is_reliable,
        "consensus_windows_checked": consensus_windows_checked,
        "consensus_windows_agree": consensus_windows_agree,
        "consensus_windows_excluded": consensus_windows_excluded,
        "consensus_tolerance_pct": round(consensus_tolerance_pct, 2),
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
    adaptive_retry: bool = True,
    variance_threshold_pct: float = 5.0,
    max_retry_windows: int = 9,
    idle_settle_min_ms: int = 1500,
    idle_settle_max_ms: int = 6000,
    idle_settle_step_ms: int = 500,
    idle_settle_std_threshold_pct: float = 3.0,
) -> Tuple[Dict[str, Dict[str, Any]], float]:
    """Profile power for all (or up to max_kernels) unique kernels.

    Measures an idle-power baseline once before iterating kernels.
    Power replay is always serial (single-GPU) — running kernels in parallel
    on separate GPUs would contaminate per-GPU NVML readings.

    Entries are grouped by duration class to reduce thermal mixing between
    ultra-short and long kernels; idle baseline is re-checked on class
    transitions and periodically every 10 kernels.

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
        adaptive_retry:    Retry unstable kernels with more windows.
        variance_threshold_pct: Thermal variance threshold for retry.
        max_retry_windows: Upper bound on retry window count.
        idle_settle_min_ms: Minimum idle-settling duration before convergence check.
        idle_settle_max_ms: Maximum idle-settling duration before proceeding.
        idle_settle_step_ms: Sampling interval for convergence checks.
        idle_settle_std_threshold_pct: Relative std-dev threshold for idle convergence.

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
    entries = _group_entries_by_duration_class(entries)

    # ── Idle baseline measurement ──────────────────────────────────────────
    idle_baseline_w = 0.0
    _idle_sampler = make_power_sampler(gpu_ids=gpu_ids, interval_ms=interval_ms, enabled=True)
    sampler_available = not isinstance(_idle_sampler, _NoOpSampler)
    if sampler_available:
        print(
            "  Measuring GPU idle power baseline "
            f"(settle {idle_settle_min_ms}-{idle_settle_max_ms} ms)..."
        )
        idle_baseline_w, settle_rel_std_pct, settle_ms, settled = _measure_idle_baseline(
            gpu_ids=gpu_ids,
            interval_ms=interval_ms,
            min_settle_ms=idle_settle_min_ms,
            max_settle_ms=idle_settle_max_ms,
            settle_step_ms=idle_settle_step_ms,
            settle_std_threshold_pct=idle_settle_std_threshold_pct,
        )
        if settled:
            print(
                f"  Idle baseline: {idle_baseline_w:.1f} W "
                f"(settled in {settle_ms} ms, rel std {settle_rel_std_pct:.2f}%)"
            )
        else:
            print(
                f"  Warning: idle baseline did not settle within {settle_ms} ms "
                f"(rel std {settle_rel_std_pct:.2f}% > {idle_settle_std_threshold_pct:.2f}%)"
            )
            print(f"  Idle baseline: {idle_baseline_w:.1f} W")
    else:
        print(
            "  Warning: NVML unavailable — power replay will return no results. "
            "Install pynvml (pip install pynvml) or ensure nvidia-smi is in PATH."
        )

    def _remeasure_idle(duration_s: float) -> float:
        """Best-effort idle baseline refresh."""
        sampler = make_power_sampler(gpu_ids=gpu_ids, interval_ms=interval_ms, enabled=True)
        if isinstance(sampler, _NoOpSampler):
            return idle_baseline_w
        sampler.start()
        time.sleep(duration_s)
        sampler.stop()
        return sampler.get_results().get("mean_power_w", idle_baseline_w)

    # ── Per-kernel replay ──────────────────────────────────────────────────
    replay_dir = output_dir / "power_replay_scripts"
    results: Dict[str, Dict[str, Any]] = {}
    total = len(entries)
    prev_class: Optional[str] = None

    for i, entry in enumerate(entries, 1):
        avg_dur_us = entry.get("statistics", {}).get("avg_duration_us", 10.0) or 10.0
        klass = _duration_class(avg_dur_us)

        if prev_class is None:
            prev_class = klass
        elif sampler_available and klass != prev_class:
            new_baseline = _remeasure_idle(duration_s=0.5)
            print(
                f"  Duration-class transition ({prev_class} -> {klass}): "
                f"idle baseline {idle_baseline_w:.1f} W -> {new_baseline:.1f} W"
            )
            idle_baseline_w = new_baseline
            prev_class = klass

        # Periodically re-measure idle baseline to track GPU thermal drift.
        # A drift > 5 W over 10 kernels (~14 s) indicates warm-up effects.
        if i > 1 and (i - 1) % 10 == 0 and sampler_available:
            new_baseline = _remeasure_idle(duration_s=0.5)
            if abs(new_baseline - idle_baseline_w) > 5.0:
                print(
                    f"  Idle baseline drift detected: {idle_baseline_w:.1f} W → "
                    f"{new_baseline:.1f} W (updating after kernel {i - 1})"
                )
                idle_baseline_w = new_baseline

        kid = entry["id"]
        kname = entry.get("kernel", {}).get("name", "?")
        op = entry.get("aten_op", {}).get("name", "?")
        print(f"  Power replay [{i}/{total}] {kid} ({klass}): {op} → {kname[:50]}")

        current_windows = num_windows
        retry_attempts = 0
        result = power_profile_kernel(
            entry=entry,
            output_dir=replay_dir,
            idle_baseline_w=idle_baseline_w,
            target_warmup_ms=target_warmup_ms,
            target_meas_ms=target_meas_ms,
            num_windows=current_windows,
            interval_ms=interval_ms,
            extra_env=extra_env,
            reliability_threshold_pct=variance_threshold_pct,
        )

        while (
            adaptive_retry
            and result is not None
            and result.get("thermal_variance_pct", 0.0) > variance_threshold_pct
            and current_windows < max_retry_windows
        ):
            retry_windows = min(max_retry_windows, max(current_windows * 2, current_windows + 1))
            if retry_windows <= current_windows:
                break
            print(
                f"  {kid}: variance {result['thermal_variance_pct']:.1f}% > "
                f"{variance_threshold_pct:.1f}% — retrying with {retry_windows} windows"
            )
            retried = power_profile_kernel(
                entry=entry,
                output_dir=replay_dir,
                idle_baseline_w=idle_baseline_w,
                target_warmup_ms=target_warmup_ms,
                target_meas_ms=target_meas_ms,
                num_windows=retry_windows,
                interval_ms=interval_ms,
                extra_env=extra_env,
                reliability_threshold_pct=variance_threshold_pct,
            )
            if retried is None:
                break
            result = retried
            current_windows = retry_windows
            retry_attempts += 1

        if result is not None:
            result["initial_windows"] = num_windows
            result["final_windows"] = current_windows
            result["retry_attempts"] = retry_attempts
            results[kid] = result

    print(
        f"\n  Power replay complete: {len(results)}/{total} kernels profiled"
        f" (idle baseline {idle_baseline_w:.1f} W)"
    )
    return results, idle_baseline_w
