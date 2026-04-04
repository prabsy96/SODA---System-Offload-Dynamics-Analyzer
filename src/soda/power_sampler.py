"""
GPU power sampling via NVML (pynvml) or nvidia-smi fallback.

Provides a background thread that polls GPU power draw during an inference
profiling window, yielding measured mean/peak/min/std power in watts.

Two backends (tried in order):
  1. pynvml  — Python binding to NVML; ~50 ms polling interval supported.
               Install with: pip install pynvml
               # TODO: verify pynvml availability on target cluster before use.
  2. nvidia-smi — subprocess fallback; higher per-sample overhead, minimum
                  ~200 ms effective interval.

Usage (as context manager)::

    from soda.power_sampler import make_power_sampler

    sampler = make_power_sampler(gpu_ids=[0], interval_ms=50, enabled=True)
    sampler.start()
    # ... run inference ...
    sampler.stop()
    results = sampler.get_results()
    # results["mean_power_w"], results["peak_power_w"], ...

Returns an empty no-op sampler when disabled or no backend is available —
callers never need to guard against None.
"""

from __future__ import annotations

import shutil
import statistics
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _try_import_pynvml() -> Optional[Any]:
    """Return the pynvml module if importable, else None."""
    try:
        import pynvml  # type: ignore[import]
        return pynvml
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# No-op sentinel (returned when sampling is disabled or unavailable)
# ---------------------------------------------------------------------------

class _NoOpSampler:
    """Sampler stub with identical interface; all operations are no-ops."""

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def get_results(self) -> Dict[str, Any]:
        return {"available": False, "backend": "none"}

    def __enter__(self) -> "_NoOpSampler":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Main sampler
# ---------------------------------------------------------------------------

class NVMLPowerSampler:
    """Background GPU power sampler using pynvml (primary) or nvidia-smi.

    Args:
        gpu_ids:     List of integer GPU device indices to sample.
        interval_ms: Polling interval in milliseconds (>= 10).
    """

    def __init__(self, gpu_ids: List[int], interval_ms: int = 50) -> None:
        if interval_ms < 10:
            raise ValueError(f"interval_ms must be >= 10, got {interval_ms}")
        self._gpu_ids = gpu_ids
        self._interval_ms = interval_ms
        self._interval_s = interval_ms / 1000.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Per-GPU samples: {gpu_id: [(timestamp_ms, power_w), ...]}
        self._samples: Dict[int, List[Tuple[float, float]]] = {
            g: [] for g in gpu_ids
        }
        self._backend: str = "none"
        self._pynvml = _try_import_pynvml()

    # ------------------------------------------------------------------
    # Polling loops (run in background thread)
    # ------------------------------------------------------------------

    def _poll_pynvml(self) -> None:
        """Poll NVML for power readings via pynvml."""
        pynvml = self._pynvml
        # TODO: verify pynvml.nvmlInit() is stable on target cluster
        pynvml.nvmlInit()
        try:
            handles = {
                g: pynvml.nvmlDeviceGetHandleByIndex(g) for g in self._gpu_ids
            }
            while not self._stop_event.is_set():
                t_ms = time.monotonic() * 1000.0
                for g, h in handles.items():
                    try:
                        # nvmlDeviceGetPowerUsage returns milliwatts
                        # TODO: verify nvmlDeviceGetPowerUsage on H200 NVL
                        mw = pynvml.nvmlDeviceGetPowerUsage(h)
                        self._samples[g].append((t_ms, mw / 1000.0))
                    except Exception:  # noqa: BLE001 — non-fatal per-sample failure
                        pass
                self._stop_event.wait(timeout=self._interval_s)
        finally:
            pynvml.nvmlShutdown()

    def _poll_nvidia_smi(self) -> None:
        """Poll GPU power via nvidia-smi subprocess.

        Each call launches a subprocess; effective minimum interval is ~200 ms
        due to process startup overhead.  Minimum enforced at 200 ms.
        """
        effective_interval_s = max(0.2, self._interval_s)
        id_str = ",".join(str(g) for g in self._gpu_ids)
        cmd = [
            "nvidia-smi",
            f"--id={id_str}",
            "--query-gpu=index,power.draw",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop_event.is_set():
            t_ms = time.monotonic() * 1000.0
            try:
                import subprocess
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                for line in result.stdout.strip().splitlines():
                    parts = line.split(",")
                    if len(parts) == 2:
                        try:
                            gpu_id = int(parts[0].strip())
                            power_w = float(parts[1].strip())
                            if gpu_id in self._samples:
                                self._samples[gpu_id].append((t_ms, power_w))
                        except ValueError:
                            pass
            except Exception:  # noqa: BLE001  — any subprocess failure is non-fatal
                pass
            self._stop_event.wait(timeout=effective_interval_s)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the background polling thread."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        if self._pynvml is not None:
            self._backend = "pynvml"
            target = self._poll_pynvml
        else:
            self._backend = "nvidia-smi"
            target = self._poll_nvidia_smi
        self._thread = threading.Thread(target=target, daemon=True, name="soda-power-sampler")
        self._thread.start()

    def stop(self) -> None:
        """Signal the polling thread to stop and wait for it to join."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                print(
                    "Warning: power sampler thread did not join within 5 s",
                    file=sys.stderr,
                )
            self._thread = None

    def get_energy_counter_mj(self, gpu_id: int = 0) -> Optional[float]:
        """Return the current cumulative energy counter in millijoules.

        Uses ``nvmlDeviceGetTotalEnergyConsumption``, which is hardware-
        integrated (not sampled), giving exact energy since driver load.
        Reading the counter before and after a workload and taking the
        difference gives ground-truth energy without polling uncertainty.

        Returns None when pynvml is unavailable, the counter is not
        supported on this GPU, or any other error occurs.

        Note: ``nvmlDeviceGetTotalEnergyConsumption`` returns millijoules.
        """
        if self._pynvml is None:
            return None
        try:
            pynvml = self._pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            return float(pynvml.nvmlDeviceGetTotalEnergyConsumption(h))  # mJ
        except Exception:
            return None

    def get_results(self) -> Dict[str, Any]:
        """Return aggregated power statistics.

        Returns:
            Dict with keys:
                available (bool), backend (str), interval_ms (int),
                sample_count (int), mean_power_w (float), peak_power_w (float),
                min_power_w (float), std_power_w (float),
                per_gpu (dict[str, {mean_w, peak_w}]).
        """
        all_watts: List[float] = []
        per_gpu: Dict[str, Dict[str, float]] = {}

        for g, readings in self._samples.items():
            watts = [w for _, w in readings]
            if watts:
                per_gpu[str(g)] = {
                    "mean_w": round(statistics.mean(watts), 2),
                    "peak_w": round(max(watts), 2),
                }
                all_watts.extend(watts)

        if not all_watts:
            return {
                "available": self._backend != "none",
                "backend": self._backend,
                "interval_ms": self._interval_ms,
                "sample_count": 0,
                "mean_power_w": 0.0,
                "peak_power_w": 0.0,
                "min_power_w": 0.0,
                "std_power_w": 0.0,
                "per_gpu": per_gpu,
            }

        std = statistics.stdev(all_watts) if len(all_watts) > 1 else 0.0
        return {
            "available": True,
            "backend": self._backend,
            "interval_ms": self._interval_ms,
            "sample_count": len(all_watts),
            "mean_power_w": round(statistics.mean(all_watts), 2),
            "peak_power_w": round(max(all_watts), 2),
            "min_power_w": round(min(all_watts), 2),
            "std_power_w": round(std, 2),
            "per_gpu": per_gpu,
        }

    def __enter__(self) -> "NVMLPowerSampler":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_power_sampler(
    gpu_ids: List[int],
    interval_ms: int = 50,
    enabled: bool = True,
) -> "NVMLPowerSampler | _NoOpSampler":
    """Create a power sampler, returning a no-op if disabled or unavailable.

    Args:
        gpu_ids:     GPU device indices to sample (e.g. [0] or [0, 1]).
        interval_ms: Polling interval in milliseconds (>= 10).
        enabled:     If False, returns a no-op sampler immediately.

    Returns:
        NVMLPowerSampler if a backend is available and enabled=True,
        otherwise a _NoOpSampler with identical interface.
    """
    if not enabled:
        return _NoOpSampler()

    has_pynvml = _try_import_pynvml() is not None
    has_nvidia_smi = shutil.which("nvidia-smi") is not None

    if not has_pynvml and not has_nvidia_smi:
        print(
            "Warning: --power-sample requested but neither pynvml nor nvidia-smi "
            "is available. Install pynvml with: pip install pynvml"
        )
        return _NoOpSampler()

    return NVMLPowerSampler(gpu_ids=gpu_ids, interval_ms=interval_ms)
