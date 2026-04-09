"""Directory-based global kernel cache for cross-job replay sharing.

Stores per-kernel replay results as individual files keyed by a SHA-256 hash
of the replay cache key.  This design eliminates lock contention: concurrent
SLURM jobs can safely read and write entries without any global lock because
each file is written atomically via ``tempfile.mkstemp`` + ``os.replace``.

Also caches the null-kernel system floor (T_sys) with a configurable TTL so
that concurrent Stage 2 jobs on the same GPU avoid redundant measurements.

Usage::

    cache = GlobalKernelCache(cache_dir, gpu_name)
    result = cache.lookup(cache_key)      # None on miss
    cache.store(cache_key, result_dict)   # atomic, immediate visibility

    floor = cache.load_t_sys(warmup=20, runs=50)
    cache.store_t_sys(floor_dict, warmup=20, runs=50)
"""

import copy
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


_ENTRY_VERSION = "1.0"


class GlobalKernelCache:
    """Per-entry file cache shared across TaxBreak pipeline runs."""

    def __init__(self, cache_dir: Path, gpu_name: str) -> None:
        self.cache_dir = Path(cache_dir)
        self.entries_dir = self.cache_dir / "entries"
        self.gpu_name = gpu_name
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Entry hashing
    # ------------------------------------------------------------------

    @staticmethod
    def _entry_hash(cache_key: str) -> str:
        """Return a 16-char hex digest of *cache_key*."""
        return hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]

    def _entry_path(self, cache_key: str) -> Path:
        """Return the filesystem path for a given cache key."""
        return self.entries_dir / f"{self._entry_hash(cache_key)}.json"

    # ------------------------------------------------------------------
    # Lookup / store
    # ------------------------------------------------------------------

    def lookup(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Return the cached replay result for *cache_key*, or ``None``."""
        path = self._entry_path(cache_key)
        if not path.exists():
            self._misses += 1
            return None
        try:
            data = json.loads(path.read_text())
            meta = data.get("_meta", {})
            if meta.get("gpu_name", "") != self.gpu_name:
                self._misses += 1
                return None
            if meta.get("version", "") != _ENTRY_VERSION:
                self._misses += 1
                return None
            self._hits += 1
            return copy.deepcopy(data["result"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            self._misses += 1
            return None

    def store(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Atomically write a replay result to the cache directory."""
        try:
            self.entries_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "_meta": {
                    "gpu_name": self.gpu_name,
                    "version": _ENTRY_VERSION,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                "cache_key": cache_key,
                "result": result,
            }
            path = self._entry_path(cache_key)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self.entries_dir),
                suffix=".tmp",
                prefix="entry_",
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(payload, f, indent=2)
                os.replace(tmp_path, str(path))
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as exc:
            print(f"  Warning: failed to store global cache entry: {exc}")

    # ------------------------------------------------------------------
    # T_sys caching
    # ------------------------------------------------------------------

    def _t_sys_path(self) -> Path:
        return self.cache_dir / "t_sys.json"

    def load_t_sys(
        self,
        warmup: int,
        runs: int,
        num_gpus: int = 1,
        ttl_hours: float = 24.0,
    ) -> Optional[Dict[str, Any]]:
        """Load a cached T_sys measurement, or ``None`` if stale/missing."""
        path = self._t_sys_path()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            meta = data.get("_meta", {})
            if meta.get("gpu_name", "") != self.gpu_name:
                return None
            cfg = data.get("config", {})
            if cfg.get("warmup") != warmup or cfg.get("runs") != runs:
                return None
            if cfg.get("num_gpus", 1) != num_gpus:
                return None
            ts = datetime.fromisoformat(data["timestamp"])
            elapsed_hours = (
                datetime.now(timezone.utc) - ts
            ).total_seconds() / 3600.0
            if elapsed_hours > ttl_hours:
                return None
            return data["floor"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    def store_t_sys(
        self,
        floor: Dict[str, Any],
        warmup: int,
        runs: int,
        num_gpus: int = 1,
        ttl_hours: float = 24.0,
    ) -> None:
        """Atomically cache a T_sys measurement."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "_meta": {
                    "gpu_name": self.gpu_name,
                    "version": _ENTRY_VERSION,
                },
                "floor": floor,
                "config": {
                    "warmup": warmup,
                    "runs": runs,
                    "num_gpus": num_gpus,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ttl_hours": ttl_hours,
            }
            path = self._t_sys_path()
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self.cache_dir),
                suffix=".tmp",
                prefix="t_sys_",
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(payload, f, indent=2)
                os.replace(tmp_path, str(path))
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as exc:
            print(f"  Warning: failed to store T_sys cache: {exc}")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        """Return cache hit/miss counters."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": self._hits + self._misses,
        }


class NullGlobalCache:
    """No-op cache used when ``--no-global-cache`` is set."""

    def lookup(self, cache_key: str) -> Optional[Dict[str, Any]]:
        return None

    def store(self, cache_key: str, result: Dict[str, Any]) -> None:
        pass

    def load_t_sys(
        self,
        warmup: int,
        runs: int,
        num_gpus: int = 1,
        ttl_hours: float = 24.0,
    ) -> Optional[Dict[str, Any]]:
        return None

    def store_t_sys(
        self,
        floor: Dict[str, Any],
        warmup: int,
        runs: int,
        num_gpus: int = 1,
        ttl_hours: float = 24.0,
    ) -> None:
        pass

    def stats(self) -> Dict[str, int]:
        return {"hits": 0, "misses": 0, "total": 0}
