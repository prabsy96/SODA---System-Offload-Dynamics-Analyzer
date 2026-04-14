"""Helpers for robust replay-cache loading and pruning."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple


def _next_quarantine_path(cache_path: Path) -> Path:
    """Return a non-conflicting quarantine path for a corrupt cache file."""
    candidate = cache_path.with_name(f"{cache_path.stem}.corrupt{cache_path.suffix}")
    index = 1
    while candidate.exists():
        candidate = cache_path.with_name(
            f"{cache_path.stem}.corrupt.{index}{cache_path.suffix}"
        )
        index += 1
    return candidate


def _quarantine_cache_file(cache_path: Path) -> Path:
    """Move a corrupt cache file aside and return the quarantine path."""
    quarantine_path = _next_quarantine_path(cache_path)
    cache_path.rename(quarantine_path)
    return quarantine_path


def save_replay_cache_payload(cache_path: Path, payload: Dict[str, Any]) -> None:
    """Write replay-cache payload to disk atomically.

    Writes to a temporary file in the same directory and then atomically
    renames it into place via ``os.replace``.  This ensures readers never
    see a partially-written file.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(cache_path.parent),
        suffix=".tmp",
        prefix=cache_path.stem,
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, str(cache_path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_replay_cache_payload(
    cache_path: Path,
    expected_gpu_name: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], bool, Optional[Path]]:
    """Load replay-cache payload, salvaging torn writes when possible.

    Returns ``(payload, recovered, quarantine_path)``. ``payload`` is ``None``
    when the cache is unusable or stale for the current GPU.
    """
    if not cache_path.exists():
        return None, False, None

    text = cache_path.read_text()
    recovered = False
    quarantine_path = None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            payload, end_idx = json.JSONDecoder().raw_decode(text)
        except json.JSONDecodeError:
            quarantine_path = _quarantine_cache_file(cache_path)
            return None, False, quarantine_path

        if not isinstance(payload, dict):
            quarantine_path = _quarantine_cache_file(cache_path)
            return None, False, quarantine_path

        if text[end_idx:].strip():
            quarantine_path = _quarantine_cache_file(cache_path)
            save_replay_cache_payload(cache_path, payload)
            recovered = True
        else:
            recovered = True

    if not isinstance(payload, dict):
        quarantine_path = _quarantine_cache_file(cache_path)
        return None, False, quarantine_path

    meta = payload.get("_meta", {})
    entries = payload.get("entries", {})
    if not isinstance(meta, dict) or not isinstance(entries, dict):
        quarantine_path = _quarantine_cache_file(cache_path)
        return None, False, quarantine_path

    cached_gpu = meta.get("gpu_name", "")
    if expected_gpu_name and cached_gpu and cached_gpu != expected_gpu_name:
        return None, recovered, quarantine_path

    return payload, recovered, quarantine_path


def prune_replay_cache_file(
    cache_path: Path,
    prune_ops: Set[str],
) -> Tuple[int, int, bool, Optional[Path]]:
    """Remove stale cache entries for selected ATen ops.

    Returns ``(removed, before, recovered, quarantine_path)``.
    """
    payload, recovered, quarantine_path = load_replay_cache_payload(cache_path)
    if payload is None:
        return 0, 0, recovered, quarantine_path

    entries = payload.get("entries", {})
    before = len(entries)
    pruned = {
        key: value
        for key, value in entries.items()
        if value.get("aten_op", "") not in prune_ops
    }
    removed = before - len(pruned)
    payload["entries"] = pruned
    save_replay_cache_payload(cache_path, payload)
    return removed, before, recovered, quarantine_path