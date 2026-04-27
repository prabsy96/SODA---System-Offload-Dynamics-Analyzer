"""
Enhanced TaxBreak pipeline orchestrator.

Reads a kernel database (from Phase 1), measures the dynamic system floor
(Phase 2), replays each kernel in isolation under nsys (Phase 3), optionally
profiles top-N kernels with ncu (Phase 4), and writes an enhanced report
(Phase 6).

Usage (Stage 2 — no model loading required):
    soda-cli --taxbreak --kernel-db-path <path> [--ncu] [--ncu-top-n 10] [--ncu-all-kernels]
"""

import copy
import json
import os
import queue
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from soda.common import utils, print_utils
from soda.taxbreak.null_kernel import measure_system_floor
from soda.taxbreak.nsys_replay import nsys_profile_pytorch_kernel
from soda.taxbreak.replay_cache_tools import (
    load_replay_cache_payload,
    save_replay_cache_payload,
)
from soda.taxbreak.global_cache import GlobalKernelCache, NullGlobalCache
from soda.taxbreak.report import generate_enhanced_report


def _duration_class_from_entry(entry: Dict[str, Any]) -> str:
    """Return coarse kernel duration class for selection and replay policy."""
    avg_dur_us = entry.get("statistics", {}).get("avg_duration_us")
    if avg_dur_us is None:
        avg_dur_us = entry.get("statistics", {}).get("total_duration_us", 10.0)
    if avg_dur_us is None:
        avg_dur_us = 10.0

    if avg_dur_us < 1.0:
        return "ultra_short"
    if avg_dur_us < 50.0:
        return "short"
    if avg_dur_us < 500.0:
        return "medium"
    return "long"


class TaxBreakPipeline:
    """Orchestrates the enhanced TaxBreak analysis pipeline."""

    def __init__(self, kernel_db_path: Path, args):
        self.kernel_db_path = Path(kernel_db_path)
        self.db = utils.load_json(str(self.kernel_db_path))
        self.args = args
        self.output_dir = self.kernel_db_path.parent / "taxbreak"
        self.num_gpus = self.db.get("metadata", {}).get("num_gpus", 1)
        self.global_cache_dir = self._resolve_global_cache_dir()

        # Instantiate directory-based global cache (or null cache if disabled)
        gpu_name = self.db.get("metadata", {}).get("gpu_name", "") or "unknown"
        if getattr(args, "no_global_cache", False):
            self.global_cache = NullGlobalCache()
        else:
            custom_dir = getattr(args, "global_cache_dir", None)
            if custom_dir:
                cache_dir = Path(custom_dir) / self._gpu_cache_slug()
            else:
                cache_dir = self.global_cache_dir
            self.global_cache = GlobalKernelCache(cache_dir, gpu_name)

    def run(self) -> Path:
        """Execute the full pipeline and return the report path.

        Steps:
            1. Measure dynamic system floor (null kernel)
            2. Isolation replay + nsys for each unique kernel
            3. (Optional) ncu profiling on top-N kernels by duration
            4. Generate enhanced TaxBreak report
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        metadata = self.db.get("metadata", {})
        summary = self.db.get("summary", {})
        kernels = self.db.get("kernels", [])

        print(f"Enhanced TaxBreak Pipeline")
        print(f"  Kernel DB : {self.kernel_db_path}")
        print(f"  Model     : {metadata.get('model', 'unknown')}")
        print(f"  Kernels   : {len(kernels)} unique")
        print(f"  Output    : {self.output_dir}")
        print()

        warmup = max(0, getattr(self.args, "warmup", 20))
        runs = max(1, getattr(self.args, "runs", 50))

        # --- Step 1: Dynamic system floor ---
        section = "Dynamic System Floor"
        print_utils.section_start(section)

        cached_floor = self.global_cache.load_t_sys(
            warmup=warmup, runs=runs, num_gpus=self.num_gpus,
        )
        if cached_floor is not None:
            floor = cached_floor
            floor["method"] = "cached"
            print(
                f"  System floor (cached): avg={floor['avg_us']:.2f} us, "
                f"std={floor.get('std_us', 0):.2f} us"
            )
        else:
            floor = measure_system_floor(
                warmup=warmup,
                runs=runs,
                num_gpus=self.num_gpus,
            )
            self.global_cache.store_t_sys(
                floor, warmup=warmup, runs=runs, num_gpus=self.num_gpus,
            )

        print_utils.section_end(section)

        # --- Step 1.5: Active-idle power (null-kernel tight-loop) ---
        # Only measured when --power-replay is requested; stored in floor dict
        # so generate_enhanced_report() can pass it to _write_power_report().
        if getattr(self.args, "power_replay", False):
            from soda.taxbreak.null_kernel import measure_active_idle_power
            ai_result = measure_active_idle_power(
                warmup_ms=getattr(self.args, "power_replay_warmup_ms", 1000),
                num_windows=getattr(self.args, "power_replay_windows", 3),
                window_ms=getattr(self.args, "power_replay_target_ms", 500),
                interval_ms=getattr(self.args, "power_replay_interval", 50),
            )
            if ai_result is not None:
                floor["active_idle_power_w"] = ai_result["active_idle_power_w"]
                floor["active_idle_std_w"] = ai_result["std_w"]
                floor["active_idle_method"] = ai_result["measurement_method"]

        # --- Step 2: Isolation replay (nsys) for each kernel ---
        section = "Isolation Replay (nsys)"
        print_utils.section_start(section)
        nsys_results = self._run_nsys_replay(kernels)
        cache_hits = getattr(self, "_cache_hits", 0)
        global_cache_hits = getattr(self, "_global_cache_hits", 0)
        if cache_hits:
            profiled = len(kernels) - cache_hits
            local_hits = cache_hits - global_cache_hits
            parts = []
            if local_hits:
                parts.append(f"{local_hits} local-cached")
            if global_cache_hits:
                parts.append(f"{global_cache_hits} global-cached")
            parts.append(f"{profiled} profiled")
            print(
                f"\nCompleted nsys replay: {len(nsys_results)}/{len(kernels)} kernels "
                f"({', '.join(parts)})"
            )
        else:
            print(f"\nCompleted nsys replay: {len(nsys_results)}/{len(kernels)} kernels")
        print_utils.section_end(section)

        # --- Step 3: Optional ncu profiling ---
        ncu_results: Dict[str, Any] = {}
        if getattr(self.args, "ncu", False):
            section = "NCU Profiling"
            print_utils.section_start(section)
            ncu_results = self._run_ncu(kernels)
            print(f"\nCompleted ncu: {len(ncu_results)} kernels")
            print_utils.section_end(section)

        # --- Step 3.5: Optional per-kernel power replay ---
        power_replay_results: Dict[str, Any] = {}
        power_idle_baseline_w: float = 0.0
        if getattr(self.args, "power_replay", False):
            section = "Per-Kernel Power Replay"
            print_utils.section_start(section)
            power_replay_results, power_idle_baseline_w = self._run_kernel_power_replay(kernels)
            print_utils.section_end(section)

        # --- Step 4: Generate report ---
        section = "Enhanced Report"
        print_utils.section_start(section)
        report_path = self._generate_report(
            floor, nsys_results, ncu_results,
            power_replay_results, power_idle_baseline_w,
        )
        print_utils.section_end(section)

        return report_path

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _make_gpu_queue(self) -> "queue.Queue[int]":
        """Return a queue pre-loaded with one token per GPU."""
        q: queue.Queue[int] = queue.Queue()
        for gpu_id in range(self.num_gpus):
            q.put(gpu_id)
        return q

    def _infer_output_root(self) -> Path:
        """Infer the shared output root that should own the global cache.

        Preference order:
        1. Explicit ``SODA_OUTPUT`` override
        2. Repository root output dir from ``SODA_ROOT``
        3. Nearest ancestor named ``output`` in ``kernel_db_path``
        4. Fallback to the kernel DB's great-grandparent directory
        """
        configured_output = os.environ.get("SODA_OUTPUT")
        if configured_output:
            return Path(configured_output).expanduser().resolve()

        soda_root = os.environ.get("SODA_ROOT")
        if soda_root:
            return (Path(soda_root).expanduser().resolve() / "output")

        for parent in self.kernel_db_path.parents:
            if parent.name == "output":
                return parent.resolve()

        try:
            return self.kernel_db_path.parents[3].resolve()
        except IndexError:
            return self.kernel_db_path.parent.parent.resolve()

    def _gpu_cache_slug(self) -> str:
        """Return a filesystem-safe GPU slug for shared cache partitioning."""
        gpu_name = self.db.get("metadata", {}).get("gpu_name", "") or "unknown"
        slug = re.sub(r"[^A-Za-z0-9]+", "_", gpu_name).strip("_")
        return slug or "unknown"

    def _resolve_global_cache_dir(self) -> Path:
        """Return the cross-job global cache directory for this GPU family."""
        return self._infer_output_root() / ".global_kernel_cache" / self._gpu_cache_slug()

    # ------------------------------------------------------------------
    # Replay cache helpers
    # ------------------------------------------------------------------

    def _make_replay_cache_key(self, entry: Dict[str, Any]) -> str:
        """Build a deterministic cache key for replay deduplication.

        The key captures the ATen-op configuration (which determines the
        replay script) and the target kernel name (which determines kernel
        matching in the nsys trace).  Grid/block are included because when
        all input_dims are empty, ``nsys_profile_pytorch_kernel`` infers
        tensor size from ``grid[0] * block[0]``.

        Two entries with the same cache key will produce identical replay
        results.
        """
        aten_op = entry.get("aten_op", {})
        kernel = entry.get("kernel", {})
        key_tuple = (
            aten_op.get("name", ""),
            utils.to_hashable(aten_op.get("input_dims", [])),
            utils.to_hashable(aten_op.get("input_type", [])),
            utils.to_hashable(aten_op.get("concrete_inputs", [])),
            kernel.get("name", ""),
            utils.to_hashable(kernel.get("grid", [0, 0, 0])),
            utils.to_hashable(kernel.get("block", [0, 0, 0])),
        )
        return str(key_tuple)

    def _load_replay_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load persistent replay cache from a prior run.

        Validates that the GPU name matches the current kernel DB.  Returns
        an empty dict if the cache is missing, corrupt, or stale.
        """
        cache_path = self.output_dir / "replay_cache.json"
        try:
            current_gpu = self.db.get("metadata", {}).get("gpu_name", "")
            data, recovered, quarantine_path = load_replay_cache_payload(
                cache_path,
                expected_gpu_name=current_gpu,
            )
            if data is None:
                if quarantine_path is not None:
                    print(
                        f"  Replay cache corrupt; moved aside to {quarantine_path}"
                    )
                return {}
            if recovered:
                print(f"  Recovered replay cache from torn write at {cache_path}")
            meta = data.get("_meta", {})
            cached_gpu = meta.get("gpu_name", "")
            if cached_gpu and current_gpu and cached_gpu != current_gpu:
                print(
                    f"  Replay cache invalidated: GPU changed "
                    f"({cached_gpu} -> {current_gpu})"
                )
                return {}
            entries = data.get("entries", {})
            if entries:
                print(f"  Loaded {len(entries)} cached replay results from prior run")
            return entries
        except Exception:
            return {}

    def _save_replay_cache(
        self, cache: Dict[str, Dict[str, Any]]
    ) -> None:
        """Persist the replay cache to disk for future runs."""
        try:
            save_replay_cache_payload(
                self.output_dir / "replay_cache.json",
                {
                    "_meta": {
                        "gpu_name": self.db.get("metadata", {}).get("gpu_name", ""),
                        "version": "1.0",
                    },
                    "entries": cache,
                },
            )
        except Exception as exc:
            print(f"  Warning: failed to save replay cache: {exc}")

    # ------------------------------------------------------------------
    # nsys replay
    # ------------------------------------------------------------------

    def _run_nsys_replay(
        self, kernels: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Replay each kernel in isolation under nsys.

        Uses a three-level cache to skip redundant replays:

        1. **Local persistent cache** (``replay_cache.json``): results from
           prior pipeline runs in the same experiment directory.
        2. **Global directory cache**: per-entry files shared across all
           experiments on the same GPU.  Entries are written immediately
           after each successful replay, giving other concurrent SLURM jobs
           instant visibility.
        3. **In-memory cache**: entries profiled earlier in this run whose
           ATen-op configuration and kernel name match a later entry.

        Single GPU: runs serially — the GPU is the serialisation point and
        there is nothing to overlap.

        Multi-GPU: runs ``num_gpus`` replays concurrently, one per GPU, using
        ``CUDA_VISIBLE_DEVICES`` to pin each subprocess to a specific device.
        A ``queue.Queue`` of GPU IDs acts as a per-device semaphore so that at
        most one replay occupies each GPU at any time.
        """
        results: Dict[str, Dict[str, Any]] = {}
        total = len(kernels)
        warmup = max(0, getattr(self.args, "warmup", 20))
        runs = max(1, getattr(self.args, "runs", 50))

        # ── load local persistent cache ──────────────────────────────────
        cache: Dict[str, Dict[str, Any]] = self._load_replay_cache()
        cache_hits = 0
        global_cache_hits = 0

        if self.num_gpus <= 1:
            # ── serial path ──────────────────────────────────────────────
            for i, entry in enumerate(kernels, 1):
                kid   = entry["id"]
                op    = entry["aten_op"].get("name", "?")
                kname = entry["kernel"]["name"]
                cache_key = self._make_replay_cache_key(entry)

                # Level 1: in-memory / local persistent cache
                if cache_key in cache:
                    cached = copy.deepcopy(cache[cache_key])
                    cached["kernel_id"] = kid
                    cached["cached"] = True
                    results[kid] = cached
                    cache_hits += 1
                    print(f"[{i}/{total}] {kid}: {op} -> {kname}  [CACHED]")
                    continue

                # Level 2: global directory cache
                global_result = self.global_cache.lookup(cache_key)
                if global_result is not None:
                    global_result["kernel_id"] = kid
                    global_result["cached"] = True
                    results[kid] = global_result
                    cache[cache_key] = copy.deepcopy(global_result)
                    cache_hits += 1
                    global_cache_hits += 1
                    print(f"[{i}/{total}] {kid}: {op} -> {kname}  [GLOBAL CACHE]")
                    continue

                # Level 3: profile via nsys
                print(f"[{i}/{total}] {kid}: {op} -> {kname}")
                result = nsys_profile_pytorch_kernel(
                    entry,
                    warmup=warmup,
                    runs=runs,
                )
                if result is not None:
                    results[kid] = result
                    cache[cache_key] = copy.deepcopy(result)
                    self.global_cache.store(cache_key, result)

            self._save_replay_cache(cache)
            self._cache_hits = cache_hits
            self._cache_misses = total - cache_hits
            self._global_cache_hits = global_cache_hits
            return results

        # ── parallel path (multi-GPU) ─────────────────────────────────────
        gpu_queue = self._make_gpu_queue()
        lock = threading.Lock()
        dispatched_count = 0  # protected by lock; nonlocal in _replay

        def _replay(entry: Dict[str, Any]):
            nonlocal dispatched_count, cache_hits, global_cache_hits
            kid   = entry["id"]
            op    = entry["aten_op"].get("name", "?")
            kname = entry["kernel"]["name"]
            cache_key = self._make_replay_cache_key(entry)

            with lock:
                dispatched_count += 1
                n = dispatched_count

                # Level 1: in-memory / local persistent cache
                if cache_key in cache:
                    cached = copy.deepcopy(cache[cache_key])
                    cached["kernel_id"] = kid
                    cached["cached"] = True
                    cache_hits += 1
                    print(f"[{n}/{total}] {kid}: {op} -> {kname}  [CACHED]")
                    return kid, cached

                # Level 2: global directory cache
                global_result = self.global_cache.lookup(cache_key)
                if global_result is not None:
                    global_result["kernel_id"] = kid
                    global_result["cached"] = True
                    cache[cache_key] = copy.deepcopy(global_result)
                    cache_hits += 1
                    global_cache_hits += 1
                    print(f"[{n}/{total}] {kid}: {op} -> {kname}  [GLOBAL CACHE]")
                    return kid, global_result

            # Level 3: profile via nsys
            gpu_id = gpu_queue.get()   # blocks until a GPU token is free
            print(f"[{n}/{total}] {kid}: {op} -> {kname}  [GPU {gpu_id}]")
            try:
                extra_env = dict(os.environ)
                extra_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                result = nsys_profile_pytorch_kernel(
                    entry,
                    warmup=warmup,
                    runs=runs,
                    extra_env=extra_env,
                )
                if result is not None:
                    with lock:
                        cache[cache_key] = copy.deepcopy(result)
                    self.global_cache.store(cache_key, result)
                return kid, result
            finally:
                gpu_queue.put(gpu_id)  # always return the GPU token

        with ThreadPoolExecutor(max_workers=self.num_gpus) as pool:
            futures = [pool.submit(_replay, entry) for entry in kernels]
            for future in as_completed(futures):
                try:
                    kid, result = future.result()
                    if result is not None:
                        results[kid] = result
                except Exception as exc:  # noqa: BLE001
                    print(f"  nsys replay error: {exc}")

        self._save_replay_cache(cache)
        self._cache_hits = cache_hits
        self._cache_misses = total - cache_hits
        self._global_cache_hits = global_cache_hits
        return results

    def _run_ncu(
        self, kernels: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Run ncu on selected kernels.

        Applies the same single-GPU serial / multi-GPU parallel split as
        ``_run_nsys_replay``.

        Selection modes:
          - default: top-N kernels by total duration (``--ncu-top-n``)
          - all-kernels: enabled by ``--ncu-all-kernels``
                    - class-targeted: ``--ncu-target-classes`` (budgeted per class)
        """
        from soda.ncu import ncu_check_available, ncu_profile_kernel

        if not ncu_check_available():
            print("Skipping ncu profiling (ncu not available).")
            return {}

        profile_all = getattr(self.args, "ncu_all_kernels", False)
        ranked = sorted(
            kernels,
            key=lambda k: k["statistics"]["total_duration_us"],
            reverse=True,
        )
        target_classes = getattr(self.args, "ncu_target_classes", None)
        per_class_n = getattr(self.args, "ncu_per_class", 5)
        if profile_all:
            selected = ranked
            print(f"  NCU scope: all kernels ({len(selected)})")
        elif target_classes:
            allowed = {c.strip() for c in target_classes if c.strip()}
            if "all" in allowed:
                allowed = {"ultra_short", "short", "medium", "long"}

            buckets: Dict[str, List[Dict[str, Any]]] = {
                "ultra_short": [],
                "short": [],
                "medium": [],
                "long": [],
            }
            for entry in ranked:
                klass = _duration_class_from_entry(entry)
                if klass in buckets:
                    buckets[klass].append(entry)

            selected = []
            for klass in ("ultra_short", "short", "medium", "long"):
                if klass not in allowed:
                    continue
                selected.extend(buckets[klass][:per_class_n])

            # De-duplicate in case of malformed class overlap and preserve order.
            seen = set()
            deduped = []
            for entry in selected:
                kid = entry["id"]
                if kid in seen:
                    continue
                seen.add(kid)
                deduped.append(entry)
            selected = deduped
            print(
                "  NCU scope: class-targeted "
                f"(classes={sorted(allowed)}, per_class={per_class_n}, total={len(selected)})"
            )
        else:
            top_n = getattr(self.args, "ncu_top_n", 10)
            selected = ranked[:top_n]
            print(f"  NCU scope: top-{top_n} by duration ({len(selected)})")

        ncu_dir = self.output_dir / "ncu"
        ncu_dir.mkdir(parents=True, exist_ok=True)

        results: Dict[str, Dict[str, Any]] = {}
        total = len(selected)

        if self.num_gpus <= 1:
            # ── serial path ──────────────────────────────────────────────
            for i, entry in enumerate(selected, 1):
                kid   = entry["id"]
                op    = entry["aten_op"].get("name", "?")
                kname = entry["kernel"]["name"]
                print(f"[{i}/{total}] {kid}: {op} -> {kname}")
                result = ncu_profile_kernel(entry, output_dir=ncu_dir)
                if result is not None:
                    results[kid] = result
            return results

        # ── parallel path (multi-GPU) ─────────────────────────────────────
        gpu_queue = self._make_gpu_queue()
        lock = threading.Lock()
        dispatched_count = 0  # protected by lock; nonlocal in _ncu

        def _ncu(entry: Dict[str, Any]):
            nonlocal dispatched_count
            gpu_id = gpu_queue.get()
            kid   = entry["id"]
            op    = entry["aten_op"].get("name", "?")
            kname = entry["kernel"]["name"]
            with lock:
                dispatched_count += 1
                n = dispatched_count
            print(f"[{n}/{total}] {kid}: {op} -> {kname}  [GPU {gpu_id}]")
            try:
                extra_env = dict(os.environ)
                extra_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                return kid, ncu_profile_kernel(
                    entry, output_dir=ncu_dir, extra_env=extra_env
                )
            finally:
                gpu_queue.put(gpu_id)

        with ThreadPoolExecutor(max_workers=self.num_gpus) as pool:
            futures = [pool.submit(_ncu, entry) for entry in selected]
            for future in as_completed(futures):
                try:
                    kid, result = future.result()
                    if result is not None:
                        results[kid] = result
                except Exception as exc:  # noqa: BLE001
                    print(f"  ncu error: {exc}")

        return results

    def _run_kernel_power_replay(
        self, kernels: List[Dict[str, Any]]
    ) -> "tuple[Dict[str, Dict[str, Any]], float]":
        """Run per-kernel NVML power replay for all unique kernels.

        Power replay is always serial — running kernels in parallel on separate
        GPUs would contaminate per-GPU NVML package-power readings.

        Returns:
            (results_dict, idle_baseline_w) where results_dict maps
            kernel_id → power result dict.
        """
        from soda.taxbreak.kernel_power_replay import power_profile_all_kernels

        max_kernels = getattr(self.args, "power_replay_max_kernels", None)
        entries = kernels
        if max_kernels is not None:
            entries = kernels[:max_kernels]

        total = len(entries)
        print(
            f"  Profiling power for {total} unique kernel"
            f"{'s' if total != 1 else ''} "
            f"(warmup={getattr(self.args, 'power_replay_warmup_ms', 500)} ms, "
            f"windows={getattr(self.args, 'power_replay_windows', 3)}×"
            f"{getattr(self.args, 'power_replay_target_ms', 300)} ms, "
            f"idle_settle={getattr(self.args, 'power_replay_idle_settle_min_ms', 1500)}-"
            f"{getattr(self.args, 'power_replay_idle_settle_max_ms', 6000)} ms)"
        )

        gpu_ids = list(range(self.num_gpus))
        results, idle_w = power_profile_all_kernels(
            kernel_db_entries=entries,
            output_dir=self.output_dir,
            gpu_ids=gpu_ids,
            target_warmup_ms=getattr(self.args, "power_replay_warmup_ms", 500),
            target_meas_ms=getattr(self.args, "power_replay_target_ms", 300),
            num_windows=getattr(self.args, "power_replay_windows", 3),
            interval_ms=getattr(self.args, "power_replay_interval", 50),
            max_kernels=None,  # already sliced above
            idle_settle_min_ms=getattr(self.args, "power_replay_idle_settle_min_ms", 1500),
            idle_settle_max_ms=getattr(self.args, "power_replay_idle_settle_max_ms", 6000),
            idle_settle_step_ms=getattr(self.args, "power_replay_idle_settle_step_ms", 500),
            idle_settle_std_threshold_pct=getattr(
                self.args,
                "power_replay_idle_std_threshold_pct",
                3.0,
            ),
        )
        return results, idle_w

    def _generate_report(
        self,
        floor: Dict[str, Any],
        nsys_results: Dict[str, Any],
        ncu_results: Dict[str, Any],
        power_replay_results: Optional["Dict[str, Any]"] = None,
        power_idle_baseline_w: float = 0.0,
    ) -> Path:
        """Delegate to the report module (Phase 6)."""
        return generate_enhanced_report(
            kernel_db=self.db,
            floor=floor,
            nsys_results=nsys_results,
            ncu_results=ncu_results,
            output_dir=self.output_dir,
            verbose=getattr(self.args, "verbose", False),
            power_replay_results=power_replay_results,
            power_idle_baseline_w=power_idle_baseline_w,
        )
