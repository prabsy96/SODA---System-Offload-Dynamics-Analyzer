"""
Enhanced TaxBreak pipeline orchestrator.

Reads a kernel database (from Phase 1), measures the dynamic system floor
(Phase 2), replays each kernel in isolation under nsys (Phase 3), optionally
profiles top-N kernels with ncu (Phase 4), and writes an enhanced report
(Phase 6).

Usage (Stage 2 — no model loading required):
    soda-cli --taxbreak --kernel-db-path <path> [--ncu] [--ncu-top-n 10]
"""

import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from soda.common import utils, print_utils
from soda.taxbreak.null_kernel import measure_system_floor
from soda.taxbreak.nsys_replay import nsys_profile_pytorch_kernel
from soda.taxbreak.report import generate_enhanced_report


class TaxBreakPipeline:
    """Orchestrates the enhanced TaxBreak analysis pipeline."""

    def __init__(self, kernel_db_path: Path, args):
        self.kernel_db_path = Path(kernel_db_path)
        self.db = utils.load_json(str(self.kernel_db_path))
        self.args = args
        self.output_dir = self.kernel_db_path.parent / "taxbreak"
        self.num_gpus = self.db.get("metadata", {}).get("num_gpus", 1)

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

        # --- Step 1: Dynamic system floor ---
        section = "Dynamic System Floor"
        print_utils.section_start(section)
        floor = measure_system_floor(num_gpus=self.num_gpus)
        print_utils.section_end(section)

        # --- Step 2: Isolation replay (nsys) for each kernel ---
        section = "Isolation Replay (nsys)"
        print_utils.section_start(section)
        nsys_results = self._run_nsys_replay(kernels)
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

        # --- Step 4: Generate report ---
        section = "Enhanced Report"
        print_utils.section_start(section)
        report_path = self._generate_report(floor, nsys_results, ncu_results)
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

    def _run_nsys_replay(
        self, kernels: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Replay each kernel in isolation under nsys.

        Single GPU: runs serially — the GPU is the serialisation point and
        there is nothing to overlap.

        Multi-GPU: runs ``num_gpus`` replays concurrently, one per GPU, using
        ``CUDA_VISIBLE_DEVICES`` to pin each subprocess to a specific device.
        A ``queue.Queue`` of GPU IDs acts as a per-device semaphore so that at
        most one replay occupies each GPU at any time.
        """
        results: Dict[str, Dict[str, Any]] = {}
        total = len(kernels)

        if self.num_gpus <= 1:
            # ── serial path ──────────────────────────────────────────────
            for i, entry in enumerate(kernels, 1):
                kid   = entry["id"]
                op    = entry["aten_op"].get("name", "?")
                kname = entry["kernel"]["name"]
                print(f"[{i}/{total}] {kid}: {op} -> {kname}")
                result = nsys_profile_pytorch_kernel(entry)
                if result is not None:
                    results[kid] = result
            return results

        # ── parallel path (multi-GPU) ─────────────────────────────────────
        gpu_queue = self._make_gpu_queue()
        lock = threading.Lock()
        dispatched_count = 0  # protected by lock; nonlocal in _replay

        def _replay(entry: Dict[str, Any]):
            nonlocal dispatched_count
            gpu_id = gpu_queue.get()   # blocks until a GPU token is free
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
                return kid, nsys_profile_pytorch_kernel(entry, extra_env=extra_env)
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

        return results

    def _run_ncu(
        self, kernels: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Run ncu on the top-N kernels by total duration.

        Applies the same single-GPU serial / multi-GPU parallel split as
        ``_run_nsys_replay``.
        """
        from soda.ncu import ncu_check_available, ncu_profile_kernel

        if not ncu_check_available():
            print("Skipping ncu profiling (ncu not available).")
            return {}

        top_n = getattr(self.args, "ncu_top_n", 10)
        ranked = sorted(
            kernels,
            key=lambda k: k["statistics"]["total_duration_us"],
            reverse=True,
        )[:top_n]

        ncu_dir = self.output_dir / "ncu"
        ncu_dir.mkdir(parents=True, exist_ok=True)

        results: Dict[str, Dict[str, Any]] = {}
        total = len(ranked)

        if self.num_gpus <= 1:
            # ── serial path ──────────────────────────────────────────────
            for i, entry in enumerate(ranked, 1):
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
            futures = [pool.submit(_ncu, entry) for entry in ranked]
            for future in as_completed(futures):
                try:
                    kid, result = future.result()
                    if result is not None:
                        results[kid] = result
                except Exception as exc:  # noqa: BLE001
                    print(f"  ncu error: {exc}")

        return results

    def _generate_report(
        self,
        floor: Dict[str, Any],
        nsys_results: Dict[str, Any],
        ncu_results: Dict[str, Any],
    ) -> Path:
        """Delegate to the report module (Phase 6)."""
        return generate_enhanced_report(
            kernel_db=self.db,
            floor=floor,
            nsys_results=nsys_results,
            ncu_results=ncu_results,
            output_dir=self.output_dir,
            verbose=getattr(self.args, "verbose", False),
        )
