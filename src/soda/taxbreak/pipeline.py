"""
Enhanced TaxBreak pipeline orchestrator.

Reads a kernel database (from Phase 1), measures the dynamic system floor
(Phase 2), replays each kernel in isolation under nsys (Phase 3), optionally
profiles top-N kernels with ncu (Phase 4), and writes an enhanced report
(Phase 6).

Usage (Stage 2 — no model loading required):
    soda-cli --taxbreak --kernel-db-path <path> [--ncu] [--ncu-top-n 10]
"""

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
        floor = measure_system_floor()
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

    def _run_nsys_replay(
        self, kernels: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Replay each kernel in isolation under nsys."""
        results: Dict[str, Dict[str, Any]] = {}

        for i, entry in enumerate(kernels, 1):
            kid = entry["id"]
            op = entry["aten_op"].get("name", "?")
            kname = entry["kernel"]["name"]
            print(f"[{i}/{len(kernels)}] {kid}: {op} -> {kname}")

            result = nsys_profile_pytorch_kernel(entry)
            if result is not None:
                results[kid] = result

        return results

    def _run_ncu(
        self, kernels: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Run ncu on the top-N kernels by total duration."""
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
        for i, entry in enumerate(ranked, 1):
            kid = entry["id"]
            op = entry["aten_op"].get("name", "?")
            kname = entry["kernel"]["name"]
            print(f"[{i}/{len(ranked)}] {kid}: {op} -> {kname}")

            result = ncu_profile_kernel(entry, output_dir=ncu_dir)
            if result is not None:
                results[kid] = result

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
