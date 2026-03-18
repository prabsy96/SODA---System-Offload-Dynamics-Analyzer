"""MoE per-expert-type memory profiling pipeline.

Two-pass design — passes are always separate invocations:

  Pass 1 — NCU isolation (always):
    ncu_profile_kernel() on sampled entries per expert type.
    Provides absolute HBM bytes (hardware counters, accurate regardless
    of cache context).  L1/L2 hit rates from NCU are isolation-only
    (self-reuse) and are labelled as such.

  Pass 2 — NVBit in-process (if --nvbit-lib provided):
    Full model.generate() under LD_PRELOAD=mem_reuse_tracker.so.
    Instruments kernels tagged by expert_type in execution order.
    Provides in-context L1/L2 cache-line reuse and cross-expert data reuse.

Usage::

    soda-cli --moe-profile --kernel-db-path <path>
    soda-cli --moe-profile --kernel-db-path <path> --nvbit-lib <path>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from soda.moe.detect import (
    classify_kernel_entries,
    get_entries_by_type,
    sample_routed_entries,
)
from soda.moe.op_profile import _detect_num_layers, generate_op_profile
from soda.moe.report import generate_moe_report

_NCU_SAMPLE_SIZE = 10  # Max entries to NCU-profile per expert type


class MoEProfilePipeline:
    """MoE memory profiling pipeline (NCU isolation + optional NVBit in-process)."""

    def __init__(self, kernel_db_path: Path, args) -> None:
        self.kernel_db_path = Path(kernel_db_path)
        self.args = args

        # Load kernel DB
        with open(self.kernel_db_path, "r", encoding="utf-8") as f:
            self.kernel_db = json.load(f)

        self.kernels: List[Dict] = self.kernel_db.get("kernels", [])

        # Derive output directory alongside the kernel DB
        self.output_dir = self.kernel_db_path.parent / "moe_profile"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optional overrides
        self.shared_dim_override: Optional[int] = getattr(args, "moe_shared_dim", None)
        self.routed_dim_override: Optional[int] = getattr(args, "moe_routed_dim", None)
        self.nvbit_lib: Optional[Path] = (
            Path(getattr(args, "nvbit_lib", None))
            if getattr(args, "nvbit_lib", None)
            else None
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> Path:
        """Run the full pipeline and return the path to moe_profile.json."""
        print(f"\n[MoE Profile] Kernel DB: {self.kernel_db_path}")
        print(f"[MoE Profile] Output:     {self.output_dir}")

        # 1. Classify entries
        hf_config = self.kernel_db.get("metadata", {}).get("model_config")
        classified = classify_kernel_entries(
            self.kernels,
            model_config=hf_config,
            shared_dim_override=self.shared_dim_override,
            routed_dim_override=self.routed_dim_override,
        )
        self._print_classification_summary(classified)

        # 2. Pass 1 — NCU (isolation HBM baseline)
        ncu_results: Dict[str, Dict] = {}
        try:
            from soda.ncu import ncu_check_available
            if ncu_check_available():
                ncu_results = self._run_ncu_pass(classified)
            else:
                print("[MoE Profile] ncu not available — skipping NCU pass")
        except ImportError:
            print("[MoE Profile] soda.ncu not importable — skipping NCU pass")

        # 3. Pass 2 — NVBit in-process (required if --nvbit-lib provided)
        nvbit_results: Optional[Dict] = None
        if self.nvbit_lib is not None:
            if not self.nvbit_lib.exists():
                print(
                    f"[MoE Profile] Error: NVBit library not found: {self.nvbit_lib}",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            nvbit_results = self._run_nvbit_pass(classified)
            if nvbit_results is None:
                print(
                    "[MoE Profile] Error: NVBit pass failed — see stderr above.",
                    file=sys.stderr,
                )
                raise SystemExit(1)
        else:
            print("[MoE Profile] --nvbit-lib not provided — skipping NVBit pass")

        # 4. Generate report
        report_path = generate_moe_report(
            classified_kernels=classified,
            ncu_results=ncu_results,
            nvbit_results=nvbit_results,
            output_dir=self.output_dir,
            args=self.args,
        )
        print(f"\n[MoE Profile] Report: {report_path}")

        # 5. Generate op_profile.json
        num_layers = self._get_num_layers(classified)
        meta = self.kernel_db.get("metadata", {})
        cfg = meta.get("config", meta)
        precision = cfg.get("precision", "bfloat16") or "bfloat16"
        op_profile_path = self.output_dir / "op_profile.json"
        records = generate_op_profile(
            classified_kernels=classified,
            num_layers=num_layers,
            precision=precision,
            ncu_results=ncu_results,
            output_path=op_profile_path,
        )
        print(
            f"[MoE Profile] Op profile ({len(records)} records, "
            f"{num_layers} layers): {op_profile_path}"
        )

        return report_path

    # ------------------------------------------------------------------
    # Pass 1: NCU isolation
    # ------------------------------------------------------------------

    def _run_ncu_pass(self, classified: List[Dict]) -> Dict[str, Dict]:
        """Run NCU on sampled entries per expert type.

        Returns dict mapping kernel_id -> NCU result dict.
        """
        from soda.ncu import ncu_profile_kernel

        ncu_dir = self.output_dir / "ncu"
        ncu_dir.mkdir(parents=True, exist_ok=True)

        results: Dict[str, Dict] = {}
        expert_types = ["shared_expert", "routed_expert", "gate"]

        for et in expert_types:
            entries = get_entries_by_type(classified, et)
            if not entries:
                continue

            if et == "routed_expert":
                to_profile = sample_routed_entries(entries, n_samples=_NCU_SAMPLE_SIZE)
            else:
                to_profile = entries[:_NCU_SAMPLE_SIZE]

            print(
                f"[MoE Profile] NCU: {et} — profiling {len(to_profile)} "
                f"of {len(entries)} entries"
            )

            for i, entry in enumerate(to_profile, 1):
                kid = entry.get("id", f"?{i}")
                op = entry.get("aten_op", {}).get("name", "?")
                kname = entry.get("kernel", {}).get("name", "?")
                print(f"  [{i}/{len(to_profile)}] {kid}: {op} -> {kname}")

                result = ncu_profile_kernel(entry, output_dir=ncu_dir)
                if result is not None:
                    result["expert_type"] = et
                    results[kid] = result

        return results

    # ------------------------------------------------------------------
    # Pass 2: NVBit in-process inference
    # ------------------------------------------------------------------

    def _run_nvbit_pass(self, classified: List[Dict]) -> Optional[Dict]:
        """Run full model inference under NVBit LD_PRELOAD.

        Returns parsed NVBit reuse metrics dict, or None on failure.
        """
        from soda.moe.nvbit_profiler import build_expert_type_map, nvbit_profile_inference
        from soda.moe.nvbit_parser import parse_reuse_log

        # Build kernel_name -> expert_type map
        expert_type_map = build_expert_type_map(classified)
        if not expert_type_map:
            print("[MoE Profile] NVBit: no tagged kernels — skipping")
            return None

        # Read model name and generation config from kernel DB metadata
        meta = self.kernel_db.get("metadata", {})
        cfg = meta.get("config", meta)
        model_name = cfg.get("model_name") or cfg.get("model")
        if not model_name:
            print(
                "[MoE Profile] NVBit: model_name not found in kernel DB metadata",
                file=sys.stderr,
            )
            return None

        generation_config = {
            "batch_size": cfg.get("batch_size", 1),
            "seq_len": cfg.get("seq_len", 128),
            "max_new_tokens": cfg.get("max_new_tokens", 1),
            "precision": cfg.get("precision", "bfloat16"),
        }

        output_log = self.output_dir / "nvbit_reuse.jsonl"
        print(
            f"[MoE Profile] NVBit: running inference under {self.nvbit_lib.name} "
            f"({len(expert_type_map)} tagged kernels)"
        )

        success, log_path, msg = nvbit_profile_inference(
            model_name=model_name,
            generation_config=generation_config,
            expert_type_map=expert_type_map,
            nvbit_lib_path=self.nvbit_lib,
            output_log=output_log,
        )

        if not success:
            print(f"[MoE Profile] NVBit failed: {msg}", file=sys.stderr)
            return None

        nvbit_results = parse_reuse_log(log_path)
        if nvbit_results is None:
            print("[MoE Profile] NVBit log empty or unparseable", file=sys.stderr)
            return None

        print(
            f"[MoE Profile] NVBit: parsed {nvbit_results.get('total_records', 0)} "
            "kernel invocation records"
        )
        return nvbit_results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_num_layers(self, classified: List[Dict]) -> int:
        """Determine number of transformer layers for op_profile layer expansion.

        Priority:
          1. CLI --moe-num-layers override.
          2. num_hidden_layers from HuggingFace AutoConfig (if model accessible).
          3. GCD-based detection from shared expert entry frequencies.
        """
        # CLI override takes highest priority.
        num_layers_override = getattr(self.args, "moe_num_layers", None)
        if num_layers_override is not None:
            return int(num_layers_override)

        # Try HuggingFace AutoConfig.
        meta = self.kernel_db.get("metadata", {})
        cfg = meta.get("config", meta)
        model_name = cfg.get("model_name") or cfg.get("model")
        if model_name:
            try:
                from transformers import AutoConfig
                hf_cfg = AutoConfig.from_pretrained(model_name)
                num_layers = getattr(hf_cfg, "num_hidden_layers", None)
                if num_layers:
                    return int(num_layers)
            except Exception:
                pass

        # GCD-based fallback.
        return _detect_num_layers(classified)

    @staticmethod
    def _print_classification_summary(classified: List[Dict]) -> None:
        from collections import Counter
        counts = Counter(e.get("expert_type", "other") for e in classified)
        print("\n[MoE Profile] Classification summary:")
        for et in ["shared_expert", "routed_expert", "gate", "attention", "other"]:
            n = counts.get(et, 0)
            if n > 0:
                print(f"  {et:<18} {n:>5} entries")
        print()
