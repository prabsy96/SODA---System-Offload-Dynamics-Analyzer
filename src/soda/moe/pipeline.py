"""MoE per-operator memory profiling pipeline via PyTorch Profiler CUPTI.

Standalone pipeline that loads a model, runs inference with curated
benchmark prompts from ``soda.moe.prompts``, and measures per-operator
HBM and L2 traffic using CUPTI hardware counters.  No dependency on
kernel_database.json or the main SODA pipeline.

Usage::

    soda-cli --moe-profile -m Qwen/Qwen1.5-MoE-A2.7B
    soda-cli --moe-profile -m Qwen/Qwen1.5-MoE-A2.7B --moe-prompts code_python science_physics
    soda-cli --moe-profile -m Qwen/Qwen1.5-MoE-A2.7B --max-seq-len 2048
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from soda.moe.cupti_profiler import profile_single_prompt
from soda.moe.ncu_bridge import (
    _NCU_ELIGIBLE_OPS,
    merge_ncu_into_events,
    run_ncu_on_profiler_events,
)
from soda.moe.op_profile import generate_op_profile_from_cupti
from soda.moe.prompts import (
    MOE_BENCHMARK_PROMPTS,
    SWEEP_REPRESENTATIVE_PROMPTS,
    get_prompts_for_categories,
)

_DEFAULT_WARMUP = 2


class MoEProfilePipeline:
    """Standalone MoE memory profiling pipeline using CUPTI hardware counters.

    Loads the model, iterates over curated prompts, runs each under
    ``torch.profiler`` with CUPTI metrics, classifies operators via
    nn.Module hierarchy, and writes per-prompt + aggregated ``op_profile.json``.
    """

    def __init__(self, model_name: str, args: Any) -> None:
        self.model_name = model_name
        self.args = args
        self.precision = getattr(args, "precision", "bfloat16") or "bfloat16"
        self.max_seq_len = getattr(args, "max_seq_len", 4096)
        self.max_new_tokens = getattr(args, "max_new_tokens", 1)
        self.warmup_iters = getattr(args, "moe_warmup", _DEFAULT_WARMUP)

        # Output directory
        output_root = Path(getattr(args, "output_dir", "output"))
        model_slug = model_name.replace("/", "_")
        self.output_dir = output_root / model_slug / "moe_cupti_profile"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.per_prompt_dir = self.output_dir / "per_prompt"
        self.per_prompt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> Path:
        """Run the full CUPTI profiling pipeline.

        Returns:
            Path to the aggregated ``op_profile.json``.
        """
        import torch

        print(f"\n[MoE CUPTI Profile] Model:        {self.model_name}")
        print(f"[MoE CUPTI Profile] Output:       {self.output_dir}")
        print(f"[MoE CUPTI Profile] Precision:    {self.precision}")
        print(f"[MoE CUPTI Profile] Max seq len:  {self.max_seq_len} (truncation cap)")
        print(f"[MoE CUPTI Profile] Max tokens:   {self.max_new_tokens}")

        # 1. Load model + tokenizer
        model, tokenizer = self._load_model()

        # 2. Select prompts
        prompts = self._select_prompts()
        print(f"[MoE CUPTI Profile] Prompts:    {len(prompts)} selected")

        # 3. Warmup
        self._warmup(model, tokenizer, prompts[0][1])

        # 4. Profile each prompt (collect events only — no file writing yet)
        all_events: Dict[str, List[Dict]] = {}
        for i, (name, text) in enumerate(prompts, 1):
            print(f"  [{i}/{len(prompts)}] Profiling: {name}")
            events = profile_single_prompt(
                model=model,
                tokenizer=tokenizer,
                prompt_text=text,
                max_seq_len=self.max_seq_len,
                max_new_tokens=self.max_new_tokens,
                precision=self.precision,
            )
            all_events[name] = events

        # 5. NCU bridge: fill HBM/L2 bytes when CUPTI returned zeros
        self._run_ncu_bridge_if_needed(all_events)

        # 6. Write per-prompt op_profile files (after NCU bridge so hbm_bytes are real)
        for name, events in all_events.items():
            per_prompt_path = self.per_prompt_dir / f"op_profile_{name}.json"
            records = generate_op_profile_from_cupti(events, output_path=per_prompt_path)
            print(f"  [per-prompt] {name}: {len(records)} records → {per_prompt_path.name}")

        # 7. Aggregate across prompts
        agg_path = self._write_aggregated_profile(all_events)
        print(f"\n[MoE CUPTI Profile] Aggregated: {agg_path}")

        # 8. Write metadata
        self._write_metadata(prompts, all_events)

        # 9. Print summary
        self._print_summary(all_events)

        # 10. Free model
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return agg_path

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load HuggingFace model and tokenizer."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[MoE CUPTI Profile] Loading model: {self.model_name}")
        dtype = getattr(torch, self.precision, torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        parallelism = getattr(self.args, "parallelism", "tp")
        if parallelism == "tp":
            device_map = "auto"
        else:
            # DP/FSDP/EP usually load on a single device first
            device_map = None 

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=device_map,
        )
        if device_map is None:
            model.to("cuda" if torch.cuda.is_available() else "cpu")

        model.eval()
        # Resolve "auto" / unknown precision strings to the actual model dtype so
        # that _dtype_bytes() and create_input_tensors() use the correct byte width.
        if self.precision in ("auto", None) or not hasattr(torch, self.precision):
            inferred = str(next(model.parameters()).dtype).split(".")[-1]
            if self.precision != inferred:
                import warnings
                warnings.warn(
                    f"[MoE] precision='{self.precision}' resolved to '{inferred}' "
                    "from model parameters. Shape-based HBM estimates will use "
                    f"{inferred}.",
                    stacklevel=2,
                )
            self.precision = inferred
        print(f"[MoE CUPTI Profile] Model loaded on {next(model.parameters()).device}")
        return model, tokenizer

    # ------------------------------------------------------------------
    # Prompt selection
    # ------------------------------------------------------------------

    def _select_prompts(self) -> List[tuple]:
        """Select prompts based on CLI args.

        Returns list of (name, text) tuples.
        """
        # Explicit prompt names
        moe_prompts = getattr(self.args, "moe_prompts", None)
        if moe_prompts:
            selected = []
            for name in moe_prompts:
                text = MOE_BENCHMARK_PROMPTS.get(name)
                if text is None:
                    print(
                        f"[MoE CUPTI Profile] Warning: unknown prompt '{name}', skipping",
                        file=sys.stderr,
                    )
                    continue
                selected.append((name, text))
            if selected:
                return selected

        # Explicit categories
        moe_categories = getattr(self.args, "moe_categories", None)
        if moe_categories:
            pairs = get_prompts_for_categories(moe_categories)
            if pairs:
                return pairs

        # Default: representative subset (7 prompts, one per domain)
        return [
            (name, MOE_BENCHMARK_PROMPTS[name])
            for name in SWEEP_REPRESENTATIVE_PROMPTS
            if name in MOE_BENCHMARK_PROMPTS
        ]

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _warmup(self, model, tokenizer, first_prompt: str) -> None:
        """Run warmup iterations to JIT-compile kernels."""
        import torch

        if self.warmup_iters <= 0:
            return

        print(f"[MoE CUPTI Profile] Warmup: {self.warmup_iters} iterations")

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        inputs = tokenizer(
            first_prompt,
            return_tensors="pt",
            max_length=self.max_seq_len,
            truncation=True,
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            for _ in range(self.warmup_iters):
                model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # NCU bridge
    # ------------------------------------------------------------------

    def _run_ncu_bridge_if_needed(
        self, all_events: Dict[str, List[Dict]]
    ) -> None:
        """If CUPTI returned all-zero HBM bytes, fill them in via NCU.

        Checks whether any CUDA-active event has nonzero ``hbm_bytes``.  If
        not, and ``ncu`` is available, deduplicates all events by
        ``(aten_op, input_shapes)`` and runs ``ncu_profile_kernel()`` on each
        unique entry.  Results are stamped back into the event dicts in-place
        (``hbm_source = "ncu"``).

        No-ops when:
        - CUPTI already returned nonzero data.
        - ``ncu`` is not in PATH.
        """
        from soda.ncu import ncu_check_available

        # Flatten all events to check if any NCU-eligible GEMM op has nonzero HBM.
        # Deliberately restrict to NCU-eligible ops: on Blackwell, non-GEMM ops
        # (layer_norm, softmax) may return nonzero l2_bytes via a different CUPTI
        # path while all GEMM HBM counters remain zero. A single non-GEMM nonzero
        # event must not suppress the bridge for GEMM ops.
        all_flat = [evt for evts in all_events.values() for evt in evts]
        has_nonzero = any(
            (evt.get("hbm_bytes") or 0.0) > 0.0
            for evt in all_flat
            if evt.get("aten_op") in _NCU_ELIGIBLE_OPS
            and (evt.get("cuda_time_us") or 0.0) > 0.0
        )
        if has_nonzero:
            return  # CUPTI data is present for GEMM ops — NCU not needed

        if not ncu_check_available():
            print(
                "[MoE CUPTI Profile] Warning: CUPTI counters are zero and ncu is not "
                "available. HBM/L2 bytes will remain zero in op_profile.json."
            )
            return

        print(
            "[MoE CUPTI Profile] CUPTI counters are zero — running NCU bridge "
            "to measure HBM/L2 bytes per unique (op, shape)."
        )
        ncu_dir = self.output_dir / "ncu_bridge"
        ncu_results = run_ncu_on_profiler_events(
            all_flat,
            ncu_output_dir=ncu_dir,
            model_dtype=self.precision,
        )

        if ncu_results:
            # Stamp back into per-prompt event lists
            for evts in all_events.values():
                merge_ncu_into_events(evts, ncu_results, model_dtype=self.precision)

        # Mark ops that were skipped because they are not NCU-replayable.
        # hbm_source="ncu_ineligible" distinguishes "we chose not to measure"
        # from "cupti_zero" which could mean either measured-zero or missing data.
        for evts in all_events.values():
            for evt in evts:
                if (
                    evt.get("hbm_source") == "cupti_zero"
                    and evt.get("aten_op") not in _NCU_ELIGIBLE_OPS
                    and (evt.get("cuda_time_us") or 0.0) > 0.0
                ):
                    evt["hbm_source"] = "ncu_ineligible"

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _write_aggregated_profile(
        self, all_events: Dict[str, List[Dict]]
    ) -> Path:
        """Aggregate per-prompt events and write a single op_profile.json.

        Concatenates all per-prompt events into one record list (each tagged
        with the prompt name) so the aggregated file captures the full
        picture across inputs.
        """
        merged: List[Dict] = []
        for prompt_name, events in all_events.items():
            for evt in events:
                evt_copy = dict(evt)
                evt_copy["prompt_name"] = prompt_name
                merged.append(evt_copy)

        agg_path = self.output_dir / "op_profile.json"
        records = generate_op_profile_from_cupti(merged, output_path=agg_path)
        return agg_path

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _write_metadata(
        self,
        prompts: List[tuple],
        all_events: Dict[str, List[Dict]],
    ) -> None:
        """Write metadata.json with run configuration and summary."""
        # Determine CUPTI/NCU status from events
        cupti_available = False
        hbm_source_counts: Dict[str, int] = {}
        for events in all_events.values():
            for evt in events:
                if not cupti_available:
                    cupti_available = evt.get("cupti_available", False)
                src = evt.get("hbm_source", "unknown")
                hbm_source_counts[src] = hbm_source_counts.get(src, 0) + 1

        metadata = {
            "model": self.model_name,
            "precision": self.precision,
            "max_seq_len": self.max_seq_len,
            "max_new_tokens": self.max_new_tokens,
            "warmup_iterations": self.warmup_iters,
            "cupti_available": cupti_available,
            "hbm_source_counts": hbm_source_counts,
            "prompts": [name for name, _ in prompts],
            "num_prompts": len(prompts),
            "events_per_prompt": {
                name: len(evts) for name, evts in all_events.items()
            },
            # op_profile.json contains one record per event per prompt (raw concat).
            # To get per-inference averages, divide latency_us / hbm_bytes by num_prompts.
            "aggregation": "per_prompt_raw",
        }
        meta_path = self.output_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    @staticmethod
    def _print_summary(all_events: Dict[str, List[Dict]]) -> None:
        """Print classification and traffic summary."""
        from collections import Counter

        total_records = 0
        type_counts: Counter = Counter()
        total_hbm = 0.0
        total_l2 = 0.0

        for events in all_events.values():
            for evt in events:
                if evt.get("num_kernels", 0) == 0 and evt.get("cuda_time_us", 0) == 0:
                    continue
                total_records += 1
                type_counts[evt.get("expert_type", "other")] += 1
                total_hbm += evt.get("hbm_bytes", 0.0)
                total_l2 += evt.get("l2_bytes", 0.0)

        print(f"\n[MoE CUPTI Profile] Summary:")
        print(f"  Total op records:   {total_records}")
        print(f"  Total HBM traffic:  {total_hbm / 1e6:.1f} MB")
        print(f"  Total L2 traffic:   {total_l2 / 1e6:.1f} MB")
        print(f"  Expert type breakdown:")
        for et in ["shared_expert", "routed_expert", "gate", "attention", "other"]:
            n = type_counts.get(et, 0)
            if n > 0:
                print(f"    {et:<18} {n:>5} ops")
