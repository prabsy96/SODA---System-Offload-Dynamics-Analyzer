#!/usr/bin/env python3
"""
ΔCT Validation Analysis Script

Compares direct trace analysis (subtractive method) against replay-based
measurement (PyTorch vs baremetal cuBLAS) to validate the ΔCT calculation.

This addresses Reviewer D's question:
  "Have the authors tried to validate the ΔCT subtractive lower bound using
   any vendor-internal or instrumented traces (e.g., cuBLAS/cudnn front-end
   timings) to quantify how much is missed? If not available, can a small
   microbenchmark isolate front-end cost more directly?"

Usage:
    python experiments/rebuttal/validate_ct.py --direct-dir <path> --replay-dir <path>

Or run via the validation experiment script:
    ./experiments/rebuttal/slurm/validate_delta_ct.sh
"""

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

def load_json(path: Path) -> Optional[Dict]:
    """Safely load JSON file."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_direct_results(direct_dir: Path) -> Dict[str, Any]:
    """Load results from direct trace analysis."""
    summary_path = direct_dir / "microbench" / "taxbreak_summary.json"
    data = load_json(summary_path)

    if not data:
        return {"error": f"Summary not found at {summary_path}"}

    # Also load unique sequences to compare kernel lists
    unique_seqs_path = direct_dir / "microbench" / "unique_all_sequences.json"
    unique_seqs = load_json(unique_seqs_path)

    return {
        "method": "direct_trace",
        "t_aten_base_us": data.get("t_aten_base_us", 0),
        "T_structural_ms": data.get("T_structural_mean_ms", data.get("T_structural_total_ms", 0)),
        "breakdown": data.get("breakdown_mean", data.get("breakdown", {})),
        "per_kernel": data.get("per_kernel", []),
        "invocations": data.get("invocations", {}),
        "unique_sequences": unique_seqs.get("sequences", []) if unique_seqs else [],
    }


def load_replay_results(replay_dir: Path) -> Dict[str, Any]:
    """Load results from replay analysis (PyTorch + baremetal)."""
    # Load summary
    summary_path = replay_dir / "microbench" / "taxbreak_summary.json"
    summary_data = load_json(summary_path)

    # Load PyTorch sequences
    pytorch_all_path = replay_dir / "microbench" / "framework" / "pytorch" / "output" / "pytorch_all_sequences.json"
    pytorch_gemm_path = replay_dir / "microbench" / "framework" / "pytorch" / "output" / "pytorch_gemm_sequences.json"
    pytorch_all = load_json(pytorch_all_path)
    pytorch_gemm = load_json(pytorch_gemm_path)

    # Load baremetal sequences
    baremetal_path = replay_dir / "microbench" / "baremetal" / "output" / "baremetal_gemm_runs.json"
    baremetal = load_json(baremetal_path)

    result = {
        "method": "replay",
        "pytorch": {
            "all_count": len(pytorch_all.get("sequences", [])) if pytorch_all else 0,
            "gemm_count": len(pytorch_gemm.get("sequences", [])) if pytorch_gemm else 0,
            "sequences": pytorch_all.get("sequences", []) if pytorch_all else [],
            "gemm_sequences": pytorch_gemm.get("sequences", []) if pytorch_gemm else [],
        },
        "baremetal": {
            "count": len(baremetal.get("sequences", [])) if baremetal else 0,
            "sequences": baremetal.get("sequences", []) if baremetal else [],
        },
    }

    if summary_data:
        result["t_aten_base_us"] = summary_data.get("t_aten_base_us", 0)
        result["T_structural_ms"] = summary_data.get("T_structural_mean_ms",
                                                      summary_data.get("T_structural_total_ms", 0))
        result["breakdown"] = summary_data.get("breakdown_mean", summary_data.get("breakdown", {}))

    return result


def compute_per_kernel_comparison(pytorch_seqs: List[Dict], baremetal_seqs: List[Dict]) -> List[Dict]:
    """
    Compare PyTorch vs baremetal overhead for matched GEMM kernels.

    Returns per-kernel comparison data showing:
    - PyTorch T_aten_xlat (includes ΔCT + ΔFT_aten)
    - Baremetal T_culib_xlat (direct cuBLAS front-end measurement)
    - Delta showing the subtractive estimate accuracy
    """
    # Build baremetal lookup by job_id
    bm_by_job = {s.get("job_id"): s for s in baremetal_seqs if s is not None}

    comparisons = []
    for i, pt_seq in enumerate(pytorch_seqs):
        if not pt_seq or not pt_seq.get("is_gemm", False):
            continue

        job_id = f"{i+1:04d}"
        bm_seq = bm_by_job.get(job_id)

        # Extract PyTorch timing
        pt_aten_xlat = pt_seq.get("aten_xlat_tax", {})
        if isinstance(pt_aten_xlat, dict):
            pt_aten_xlat_avg = pt_aten_xlat.get("avg", 0)
        else:
            pt_aten_xlat_avg = pt_aten_xlat or 0

        pt_py_tax = pt_seq.get("py_tax", {})
        if isinstance(pt_py_tax, dict):
            pt_py_tax_avg = pt_py_tax.get("avg", 0)
        else:
            pt_py_tax_avg = pt_py_tax or 0

        pt_launch_tax = pt_seq.get("launch_tax", {})
        if isinstance(pt_launch_tax, dict):
            pt_launch_tax_avg = pt_launch_tax.get("avg", 0)
        else:
            pt_launch_tax_avg = pt_launch_tax or 0

        # Extract baremetal timing
        bm_culib_xlat_avg = None
        bm_shim_avg = None
        bm_launch_tax_avg = None
        bm_setup_avg = None
        bm_heur_avg = None

        if bm_seq:
            bm_culib_xlat = bm_seq.get("culib_xlat_tax", {})
            if isinstance(bm_culib_xlat, dict):
                bm_culib_xlat_avg = bm_culib_xlat.get("avg")
            else:
                bm_culib_xlat_avg = bm_culib_xlat

            bm_shim = bm_seq.get("shim_tax", {})
            if isinstance(bm_shim, dict):
                bm_shim_avg = bm_shim.get("avg")
            else:
                bm_shim_avg = bm_shim

            bm_launch = bm_seq.get("launch_tax", {})
            if isinstance(bm_launch, dict):
                bm_launch_tax_avg = bm_launch.get("avg")
            else:
                bm_launch_tax_avg = bm_launch

            # cuBLAS breakdown (setup + heuristic phases)
            culib = bm_seq.get("culib", {})
            if culib:
                setup = culib.get("setup", {}).get("dur", {})
                if isinstance(setup, dict):
                    bm_setup_avg = setup.get("avg")
                heur = culib.get("heur", {}).get("dur", {})
                if isinstance(heur, dict):
                    bm_heur_avg = heur.get("avg")

        # Compute delta (PyTorch ATen - baremetal cuBLAS)
        delta_xlat = None
        if pt_aten_xlat_avg is not None and bm_culib_xlat_avg is not None:
            delta_xlat = pt_aten_xlat_avg - bm_culib_xlat_avg

        kernel_name = pt_seq.get("kernel", {}).get("name", "unknown")
        aten_op = pt_seq.get("aten_op", {}).get("name", "unknown")

        comparisons.append({
            "job_id": job_id,
            "kernel": kernel_name,
            "aten_op": aten_op,
            "pytorch": {
                "T_py": pt_py_tax_avg,
                "T_aten_xlat": pt_aten_xlat_avg,
                "T_sys": pt_launch_tax_avg,
                "T_total": (pt_py_tax_avg or 0) + (pt_aten_xlat_avg or 0) + (pt_launch_tax_avg or 0),
            },
            "baremetal": {
                "T_culib_xlat": bm_culib_xlat_avg,
                "T_culib_setup": bm_setup_avg,
                "T_culib_heur": bm_heur_avg,
                "T_shim": bm_shim_avg,
                "T_sys": bm_launch_tax_avg,
                "T_total": (
                    (bm_culib_xlat_avg or 0) +
                    (bm_shim_avg or 0) +
                    (bm_launch_tax_avg or 0)
                ) if bm_culib_xlat_avg else None,
            },
            "delta": {
                "xlat_diff": delta_xlat,  # PyTorch ATen - baremetal cuBLAS
                "note": "Positive delta indicates PyTorch framework overhead beyond cuBLAS",
            },
            "matched": bm_seq is not None,
        })

    return comparisons


def generate_validation_report(
    direct_results: Dict[str, Any],
    replay_results: Dict[str, Any],
    output_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive ΔCT validation report.

    Compares:
    1. Direct trace method: ΔCT = max(0, T_aten_xlat - T_aten_base)
    2. Replay method: Direct cuBLAS measurement via baremetal profiling
    """
    report = {
        "title": "ΔCT Validation Report (Reviewer D)",
        "methodology": {
            "direct_trace": {
                "description": "Subtractive method using non-GEMM kernels as baseline",
                "formula": "ΔCT = max(0, T_aten_xlat_GEMM - T_aten_base)",
                "note": "T_aten_base = median(T_aten_xlat of non-GEMM kernels)"
            },
            "replay": {
                "description": "Isolated kernel profiling comparing PyTorch vs baremetal cuBLAS",
                "pytorch": "Full framework overhead (Python + ATen + cuBLAS + launch)",
                "baremetal": "Direct cuBLAS overhead (cuBLAS API + launch only)",
            }
        }
    }

    # Direct trace results
    if "error" not in direct_results:
        report["direct_trace"] = {
            "t_aten_base_us": direct_results.get("t_aten_base_us"),
            "T_structural_ms": direct_results.get("T_structural_ms"),
            "breakdown": direct_results.get("breakdown"),
        }
    else:
        report["direct_trace"] = {"error": direct_results["error"]}

    # Replay results
    report["replay"] = {
        "pytorch": {
            "total_sequences": replay_results["pytorch"]["all_count"],
            "gemm_sequences": replay_results["pytorch"]["gemm_count"],
        },
        "baremetal": {
            "sequences_profiled": replay_results["baremetal"]["count"],
        },
    }

    if replay_results.get("breakdown"):
        report["replay"]["breakdown"] = replay_results["breakdown"]

    # Verify kernel list consistency between direct and replay
    direct_gemm_ops = set()
    if "error" not in direct_results:
        for seq in direct_results.get("unique_sequences", []):
            if seq.get("is_gemm", False):
                aten_op = seq.get("aten_op", {})
                key = (aten_op.get("name"), tuple(tuple(d) if isinstance(d, list) else d
                       for d in aten_op.get("input_dims", [])))
                direct_gemm_ops.add(key)

    replay_gemm_ops = set()
    for seq in replay_results["pytorch"].get("gemm_sequences", []):
        aten_op = seq.get("aten_op", {})
        key = (aten_op.get("name"), tuple(tuple(d) if isinstance(d, list) else d
               for d in aten_op.get("input_dims", [])))
        replay_gemm_ops.add(key)

    report["kernel_matching"] = {
        "direct_gemm_count": len(direct_gemm_ops),
        "replay_gemm_count": len(replay_gemm_ops),
        "common_ops": len(direct_gemm_ops & replay_gemm_ops),
        "only_in_direct": len(direct_gemm_ops - replay_gemm_ops),
        "only_in_replay": len(replay_gemm_ops - direct_gemm_ops),
        "consistent": direct_gemm_ops == replay_gemm_ops if direct_gemm_ops and replay_gemm_ops else None,
    }

    # Per-kernel comparison
    pytorch_seqs = replay_results["pytorch"].get("gemm_sequences", [])
    baremetal_seqs = replay_results["baremetal"].get("sequences", [])

    if pytorch_seqs:
        comparisons = compute_per_kernel_comparison(pytorch_seqs, baremetal_seqs)
        matched = [c for c in comparisons if c["matched"]]

        report["per_kernel_comparison"] = {
            "total_gemm_kernels": len(comparisons),
            "matched_with_baremetal": len(matched),
            "kernels": comparisons[:10],  # First 10 for brevity
        }

        # Aggregate statistics
        if matched:
            xlat_diffs = [c["delta"]["xlat_diff"] for c in matched
                         if c["delta"]["xlat_diff"] is not None]
            if xlat_diffs:
                report["validation_summary"] = {
                    "matched_kernels": len(matched),
                    "avg_xlat_diff_us": statistics.mean(xlat_diffs),
                    "median_xlat_diff_us": statistics.median(xlat_diffs),
                    "min_xlat_diff_us": min(xlat_diffs),
                    "max_xlat_diff_us": max(xlat_diffs),
                    "interpretation": (
                        "Positive xlat_diff indicates PyTorch ATen overhead exceeds bare cuBLAS. "
                        "This excess is the 'hidden' framework cost captured by the subtractive ΔCT."
                    ),
                }
        elif not matched:
            report["validation_summary"] = {
                "note": "No baremetal kernel matches found.",
                "reason": (
                    "On H100/H200/GB200, PyTorch often uses internal optimized kernels "
                    "instead of cuBLAS. In this case, direct subtractive ΔCT is the "
                    "appropriate method since there's no external library call to isolate."
                ),
            }

    # Save report
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Saved validation report to: {output_path}")

    return report


def print_validation_summary(report: Dict[str, Any]):
    """Print human-readable validation summary."""
    print("\n" + "=" * 70)
    print("ΔCT VALIDATION REPORT (Reviewer D)")
    print("=" * 70)

    print("\n1. METHODOLOGY")
    print("-" * 50)
    meth = report.get("methodology", {})
    print("Direct Trace (Subtractive Method):")
    print(f"  Formula: {meth.get('direct_trace', {}).get('formula', 'N/A')}")
    print(f"  Note: {meth.get('direct_trace', {}).get('note', 'N/A')}")
    print()
    print("Replay (Isolated Comparison):")
    print(f"  PyTorch: {meth.get('replay', {}).get('pytorch', 'N/A')}")
    print(f"  Baremetal: {meth.get('replay', {}).get('baremetal', 'N/A')}")

    print("\n2. DIRECT TRACE RESULTS")
    print("-" * 50)
    dt = report.get("direct_trace", {})
    if "error" not in dt:
        print(f"  T_aten_base (derived): {dt.get('t_aten_base_us', 0):.3f} μs")
        print(f"  T_structural (total):  {dt.get('T_structural_ms', 0):.3f} ms")
        bd = dt.get("breakdown", {})
        if bd:
            print(f"  Breakdown:")
            print(f"    ΔFT (Python):    {bd.get('FT_python_ms', 0):.3f} ms")
            print(f"    ΔCT (ATen):      {bd.get('CT_aten_ms', 0):.3f} ms")
            print(f"    ΔCudaT:          {bd.get('CudaT_cuda_runtime_ms', 0):.3f} ms")
            print(f"    ΔKT (Launch):    {bd.get('KT_kernel_launch_ms', 0):.3f} ms")
    else:
        print(f"  Error: {dt['error']}")

    print("\n3. REPLAY RESULTS")
    print("-" * 50)
    rp = report.get("replay", {})
    pt = rp.get("pytorch", {})
    bm = rp.get("baremetal", {})
    print(f"  PyTorch sequences: {pt.get('total_sequences', 0)} total, {pt.get('gemm_sequences', 0)} GEMM")
    print(f"  Baremetal sequences: {bm.get('sequences_profiled', 0)}")

    print("\n4. KERNEL MATCHING VERIFICATION")
    print("-" * 50)
    km = report.get("kernel_matching", {})
    print(f"  GEMM ops in direct trace:  {km.get('direct_gemm_count', 'N/A')}")
    print(f"  GEMM ops in replay:        {km.get('replay_gemm_count', 'N/A')}")
    print(f"  Common ops:                {km.get('common_ops', 'N/A')}")
    if km.get("only_in_direct", 0) > 0:
        print(f"  Only in direct:            {km.get('only_in_direct', 0)}")
    if km.get("only_in_replay", 0) > 0:
        print(f"  Only in replay:            {km.get('only_in_replay', 0)}")
    if km.get("consistent") is not None:
        status = "CONSISTENT" if km.get("consistent") else "MISMATCH"
        print(f"  Status:                    {status}")

    print("\n5. PER-KERNEL COMPARISON")
    print("-" * 50)
    comp = report.get("per_kernel_comparison", {})
    print(f"  GEMM kernels: {comp.get('total_gemm_kernels', 0)}")
    print(f"  Matched with baremetal: {comp.get('matched_with_baremetal', 0)}")

    kernels = comp.get("kernels", [])[:5]
    if kernels:
        print("\n  Sample kernels (μs):")
        print("  " + "-" * 66)
        print(f"  {'Kernel':<30} {'PT T_aten':>12} {'BM T_culib':>12} {'Δ':>10}")
        print("  " + "-" * 66)
        for k in kernels:
            name = k.get("kernel", "")[:28]
            pt_xlat = k.get("pytorch", {}).get("T_aten_xlat")
            bm_xlat = k.get("baremetal", {}).get("T_culib_xlat")
            delta = k.get("delta", {}).get("xlat_diff")

            pt_str = f"{pt_xlat:.2f}" if pt_xlat is not None else "N/A"
            bm_str = f"{bm_xlat:.2f}" if bm_xlat is not None else "N/A"
            d_str = f"{delta:+.2f}" if delta is not None else "N/A"

            print(f"  {name:<30} {pt_str:>12} {bm_str:>12} {d_str:>10}")

    print("\n6. VALIDATION SUMMARY")
    print("-" * 50)
    vs = report.get("validation_summary", {})
    if "note" in vs:
        print(f"  {vs['note']}")
        print(f"  Reason: {vs.get('reason', 'N/A')}")
    else:
        print(f"  Matched kernels: {vs.get('matched_kernels', 0)}")
        print(f"  Avg xlat diff:   {vs.get('avg_xlat_diff_us', 0):.2f} μs")
        print(f"  Median xlat diff: {vs.get('median_xlat_diff_us', 0):.2f} μs")
        print(f"  Range: [{vs.get('min_xlat_diff_us', 0):.2f}, {vs.get('max_xlat_diff_us', 0):.2f}] μs")
        print()
        print(f"  Interpretation: {vs.get('interpretation', 'N/A')}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Validate ΔCT calculation by comparing direct trace vs replay analysis"
    )
    parser.add_argument(
        "--direct-dir",
        type=Path,
        required=True,
        help="Directory containing direct trace analysis results"
    )
    parser.add_argument(
        "--replay-dir",
        type=Path,
        required=True,
        help="Directory containing replay analysis results"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for validation report JSON"
    )

    args = parser.parse_args()

    print("Loading direct trace results...")
    direct_results = load_direct_results(args.direct_dir)

    print("Loading replay results...")
    replay_results = load_replay_results(args.replay_dir)

    print("Generating validation report...")
    report = generate_validation_report(
        direct_results,
        replay_results,
        args.output
    )

    print_validation_summary(report)


if __name__ == "__main__":
    main()
