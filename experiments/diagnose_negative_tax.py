#!/usr/bin/env python3
"""
Diagnose negative launch tax patterns in MoE traces.
Usage: python diagnose_negative_tax.py <trace.json>
"""
import json
import sys
from collections import defaultdict

def analyze_trace(trace_path):
    with open(trace_path, 'r') as f:
        trace = json.load(f)
    
    # Build lookup tables
    launches_by_corr = {}
    kernels = []
    
    for event in trace["traceEvents"]:
        name = event.get("name", "")
        cat = event.get("cat", "")
        args = event.get("args", {})
        
        if "LaunchKernel" in name and cat in ("cuda_runtime", "cuda_driver"):
            corr = args.get("correlation")
            if corr:
                launches_by_corr[corr] = {
                    "name": name,
                    "ts": event.get("ts", 0),
                    "dur": event.get("dur", 0),
                    "stream": args.get("stream"),
                }
        
        if cat == "kernel":
            corr = args.get("correlation")
            kernels.append({
                "name": name,
                "ts": event.get("ts", 0),
                "dur": event.get("dur", 0),
                "correlation": corr,
                "stream": args.get("stream"),
            })
    
    # Analyze negative tax kernels
    small_neg = []   # -10μs to 0
    medium_neg = []  # -50μs to -10μs
    large_neg = []   # < -50μs
    
    for kernel in kernels:
        corr = kernel["correlation"]
        if corr not in launches_by_corr:
            continue
        
        launch = launches_by_corr[corr]
        launch_end = launch["ts"] + launch["dur"]
        kernel_start = kernel["ts"]
        tax = kernel_start - launch_end
        
        if tax < 0:
            entry = {
                "kernel_name": kernel["name"].split("<")[0][:50],
                "kernel_dur": kernel["dur"],
                "tax": tax,
                "launch_name": launch["name"],
                "launch_dur": launch["dur"],
                "stream": kernel["stream"],
                # Time between launch START and kernel START
                "launch_to_kernel": kernel_start - launch["ts"],
            }
            
            if tax >= -10:
                small_neg.append(entry)
            elif tax >= -50:
                medium_neg.append(entry)
            else:
                large_neg.append(entry)
    
    print(f"\n{'='*60}")
    print(f"NEGATIVE LAUNCH TAX DIAGNOSIS")
    print(f"{'='*60}")
    print(f"Total kernels: {len(kernels)}")
    print(f"Small negative (-10 to 0 μs): {len(small_neg)} kernels")
    print(f"Medium negative (-50 to -10 μs): {len(medium_neg)} kernels")
    print(f"Large negative (< -50 μs): {len(large_neg)} kernels")
    
    # Analyze large negatives
    if large_neg:
        print(f"\n{'='*60}")
        print(f"LARGE NEGATIVE ANALYSIS (< -50μs)")
        print(f"{'='*60}")
        
        # Group by kernel type
        by_kernel = defaultdict(list)
        for entry in large_neg:
            by_kernel[entry["kernel_name"]].append(entry)
        
        print(f"\nTop kernel types with large negatives:")
        for kname, entries in sorted(by_kernel.items(), key=lambda x: -len(x[1]))[:10]:
            avg_tax = sum(e["tax"] for e in entries) / len(entries)
            avg_dur = sum(e["kernel_dur"] for e in entries) / len(entries)
            avg_launch_dur = sum(e["launch_dur"] for e in entries) / len(entries)
            print(f"\n  {kname}")
            print(f"    Count: {len(entries)}")
            print(f"    Avg tax: {avg_tax:.1f}μs")
            print(f"    Avg kernel dur: {avg_dur:.1f}μs")
            print(f"    Avg launch dur: {avg_launch_dur:.1f}μs")
        
        # Check for stream distribution
        streams = defaultdict(int)
        for entry in large_neg:
            streams[entry["stream"]] += 1
        print(f"\n  Stream distribution: {dict(streams)}")
        
        # Sample some entries
        print(f"\n  Sample entries:")
        for entry in sorted(large_neg, key=lambda x: x["tax"])[:5]:
            print(f"    {entry['kernel_name'][:40]}: tax={entry['tax']:.1f}μs, "
                  f"kernel_dur={entry['kernel_dur']:.1f}μs, "
                  f"launch_dur={entry['launch_dur']:.1f}μs")

    # Check for pattern: Are large negatives clustered in time?
    if large_neg:
        print(f"\n{'='*60}")
        print(f"TEMPORAL CLUSTERING CHECK")
        print(f"{'='*60}")
        
        # Find the 10 most negative entries and check their neighbors
        worst = sorted(large_neg, key=lambda x: x["tax"])[:10]
        
        for entry in worst[:3]:
            kname = entry["kernel_name"]
            tax = entry["tax"]
            print(f"\n  Worst: {kname[:40]} (tax={tax:.1f}μs)")
            print(f"    This suggests the kernel started {-tax:.1f}μs BEFORE the launch ended.")
            print(f"    Launch duration was {entry['launch_dur']:.1f}μs")
            print(f"    Kernel duration was {entry['kernel_dur']:.1f}μs")
            
            if entry["launch_to_kernel"] > 0:
                print(f"    ✓ Kernel DID start after launch began (gap: {entry['launch_to_kernel']:.1f}μs)")
            else:
                print(f"    ✗ Kernel started BEFORE launch began! (gap: {entry['launch_to_kernel']:.1f}μs)")
                print(f"      → This indicates a profiler/correlation mismatch, not clock skew.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_negative_tax.py <trace.json>")
        sys.exit(1)
    analyze_trace(sys.argv[1])