#!/usr/bin/env python3
"""
Optimization Stack Comparison for Reviewer C

Runs TaxBreak decomposition under different optimization modes:
1. Eager mode (baseline)
2. FlashAttention (flash_attention_2)
3. torch.compile (inductor with CUDA Graphs via reduce-overhead)
4. Manual CUDA Graphs (expected to fail on HF models, for completeness)

This addresses Reviewer C's concern:
  "CUDA Graphs and kernel fusion are widely used to eliminate CPU launch overhead.
   These optimizations can fundamentally change host-side execution paths...
   evaluation should include these widely adopted techniques."

TaxBreak-Aligned Methodology (aligned with calculate_hdbi in utils.py):
  HDBI = T_DeviceActive / (T_DeviceActive + T_Orchestrate)
  T_Orchestrate = T_xlat_tax + ΔKT
  ΔKT = num_kernels × T_floor_sys  (from baremetal null-kernel measurement)
  T_xlat_tax = Σ (aten_op.dur - cuda_launch.dur)  (per-sequence, CPU timeline)

  T_xlat_tax is measured from linked CPU events, NOT derived as a residual.

Usage:
    python experiments/rebuttal/optimization_comparison.py --output-dir DIR [options]
"""

import argparse
import gc
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.profiler import profile, ProfilerActivity

# Check for optional dependencies
try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


# =============================================================================
# TaxBreak Constants (aligned with paper and CLAUDE.md)
# =============================================================================

# T_floor_sys: Null-kernel baseline from baremetal profiling (μs)
T_FLOOR_SYS_US = 4.50  # H100 default, in microseconds
T_FLOOR_SYS_MS = T_FLOOR_SYS_US / 1000.0


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class OptimizationResult:
    """Results for a single optimization mode."""
    mode: str
    inference_time_ms: float          # Wall-clock via CUDA events (T_wall)
    gpu_busy_time_ms: float           # Merged kernel intervals on GPU timeline
    gpu_idle_time_ms: float           # GPU span - GPU busy (GPU timeline only)
    total_kernel_exec_time_ms: float  # Sum of all kernel durations (T_DeviceActive)
    num_kernels: int
    num_cuda_launches: int
    num_linked_sequences: int         # Kernels with matched aten_op + launch
    num_orphan_kernels: int           # Kernels without CPU-side match
    t_orchestrate_ms: float           # T_xlat_tax + ΔKT
    t_xlat_tax_ms: float              # Σ (aten_dur - launch_dur) per linked sequence
    delta_kt_ms: float                # num_kernels × T_floor_sys
    hdbi: float                       # T_DeviceActive / (T_DeviceActive + T_Orchestrate)
    hdbi_class: str                   # "host-bound", "balanced", "device-bound"
    avg_kernel_gap_us: float          # Mean inter-kernel gap (GPU timeline only)
    status: str = "ok"
    error: Optional[str] = None


@dataclass
class ComparisonReport:
    """Complete comparison report across all modes."""
    model_name: str
    batch_size: int
    seq_len: int
    gpu_name: str
    pytorch_version: str
    cuda_version: str
    timestamp: str = ""
    results: Dict[str, OptimizationResult] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "gpu_name": self.gpu_name,
            "pytorch_version": self.pytorch_version,
            "cuda_version": self.cuda_version,
            "timestamp": self.timestamp,
            "t_floor_sys_us": T_FLOOR_SYS_US,
            "results": {k: vars(v) for k, v in self.results.items()},
        }


# =============================================================================
# HDBI Computation (aligned with calculate_hdbi in utils.py)
# =============================================================================

def compute_hdbi(metrics: Dict[str, Any]) -> Tuple[float, str, Dict[str, float]]:
    """
    Compute HDBI per TaxBreak paper, aligned with utils.py::calculate_hdbi.

    HDBI = T_DeviceActive / (T_DeviceActive + T_Orchestrate)

    Where:
        T_DeviceActive = total kernel execution time (sum of all kernel durations)
        T_Orchestrate  = T_xlat_tax + ΔKT
        ΔKT            = num_kernels × T_floor_sys  (baremetal constant)
        T_xlat_tax     = Σ (aten_op.dur - cuda_launch.dur)  (MEASURED, not residual)

    This matches the main pipeline: T_xlat_tax is independently measured from
    linked CPU event sequences (aten_op → cuda_launch), NOT derived as
    T_wall - T_dev - ΔKT (which would make ΔKT cancel and HDBI = T_dev/T_wall).

    Returns:
        (hdbi_value, classification, breakdown_dict)
    """
    t_device_active_ms = metrics.get("total_kernel_exec_time_ms", 0.0)
    num_kernels = metrics.get("num_kernels", 0)
    t_xlat_tax_ms = metrics.get("total_xlat_tax_ms", 0.0)
    inference_time_ms = metrics.get("inference_time_ms", 0.0)

    # ΔKT: Kernel launch tax (paper Section III-C)
    delta_kt_ms = num_kernels * T_FLOOR_SYS_MS

    # T_Orchestrate = T_xlat_tax + ΔKT (both independently measured/computed)
    t_orchestrate_ms = t_xlat_tax_ms + delta_kt_ms

    denominator = t_device_active_ms + t_orchestrate_ms
    if denominator > 0:
        hdbi = t_device_active_ms / denominator
    else:
        hdbi = 0.0

    hdbi = max(0.0, min(1.0, hdbi))

    # Classification per TaxBreak paper
    if hdbi >= 0.5:
        classification = "device-bound"
    elif hdbi >= 0.2:
        classification = "balanced"
    else:
        classification = "host-bound"

    breakdown = {
        "T_DeviceActive_ms": t_device_active_ms,
        "T_Orchestrate_ms": t_orchestrate_ms,
        "delta_KT_ms": delta_kt_ms,
        "T_xlat_tax_ms": t_xlat_tax_ms,
        "inference_time_ms": inference_time_ms,
    }

    return hdbi, classification, breakdown


# =============================================================================
# Trace Analysis — Sequence Linking (aligned with utils.py::link_sequences)
# =============================================================================

def analyze_trace(trace_path: Path) -> Dict[str, Any]:
    """
    Analyze a PyTorch profiler trace to extract TaxBreak metrics.

    This performs the SAME sequence linking as the main pipeline
    (utils.py::collect_events + link_sequences + calculate_sequence_metrics):

    1. Collect events by category (aten_ops, cuda_launches, kernels)
    2. Link kernels → cuda_launches via correlation ID
    3. Link cuda_launches → aten_ops via external_id / parent hierarchy
    4. Compute per-sequence xlat_tax = aten_op.dur - cuda_launch.dur
    5. Sum xlat_tax across all linked sequences

    GPU metrics (busy, idle, gaps) use GPU timeline only (no cross-timeline math).
    """
    with open(trace_path) as f:
        trace_data = json.load(f)

    events = trace_data.get("traceEvents", [])

    # =========================================================================
    # Phase 1: Collect events by category
    # =========================================================================
    kernel_events = []          # GPU cat="kernel"
    cuda_launch_events = {}     # correlation_id → event (CPU cudaLaunchKernel)
    aten_op_events = {}         # external_id → event (CPU aten::* ops)

    for event in events:
        cat = event.get("cat", "")
        name = event.get("name", "")
        ph = event.get("ph", "")

        if ph != "X":
            continue

        if cat == "kernel":
            kernel_events.append(event)
        elif "cudaLaunchKernel" in name or "cudaGraphLaunch" in name:
            # Index by correlation to match with kernel
            args = event.get("args", {})
            corr_id = args.get("correlation", None)
            if corr_id is not None:
                cuda_launch_events[corr_id] = event
        elif cat == "cpu_op" or (name.startswith("aten::") and cat in ("", "cpu_op", "operator")):
            # ATen ops — index by external_id for linking
            args = event.get("args", {})
            ext_id = args.get("External id", None)
            if ext_id is not None:
                aten_op_events[ext_id] = event

    if not kernel_events:
        return {
            "inference_time_ms": 0,
            "gpu_busy_time_ms": 0,
            "gpu_idle_time_ms": 0,
            "total_kernel_exec_time_ms": 0,
            "num_kernels": 0,
            "num_cuda_launches": len(cuda_launch_events),
            "num_linked_sequences": 0,
            "num_orphan_kernels": 0,
            "total_xlat_tax_ms": 0,
            "avg_kernel_gap_us": 0,
        }

    # =========================================================================
    # Phase 2: Link sequences (kernel → launch → aten_op)
    # =========================================================================
    linked_sequences = []
    orphan_kernels = 0

    for kernel in kernel_events:
        k_args = kernel.get("args", {})
        corr_id = k_args.get("correlation", None)

        if corr_id is None or corr_id not in cuda_launch_events:
            orphan_kernels += 1
            continue

        launch = cuda_launch_events[corr_id]
        l_args = launch.get("args", {})

        # Find the parent aten_op via external_id on the launch event
        ext_id = l_args.get("External id", None)
        aten_op = aten_op_events.get(ext_id, None) if ext_id is not None else None

        linked_sequences.append({
            "kernel": kernel,
            "cuda_launch": launch,
            "aten_op": aten_op,  # May be None
        })

    # =========================================================================
    # Phase 3: Compute per-sequence xlat_tax (CPU timeline, same clock domain)
    # =========================================================================
    # xlat_tax = aten_op.dur - cuda_launch.dur
    # This is the CPU-side translation overhead ABOVE the launch call itself.
    # Both aten_op and cuda_launch are on the CPU timeline (same clock).
    xlat_tax_values_us = []

    for seq in linked_sequences:
        aten_op = seq.get("aten_op")
        launch = seq.get("cuda_launch")

        if aten_op is not None and launch is not None:
            aten_dur = aten_op.get("dur", 0)
            launch_dur = launch.get("dur", 0)
            xlat = max(0.0, aten_dur - launch_dur)
            xlat_tax_values_us.append(xlat)

    total_xlat_tax_us = sum(xlat_tax_values_us)
    total_xlat_tax_ms = total_xlat_tax_us / 1000.0

    # =========================================================================
    # Phase 4: GPU timeline metrics (same clock domain, no cross-timeline)
    # =========================================================================
    kernel_events.sort(key=lambda e: e.get("ts", 0))

    intervals = [
        (e.get("ts", 0), e.get("ts", 0) + e.get("dur", 0))
        for e in kernel_events
    ]
    intervals.sort()

    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    gpu_busy_us = sum(end - start for start, end in merged)
    gpu_span_us = merged[-1][1] - merged[0][0]
    gpu_idle_us = max(0, gpu_span_us - gpu_busy_us)

    total_kernel_exec_us = sum(e.get("dur", 0) for e in kernel_events)

    kernel_gaps = []
    for i in range(1, len(merged)):
        gap = merged[i][0] - merged[i - 1][1]
        if gap > 0:
            kernel_gaps.append(gap)
    avg_kernel_gap_us = (
        sum(kernel_gaps) / len(kernel_gaps) if kernel_gaps else 0.0
    )

    return {
        "inference_time_ms": gpu_span_us / 1000.0,  # placeholder, overridden by CUDA events
        "gpu_busy_time_ms": gpu_busy_us / 1000.0,
        "gpu_idle_time_ms": gpu_idle_us / 1000.0,
        "total_kernel_exec_time_ms": total_kernel_exec_us / 1000.0,
        "num_kernels": len(kernel_events),
        "num_cuda_launches": len(cuda_launch_events),
        "num_linked_sequences": len(linked_sequences),
        "num_orphan_kernels": orphan_kernels,
        "total_xlat_tax_ms": total_xlat_tax_ms,
        "avg_kernel_gap_us": avg_kernel_gap_us,
        "avg_xlat_tax_us": (total_xlat_tax_us / len(xlat_tax_values_us)) if xlat_tax_values_us else 0.0,
    }


# =============================================================================
# Model Loading
# =============================================================================

def load_model(
    model_name: str,
    attn_implementation: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> Tuple[Any, Any]:
    """Load a HuggingFace model with specified attention implementation."""
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers library required")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=True
    )
    if hasattr(config, "pad_token_id") and config.pad_token_id is None:
        config.pad_token_id = tokenizer.eos_token_id

    # Handle transformers version compatibility for dtype parameter
    tv = transformers.__version__
    major, minor = int(tv.split(".")[0]), int(tv.split(".")[1])
    use_new_dtype = (major > 4) or (major == 4 and minor >= 47)

    load_kwargs = dict(
        config=config,
        device_map=device,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )

    if use_new_dtype:
        load_kwargs["dtype"] = dtype
    else:
        load_kwargs["torch_dtype"] = dtype

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, **load_kwargs
    ).eval()

    return model, tokenizer


def generate_inputs(
    tokenizer: Any,
    batch_size: int,
    seq_len: int,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """Generate synthetic inputs for the model."""
    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(
        low=1,
        high=vocab_size - 1,
        size=(batch_size, seq_len),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


# =============================================================================
# Profiling Core
# =============================================================================

def _profile_mode(
    mode_name: str,
    model: Any,
    inputs: Dict[str, torch.Tensor],
    output_dir: Path,
    warmup: int,
    runs: int,
    forward_fn=None,
) -> OptimizationResult:
    """
    Generic profiling function for any optimization mode.

    Strategy:
    - Phase 1: Wall-clock timing via CUDA events (multi-run average)
    - Phase 2: Single-run trace capture for sequence linking & kernel counts
    - Phase 3: Compute HDBI from independently measured components
    """
    if forward_fn is None:
        def forward_fn():
            with torch.no_grad():
                return model(**inputs)

    # Warmup
    print(f"Warmup ({warmup} iterations)...")
    sys.stdout.flush()
    for _ in range(warmup):
        forward_fn()
    torch.cuda.synchronize()

    # -------------------------------------------------------------------------
    # Phase 1: Wall-clock timing via CUDA events (accurate, multi-run)
    # -------------------------------------------------------------------------
    print(f"Timing ({runs} iterations)...")
    sys.stdout.flush()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(runs):
        forward_fn()
    end_event.record()
    torch.cuda.synchronize()

    wall_time_ms = start_event.elapsed_time(end_event) / runs

    # -------------------------------------------------------------------------
    # Phase 2: Trace capture for SINGLE run (sequence linking)
    # -------------------------------------------------------------------------
    print("Capturing trace (1 iteration)...")
    sys.stdout.flush()

    trace_dir = output_dir / mode_name
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / "trace.json"

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        forward_fn()
        torch.cuda.synchronize()

    prof.export_chrome_trace(str(trace_path))

    # -------------------------------------------------------------------------
    # Phase 3: Analyze trace with sequence linking
    # -------------------------------------------------------------------------
    metrics = analyze_trace(trace_path)

    # Override inference_time with CUDA event measurement
    metrics["inference_time_ms"] = wall_time_ms

    # Compute HDBI using independently measured components
    hdbi, hdbi_class, breakdown = compute_hdbi(metrics)

    # Sanity check: T_DeviceActive + T_Orchestrate vs T_wall
    t_dev = breakdown["T_DeviceActive_ms"]
    t_orch = breakdown["T_Orchestrate_ms"]
    t_sum = t_dev + t_orch
    coverage = (t_sum / wall_time_ms * 100) if wall_time_ms > 0 else 0

    result = OptimizationResult(
        mode=mode_name,
        inference_time_ms=wall_time_ms,
        gpu_busy_time_ms=metrics["gpu_busy_time_ms"],
        gpu_idle_time_ms=metrics["gpu_idle_time_ms"],
        total_kernel_exec_time_ms=breakdown["T_DeviceActive_ms"],
        num_kernels=metrics["num_kernels"],
        num_cuda_launches=metrics["num_cuda_launches"],
        num_linked_sequences=metrics["num_linked_sequences"],
        num_orphan_kernels=metrics["num_orphan_kernels"],
        t_orchestrate_ms=breakdown["T_Orchestrate_ms"],
        t_xlat_tax_ms=breakdown["T_xlat_tax_ms"],
        delta_kt_ms=breakdown["delta_KT_ms"],
        hdbi=hdbi,
        hdbi_class=hdbi_class,
        avg_kernel_gap_us=metrics["avg_kernel_gap_us"],
    )

    print(f"  T_wall:             {result.inference_time_ms:.2f} ms (CUDA event, avg of {runs})")
    print(f"  T_DeviceActive:     {result.total_kernel_exec_time_ms:.2f} ms (Σ kernel durations)")
    print(f"  GPU busy time:      {result.gpu_busy_time_ms:.2f} ms (merged intervals)")
    print(f"  T_Orchestrate:      {result.t_orchestrate_ms:.2f} ms")
    print(f"    ├─ T_xlat_tax:    {result.t_xlat_tax_ms:.2f} ms (Σ aten.dur − launch.dur, {result.num_linked_sequences} sequences)")
    print(f"    └─ ΔKT:           {result.delta_kt_ms:.2f} ms ({result.num_kernels} × {T_FLOOR_SYS_US:.1f}µs)")
    print(f"  Num kernels:        {result.num_kernels} ({result.num_orphan_kernels} orphans)")
    print(f"  Num CUDA launches:  {result.num_cuda_launches}")
    print(f"  Avg xlat tax:       {metrics.get('avg_xlat_tax_us', 0):.1f} µs/sequence")
    print(f"  Avg kernel gap:     {result.avg_kernel_gap_us:.1f} µs (GPU timeline)")
    print(f"  HDBI:               {result.hdbi:.3f} ({result.hdbi_class})")
    print(f"  Coverage:           {coverage:.1f}% (T_dev+T_orch vs T_wall)")
    sys.stdout.flush()

    return result


# =============================================================================
# Optimization Mode Profilers
# =============================================================================

def _make_error_result(mode: str, error: str) -> OptimizationResult:
    """Create a failed OptimizationResult."""
    return OptimizationResult(
        mode=mode,
        inference_time_ms=0, gpu_busy_time_ms=0, gpu_idle_time_ms=0,
        total_kernel_exec_time_ms=0,
        num_kernels=0, num_cuda_launches=0,
        num_linked_sequences=0, num_orphan_kernels=0,
        t_orchestrate_ms=0, t_xlat_tax_ms=0, delta_kt_ms=0,
        hdbi=0, hdbi_class="unknown", avg_kernel_gap_us=0,
        status="error", error=error,
    )


def _make_skipped_result(mode: str, reason: str) -> OptimizationResult:
    """Create a skipped OptimizationResult."""
    return OptimizationResult(
        mode=mode,
        inference_time_ms=0, gpu_busy_time_ms=0, gpu_idle_time_ms=0,
        total_kernel_exec_time_ms=0,
        num_kernels=0, num_cuda_launches=0,
        num_linked_sequences=0, num_orphan_kernels=0,
        t_orchestrate_ms=0, t_xlat_tax_ms=0, delta_kt_ms=0,
        hdbi=0, hdbi_class="unknown", avg_kernel_gap_us=0,
        status="skipped", error=reason,
    )


def profile_eager_mode(
    model_name: str,
    batch_size: int,
    seq_len: int,
    output_dir: Path,
    warmup: int = 5,
    runs: int = 3,
    dtype: torch.dtype = torch.bfloat16,
) -> OptimizationResult:
    """Profile model in eager mode (baseline)."""
    print("\n" + "=" * 60)
    print("EAGER MODE (Baseline)")
    print("=" * 60)
    sys.stdout.flush()

    try:
        model, tokenizer = load_model(model_name, "eager", dtype)
        inputs = generate_inputs(tokenizer, batch_size, seq_len)
        result = _profile_mode("eager", model, inputs, output_dir, warmup, runs)

        del model, tokenizer, inputs
        gc.collect()
        torch.cuda.empty_cache()
        return result

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        return _make_error_result("eager", str(e))


def profile_flash_attention(
    model_name: str,
    batch_size: int,
    seq_len: int,
    output_dir: Path,
    warmup: int = 5,
    runs: int = 3,
    dtype: torch.dtype = torch.bfloat16,
) -> OptimizationResult:
    """Profile model with FlashAttention-2."""
    print("\n" + "=" * 60)
    print("FLASH ATTENTION MODE")
    print("=" * 60)
    sys.stdout.flush()

    if not HAS_FLASH_ATTN:
        print("  SKIPPED: flash_attn package not installed")
        sys.stdout.flush()
        return _make_skipped_result("flash_attention", "flash_attn not installed")

    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability(0)
        if cc[0] < 8:
            msg = f"GPU CC {cc[0]}.{cc[1]} < 8.0"
            print(f"  SKIPPED: {msg}")
            sys.stdout.flush()
            return _make_skipped_result("flash_attention", msg)

    try:
        model, tokenizer = load_model(model_name, "flash_attention_2", dtype)
        inputs = generate_inputs(tokenizer, batch_size, seq_len)
        result = _profile_mode("flash_attention", model, inputs, output_dir, warmup, runs)

        del model, tokenizer, inputs
        gc.collect()
        torch.cuda.empty_cache()
        return result

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        return _make_error_result("flash_attention", str(e))


def profile_torch_compile(
    model_name: str,
    batch_size: int,
    seq_len: int,
    output_dir: Path,
    warmup: int = 5,
    runs: int = 3,
    dtype: torch.dtype = torch.bfloat16,
) -> OptimizationResult:
    """
    Profile model with torch.compile(mode="reduce-overhead").

    Uses CUDA Graphs internally via Inductor.
    """
    print("\n" + "=" * 60)
    print("TORCH.COMPILE MODE (reduce-overhead / CUDA Graphs)")
    print("=" * 60)
    sys.stdout.flush()

    try:
        model, tokenizer = load_model(model_name, "sdpa", dtype)
        inputs = generate_inputs(tokenizer, batch_size, seq_len)

        print("Compiling model with torch.compile(mode='reduce-overhead')...")
        sys.stdout.flush()

        compiled_model = torch.compile(model, mode="reduce-overhead")

        compile_warmup = max(warmup, 10)

        def forward_fn():
            with torch.no_grad():
                return compiled_model(**inputs)

        result = _profile_mode(
            "torch_compile", compiled_model, inputs, output_dir,
            compile_warmup, runs, forward_fn=forward_fn,
        )

        del compiled_model, model, tokenizer, inputs
        gc.collect()
        torch.cuda.empty_cache()
        torch._dynamo.reset()
        return result

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        try:
            torch._dynamo.reset()
        except Exception:
            pass
        return _make_error_result("torch_compile", str(e))


def profile_cuda_graphs_manual(
    model_name: str,
    batch_size: int,
    seq_len: int,
    output_dir: Path,
    warmup: int = 5,
    runs: int = 3,
    dtype: torch.dtype = torch.bfloat16,
) -> OptimizationResult:
    """
    Profile model with manual CUDA Graph capture.
    Falls back gracefully if capture fails (common with HF models).
    """
    print("\n" + "=" * 60)
    print("CUDA GRAPHS MODE (Manual Capture)")
    print("=" * 60)
    sys.stdout.flush()

    try:
        model, tokenizer = load_model(model_name, "sdpa", dtype)
        inputs = generate_inputs(tokenizer, batch_size, seq_len)

        static_input_ids = inputs["input_ids"].clone()
        static_attention_mask = inputs["attention_mask"].clone()

        print(f"Warmup ({warmup} iterations)...")
        sys.stdout.flush()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                with torch.no_grad():
                    _ = model(
                        input_ids=static_input_ids,
                        attention_mask=static_attention_mask,
                    )
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        print("Capturing CUDA Graph...")
        sys.stdout.flush()
        g = torch.cuda.CUDAGraph()

        try:
            with torch.cuda.graph(g):
                with torch.no_grad():
                    static_output = model(
                        input_ids=static_input_ids,
                        attention_mask=static_attention_mask,
                    )
            print("CUDA Graph captured successfully!")
            sys.stdout.flush()
        except RuntimeError as capture_error:
            msg = str(capture_error)[:200]
            print(f"  CUDA Graph capture failed: {msg}")
            print("  This is expected for models with dynamic ops.")
            print("  Use torch.compile(mode='reduce-overhead') instead.")
            sys.stdout.flush()
            del model, tokenizer, inputs
            gc.collect()
            torch.cuda.empty_cache()
            return _make_error_result(
                "cuda_graphs_manual", f"Capture failed: {msg}"
            )

        def replay_fn():
            g.replay()

        result = _profile_mode(
            "cuda_graphs_manual", None, None, output_dir,
            warmup=3, runs=runs, forward_fn=replay_fn,
        )

        del g, model, tokenizer, inputs, static_output
        gc.collect()
        torch.cuda.empty_cache()
        return result

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        return _make_error_result("cuda_graphs_manual", str(e))


# =============================================================================
# Main Comparison
# =============================================================================

def run_optimization_comparison(
    model_name: str,
    output_dir: Path,
    batch_size: int = 1,
    seq_len: int = 512,
    warmup: int = 5,
    runs: int = 3,
    dtype_str: str = "bfloat16",
) -> ComparisonReport:
    """Run TaxBreak decomposition under all optimization modes."""
    print("=" * 80)
    print("OPTIMIZATION STACK COMPARISON (Reviewer C)")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Transformers: {transformers.__version__ if HAS_TRANSFORMERS else 'N/A'}")
    print(f"FlashAttention: {'available' if HAS_FLASH_ATTN else 'not installed'}")
    print(f"T_floor_sys: {T_FLOOR_SYS_US} µs")
    print("=" * 80)
    sys.stdout.flush()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str, torch.bfloat16)

    report = ComparisonReport(
        model_name=model_name,
        batch_size=batch_size,
        seq_len=seq_len,
        gpu_name=torch.cuda.get_device_name(0),
        pytorch_version=torch.__version__,
        cuda_version=torch.version.cuda,
    )

    report.results["eager"] = profile_eager_mode(
        model_name, batch_size, seq_len, output_dir, warmup, runs, dtype
    )
    report.results["flash_attention"] = profile_flash_attention(
        model_name, batch_size, seq_len, output_dir, warmup, runs, dtype
    )
    report.results["torch_compile"] = profile_torch_compile(
        model_name, batch_size, seq_len, output_dir, warmup, runs, dtype
    )
    report.results["cuda_graphs_manual"] = profile_cuda_graphs_manual(
        model_name, batch_size, seq_len, output_dir, warmup, runs, dtype
    )

    # =========================================================================
    # Summary Table
    # =========================================================================
    print("\n" + "=" * 150)
    print("SUMMARY COMPARISON")
    print("=" * 150)
    header = (
        f"{'Mode':<25} {'T_wall':>8} {'T_Dev':>8} {'T_Orch':>8} "
        f"{'T_xlat':>8} {'ΔKT':>6} {'Kern':>6} {'Linked':>7} {'Orphan':>7} "
        f"{'Launch':>7} {'Gap(µs)':>8} {'HDBI':>6} {'Class':>13} {'Cover%':>7}"
    )
    print(header)
    print("-" * 150)
    sys.stdout.flush()

    baseline = report.results.get("eager")

    for mode, result in report.results.items():
        if result.status == "ok":
            speedup = ""
            if (baseline and baseline.status == "ok"
                    and baseline.inference_time_ms > 0
                    and mode != "eager"):
                ratio = baseline.inference_time_ms / max(result.inference_time_ms, 1e-6)
                speedup = f" ({ratio:.2f}x)"

            t_sum = result.total_kernel_exec_time_ms + result.t_orchestrate_ms
            coverage = (t_sum / result.inference_time_ms * 100) if result.inference_time_ms > 0 else 0

            row = (
                f"{mode:<25} "
                f"{result.inference_time_ms:>8.2f}{speedup} "
                f"{result.total_kernel_exec_time_ms:>8.2f} "
                f"{result.t_orchestrate_ms:>8.2f} "
                f"{result.t_xlat_tax_ms:>8.2f} "
                f"{result.delta_kt_ms:>6.2f} "
                f"{result.num_kernels:>6} "
                f"{result.num_linked_sequences:>7} "
                f"{result.num_orphan_kernels:>7} "
                f"{result.num_cuda_launches:>7} "
                f"{result.avg_kernel_gap_us:>8.1f} "
                f"{result.hdbi:>6.3f} "
                f"{result.hdbi_class:>13} "
                f"{coverage:>6.1f}%"
            )
            print(row)
        elif result.status == "skipped":
            print(f"{mode:<25} {'SKIP':>8}  {result.error or ''}")
        else:
            print(f"{mode:<25} {'ERR':>8}  {result.error or ''}")

    print("=" * 150)
    sys.stdout.flush()

    # =========================================================================
    # Key Insights
    # =========================================================================
    print("\n" + "=" * 80)
    print("KEY INSIGHTS FOR REVIEWER C")
    print("=" * 80)

    eager = report.results.get("eager")
    flash = report.results.get("flash_attention")
    compiled = report.results.get("torch_compile")
    manual_cg = report.results.get("cuda_graphs_manual")

    if eager and eager.status == "ok":
        print(f"\n1. EAGER MODE (Baseline):")
        print(f"   T_wall = {eager.inference_time_ms:.2f} ms")
        print(f"   T_DeviceActive = {eager.total_kernel_exec_time_ms:.2f} ms")
        print(f"   T_Orchestrate = {eager.t_orchestrate_ms:.2f} ms "
              f"(T_xlat={eager.t_xlat_tax_ms:.2f} + ΔKT={eager.delta_kt_ms:.2f})")
        print(f"   HDBI = {eager.hdbi:.3f} ({eager.hdbi_class})")
        print(f"   {eager.num_linked_sequences} linked sequences, "
              f"{eager.num_orphan_kernels} orphan kernels")

    if flash and flash.status == "ok" and eager and eager.status == "ok":
        kernel_reduction = 100 * (1 - flash.num_kernels / max(eager.num_kernels, 1))
        dkt_reduction = 100 * (1 - flash.delta_kt_ms / max(eager.delta_kt_ms, 1e-6))
        xlat_reduction = 100 * (1 - flash.t_xlat_tax_ms / max(eager.t_xlat_tax_ms, 1e-6))
        print(f"\n2. FLASH ATTENTION (Kernel Fusion):")
        print(f"   Kernels: {flash.num_kernels} vs {eager.num_kernels} "
              f"(-{kernel_reduction:.0f}%)")
        print(f"   ΔKT: {flash.delta_kt_ms:.2f} vs {eager.delta_kt_ms:.2f} ms "
              f"(-{dkt_reduction:.0f}%)")
        print(f"   T_xlat_tax: {flash.t_xlat_tax_ms:.2f} vs {eager.t_xlat_tax_ms:.2f} ms "
              f"({xlat_reduction:+.0f}%)")
        print(f"   T_Orchestrate: {flash.t_orchestrate_ms:.2f} vs "
              f"{eager.t_orchestrate_ms:.2f} ms")
        print(f"   HDBI: {flash.hdbi:.3f} vs {eager.hdbi:.3f} "
              f"(Δ={flash.hdbi - eager.hdbi:+.3f})")

    if compiled and compiled.status == "ok" and eager and eager.status == "ok":
        launch_reduction = 100 * (1 - compiled.num_cuda_launches / max(eager.num_cuda_launches, 1))
        print(f"\n3. TORCH.COMPILE (reduce-overhead / CUDA Graphs via Inductor):")
        print(f"   Launches: {compiled.num_cuda_launches} vs {eager.num_cuda_launches} "
              f"(-{launch_reduction:.0f}%)")
        print(f"   T_xlat_tax: {compiled.t_xlat_tax_ms:.2f} vs {eager.t_xlat_tax_ms:.2f} ms")
        print(f"   T_Orchestrate: {compiled.t_orchestrate_ms:.2f} vs "
              f"{eager.t_orchestrate_ms:.2f} ms")
        print(f"   HDBI: {compiled.hdbi:.3f} vs {eager.hdbi:.3f} "
              f"(Δ={compiled.hdbi - eager.hdbi:+.3f})")
        if compiled.inference_time_ms > 0 and eager.inference_time_ms > 0:
            speedup = eager.inference_time_ms / compiled.inference_time_ms
            print(f"   Speedup: {speedup:.2f}x vs eager")
    elif compiled and compiled.status == "error":
        print(f"\n3. TORCH.COMPILE: FAILED — {compiled.error}")

    if manual_cg and manual_cg.status == "error":
        print(f"\n4. MANUAL CUDA GRAPHS: FAILED (expected)")
        print(f"   HF models use dynamic ops incompatible with manual capture.")
        print(f"   torch.compile(mode='reduce-overhead') is the recommended path.")

    print()
    print("METHODOLOGY (aligned with main SODA pipeline):")
    print(f"  HDBI = T_DeviceActive / (T_DeviceActive + T_Orchestrate)")
    print(f"  T_Orchestrate = T_xlat_tax + ΔKT")
    print(f"  T_xlat_tax = Σ (aten_op.dur − cuda_launch.dur)  [measured, not residual]")
    print(f"  ΔKT = num_kernels × T_floor_sys ({T_FLOOR_SYS_US} µs)")
    print(f"  T_wall: CUDA events (avg of {runs} runs)")
    print(f"  Trace: single-run capture (accurate sequence linking)")
    print(f"  Coverage = (T_dev + T_orch) / T_wall  [sanity check]")
    print("\n" + "=" * 80)
    sys.stdout.flush()

    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "optimization_comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"\nReport saved to: {report_path}")
    sys.stdout.flush()

    return report


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Optimization Stack Comparison for Reviewer C"
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/optimization_comparison"),
        help="Output directory",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        return 1

    try:
        run_optimization_comparison(
            model_name=args.model,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            warmup=args.warmup,
            runs=args.runs,
            dtype_str=args.dtype,
        )
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())