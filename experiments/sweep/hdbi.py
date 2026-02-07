#!/usr/bin/env python3
"""
HSB (Hardware-Software Inversion) Metric Calculation

This module implements the HSB metric as described in the ISPASS paper.
HSB measures the balance between hardware utilization and software overhead.

HSB = 1 - (T_Exposed / T_Structural)

Where:
- T_Exposed: Total GPU idle time during inference
- T_Structural: Sum of per-kernel framework overheads

Interpretation:
- HSB = 1: Fully hardware-bound (all overhead is hidden by GPU computation)
- HSB = 0: Balanced (exposed overhead equals structural overhead)
- HSB < 0: Framework-bound (CPU overhead is the bottleneck)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Union


@dataclass
class HSBResult:
    """Container for HSB calculation results."""
    batch_size: int
    seq_len: int
    max_new_tokens: int
    inference_time_ms: Optional[float]
    gpu_busy_time_ms: Optional[float]
    gpu_idle_time_ms: Optional[float]
    t_exposed_us: Optional[float]
    t_structural_us: Optional[float]
    hsb: Optional[float]
    num_kernels: int = 0
    num_kernels_in_lut: int = 0
    status: str = "ok"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "max_new_tokens": self.max_new_tokens,
            "inference_time_ms": self.inference_time_ms,
            "gpu_busy_time_ms": self.gpu_busy_time_ms,
            "gpu_idle_time_ms": self.gpu_idle_time_ms,
            "t_exposed_us": self.t_exposed_us,
            "t_structural_us": self.t_structural_us,
            "hsb": self.hsb,
            "num_kernels": self.num_kernels,
            "num_kernels_in_lut": self.num_kernels_in_lut,
            "status": self.status,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'HSBResult':
        """Create from dictionary."""
        return cls(
            batch_size=d.get("batch_size", 0),
            seq_len=d.get("seq_len", 0),
            max_new_tokens=d.get("max_new_tokens", 0),
            inference_time_ms=d.get("inference_time_ms"),
            gpu_busy_time_ms=d.get("gpu_busy_time_ms"),
            gpu_idle_time_ms=d.get("gpu_idle_time_ms"),
            t_exposed_us=d.get("t_exposed_us"),
            t_structural_us=d.get("t_structural_us"),
            hsb=d.get("hsb"),
            num_kernels=d.get("num_kernels", 0),
            num_kernels_in_lut=d.get("num_kernels_in_lut", 0),
            status=d.get("status", "ok"),
        )


def load_taxbreak_lut(taxbreak_file: Union[str, Path]) -> Dict[str, float]:
    """
    Load taxbreak.json and build a kernel -> T_fo lookup table.
    
    The taxbreak.json contains per-kernel framework overhead measurements.
    We extract the T_fo (total framework overhead) for each kernel.
    
    Args:
        taxbreak_file: Path to taxbreak.json
    
    Returns:
        Dictionary mapping kernel_name -> T_fo (in microseconds)
    """
    taxbreak_file = Path(taxbreak_file)
    
    if not taxbreak_file.exists():
        print(f"Warning: taxbreak.json not found at {taxbreak_file}")
        return {}
    
    with open(taxbreak_file, "r") as f:
        taxbreak_data = json.load(f)
    
    lut: Dict[str, float] = {}
    
    # Extract per-kernel T_fo from the taxbreak data
    data = taxbreak_data.get("data", [])
    
    for entry in data:
        kernel_name = entry.get("kernel")
        if not kernel_name:
            continue
        
        # T_fo is the total framework overhead
        t_fo = entry.get("T_fo")
        
        if t_fo is not None:
            lut[kernel_name] = float(t_fo)
        else:
            # Fallback: try to compute from components
            t_py = entry.get("T_py") or 0
            t_aten_xlat = entry.get("T_aten_xlat") or entry.get("T_aten") or 0
            t_sys = entry.get("T_sys") or 0
            
            # Also include library overhead for GEMM kernels
            t_lib_setup = entry.get("T_lib_setup") or 0
            t_lib_heur = entry.get("T_lib_heur") or 0
            t_lib_shim = entry.get("T_lib_shim") or 0
            
            computed_t_fo = t_py + t_aten_xlat + t_sys + t_lib_setup + t_lib_heur + t_lib_shim
            
            if computed_t_fo > 0:
                lut[kernel_name] = computed_t_fo
    
    return lut


def calculate_t_exposed(
    inference_time_us: float,
    gpu_busy_time_us: float,
) -> float:
    """
    Calculate T_Exposed (exposed framework overhead).
    
    T_Exposed represents the GPU idle time during inference - the time
    when the GPU is waiting for the CPU to prepare/launch the next kernel.
    
    T_Exposed = Total_Inference_Time - GPU_Busy_Time
    
    Args:
        inference_time_us: Total end-to-end inference time in microseconds
        gpu_busy_time_us: Total time GPU was actively executing kernels
    
    Returns:
        T_Exposed in microseconds (clamped to >= 0)
    """
    t_exposed = inference_time_us - gpu_busy_time_us
    return max(0.0, t_exposed)


def calculate_t_structural(
    sequences: List[Dict[str, Any]],
    taxbreak_lut: Dict[str, float],
    default_t_fo: Optional[float] = None,
) -> float:
    """
    Calculate T_Structural (total structural framework overhead).
    
    T_Structural is the sum of per-kernel framework overheads (T_fo) for
    all kernels executed during inference.
    
    Args:
        sequences: List of kernel sequences from the trace
        taxbreak_lut: Lookup table mapping kernel_name -> T_fo
        default_t_fo: Default T_fo to use for kernels not in LUT.
                      If None, uses average of known kernels or 0.
    
    Returns:
        T_Structural in microseconds
    """
    if not sequences:
        return 0.0
    
    t_structural = 0.0
    kernels_found = 0
    kernels_missing = 0
    
    # Calculate default T_fo from LUT average if not provided
    if default_t_fo is None and taxbreak_lut:
        default_t_fo = sum(taxbreak_lut.values()) / len(taxbreak_lut)
    elif default_t_fo is None:
        default_t_fo = 0.0
    
    for seq in sequences:
        kernel = seq.get("kernel", {})
        if isinstance(kernel, dict):
            kernel_name = kernel.get("name", "")
        else:
            kernel_name = str(kernel) if kernel else ""
        
        if not kernel_name:
            continue
        
        # Look up T_fo in LUT
        if kernel_name in taxbreak_lut:
            t_fo = taxbreak_lut[kernel_name]
            kernels_found += 1
        else:
            # Try partial match (kernel names may be truncated)
            matched = False
            for lut_name, lut_t_fo in taxbreak_lut.items():
                if kernel_name in lut_name or lut_name in kernel_name:
                    t_fo = lut_t_fo
                    kernels_found += 1
                    matched = True
                    break
            
            if not matched:
                # Use default for missing kernels
                t_fo = default_t_fo
                kernels_missing += 1
        
        # Get frequency (number of times this kernel was invoked)
        freq = seq.get("freq", 1)
        if freq is None:
            freq = 1
        
        t_structural += t_fo * freq
    
    return t_structural


def calculate_hsb(
    t_exposed: float,
    t_structural: float,
    epsilon: float = 1e-6,
) -> float:
    """
    Calculate the Hardware-Software Inversion (HSB) metric.
    
    HSB = 1 - (T_Exposed / T_Structural)
    
    Interpretation:
    - HSB = 1: Fully hardware-bound (all overhead hidden by GPU)
    - HSB = 0: Balanced state
    - HSB < 0: Framework-bound (CPU overhead is bottleneck)
    - HSB = -∞: T_Structural ≈ 0 but T_Exposed > 0 (edge case)
    
    Args:
        t_exposed: Exposed framework overhead (GPU idle time) in microseconds
        t_structural: Structural framework overhead (sum of T_fo) in microseconds
        epsilon: Small value to avoid division by zero
    
    Returns:
        HSB value (typically in range [-inf, 1])
    """
    if t_structural < epsilon:
        # Edge case: no structural overhead
        if t_exposed < epsilon:
            # Both are essentially zero - perfectly balanced
            return 1.0
        else:
            # Exposed overhead but no structural overhead - severely framework-bound
            # Clamp to a reasonable minimum
            return -10.0
    
    ratio = t_exposed / t_structural
    hsb = 1.0 - ratio
    
    return hsb


def classify_hsb(hsb: float) -> str:
    """
    Classify HSB value into a human-readable category.
    
    Args:
        hsb: HSB value
    
    Returns:
        Classification string
    """
    if hsb is None:
        return "unknown"
    
    if hsb >= 0.9:
        return "hardware-bound"
    elif hsb >= 0.7:
        return "mostly-hardware-bound"
    elif hsb >= 0.3:
        return "balanced"
    elif hsb >= 0.0:
        return "mostly-framework-bound"
    else:
        return "framework-bound"


def compute_hsb_from_report(
    report_file: Union[str, Path],
    taxbreak_lut: Dict[str, float],
) -> Optional[HSBResult]:
    """
    Compute HSB from a SODA report.json file.
    
    Args:
        report_file: Path to report.json
        taxbreak_lut: Pre-loaded taxbreak lookup table
    
    Returns:
        HSBResult or None if computation fails
    """
    report_file = Path(report_file)
    
    if not report_file.exists():
        return None
    
    with open(report_file, "r") as f:
        report = json.load(f)
    
    metadata = report.get("metadata", {})
    config = metadata.get("config", {})
    metrics = report.get("performance_metrics", {})
    
    # Check for OOM
    if metrics.get("inference_time_ms") == "OOM":
        return HSBResult(
            batch_size=config.get("batch_size", 0),
            seq_len=config.get("seq_len", 0),
            max_new_tokens=config.get("max_new_tokens", 0),
            inference_time_ms=None,
            gpu_busy_time_ms=None,
            gpu_idle_time_ms=None,
            t_exposed_us=None,
            t_structural_us=None,
            hsb=None,
            status="oom",
        )
    
    # Extract metrics
    inference_time_ms = metrics.get("inference_time_ms", 0)
    gpu_busy_time_ms = metrics.get("gpu_busy_time_ms", 0)
    gpu_idle_time_ms = metrics.get("gpu_idle_time_ms", 0)
    num_kernels = metrics.get("num_total_kernels", 0)
    
    # Convert to microseconds
    inference_time_us = inference_time_ms * 1000.0
    gpu_busy_time_us = gpu_busy_time_ms * 1000.0
    
    # Calculate T_Exposed
    t_exposed = calculate_t_exposed(inference_time_us, gpu_busy_time_us)
    
    # For T_Structural, we need the kernel sequences
    # If not available in report, estimate from framework_overhead
    framework_overhead = metrics.get("framework_overhead", {})
    t_exposed_from_report = framework_overhead.get("T_exposed_ms", 0) * 1000.0
    
    # Estimate T_Structural from taxbreak LUT average * num_kernels
    # This is an approximation when we don't have per-kernel data
    if taxbreak_lut:
        avg_t_fo = sum(taxbreak_lut.values()) / len(taxbreak_lut)
        t_structural = avg_t_fo * num_kernels
    else:
        # Fallback: use total_xlat_tax + total_launch_tax as proxy
        total_xlat_tax_ms = metrics.get("total_xlat_tax_ms", 0)
        total_launch_tax_ms = metrics.get("total_launch_tax_ms", 0)
        t_structural = (total_xlat_tax_ms + total_launch_tax_ms) * 1000.0
    
    # Calculate HSB
    hsb = calculate_hsb(t_exposed, t_structural)
    
    return HSBResult(
        batch_size=config.get("batch_size", 0),
        seq_len=config.get("seq_len", 0),
        max_new_tokens=config.get("max_new_tokens", 0),
        inference_time_ms=inference_time_ms,
        gpu_busy_time_ms=gpu_busy_time_ms,
        gpu_idle_time_ms=gpu_idle_time_ms,
        t_exposed_us=t_exposed,
        t_structural_us=t_structural,
        hsb=hsb,
        num_kernels=num_kernels,
        num_kernels_in_lut=len(taxbreak_lut),
        status="ok",
    )