"""
Carbon footprint estimation for LLM inference on NVIDIA GPUs.

Methodology:
    estimated_power_W = GPU_TDP_W × (gpu_utilization / 100)
    gpu_energy_Wh     = estimated_power_W × inference_time_s / 3600
    total_energy_Wh   = gpu_energy_Wh × PUE          (system + cooling)
    carbon_gCO2eq     = total_energy_Wh × carbon_intensity_g_kwh / 1000

TDP values from NVIDIA datasheets. Carbon intensity presets from IEA/EPA 2023.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# GPU TDP lookup (W at full load) — NVIDIA datasheets
# ---------------------------------------------------------------------------

GPU_TDP_W: Dict[str, float] = {
    "H100 SXM":  700.0,
    "H100 PCIe": 350.0,
    "H200 SXM":  700.0,
    "H200 NVL":  700.0,   # PCIe form-factor, 96 GB HBM3e; same per-card TDP as SXM
    "A100 SXM":  400.0,
    "A100 PCIe": 300.0,
    "V100 SXM2": 300.0,
    "A6000":     300.0,
    "L40S":      350.0,
    # Blackwell workstation — RTX 6000 Blackwell (96 GB GDDR7)
    "RTX 6000 Blackwell": 600.0,
    # Blackwell data-center — B200 SXM5 / GB200 NVL (per-GPU)
    "B200 SXM":  1000.0,
    "GB200 NVL": 1000.0,
}

# Fuzzy match patterns — same order / logic as roofline.py _GPU_PATTERNS
_TDP_PATTERNS = [
    # Blackwell — match before any generic H/A patterns
    (re.compile(r"RTX\s*6000.*Blackwell", re.IGNORECASE), "RTX 6000 Blackwell"),
    (re.compile(r"GB200.*NVL|NVL.*GB200", re.IGNORECASE), "GB200 NVL"),
    (re.compile(r"B200.*SXM|B200", re.IGNORECASE), "B200 SXM"),
    # Hopper
    (re.compile(r"H200.*NVL", re.IGNORECASE), "H200 NVL"),
    (re.compile(r"H200.*SXM", re.IGNORECASE), "H200 SXM"),
    # H200 without variant qualifier — default to SXM (most common data-centre config)
    (re.compile(r"H200", re.IGNORECASE), "H200 SXM"),
    (re.compile(r"H100.*SXM", re.IGNORECASE), "H100 SXM"),
    (re.compile(r"H100.*PCIe|H100.*PCI", re.IGNORECASE), "H100 PCIe"),
    (re.compile(r"A100.*SXM", re.IGNORECASE), "A100 SXM"),
    (re.compile(r"A100.*PCIe|A100.*PCI", re.IGNORECASE), "A100 PCIe"),
    (re.compile(r"V100.*SXM", re.IGNORECASE), "V100 SXM2"),
    (re.compile(r"A6000", re.IGNORECASE), "A6000"),
    (re.compile(r"L40S", re.IGNORECASE), "L40S"),
    # Fallback: variant not in name
    (re.compile(r"H100", re.IGNORECASE), "H100 SXM"),
    (re.compile(r"A100", re.IGNORECASE), "A100 SXM"),
]


# ---------------------------------------------------------------------------
# Regional carbon intensity presets (gCO2eq / kWh)
# ---------------------------------------------------------------------------

CARBON_INTENSITY_PRESETS: Dict[str, float] = {
    "us":     386.0,  # US national average (EPA 2023)
    "eu":     295.0,  # EU average (EEA 2023)
    "global": 475.0,  # Global average (IEA 2023)
    "fr":      58.0,  # France — nuclear-heavy grid
    "de":     380.0,  # Germany
    "cn":     581.0,  # China
}

DEFAULT_CARBON_INTENSITY_G_KWH: float = 400.0  # conservative global estimate
DEFAULT_PUE: float = 1.1                        # modest data-center overhead


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_gpu_tdp(device_name: str) -> Optional[float]:
    """Return GPU TDP (watts) by fuzzy-matching a device name string.

    Args:
        device_name: String from ``torch.cuda.get_device_name()``,
            e.g. ``"NVIDIA H100 80GB HBM3"``.

    Returns:
        TDP in watts, or ``None`` if the GPU is not recognised.
    """
    for pattern, key in _TDP_PATTERNS:
        if pattern.search(device_name):
            return GPU_TDP_W[key]
    return None


def compute_carbon_footprint(
    inference_time_s: float,
    gpu_tdp_w: float,
    gpu_util_pct: float,
    batch_size: int,
    num_tokens: int,
    carbon_intensity_g_kwh: float = DEFAULT_CARBON_INTENSITY_G_KWH,
    pue: float = DEFAULT_PUE,
) -> Dict[str, Any]:
    """Estimate carbon footprint for a single inference pass.

    Methodology:
        estimated_power_W  = gpu_tdp_w × (gpu_util_pct / 100)
        gpu_energy_Wh      = estimated_power_W × inference_time_s / 3600
        total_energy_Wh    = gpu_energy_Wh × pue           (system + cooling)
        carbon_gCO2eq      = total_energy_Wh × carbon_intensity_g_kwh / 1000

    Args:
        inference_time_s:       Wall time for one inference pass (seconds).
        gpu_tdp_w:              GPU thermal design power (watts).
        gpu_util_pct:           Observed GPU utilization (0–100).
        batch_size:             Number of sequences in the batch.
        num_tokens:             Tokens generated/processed per batch element.
        carbon_intensity_g_kwh: Grid carbon intensity (gCO2eq/kWh).
        pue:                    Power Usage Effectiveness (≥ 1.0).

    Returns:
        Dict with keys:
            gpu_tdp_w, estimated_power_w, gpu_util_pct, pue,
            carbon_intensity_g_kwh, inference_time_s,
            energy_per_inference_mwh, carbon_per_inference_mgco2eq,
            (when num_tokens > 0):
                total_tokens_per_inference, energy_per_token_mwh,
                carbon_per_token_mgco2eq, carbon_per_million_tokens_gco2eq,
                energy_per_million_tokens_kwh.
    """
    estimated_power_w = gpu_tdp_w * (gpu_util_pct / 100.0)
    gpu_energy_wh = estimated_power_w * inference_time_s / 3600.0
    total_energy_wh = gpu_energy_wh * pue
    total_energy_mwh = total_energy_wh * 1000.0

    carbon_gco2eq = total_energy_wh * carbon_intensity_g_kwh / 1000.0
    carbon_mgco2eq = carbon_gco2eq * 1000.0

    result: Dict[str, Any] = {
        "gpu_tdp_w": gpu_tdp_w,
        "estimated_power_w": round(estimated_power_w, 2),
        "gpu_util_pct": gpu_util_pct,
        "pue": pue,
        "carbon_intensity_g_kwh": carbon_intensity_g_kwh,
        "inference_time_s": round(inference_time_s, 6),
        "energy_per_inference_mwh": round(total_energy_mwh, 6),
        "carbon_per_inference_mgco2eq": round(carbon_mgco2eq, 6),
    }

    total_tokens = batch_size * num_tokens if (batch_size > 0 and num_tokens > 0) else 0
    if total_tokens > 0:
        energy_mwh_per_token = total_energy_mwh / total_tokens
        carbon_mgco2eq_per_token = carbon_mgco2eq / total_tokens
        result["total_tokens_per_inference"] = total_tokens
        result["energy_per_token_mwh"] = round(energy_mwh_per_token, 9)
        result["carbon_per_token_mgco2eq"] = round(carbon_mgco2eq_per_token, 9)
        # Scale to 1M tokens for intuitive comparison
        result["carbon_per_million_tokens_gco2eq"] = round(
            carbon_mgco2eq_per_token * 1e6 / 1000.0, 4
        )
        result["energy_per_million_tokens_kwh"] = round(
            energy_mwh_per_token * 1e6 / 1e6, 6
        )

    return result
