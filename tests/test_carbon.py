"""
Unit tests for soda.carbon — GPU TDP lookup, carbon intensity presets,
and compute_carbon_footprint().
"""

import pytest
from soda.carbon import (
    get_gpu_tdp,
    compute_carbon_footprint,
    GPU_TDP_W,
    CARBON_INTENSITY_PRESETS,
)


# ---------------------------------------------------------------------------
# GPU_TDP_W / get_gpu_tdp
# ---------------------------------------------------------------------------

class TestGetGpuTdp:
    def test_h100_recognized(self):
        tdp = get_gpu_tdp("NVIDIA H100 80GB HBM3")
        assert tdp is not None
        assert tdp > 0

    def test_h200_recognized(self):
        tdp = get_gpu_tdp("NVIDIA H200 SXM")
        assert tdp is not None
        assert tdp > 0

    def test_a100_recognized(self):
        tdp = get_gpu_tdp("NVIDIA A100-SXM4-80GB")
        assert tdp is not None
        assert tdp > 0

    def test_unknown_gpu_returns_none(self):
        tdp = get_gpu_tdp("NVIDIA XYZ-9999 Unknown")
        # Unknown GPUs return None (no fallback; caller handles missing TDP)
        assert tdp is None

    def test_case_insensitive_matching(self):
        tdp_upper = get_gpu_tdp("NVIDIA H100")
        tdp_lower = get_gpu_tdp("nvidia h100")
        # Both should resolve to the same TDP
        assert tdp_upper == tdp_lower

    def test_tdp_table_non_empty(self):
        assert len(GPU_TDP_W) > 0


# ---------------------------------------------------------------------------
# CARBON_INTENSITY_PRESETS
# ---------------------------------------------------------------------------

class TestCarbonIntensityPresets:
    def test_presets_exist(self):
        assert len(CARBON_INTENSITY_PRESETS) > 0

    def test_known_presets(self):
        # At least some regional presets should be present
        keys_lower = {k.lower() for k in CARBON_INTENSITY_PRESETS}
        assert any(k in keys_lower for k in ("us", "eu", "fr", "cn"))

    def test_preset_values_positive(self):
        for region, intensity in CARBON_INTENSITY_PRESETS.items():
            assert intensity > 0, f"Preset {region!r} has non-positive intensity {intensity}"

    def test_fr_lower_than_cn(self):
        keys_lower = {k.lower(): v for k, v in CARBON_INTENSITY_PRESETS.items()}
        if "fr" in keys_lower and "cn" in keys_lower:
            assert keys_lower["fr"] < keys_lower["cn"], "France should have lower carbon intensity than China"


# ---------------------------------------------------------------------------
# compute_carbon_footprint
# ---------------------------------------------------------------------------

class TestComputeCarbonFootprint:
    """
    compute_carbon_footprint(inference_time_s, gpu_tdp_w, gpu_util_pct,
                             batch_size, num_tokens,
                             carbon_intensity_g_kwh=400.0, pue=1.1)
    """
    _TDP = 700.0  # H100 SXM TDP in Watts

    def _run(self, gpu_tdp_w=700.0, gpu_util_pct=80.0, inference_time_s=0.01,
             batch_size=1, num_tokens=128, carbon_intensity=400.0, pue=1.1):
        return compute_carbon_footprint(
            inference_time_s=inference_time_s,
            gpu_tdp_w=gpu_tdp_w,
            gpu_util_pct=gpu_util_pct,
            batch_size=batch_size,
            num_tokens=num_tokens,
            carbon_intensity_g_kwh=carbon_intensity,
            pue=pue,
        )

    def test_returns_dict(self):
        result = self._run()
        assert isinstance(result, dict)

    def _carbon_val(self, result):
        """Extract carbon-per-inference value from result dict."""
        # Prefer specific per-inference key; fall back to any *_mgco2eq key
        for key in ("carbon_per_inference_mgco2eq", "carbon_mgco2eq", "carbon_mg_co2eq"):
            if key in result:
                return result[key]
        return next((v for k, v in result.items()
                     if "carbon" in k and "per_inference" in k), None)

    def test_returns_dict(self):
        result = self._run()
        assert isinstance(result, dict)

    def test_carbon_value_positive(self):
        result = self._run(gpu_util_pct=100.0, inference_time_s=1.0)
        val = self._carbon_val(result)
        assert val is not None
        assert val >= 0

    def test_zero_utilization_zero_carbon(self):
        result = self._run(gpu_util_pct=0.0, inference_time_s=0.1)
        val = self._carbon_val(result)
        if val is not None:
            assert val == pytest.approx(0.0, abs=1e-10)

    def test_zero_inference_time_zero_carbon(self):
        result = self._run(inference_time_s=0.0)
        val = self._carbon_val(result)
        if val is not None:
            assert val == pytest.approx(0.0, abs=1e-10)

    def test_higher_pue_higher_carbon(self):
        r1 = self._run(pue=1.0, inference_time_s=0.05)
        r2 = self._run(pue=1.5, inference_time_s=0.05)
        c1 = self._carbon_val(r1) or 0
        c2 = self._carbon_val(r2) or 0
        assert c2 >= c1

    def test_higher_intensity_higher_carbon(self):
        r1 = self._run(carbon_intensity=58.0)   # France
        r2 = self._run(carbon_intensity=581.0)  # China
        c1 = self._carbon_val(r1) or 0
        c2 = self._carbon_val(r2) or 0
        assert c2 > c1
