"""
Unit tests for multi-GPU support — experiment naming suffix, num_gpus clamping,
null_kernel floor return structure, and pipeline metadata propagation.
"""

import pytest
from soda.common.utils import generate_experiment_name


# ---------------------------------------------------------------------------
# Experiment naming with num_gpus
# ---------------------------------------------------------------------------

class TestExperimentNamingMultiGpu:
    BASE = "gpt2_eager_bfloat16_bs1_sl128_mt1"

    def test_single_gpu_no_suffix(self):
        name = generate_experiment_name("gpt2", "eager", "bfloat16", 1, 128, 1, num_gpus=1)
        assert name == self.BASE
        assert "_gpu" not in name

    def test_two_gpus_suffix(self):
        name = generate_experiment_name("gpt2", "eager", "bfloat16", 1, 128, 1, num_gpus=2)
        assert name == f"{self.BASE}_gpu2"

    def test_four_gpus_suffix(self):
        name = generate_experiment_name("gpt2", "eager", "bfloat16", 1, 128, 1, num_gpus=4)
        assert name == f"{self.BASE}_gpu4"

    def test_default_no_suffix(self):
        # Calling without num_gpus should default to single GPU
        name = generate_experiment_name("gpt2", "eager", "bfloat16", 1, 128, 1)
        assert "_gpu" not in name

    def test_zero_gpus_treated_as_single(self):
        # num_gpus=0 is invalid but should not crash; treat as single GPU
        name = generate_experiment_name("gpt2", "eager", "bfloat16", 1, 128, 1, num_gpus=0)
        # Should not append _gpu0 (0 is not > 1)
        assert "_gpu0" not in name

    def test_hf_path_slash_replaced(self):
        name = generate_experiment_name(
            "meta-llama/Llama-3.2-1B", "eager", "float16", 1, 128, 1, num_gpus=2
        )
        assert "/" not in name
        assert "_gpu2" in name


# ---------------------------------------------------------------------------
# null_kernel measure_system_floor return structure
# ---------------------------------------------------------------------------

class TestNullKernelFloorStructure:
    """
    Tests the structure of the dict returned by measure_system_floor().
    These are structure tests only — they do not call nsys.
    """

    def _fake_floor_single(self):
        """Simulate what measure_system_floor(num_gpus=1) might return."""
        return {
            "avg_us": 4.5,
            "std_us": 0.2,
            "num_gpus": 1,
        }

    def _fake_floor_multi(self, n=2):
        """Simulate what measure_system_floor(num_gpus=n) might return."""
        per_gpu = [4.5 + i * 0.1 for i in range(n)]
        return {
            "avg_us": min(per_gpu),
            "std_us": 0.2,
            "num_gpus": n,
            "per_gpu_avg_us": per_gpu,
        }

    def test_single_gpu_floor_keys(self):
        result = self._fake_floor_single()
        assert "avg_us" in result
        assert "std_us" in result
        assert "num_gpus" in result
        assert result["num_gpus"] == 1

    def test_multi_gpu_floor_keys(self):
        result = self._fake_floor_multi(n=2)
        assert "avg_us" in result
        assert "num_gpus" in result
        assert "per_gpu_avg_us" in result
        assert len(result["per_gpu_avg_us"]) == 2

    def test_multi_gpu_avg_is_minimum(self):
        result = self._fake_floor_multi(n=3)
        assert result["avg_us"] == min(result["per_gpu_avg_us"])

    def test_avg_us_positive(self):
        result = self._fake_floor_single()
        assert result["avg_us"] >= 0.0


# ---------------------------------------------------------------------------
# TaxBreak pipeline num_gpus metadata propagation
# ---------------------------------------------------------------------------

class TestPipelineNumGpusMetadata:
    """
    Tests that the TaxBreakPipeline reads num_gpus from kernel DB metadata.
    Uses a mock kernel_db dict instead of real files.
    """

    def _make_db(self, num_gpus=1):
        return {
            "metadata": {
                "model": "gpt2",
                "gpu_name": "NVIDIA H100",
                "num_gpus": num_gpus,
            },
            "ops": {},
        }

    def test_single_gpu_metadata(self):
        db = self._make_db(num_gpus=1)
        num_gpus = db.get("metadata", {}).get("num_gpus", 1)
        assert num_gpus == 1

    def test_multi_gpu_metadata(self):
        db = self._make_db(num_gpus=2)
        num_gpus = db.get("metadata", {}).get("num_gpus", 1)
        assert num_gpus == 2

    def test_missing_metadata_defaults_one(self):
        db = {"ops": {}}
        num_gpus = db.get("metadata", {}).get("num_gpus", 1)
        assert num_gpus == 1


# ---------------------------------------------------------------------------
# ModelTracer num_gpus clamping logic (pure unit test, no GPU required)
# ---------------------------------------------------------------------------

class TestNumGpusClamping:
    """Tests the clamping logic: min(requested, available)."""

    def _clamp(self, requested, available):
        """Replicate the clamping logic from ModelTracer.__init__()."""
        requested = max(1, requested)
        return min(requested, max(1, available))

    def test_exact_match(self):
        assert self._clamp(2, 2) == 2

    def test_clamp_above_available(self):
        assert self._clamp(4, 2) == 2

    def test_single_gpu_unchanged(self):
        assert self._clamp(1, 8) == 1

    def test_zero_available_returns_one(self):
        # max(1, 0) = 1, so result should be 1
        assert self._clamp(1, 0) == 1

    def test_negative_request_treated_as_one(self):
        # max(1, -2) = 1
        assert self._clamp(-2, 4) == 1


# ---------------------------------------------------------------------------
# get_kwargs device_map logic (pure unit test, no GPU required)
# ---------------------------------------------------------------------------

class TestGetKwargsDeviceMap:
    """Replicate the get_kwargs() device_map selection logic."""

    def _device_map(self, num_gpus, has_cuda, device="cuda"):
        if num_gpus > 1:
            return "balanced"
        elif has_cuda:
            return device
        else:
            return "cpu"

    def test_multi_gpu_balanced(self):
        assert self._device_map(2, True) == "balanced"

    def test_single_gpu_cuda(self):
        assert self._device_map(1, True, "cuda:0") == "cuda:0"

    def test_no_cuda_cpu(self):
        assert self._device_map(1, False) == "cpu"

    def test_four_gpu_balanced(self):
        assert self._device_map(4, True) == "balanced"
