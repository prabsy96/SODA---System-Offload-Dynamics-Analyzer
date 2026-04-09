"""
Unit tests for soda.kerneldb — kernel classification helpers and
_extract_last_run_sequences().
"""

import pytest
from soda.kerneldb import (
    _is_vendor_replayable,
    _extract_last_run_sequences,
    VENDOR_REPLAY_PATTERNS,
    INTERNAL_GEMM_PATTERNS,
)


# ---------------------------------------------------------------------------
# _is_vendor_replayable
# ---------------------------------------------------------------------------

class TestIsVendorReplayable:
    def test_cublas_replayable(self):
        assert _is_vendor_replayable("cublasGemmEx") is True

    def test_cublaslt_replayable(self):
        assert _is_vendor_replayable("cublasLtMatmul") is True

    def test_cutlass_replayable(self):
        assert _is_vendor_replayable("cutlass_gemm_kernel<float>") is True

    def test_nvjet_not_replayable(self):
        # nvjet is an internal GEMM pattern → not vendor-replayable
        assert _is_vendor_replayable("nvjet_gemm_kernel") is False

    def test_wgmma_not_replayable(self):
        assert _is_vendor_replayable("wgmma_fp8_kernel") is False

    def test_s884gemm_not_replayable(self):
        assert _is_vendor_replayable("s884gemm_kernel") is False

    def test_elementwise_not_replayable(self):
        assert _is_vendor_replayable("vectorized_elementwise_kernel") is False

    def test_empty_not_replayable(self):
        assert _is_vendor_replayable("") is False

    def test_case_insensitive(self):
        # Pattern matching uses .lower()
        assert _is_vendor_replayable("CUBLAS_GEMM") is True


# ---------------------------------------------------------------------------
# Three-way kernel_class assignment logic
# ---------------------------------------------------------------------------

class TestKernelClassLogic:
    """
    Tests the kernel_class three-way logic (extracted from generate_kernel_database).
    We replicate the logic directly rather than calling generate_kernel_database
    (which requires a live ModelTracer with GPU).
    """
    from soda.common.utils import is_gemm_op, is_gemm_kernel

    def _classify(self, aten_op_name, raw_kernel_name):
        from soda.common.utils import is_gemm_op, is_gemm_kernel
        is_gemm_op_flag = is_gemm_op(aten_op_name)
        is_gemm_kernel_flag = is_gemm_kernel(raw_kernel_name)
        is_gemm = is_gemm_op_flag or is_gemm_kernel_flag

        if is_gemm_kernel_flag:
            kernel_class = "gemm"
        elif is_gemm_op_flag:
            kernel_class = "unknown"
        else:
            kernel_class = "non_gemm"

        is_vendor = _is_vendor_replayable(raw_kernel_name)
        i_lib = 1 if is_vendor else 0
        return {"is_gemm": is_gemm, "kernel_class": kernel_class,
                "i_lib": i_lib, "is_vendor": is_vendor}

    def test_cublas_gemm_class(self):
        result = self._classify("aten::mm", "cublasGemmEx")
        assert result["kernel_class"] == "gemm"
        assert result["is_gemm"] is True
        assert result["i_lib"] == 1

    def test_unknown_gemm_class(self):
        # ATen op is GEMM but kernel name is unrecognized
        result = self._classify("aten::mm", "some_triton_gemm_impl_v2")
        # "some_triton_gemm_impl_v2" contains "gemm" → is_gemm_kernel = True
        # Adjust: use a name without gemm to test "unknown" class
        result = self._classify("aten::mm", "completely_unknown_kernel")
        assert result["kernel_class"] == "unknown"
        assert result["is_gemm"] is True  # from ATen op
        assert result["i_lib"] == 0

    def test_non_gemm_class(self):
        result = self._classify("aten::relu", "vectorized_elementwise_kernel")
        assert result["kernel_class"] == "non_gemm"
        assert result["is_gemm"] is False
        assert result["i_lib"] == 0

    def test_nvjet_gemm_but_i_lib_zero(self):
        result = self._classify("aten::mm", "nvjet_gemm_sm90_kernel")
        assert result["is_gemm"] is True
        assert result["i_lib"] == 0  # nvjet not vendor-replayable

    def test_cutlass_i_lib_one(self):
        result = self._classify("aten::mm", "cutlass_gemm_universal")
        assert result["i_lib"] == 1
        assert result["is_vendor"] is True

    def test_wgmma_gemm_class_i_lib_zero(self):
        # wgmma in kernel name → is_gemm_kernel=True → "gemm" class
        # but wgmma is internal → i_lib=0
        result = self._classify("aten::mm", "wgmma_bf16_kernel")
        assert result["kernel_class"] == "gemm"
        assert result["i_lib"] == 0


# ---------------------------------------------------------------------------
# _extract_last_run_sequences
# ---------------------------------------------------------------------------

class TestExtractLastRunSequences:
    def _make_seq(self, ts):
        """Minimal sequence dict with a kernel timestamp."""
        return {
            "kernel": {"name": "k", "ts": ts, "dur": 1},
            "aten_op": {"name": "aten::mm", "ts": ts - 5, "dur": 3},
            "cuda_launch": {"ts": ts - 2, "dur": 1},
        }

    def test_single_run_returns_all(self):
        seqs = [self._make_seq(100), self._make_seq(200), self._make_seq(300)]
        result = _extract_last_run_sequences(seqs, num_profiled_runs=1)
        assert len(result) == len(seqs)

    def test_two_runs_returns_last_half(self):
        # Timestamps: 0,1,2,...,9  → last run: ts >= 5
        seqs = [self._make_seq(float(i)) for i in range(10)]
        result = _extract_last_run_sequences(seqs, num_profiled_runs=2)
        # Last half: ts >= 4.5, so ts in {5,6,7,8,9}
        assert len(result) == 5
        for s in result:
            assert s["kernel"]["ts"] >= 4.5

    def test_empty_sequences(self):
        result = _extract_last_run_sequences([], num_profiled_runs=5)
        assert result == []

    def test_all_same_timestamp(self):
        # total_span = 0 → returns all sequences unchanged
        seqs = [self._make_seq(100.0) for _ in range(5)]
        result = _extract_last_run_sequences(seqs, num_profiled_runs=3)
        assert len(result) == 5

    def test_three_runs(self):
        # 9 seqs spanning ts=0..8, 3 runs → last run: ts >= 6
        seqs = [self._make_seq(float(i)) for i in range(9)]
        result = _extract_last_run_sequences(seqs, num_profiled_runs=3)
        for s in result:
            assert s["kernel"]["ts"] >= 6.0 - 1e-9  # boundary at 6.0


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------

class TestKerneldbConstants:
    def test_vendor_patterns_non_empty(self):
        assert len(VENDOR_REPLAY_PATTERNS) > 0

    def test_internal_patterns_non_empty(self):
        assert len(INTERNAL_GEMM_PATTERNS) > 0

    def test_no_overlap_vendor_internal(self):
        # Vendor and internal patterns should not overlap
        vendor_set = set(VENDOR_REPLAY_PATTERNS)
        internal_set = set(INTERNAL_GEMM_PATTERNS)
        assert vendor_set.isdisjoint(internal_set), \
            f"Overlap between vendor and internal patterns: {vendor_set & internal_set}"
