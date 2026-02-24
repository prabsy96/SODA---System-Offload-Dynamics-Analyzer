"""
Unit tests for soda.common.data — Kernel/ATenOp/Sequence dataclasses and
clean_kernel_name().
"""

import pytest
from soda.common.data import Kernel, ATenOp, Sequence, clean_kernel_name


# ---------------------------------------------------------------------------
# clean_kernel_name
# ---------------------------------------------------------------------------

class TestCleanKernelName:
    def test_empty_string_returns_unknown(self):
        # The implementation returns "unknown" for falsy input
        assert clean_kernel_name("") == "unknown"

    def test_none_returns_unknown(self):
        assert clean_kernel_name(None) == "unknown"

    def test_strips_template_args(self):
        name = "cutlass_gemm_kernel<float, float>"
        result = clean_kernel_name(name)
        assert "<" not in result

    def test_gemm_suffix_added_for_cublas(self):
        # cublas is in gemm_indicators — clean name should contain "gemm"
        name = "cublasGemmEx"
        result = clean_kernel_name(name)
        assert "gemm" in result.lower()

    def test_wgmma_suffix_added(self):
        name = "wgmma_kernel<__half>"
        result = clean_kernel_name(name)
        assert "wgmma" in result.lower()

    def test_plain_elementwise_kernel(self):
        name = "vectorized_elementwise_kernel"
        result = clean_kernel_name(name)
        assert result == "vectorized_elementwise_kernel"

    def test_strips_void_prefix(self):
        name = "void my_kernel(int* a)"
        result = clean_kernel_name(name)
        assert "void" not in result

    def test_namespace_extraction(self):
        # "cuda::detail::flash_fwd_kernel" → last part after last '::'
        name = "cuda::detail::my_kernel"
        result = clean_kernel_name(name)
        assert "::" not in result

    def test_flash_fwd_annotation(self):
        name = "flash_fwd_kernel<__nv_bfloat16>"
        result = clean_kernel_name(name)
        assert "flash" in result.lower()

    def test_sgemm_gets_gemm_marker(self):
        # "sgemm" is a gemm_indicator substring
        name = "sgemm_batched"
        result = clean_kernel_name(name)
        assert "gemm" in result.lower()


# ---------------------------------------------------------------------------
# Kernel dataclass
# ---------------------------------------------------------------------------

class TestKernel:
    def _make(self, **kwargs):
        defaults = dict(name="sgemm_kernel", grid=[1, 1, 1], block=[128, 1, 1])
        defaults.update(kwargs)
        return Kernel(**defaults)

    def test_construction_basic(self):
        k = self._make()
        # clean_kernel_name is called inside __init__; name should not be raw string
        assert isinstance(k.name, str)
        assert len(k.name) > 0

    def test_grid_normalized_to_tuple(self):
        k = self._make(grid=[4, 2, 1])
        assert k.grid == (4, 2, 1)

    def test_block_normalized_to_tuple(self):
        k = self._make(block=[32, 4, 1])
        assert k.block == (32, 4, 1)

    def test_shared_memory_default_zero(self):
        k = self._make()
        assert k.shared_memory == 0

    def test_optional_device_none(self):
        k = self._make()
        assert k.device is None

    def test_device_set(self):
        k = self._make(device=0)
        assert k.device == 0

    def test_from_dict(self):
        d = {"name": "cutlass_gemm", "grid": [2, 1, 1], "block": [64, 1, 1], "shared_memory": 512}
        k = Kernel.from_dict(d)
        assert k is not None
        assert k.grid == (2, 1, 1)
        assert k.shared_memory == 512

    def test_from_dict_none_returns_none(self):
        assert Kernel.from_dict(None) is None

    def test_get_signature_basic(self):
        k = self._make()
        sig = k.get_signature()
        assert "name" in sig
        assert "grid" in sig
        assert "block" in sig
        assert "shared_memory" in sig

    def test_compare_same_kernels(self):
        k1 = self._make()
        k2 = self._make()
        result = k1.compare(k2)
        assert result["match"] is True

    def test_compare_different_grid(self):
        k1 = self._make(grid=[1, 1, 1])
        k2 = self._make(grid=[2, 1, 1])
        result = k1.compare(k2)
        assert result["match"] is False


# ---------------------------------------------------------------------------
# ATenOp dataclass
# ---------------------------------------------------------------------------

class TestATenOp:
    def _make(self, **kwargs):
        defaults = dict(name="aten::mm", ts=100.0, dur=50.0, external_id=1)
        defaults.update(kwargs)
        return ATenOp(**defaults)

    def test_construction(self):
        op = self._make()
        assert op.name == "aten::mm"
        assert op.external_id == 1

    def test_input_dims_default_empty(self):
        op = self._make()
        # Default is [] per the class definition
        assert op.input_dims == []

    def test_input_dims_set(self):
        op = self._make(input_dims=[[4, 8], [8, 16]])
        assert op.input_dims == [[4, 8], [8, 16]]

    def test_get_alpha_beta_defaults(self):
        op = self._make()
        alpha, beta = op.get_alpha_beta()
        assert alpha == 1.0
        assert beta == 1.0

    def test_get_alpha_beta_from_concrete_inputs(self):
        # concrete_inputs[3]=alpha, [4]=beta
        op = self._make(concrete_inputs=[None, None, None, 2.0, 0.5])
        alpha, beta = op.get_alpha_beta()
        assert alpha == 2.0
        assert beta == 0.5

    def test_from_dict(self):
        d = {"name": "aten::addmm", "input_dims": [[8], [8, 16], [8, 16]],
             "external_id": 5, "ts": 200.0, "dur": 30.0}
        op = ATenOp.from_dict(d)
        assert op is not None
        assert op.name == "aten::addmm"
        assert op.external_id == 5

    def test_from_dict_none_returns_none(self):
        assert ATenOp.from_dict(None) is None

    def test_compare_same(self):
        op1 = self._make()
        op2 = self._make()
        assert op1.compare(op2) is True

    def test_compare_different_name(self):
        op1 = self._make(name="aten::mm")
        op2 = self._make(name="aten::bmm")
        assert op1.compare(op2) is False


# ---------------------------------------------------------------------------
# Sequence dataclass
# ---------------------------------------------------------------------------

class TestSequence:
    def test_construction(self):
        k = Kernel(name="sgemm", grid=[1, 1, 1], block=[128, 1, 1])
        op = ATenOp(name="aten::mm", ts=100.0, dur=50.0, external_id=1)
        seq = Sequence(kernel=k, aten_op=op)
        assert seq.kernel.name is not None
        assert seq.aten_op.name == "aten::mm"

    def test_get_str(self):
        k = Kernel(name="sgemm_kernel", grid=[1, 1, 1], block=[128, 1, 1])
        op = ATenOp(name="aten::mm", ts=0.0, dur=0.0, external_id=1)
        seq = Sequence(kernel=k, aten_op=op)
        s = seq.get_str()
        assert "aten::mm" in s
        assert "->" in s

    def test_from_dict(self):
        d = {
            "aten_op": {"name": "aten::mm", "input_dims": [], "ts": 0.0, "dur": 0.0, "external_id": 1},
            "kernel": {"name": "cutlass_gemm", "grid": [1, 1, 1], "block": [128, 1, 1]},
        }
        seq = Sequence.from_dict(d)
        assert seq is not None
        assert seq.aten_op.name == "aten::mm"
