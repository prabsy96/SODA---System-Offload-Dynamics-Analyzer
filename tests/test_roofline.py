"""
Unit tests for soda.roofline — GPU specs, GEMM FLOPs, Pareto frontier,
and roofline data computation.
"""

import pytest
from soda.roofline import (
    get_gpu_specs,
    compute_gemm_flops,
    compute_pareto_frontier,
    GPU_SPECS,
)


# ---------------------------------------------------------------------------
# GPU_SPECS / get_gpu_specs
# ---------------------------------------------------------------------------

class TestGetGpuSpecs:
    def test_h100_recognized(self):
        specs = get_gpu_specs("NVIDIA H100 80GB HBM3")
        assert specs is not None
        assert specs["peak_gflops"] > 0
        assert specs["peak_bw_bytes_s"] > 0
        assert specs["ridge_point"] > 0

    def test_a100_recognized(self):
        specs = get_gpu_specs("NVIDIA A100-SXM4-80GB")
        assert specs is not None

    def test_unknown_returns_none(self):
        specs = get_gpu_specs("NVIDIA UNKNOWN-9999")
        # Unknown GPU returns None
        assert specs is None or isinstance(specs, dict)

    def test_ridge_point_computed(self):
        # ridge_point = peak_gflops / (peak_bw_bytes_s / 1e9)  [GFLOP / (GB/s)]
        specs = get_gpu_specs("NVIDIA H100 80GB HBM3")
        if specs:
            bw_gb_s = specs["peak_bw_bytes_s"] / 1e9
            expected_ridge = specs["peak_gflops"] / bw_gb_s
            assert specs["ridge_point"] == pytest.approx(expected_ridge, rel=0.01)

    def test_gpu_specs_table_non_empty(self):
        assert len(GPU_SPECS) > 0

    def test_specs_have_required_keys(self):
        # GPU_SPECS stores raw values; get_gpu_specs() adds derived keys
        for gpu_key, specs in GPU_SPECS.items():
            assert "peak_tflops_fp16" in specs or "peak_gflops" in specs, \
                f"Missing peak compute spec for {gpu_key}"
            assert "peak_bw_tb_s" in specs or "peak_bw_bytes_s" in specs, \
                f"Missing bandwidth spec for {gpu_key}"


# ---------------------------------------------------------------------------
# compute_gemm_flops
# ---------------------------------------------------------------------------

class TestComputeGemmFlops:
    def test_mm_flops(self):
        # aten::mm: [M, K] @ [K, N] → 2*M*K*N
        flops = compute_gemm_flops("aten::mm", [[4, 8], [8, 16]])
        assert flops == pytest.approx(2 * 4 * 8 * 16)

    def test_bmm_flops(self):
        # aten::bmm: [B, M, K] @ [B, K, N] → 2*B*M*K*N
        flops = compute_gemm_flops("aten::bmm", [[2, 4, 8], [2, 8, 16]])
        assert flops == pytest.approx(2 * 2 * 4 * 8 * 16)

    def test_addmm_flops(self):
        # aten::addmm: [M, K] @ [K, N] → same as mm
        flops = compute_gemm_flops("aten::addmm", [[4], [4, 8], [8, 16]])
        assert flops == pytest.approx(2 * 4 * 8 * 16)

    def test_linear_flops(self):
        # aten::linear: input[M, K], weight[N, K] → 2*M*N*K
        flops = compute_gemm_flops("aten::linear", [[4, 8], [16, 8]])
        assert flops == pytest.approx(2 * 4 * 16 * 8)

    def test_scaled_mm_flops(self):
        # aten::_scaled_mm: A[M,K] @ B[K,N]
        flops = compute_gemm_flops("aten::_scaled_mm", [[4, 8], [8, 16]])
        assert flops == pytest.approx(2 * 4 * 8 * 16)

    def test_invalid_op_returns_none(self):
        flops = compute_gemm_flops("aten::relu", [[4, 8]])
        assert flops is None

    def test_missing_dims_returns_none(self):
        flops = compute_gemm_flops("aten::mm", [])
        assert flops is None

    def test_flops_positive(self):
        flops = compute_gemm_flops("aten::mm", [[128, 512], [512, 1024]])
        assert flops is not None
        assert flops > 0


# ---------------------------------------------------------------------------
# compute_pareto_frontier
# ---------------------------------------------------------------------------

class TestComputeParetoFrontier:
    """
    compute_pareto_frontier expects dicts with
    'throughput_tok_s' and 'interactivity_tok_s' keys.
    Maximises both dimensions simultaneously.
    """

    def _make_points(self, thr_int_list):
        """Create points with throughput_tok_s and interactivity_tok_s."""
        return [
            {"throughput_tok_s": t, "interactivity_tok_s": i,
             "label": f"p{idx}"}
            for idx, (t, i) in enumerate(thr_int_list)
        ]

    def test_single_point_is_pareto(self):
        points = self._make_points([(100, 50)])
        frontier = compute_pareto_frontier(points)
        assert len(frontier) == 1

    def test_dominated_point_excluded(self):
        # (1,1) is dominated by (2,2) — lower in both dimensions
        points = self._make_points([(1, 1), (2, 2)])
        frontier = compute_pareto_frontier(points)
        throughputs = [p["throughput_tok_s"] for p in frontier]
        assert 2 in throughputs
        assert 1 not in throughputs

    def test_incomparable_points_both_on_frontier(self):
        # (3,1) and (1,3): higher throughput vs higher interactivity
        points = self._make_points([(3, 1), (1, 3)])
        frontier = compute_pareto_frontier(points)
        assert len(frontier) == 2

    def test_empty_returns_empty(self):
        frontier = compute_pareto_frontier([])
        assert frontier == []

    def test_all_same_value(self):
        points = self._make_points([(5, 5), (5, 5), (5, 5)])
        frontier = compute_pareto_frontier(points)
        assert len(frontier) >= 1

    def test_frontier_subset_of_input(self):
        points = self._make_points([(1, 3), (2, 2), (3, 1), (1, 1)])
        frontier = compute_pareto_frontier(points)
        frontier_keys = {(p["throughput_tok_s"], p["interactivity_tok_s"]) for p in frontier}
        all_keys = {(p["throughput_tok_s"], p["interactivity_tok_s"]) for p in points}
        assert frontier_keys.issubset(all_keys)
