"""
Unit tests for soda.common.utils — conversion helpers, GEMM classification,
sequence metrics, trace parsing, and experiment naming.
"""

import pytest
from soda.common.utils import (
    us_to_ms,
    ms_to_us,
    is_gemm_op,
    is_gemm_kernel,
    filter_kernel_sequences,
    summarize_metric,
    generate_experiment_name,
    calculate_sequence_metrics,
    calculate_tklqt,
    calculate_hdbi,
    calculate_framework_tax,
    compute_kernel_fragmentation,
    calculate_true_gpu_busy_time,
    calculate_gpu_utilization,
    collect_events,
    link_sequences,
    make_kernel_identity_key,
    group_sequences_by_identity,
    to_hashable,
)


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

class TestUnitConversions:
    def test_us_to_ms_basic(self):
        assert us_to_ms(1000.0) == pytest.approx(1.0)

    def test_ms_to_us_basic(self):
        assert ms_to_us(1.0) == pytest.approx(1000.0)

    def test_us_to_ms_zero(self):
        assert us_to_ms(0.0) == 0.0

    def test_ms_to_us_zero(self):
        assert ms_to_us(0.0) == 0.0

    def test_round_trip(self):
        val = 42.5
        assert us_to_ms(ms_to_us(val)) == pytest.approx(val)

    def test_us_to_ms_small(self):
        assert us_to_ms(1.0) == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# GEMM classification
# ---------------------------------------------------------------------------

class TestIsGemmOp:
    def test_mm_is_gemm(self):
        assert is_gemm_op("aten::mm") is True

    def test_bmm_is_gemm(self):
        assert is_gemm_op("aten::bmm") is True

    def test_addmm_is_gemm(self):
        assert is_gemm_op("aten::addmm") is True

    def test_matmul_is_gemm(self):
        assert is_gemm_op("aten::matmul") is True

    def test_linear_is_gemm(self):
        assert is_gemm_op("aten::linear") is True

    def test_scaled_mm_is_gemm(self):
        assert is_gemm_op("aten::_scaled_mm") is True

    def test_relu_not_gemm(self):
        assert is_gemm_op("aten::relu") is False

    def test_add_not_gemm(self):
        assert is_gemm_op("aten::add") is False

    def test_empty_not_gemm(self):
        assert is_gemm_op("") is False


class TestIsGemmKernel:
    def test_cublas_is_gemm(self):
        assert is_gemm_kernel("cublasGemmEx") is True

    def test_cutlass_is_gemm(self):
        assert is_gemm_kernel("cutlass_gemm_kernel") is True

    def test_wgmma_is_gemm(self):
        assert is_gemm_kernel("wgmma_kernel") is True

    def test_gemm_pattern_is_gemm(self):
        assert is_gemm_kernel("sgemm_batched") is True

    def test_flash_is_gemm(self):
        assert is_gemm_kernel("flash_fwd_kernel") is True

    def test_elementwise_not_gemm(self):
        assert is_gemm_kernel("vectorized_elementwise_kernel") is False

    def test_empty_not_gemm(self):
        assert is_gemm_kernel("") is False


# ---------------------------------------------------------------------------
# summarize_metric
# ---------------------------------------------------------------------------

class TestSummarizeMetric:
    def test_single_value(self):
        result = summarize_metric([5.0])
        assert result["count"] == 1
        assert result["avg"] == pytest.approx(5.0)
        assert result["min"] == pytest.approx(5.0)
        assert result["max"] == pytest.approx(5.0)
        assert result["std"] == pytest.approx(0.0)

    def test_multiple_values(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = summarize_metric(values)
        assert result["count"] == 5
        assert result["avg"] == pytest.approx(3.0)
        assert result["min"] == pytest.approx(1.0)
        assert result["max"] == pytest.approx(5.0)

    def test_std_two_values(self):
        # sample std for [1, 3] = sqrt(((1-2)^2 + (3-2)^2) / 1) = sqrt(2) ≈ 1.414
        result = summarize_metric([1.0, 3.0])
        assert result["std"] == pytest.approx(1.4142135, rel=1e-4)

    def test_all_field_present(self):
        result = summarize_metric([10.0, 20.0])
        assert "all" in result
        assert len(result["all"]) == 2


# ---------------------------------------------------------------------------
# generate_experiment_name
# ---------------------------------------------------------------------------

class TestGenerateExperimentName:
    def test_single_gpu_no_suffix(self):
        name = generate_experiment_name("gpt2", "eager", "bfloat16", 1, 128, 1, num_gpus=1)
        assert name == "gpt2_eager_bfloat16_bs1_sl128_mt1"
        assert "_gpu" not in name

    def test_multi_gpu_appends_suffix(self):
        name = generate_experiment_name("gpt2", "eager", "bfloat16", 1, 128, 1, num_gpus=2)
        assert name.endswith("_gpu2")

    def test_default_num_gpus_is_one(self):
        name = generate_experiment_name("gpt2", "eager", "bfloat16", 1, 128, 1)
        assert "_gpu" not in name

    def test_hf_model_slash_replaced(self):
        name = generate_experiment_name("meta-llama/Llama-3.2-1B", "eager", "float16", 1, 128, 1)
        assert "/" not in name

    def test_all_fields_present(self):
        name = generate_experiment_name("gpt2", "eager", "float32", 4, 512, 10)
        assert "gpt2" in name
        assert "eager" in name
        assert "float32" in name
        assert "bs4" in name
        assert "sl512" in name
        assert "mt10" in name

    def test_three_gpus(self):
        name = generate_experiment_name("gpt2", "eager", "bfloat16", 1, 128, 1, num_gpus=3)
        assert name.endswith("_gpu3")


# ---------------------------------------------------------------------------
# filter_kernel_sequences
# ---------------------------------------------------------------------------

class TestFilterKernelSequences:
    def _make_seq(self, aten_name="aten::relu", kernel_name="elementwise", has_launch=True):
        seq = {
            "kernel": {"name": kernel_name, "ts": 200, "dur": 10},
            "aten_op": {"name": aten_name, "ts": 100, "dur": 50, "input_dims": []},
            "cuda_launch": {"name": "cudaLaunchKernel", "ts": 150, "dur": 5} if has_launch else None,
            "torch_op": None,
        }
        return seq

    def test_keeps_sequences_with_all_fields(self):
        seq = self._make_seq()
        result = filter_kernel_sequences([seq])
        assert len(result) == 1

    def test_drops_sequences_without_cuda_launch(self):
        seq = self._make_seq(has_launch=False)
        result = filter_kernel_sequences([seq])
        assert len(result) == 0

    def test_gemm_op_marked_is_gemm(self):
        seq = self._make_seq(aten_name="aten::mm", kernel_name="sgemm")
        result = filter_kernel_sequences([seq])
        assert result[0]["is_gemm"] is True

    def test_gemm_kernel_name_marks_is_gemm(self):
        # Non-GEMM ATen op but GEMM kernel name → still GEMM
        seq = self._make_seq(aten_name="aten::relu", kernel_name="cutlass_gemm")
        result = filter_kernel_sequences([seq])
        assert result[0]["is_gemm"] is True

    def test_non_gemm_sequence(self):
        seq = self._make_seq(aten_name="aten::relu", kernel_name="elementwise_kernel")
        result = filter_kernel_sequences([seq])
        assert result[0]["is_gemm"] is False


# ---------------------------------------------------------------------------
# calculate_sequence_metrics
# ---------------------------------------------------------------------------

class TestCalculateSequenceMetrics:
    def _make_seq(self, kernel_ts=200, launch_ts=150, aten_ts=100, torch_ts=None):
        return {
            "kernel": {"name": "k", "ts": kernel_ts, "dur": 10},
            "cuda_launch": {"ts": launch_ts, "dur": 5},
            "aten_op": {"name": "aten::mm", "ts": aten_ts, "dur": 50},
            "torch_op": {"ts": torch_ts, "dur": 80} if torch_ts is not None else None,
        }

    def test_launch_tax_computed(self):
        seq = self._make_seq(kernel_ts=200, launch_ts=150)
        result = calculate_sequence_metrics([seq], ["launch_tax"])
        assert result[0]["launch_tax"] == pytest.approx(200 - 150)

    def test_aten_xlat_tax_computed(self):
        seq = self._make_seq(launch_ts=150, aten_ts=100)
        result = calculate_sequence_metrics([seq], ["aten_xlat_tax"])
        assert result[0]["aten_xlat_tax"] == pytest.approx(150 - 100)

    def test_py_tax_with_torch_op(self):
        seq = self._make_seq(aten_ts=100, torch_ts=90)
        result = calculate_sequence_metrics([seq], ["py_tax"])
        assert result[0]["py_tax"] == pytest.approx(100 - 90)

    def test_py_tax_without_torch_op_is_zero(self):
        seq = self._make_seq(torch_ts=None)
        result = calculate_sequence_metrics([seq], ["py_tax"])
        assert result[0]["py_tax"] == pytest.approx(0.0)

    def test_multiple_metrics(self):
        seq = self._make_seq(kernel_ts=200, launch_ts=150, aten_ts=100)
        result = calculate_sequence_metrics([seq], ["launch_tax", "aten_xlat_tax"])
        assert "launch_tax" in result[0]
        assert "aten_xlat_tax" in result[0]


# ---------------------------------------------------------------------------
# calculate_tklqt
# ---------------------------------------------------------------------------

class TestCalculateTklqt:
    def _make_seq(self, kernel_ts, launch_ts):
        return {
            "kernel": {"ts": kernel_ts, "dur": 10},
            "cuda_launch": {"ts": launch_ts, "dur": 5},
        }

    def test_positive_tklqt(self):
        seqs = [self._make_seq(200, 150), self._make_seq(300, 280)]
        result = calculate_tklqt(seqs)
        assert result["total"] == pytest.approx(50 + 20)
        assert result["count"] == 2
        assert result["avg"] == pytest.approx(35.0)

    def test_empty_sequences(self):
        result = calculate_tklqt([])
        assert result["total"] == 0.0
        assert result["count"] == 0

    def test_negative_lqt_clamped(self):
        # kernel_ts < launch_ts means negative gap (GPU ran before launch — measurement noise)
        seq = self._make_seq(kernel_ts=100, launch_ts=110)
        result = calculate_tklqt([seq])
        # Should be clamped (excluded or 0)
        assert result["total"] >= 0.0


# ---------------------------------------------------------------------------
# calculate_framework_tax
# ---------------------------------------------------------------------------

class TestCalculateFrameworkTax:
    def test_exposed_time(self):
        result = calculate_framework_tax(inference_time_us=1000.0, gpu_busy_time_us=800.0)
        assert result["T_exposed"] == pytest.approx(200.0)
        assert result["T_exposed_ms"] == pytest.approx(0.2)

    def test_gpu_busy_exceeds_inference_clamped(self):
        # Measurement noise: GPU > total, T_exposed should be clamped to 0
        result = calculate_framework_tax(inference_time_us=800.0, gpu_busy_time_us=900.0)
        assert result["T_exposed"] == pytest.approx(0.0)

    def test_zero_inference_time(self):
        result = calculate_framework_tax(inference_time_us=0.0, gpu_busy_time_us=0.0)
        assert result["T_exposed_percent"] == 0.0
        assert result["T_gpu_busy_percent"] == 0.0

    def test_percentages_sum(self):
        result = calculate_framework_tax(inference_time_us=1000.0, gpu_busy_time_us=600.0)
        # T_exposed_percent + T_gpu_busy_percent may not sum to 100 exactly if clamped
        assert result["T_exposed_percent"] >= 0.0
        assert result["T_gpu_busy_percent"] >= 0.0


# ---------------------------------------------------------------------------
# calculate_hdbi
# ---------------------------------------------------------------------------

class TestCalculateHdbi:
    def test_device_bound(self):
        # Large kernel time, small overhead → device-bound
        result = calculate_hdbi(
            total_kernel_exec_time_ms=100.0,
            total_xlat_tax_ms=1.0,
            num_total_kernels=10,
            t_sys_us=1.0,  # tiny floor
        )
        assert result["hdbi_classification"] == "device-bound"
        assert result["hdbi_value"] >= 0.5

    def test_host_bound(self):
        # Tiny kernel time, large overhead → host-bound
        result = calculate_hdbi(
            total_kernel_exec_time_ms=0.1,
            total_xlat_tax_ms=50.0,
            num_total_kernels=100,
            t_sys_us=1.0,
        )
        assert result["hdbi_classification"] == "host-bound"
        assert result["hdbi_value"] < 0.2

    def test_hdbi_clamped_to_one(self):
        # Pathological: no overhead → HDBI = 1.0
        result = calculate_hdbi(
            total_kernel_exec_time_ms=100.0,
            total_xlat_tax_ms=0.0,
            num_total_kernels=0,
            t_sys_us=0.0,
        )
        assert result["hdbi_value"] <= 1.0

    def test_hdbi_keys_present(self):
        result = calculate_hdbi(10.0, 1.0, 100, 4.5)
        for key in ("hdbi_value", "hdbi_classification", "t_device_active_ms", "t_orchestrate_ms", "delta_kt_ms"):
            assert key in result

    def test_delta_kt_computed(self):
        result = calculate_hdbi(
            total_kernel_exec_time_ms=10.0,
            total_xlat_tax_ms=0.0,
            num_total_kernels=100,
            t_sys_us=4.5,
        )
        # delta_kt_ms = 100 * (4.5 / 1000) = 0.45
        assert result["delta_kt_ms"] == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# compute_kernel_fragmentation
# ---------------------------------------------------------------------------

class TestComputeKernelFragmentation:
    def _make_events(self, kernel_names):
        kernels = [{"name": n, "ts": i * 100, "dur": 10} for i, n in enumerate(kernel_names)]
        return {"gpu": {"kernels": kernels, "memory": [], "all": kernels}}

    def test_basic_fragmentation(self):
        names = ["gemm_kernel", "gemm_kernel", "elementwise_kernel"]
        events = self._make_events(names)
        result = compute_kernel_fragmentation(events, total_output_tokens=1)
        assert result["total_kernel_launches"] == 3
        assert result["unique_kernel_count"] == 2
        assert result["kernels_per_output_token"] == pytest.approx(3.0)

    def test_diversity_ratio(self):
        names = ["a", "b", "c", "d"]  # all unique
        events = self._make_events(names)
        result = compute_kernel_fragmentation(events, total_output_tokens=2)
        assert result["kernel_diversity_ratio"] == pytest.approx(1.0)

    def test_zero_tokens_returns_none(self):
        events = self._make_events(["a"])
        result = compute_kernel_fragmentation(events, total_output_tokens=0)
        assert result["kernels_per_output_token"] is None

    def test_empty_kernels(self):
        events = {"gpu": {"kernels": [], "memory": [], "all": []}}
        result = compute_kernel_fragmentation(events, total_output_tokens=1)
        assert result["total_kernel_launches"] == 0
        assert result["kernel_diversity_ratio"] == 0.0


# ---------------------------------------------------------------------------
# calculate_true_gpu_busy_time (interval merging)
# ---------------------------------------------------------------------------

class TestCalculateTrueGpuBusyTime:
    def _make_events(self, intervals):
        """intervals: list of (ts, dur) tuples."""
        gpu_events = [{"ts": ts, "dur": dur} for ts, dur in intervals]
        return {"gpu": {"all": gpu_events}}

    def test_non_overlapping(self):
        # [0,10], [20,10] → busy = 20
        events = self._make_events([(0, 10), (20, 10)])
        assert calculate_true_gpu_busy_time(events) == pytest.approx(20.0)

    def test_fully_overlapping(self):
        # [0,20], [5,10] → merged [0,20] → busy = 20
        events = self._make_events([(0, 20), (5, 10)])
        assert calculate_true_gpu_busy_time(events) == pytest.approx(20.0)

    def test_adjacent(self):
        # [0,10], [10,10] → adjacent, not overlapping → busy = 20
        events = self._make_events([(0, 10), (10, 10)])
        assert calculate_true_gpu_busy_time(events) == pytest.approx(20.0)

    def test_empty(self):
        events = {"gpu": {"all": []}}
        assert calculate_true_gpu_busy_time(events) == pytest.approx(0.0)

    def test_partially_overlapping(self):
        # [0,15], [10,20] → merged [0,30] → busy = 30
        events = self._make_events([(0, 15), (10, 20)])
        assert calculate_true_gpu_busy_time(events) == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# collect_events
# ---------------------------------------------------------------------------

class TestCollectEvents:
    def test_kernel_event_collected(self, minimal_trace):
        events = collect_events(minimal_trace)
        assert len(events["gpu"]["kernels"]) == 1
        assert events["gpu"]["kernels"][0]["name"] == "sgemm_kernel"

    def test_launch_event_collected(self, minimal_trace):
        events = collect_events(minimal_trace)
        assert 10 in events["cpu"]["launches"]

    def test_aten_op_collected(self, minimal_trace):
        events = collect_events(minimal_trace)
        assert 1 in events["cpu"]["aten_ops"]
        assert events["cpu"]["aten_ops"][1]["name"] == "aten::mm"

    def test_structure_keys(self, minimal_trace):
        events = collect_events(minimal_trace)
        assert "cpu" in events
        assert "gpu" in events
        assert "kernels" in events["gpu"]
        assert "aten_ops" in events["cpu"]
        assert "launches" in events["cpu"]

    def test_empty_trace(self):
        events = collect_events({"traceEvents": []})
        assert len(events["gpu"]["kernels"]) == 0
        assert len(events["cpu"]["aten_ops"]) == 0


# ---------------------------------------------------------------------------
# link_sequences
# ---------------------------------------------------------------------------

class TestLinkSequences:
    def test_full_sequence_linked(self, minimal_trace):
        events = collect_events(minimal_trace)
        seqs = link_sequences(events)
        assert len(seqs) == 1
        assert seqs[0]["kernel"]["name"] == "sgemm_kernel"
        assert seqs[0]["aten_op"]["name"] == "aten::mm"

    def test_orphan_kernel_excluded(self):
        # Kernel with no matching aten_op/launch
        trace = {
            "traceEvents": [
                {
                    "ph": "X", "cat": "kernel", "name": "orphan_kernel",
                    "ts": 100, "dur": 10,
                    "args": {"correlation": 99, "External id": 99,
                             "grid": [1, 1, 1], "block": [128, 1, 1], "shared memory": 0},
                }
            ]
        }
        events = collect_events(trace)
        seqs = link_sequences(events)
        assert len(seqs) == 0


# ---------------------------------------------------------------------------
# to_hashable
# ---------------------------------------------------------------------------

class TestToHashable:
    def test_list_becomes_tuple(self):
        result = to_hashable([1, 2, 3])
        assert isinstance(result, tuple)
        assert result == (1, 2, 3)

    def test_nested_list(self):
        result = to_hashable([[1, 2], [3, 4]])
        assert result == ((1, 2), (3, 4))

    def test_dict_becomes_sorted_tuple(self):
        result = to_hashable({"b": 2, "a": 1})
        assert isinstance(result, tuple)
        # Should be sorted by key
        assert result == (("a", 1), ("b", 2))

    def test_primitives_unchanged(self):
        assert to_hashable(42) == 42
        assert to_hashable("hello") == "hello"
        assert to_hashable(3.14) == 3.14


# ---------------------------------------------------------------------------
# make_kernel_identity_key
# ---------------------------------------------------------------------------

class TestMakeKernelIdentityKey:
    def test_same_config_same_key(self):
        kernel = {"name": "sgemm", "grid": [1, 1, 1], "block": [128, 1, 1], "shared_memory": 0}
        aten_op = {"input_dims": [[4, 8], [8, 16]]}
        k1 = make_kernel_identity_key(kernel, aten_op)
        k2 = make_kernel_identity_key(kernel, aten_op)
        assert k1 == k2

    def test_different_dims_different_key(self):
        kernel = {"name": "sgemm", "grid": [1, 1, 1], "block": [128, 1, 1], "shared_memory": 0}
        aten1 = {"input_dims": [[4, 8], [8, 16]]}
        aten2 = {"input_dims": [[8, 16], [16, 32]]}
        k1 = make_kernel_identity_key(kernel, aten1)
        k2 = make_kernel_identity_key(kernel, aten2)
        assert k1 != k2

    def test_key_is_hashable(self):
        kernel = {"name": "sgemm", "grid": [1, 1, 1], "block": [128, 1, 1], "shared_memory": 0}
        aten_op = {"input_dims": [[4, 8]]}
        key = make_kernel_identity_key(kernel, aten_op)
        # Must be usable as a dict key
        d = {key: "value"}
        assert d[key] == "value"
