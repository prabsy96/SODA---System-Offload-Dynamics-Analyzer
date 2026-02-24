"""
Shared pytest fixtures for SODA unit tests.
"""

import pytest


# ---------------------------------------------------------------------------
# Minimal fake trace events (Chrome trace format)
# ---------------------------------------------------------------------------

def _make_kernel_event(name, ts, dur, corr, ext_id, grid=None, block=None):
    return {
        "ph": "X",
        "cat": "kernel",
        "name": name,
        "ts": ts,
        "dur": dur,
        "args": {
            "correlation": corr,
            "External id": ext_id,
            "grid": grid or [1, 1, 1],
            "block": block or [128, 1, 1],
            "shared memory": 0,
        },
    }


def _make_launch_event(name, ts, dur, corr, ext_id):
    return {
        "ph": "X",
        "cat": "cuda_runtime",
        "name": "cudaLaunchKernel",
        "ts": ts,
        "dur": dur,
        "args": {"correlation": corr, "External id": ext_id},
    }


def _make_aten_event(name, ts, dur, ext_id, input_dims=None):
    return {
        "ph": "X",
        "cat": "cpu_op",
        "name": name,
        "ts": ts,
        "dur": dur,
        "args": {
            "External id": ext_id,
            "Input Dims": input_dims or [],
            "Input type": [],
            "Input Strides": [],
            "Concrete Inputs": [],
        },
    }


@pytest.fixture
def minimal_trace():
    """A minimal Chrome trace with one aten::mm sequence."""
    return {
        "traceEvents": [
            # ATen op on CPU
            _make_aten_event("aten::mm", ts=100, dur=50, ext_id=1,
                             input_dims=[[4, 8], [8, 16]]),
            # CUDA launch
            _make_launch_event("cudaLaunchKernel", ts=140, dur=5, corr=10, ext_id=1),
            # GPU kernel
            _make_kernel_event("sgemm_kernel", ts=200, dur=30, corr=10, ext_id=1),
        ]
    }


@pytest.fixture
def multi_kernel_trace():
    """Trace with two kernels: one GEMM (aten::mm), one non-GEMM (aten::relu)."""
    return {
        "traceEvents": [
            _make_aten_event("aten::mm", ts=100, dur=60, ext_id=1,
                             input_dims=[[8, 16], [16, 32]]),
            _make_launch_event("cudaLaunchKernel", ts=150, dur=5, corr=10, ext_id=1),
            _make_kernel_event("cutlass_gemm_kernel", ts=200, dur=40, corr=10, ext_id=1),

            _make_aten_event("aten::relu", ts=300, dur=20, ext_id=2),
            _make_launch_event("cudaLaunchKernel", ts=310, dur=3, corr=20, ext_id=2),
            _make_kernel_event("vectorized_elementwise_kernel", ts=330, dur=10, corr=20, ext_id=2),
        ]
    }


@pytest.fixture
def fake_events(multi_kernel_trace):
    """Pre-parsed events dict from the multi_kernel_trace fixture."""
    from soda.common.utils import collect_events
    return collect_events(multi_kernel_trace)


@pytest.fixture
def fake_sequence():
    """A single hand-crafted sequence dict (no trace parsing required)."""
    return {
        "kernel": {"name": "sgemm_kernel", "ts": 200, "dur": 30,
                   "grid": [1, 1, 1], "block": [128, 1, 1], "shared_memory": 0},
        "cuda_launch": {"name": "cudaLaunchKernel", "ts": 140, "dur": 5, "correlation": 10},
        "aten_op": {"name": "aten::mm", "ts": 100, "dur": 50,
                    "input_dims": [[4, 8], [8, 16]]},
        "torch_op": {"name": "torch.nn.Linear", "ts": 90, "dur": 70},
    }


@pytest.fixture
def fake_args():
    """A minimal argparse-like namespace for testing."""
    import argparse
    return argparse.Namespace(
        model="gpt2",
        compile_type="eager",
        precision="bfloat16",
        batch_size=1,
        seq_len=128,
        max_new_tokens=1,
        num_gpus=1,
        carbon_intensity=400.0,
        pue=1.1,
        verbose=False,
    )
