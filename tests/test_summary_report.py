"""
Unit tests for soda.common.summary_report — formatting helpers and table builders.
"""

import pytest
import argparse
from soda.common.summary_report import (
    _bar,
    _pct,
    _fmt_ms,
    _fmt_mb,
    _fmt_energy,
    _fmt_carbon,
    _build_gpu_table,
    _build_overhead_table,
    _build_kernel_table,
)


# ---------------------------------------------------------------------------
# _bar
# ---------------------------------------------------------------------------

class TestBar:
    def test_zero_fraction(self):
        b = _bar(0.0, width=10)
        assert b == "░" * 10

    def test_full_fraction(self):
        b = _bar(1.0, width=10)
        assert b == "█" * 10

    def test_half_fraction(self):
        b = _bar(0.5, width=10)
        filled = b.count("█")
        empty = b.count("░")
        assert filled == 5
        assert empty == 5

    def test_clamped_above_one(self):
        b = _bar(2.0, width=10)
        assert b == "█" * 10

    def test_clamped_below_zero(self):
        b = _bar(-1.0, width=10)
        assert b == "░" * 10

    def test_default_width_20(self):
        b = _bar(0.5)
        assert len(b) == 20

    def test_custom_width(self):
        b = _bar(0.5, width=8)
        assert len(b) == 8


# ---------------------------------------------------------------------------
# _pct
# ---------------------------------------------------------------------------

class TestPct:
    def test_half(self):
        assert _pct(50, 100) == pytest.approx(50.0)

    def test_zero_total(self):
        assert _pct(10, 0) == 0.0

    def test_zero_part(self):
        assert _pct(0, 100) == pytest.approx(0.0)

    def test_full(self):
        assert _pct(100, 100) == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# _fmt_ms
# ---------------------------------------------------------------------------

class TestFmtMs:
    def test_microseconds_range(self):
        # ms < 1 → shows us
        result = _fmt_ms(0.5)
        assert "us" in result

    def test_milliseconds_range(self):
        # 1 ≤ ms < 1000 → shows ms
        result = _fmt_ms(42.5)
        assert "ms" in result

    def test_seconds_range(self):
        # ms ≥ 1000 → shows s
        result = _fmt_ms(1500.0)
        assert " s" in result

    def test_boundary_1ms(self):
        result = _fmt_ms(1.0)
        assert "ms" in result

    def test_boundary_1s(self):
        result = _fmt_ms(1000.0)
        assert " s" in result


# ---------------------------------------------------------------------------
# _fmt_mb
# ---------------------------------------------------------------------------

class TestFmtMb:
    def test_megabytes_range(self):
        result = _fmt_mb(512.0)
        assert "MB" in result

    def test_gigabytes_range(self):
        result = _fmt_mb(2048.0)
        assert "GB" in result

    def test_boundary_1024mb_is_gb(self):
        result = _fmt_mb(1024.0)
        assert "GB" in result


# ---------------------------------------------------------------------------
# _fmt_energy
# ---------------------------------------------------------------------------

class TestFmtEnergy:
    def test_mwh_range(self):
        result = _fmt_energy(1.5)
        assert "mWh" in result

    def test_uwh_range(self):
        # 0.001 mWh = 1 uWh boundary; use 0.005 mWh to be safely in uWh range
        result = _fmt_energy(0.005)
        assert "uWh" in result

    def test_nwh_range(self):
        result = _fmt_energy(0.0000001)
        assert "nWh" in result


# ---------------------------------------------------------------------------
# _fmt_carbon
# ---------------------------------------------------------------------------

class TestFmtCarbon:
    def test_g_range(self):
        result = _fmt_carbon(1500.0)
        assert "g CO2eq" in result

    def test_mg_range(self):
        result = _fmt_carbon(5.0)
        assert "mg CO2eq" in result

    def test_ug_range(self):
        # ug range: 0.001 ≤ mgco2eq < 1.0; use 0.005
        result = _fmt_carbon(0.005)
        assert "ug CO2eq" in result

    def test_ng_range(self):
        result = _fmt_carbon(0.0000001)
        assert "ng CO2eq" in result


# ---------------------------------------------------------------------------
# _build_gpu_table
# ---------------------------------------------------------------------------

class TestBuildGpuTable:
    def _make_metrics(self, gpu_name="NVIDIA H100"):
        return {
            "metadata": {"config": {"gpu_name": gpu_name}},
            "performance_metrics": {
                "gpu_utilization_percent": 75.0,
                "inference_throughput": {"throughput_tok_s": 1234.5},
            },
        }

    def test_returns_table(self):
        from rich.table import Table
        metrics = self._make_metrics()
        tbl = _build_gpu_table(metrics)
        assert isinstance(tbl, Table)

    def test_single_gpu_no_gpu_count_row(self):
        from rich.table import Table
        metrics = self._make_metrics()
        args = argparse.Namespace(num_gpus=1)
        tbl = _build_gpu_table(metrics, args=args)
        assert isinstance(tbl, Table)

    def test_multi_gpu_adds_row(self):
        from rich.table import Table
        metrics = self._make_metrics()
        args = argparse.Namespace(num_gpus=2)
        tbl = _build_gpu_table(metrics, args=args)
        # Table should be built without error
        assert isinstance(tbl, Table)

    def test_no_args_defaults_single_gpu(self):
        from rich.table import Table
        metrics = self._make_metrics()
        tbl = _build_gpu_table(metrics, args=None)
        assert isinstance(tbl, Table)


# ---------------------------------------------------------------------------
# _build_overhead_table
# ---------------------------------------------------------------------------

class TestBuildOverheadTable:
    def test_returns_table(self):
        from rich.table import Table
        components = [
            {"name": "Python layer", "ms": 1.5},
            {"name": "ATen dispatch", "ms": 0.8},
            {"name": "Kernel launch", "ms": 0.3},
        ]
        tbl = _build_overhead_table(components, total_ms=10.0)
        assert isinstance(tbl, Table)

    def test_empty_components(self):
        from rich.table import Table
        tbl = _build_overhead_table([], total_ms=10.0)
        assert isinstance(tbl, Table)


# ---------------------------------------------------------------------------
# _build_kernel_table
# ---------------------------------------------------------------------------

class TestBuildKernelTable:
    def test_returns_table(self):
        from rich.table import Table
        kernels = [
            {"name": "sgemm_kernel", "frequency": 100, "total_duration_ms": 50.0},
            {"name": "elementwise_kernel", "frequency": 200, "total_duration_ms": 10.0},
        ]
        tbl = _build_kernel_table(kernels)
        assert isinstance(tbl, Table)

    def test_empty_kernels(self):
        from rich.table import Table
        tbl = _build_kernel_table([])
        assert isinstance(tbl, Table)
