from __future__ import annotations

import pytest

from scripts.benchmark_phase0_parity import _regression, _summary


def test_phase0_summary_reports_median_and_nearest_rank_p95() -> None:
    summary = _summary([float(value) for value in range(1, 101)])
    assert summary["p50"] == 50.5
    assert summary["p95"] == 95.0


def test_phase0_regression_is_relative_to_baseline() -> None:
    assert _regression(101.0, 100.0) == pytest.approx(0.01)
    assert _regression(99.0, 100.0) == pytest.approx(-0.01)
