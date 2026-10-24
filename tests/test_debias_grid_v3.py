"""Binding SPEC §8 2026-10-24 21:15: structural grid coverage, unchanged rules."""

import math

import numpy as np
import pytest

from src.debias import Calibration

GRID = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)


def measured(ratio, ce=0.9):
    c = Calibration()
    for step in range(275, 1251, 25):
        c.record(step, ratio, ce, 1.0)
    return c


def test_v3_dense_log_coverage_and_exact_endpoints():
    assert Calibration.grid == GRID
    for ratio in np.geomspace(0.05, 2e4, 10001):
        result = measured(float(ratio)).finish()
        assert result["lambda_max"] in GRID
        assert 0.05 <= result["lambda_max"] * ratio <= 0.20
    assert 0.05 / max(GRID) == 0.05 and 0.20 / min(GRID) == 20000.0
    # Each covered interval overlaps its neighbor; no log-grid sample hides a gap.
    assert max(b / a for a, b in zip(GRID, GRID[1:])) < 4


@pytest.mark.parametrize("ratio", [0.01, 1e5])
def test_outside_v3_coverage_is_still_inconclusive(ratio):
    with pytest.raises(RuntimeError, match="No registered positive lambda"):
        measured(ratio).finish()


def test_cal_a_ratio_selects_point03_and_records_grid_policy():
    result = measured(2.257).finish()
    assert result["lambda_max"] == 0.03
    assert result["grid_version"] == "v3"
    assert result["grid_policy"] == "SPEC-8-2026-10-24-21:15-grid-v3"
    assert result["policy"] == "SPEC-8-2026-10-24-15:40"


@pytest.mark.parametrize("ratio,expected", [(1.7, 0.1), (1.9, 0.03)])
def test_two_qualifying_candidates_use_log_distance(ratio, expected):
    assert sum(0.05 <= a * ratio <= 0.2 for a in GRID) == 2
    assert measured(ratio).finish()["lambda_max"] == expected


def test_equal_log_distance_tie_selects_smaller_lambda():
    ratio = 0.1 / math.sqrt(0.03 * 0.1)
    assert abs(math.log(0.03 * ratio / 0.1)) == abs(math.log(0.1 * ratio / 0.1))
    assert measured(ratio).finish()["lambda_max"] == 0.03


def test_q2_gate_precedes_budget_failure_and_measurement_steps_unchanged():
    with pytest.raises(RuntimeError, match="audit CE"):
        measured(0.01, ce=0.96).finish()
    assert [s for s in range(1, 1251) if Calibration.is_measurement(s)] == list(
        range(275, 1251, 25)
    )
