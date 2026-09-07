from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting import (
    FaceTimeSeries,
    build_self_history_design_matrix,
)


def _series() -> FaceTimeSeries:
    X = np.zeros((6, 2, 2), dtype=float)
    X[:, 0, 0] = np.array([10, 11, 12, 13, 14, 15], dtype=float)
    X[:, 0, 1] = np.array([20, 21, 22, 23, 24, 25], dtype=float)
    X[:, 1, 0] = np.array([100, 101, 102, 103, 104, 105], dtype=float)
    X[:, 1, 1] = np.array([200, 201, 202, 203, 204, 205], dtype=float)
    return FaceTimeSeries(
        X=X,
        subject_id="s01",
        time_index=np.arange(6, dtype=float),
        region_id=("mouth", "jaw"),
        dimension=("vx", "vy"),
        valid_mask=np.ones_like(X, dtype=bool),
        sampling_rate=30.0,
    )


def test_self_history_uses_only_target_region_and_correct_target_relative_lags() -> None:
    matrix = build_self_history_design_matrix(
        _series(), target_region="mouth", lags=(1, 2), horizon=1
    )

    assert matrix.X.shape == (4, 4)
    assert np.array_equal(
        matrix.X,
        np.array(
            [
                [11, 21, 10, 20],
                [12, 22, 11, 21],
                [13, 23, 12, 22],
                [14, 24, 13, 23],
            ],
            dtype=float,
        ),
    )
    assert np.array_equal(
        matrix.y,
        np.array([[12, 22], [13, 23], [14, 24], [15, 25]], dtype=float),
    )
    assert matrix.feature_names == (
        "mouth.vx",
        "mouth.vy",
        "mouth.vx",
        "mouth.vy",
    )
    assert matrix.feature_lags == (1, 1, 2, 2)
    assert matrix.subject_id == ("s01",) * 4
    assert matrix.region_id == ("mouth",) * 4
    assert np.array_equal(matrix.forecast_origin, np.array([1, 2, 3, 4], dtype=float))
    assert np.array_equal(matrix.target_time, np.array([2, 3, 4, 5], dtype=float))


def test_self_history_mask_combines_source_and_target_validity() -> None:
    series = _series()
    mask = series.valid_mask.copy()
    mask[1, 0, 0] = False  # invalid lag-1 source for target index 2
    mask[4, 0, 1] = False  # invalid target for target index 4
    series = FaceTimeSeries(
        X=series.X,
        subject_id=series.subject_id,
        time_index=series.time_index,
        region_id=series.region_id,
        dimension=series.dimension,
        valid_mask=mask,
        sampling_rate=series.sampling_rate,
    )

    matrix = build_self_history_design_matrix(
        series, target_region="mouth", lags=(1, 2), horizon=1
    )

    assert np.array_equal(matrix.valid_mask, np.array([False, True, False, False]))


def test_rejects_unknown_target_region() -> None:
    with pytest.raises(ValueError, match="unknown target_region"):
        build_self_history_design_matrix(_series(), target_region="eye", lags=(1,))


def test_rejects_duplicate_lags() -> None:
    with pytest.raises(ValueError, match="lags must be unique"):
        build_self_history_design_matrix(_series(), target_region="mouth", lags=(1, 1))


def test_rejects_non_primary_horizon() -> None:
    with pytest.raises(ValueError, match="frozen to h=1"):
        build_self_history_design_matrix(
            _series(), target_region="mouth", lags=(2,), horizon=2
        )
