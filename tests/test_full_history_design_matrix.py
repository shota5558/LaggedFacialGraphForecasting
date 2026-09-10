from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting import FaceTimeSeries
from lagged_facial_graph_forecasting.design_matrix import (
    build_full_history_design_matrix,
)


def _series() -> FaceTimeSeries:
    X = np.zeros((5, 2, 2), dtype=float)
    X[:, 0, 0] = np.array([10, 11, 12, 13, 14], dtype=float)
    X[:, 0, 1] = np.array([20, 21, 22, 23, 24], dtype=float)
    X[:, 1, 0] = np.array([100, 101, 102, 103, 104], dtype=float)
    X[:, 1, 1] = np.array([200, 201, 202, 203, 204], dtype=float)
    return FaceTimeSeries(
        X=X,
        subject_id="s01",
        time_index=np.arange(5, dtype=float),
        region_id=("mouth", "jaw"),
        dimension=("vx", "vy"),
        valid_mask=np.ones_like(X, dtype=bool),
        sampling_rate=30.0,
    )


def test_full_history_uses_all_regions_and_requested_lags_in_deterministic_order() -> None:
    matrix = build_full_history_design_matrix(
        _series(), target_region="mouth", lags=(2, 1), horizon=1
    )

    assert np.array_equal(
        matrix.X,
        np.array(
            [
                [11, 21, 101, 201, 10, 20, 100, 200],
                [12, 22, 102, 202, 11, 21, 101, 201],
                [13, 23, 103, 203, 12, 22, 102, 202],
            ],
            dtype=float,
        ),
    )
    assert np.array_equal(
        matrix.y,
        np.array([[12, 22], [13, 23], [14, 24]], dtype=float),
    )
    assert matrix.feature_names == (
        "mouth.vx",
        "mouth.vy",
        "jaw.vx",
        "jaw.vy",
        "mouth.vx",
        "mouth.vy",
        "jaw.vx",
        "jaw.vy",
    )
    assert matrix.feature_lags == (1, 1, 1, 1, 2, 2, 2, 2)
    assert np.array_equal(matrix.forecast_origin, np.array([1, 2, 3], dtype=float))
    assert np.array_equal(matrix.target_time, np.array([2, 3, 4], dtype=float))


def test_full_history_missingness_requires_all_features_and_target() -> None:
    series = _series()
    mask = series.valid_mask.copy()
    mask[0, 1, 0] = False
    mask[4, 0, 1] = False
    masked = FaceTimeSeries(
        X=series.X,
        subject_id=series.subject_id,
        time_index=series.time_index,
        region_id=series.region_id,
        dimension=series.dimension,
        valid_mask=mask,
        sampling_rate=series.sampling_rate,
    )

    matrix = build_full_history_design_matrix(
        masked, target_region="mouth", lags=(1, 2), horizon=1
    )

    assert np.array_equal(matrix.valid_mask, np.array([False, True, False]))


def test_full_history_rejects_unknown_target_region() -> None:
    with pytest.raises(ValueError, match="unknown target_region"):
        build_full_history_design_matrix(
            _series(), target_region="eye", lags=(1, 2)
        )


def test_full_history_rejects_non_primary_horizon() -> None:
    with pytest.raises(ValueError, match="frozen to h=1"):
        build_full_history_design_matrix(
            _series(), target_region="mouth", lags=(2,), horizon=2
        )
