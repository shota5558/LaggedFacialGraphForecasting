from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting import FaceTimeSeries
from lagged_facial_graph_forecasting.design_matrix import (
    build_persistence_design_matrix,
    build_self_history_design_matrix,
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


def test_persistence_uses_only_current_target_region_value() -> None:
    matrix = build_persistence_design_matrix(_series(), target_region="mouth")

    assert np.array_equal(
        matrix.X,
        np.array([[10, 20], [11, 21], [12, 22], [13, 23]], dtype=float),
    )
    assert np.array_equal(
        matrix.y,
        np.array([[11, 21], [12, 22], [13, 23], [14, 24]], dtype=float),
    )
    assert matrix.feature_names == ("mouth.vx", "mouth.vy")
    assert matrix.feature_lags == (1, 1)
    assert np.array_equal(matrix.forecast_origin, np.array([0, 1, 2, 3], dtype=float))
    assert np.array_equal(matrix.target_time, np.array([1, 2, 3, 4], dtype=float))


def test_persistence_is_exact_self_history_lag_one_specialization() -> None:
    series = _series()
    persistence = build_persistence_design_matrix(series, target_region="mouth")
    self_lag_one = build_self_history_design_matrix(
        series,
        target_region="mouth",
        lags=(1,),
        horizon=1,
    )

    assert np.array_equal(persistence.X, self_lag_one.X)
    assert np.array_equal(persistence.y, self_lag_one.y)
    assert persistence.feature_names == self_lag_one.feature_names
    assert persistence.feature_lags == self_lag_one.feature_lags
    assert np.array_equal(persistence.valid_mask, self_lag_one.valid_mask)


def test_persistence_propagates_source_and_target_missingness() -> None:
    series = _series()
    mask = series.valid_mask.copy()
    mask[1, 0, 0] = False
    mask[3, 0, 1] = False
    masked = FaceTimeSeries(
        X=series.X,
        subject_id=series.subject_id,
        time_index=series.time_index,
        region_id=series.region_id,
        dimension=series.dimension,
        valid_mask=mask,
        sampling_rate=series.sampling_rate,
    )

    matrix = build_persistence_design_matrix(masked, target_region="mouth")

    assert np.array_equal(matrix.valid_mask, np.array([False, False, False, False]))


def test_persistence_rejects_non_primary_horizon() -> None:
    with pytest.raises(ValueError, match="frozen to h=1"):
        build_persistence_design_matrix(
            _series(), target_region="mouth", horizon=2
        )
