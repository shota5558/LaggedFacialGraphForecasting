from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting import (
    DesignMatrix,
    PredictionArtifact,
    fit_ridge_forecaster,
    predict_ridge_forecaster,
)


def _matrix(
    X: np.ndarray,
    y: np.ndarray,
    *,
    valid_mask: np.ndarray,
    feature_names: tuple[str, ...] = ("mouth.vx", "mouth.vy"),
    feature_lags: tuple[int, ...] = (1, 1),
) -> DesignMatrix:
    n_rows = X.shape[0]
    return DesignMatrix(
        X=X,
        y=y,
        subject_id=("s01",) * n_rows,
        region_id=("mouth",) * n_rows,
        target_dimensions=("vx", "vy"),
        forecast_origin=np.arange(n_rows, dtype=float),
        target_time=np.arange(1, n_rows + 1, dtype=float),
        feature_names=feature_names,
        feature_lags=feature_lags,
        valid_mask=valid_mask,
    )


def _fitted():
    train = _matrix(
        np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]),
        valid_mask=np.ones(4, dtype=bool),
    )
    return fit_ridge_forecaster(train, alpha=1.0)


def test_prediction_returns_row_aligned_artifact() -> None:
    matrix = _matrix(
        np.array([[1.5, 1.5], [2.5, 2.5]]),
        np.array([[1.5, 3.0], [2.5, 5.0]]),
        valid_mask=np.ones(2, dtype=bool),
    )

    artifact = predict_ridge_forecaster(
        _fitted(), matrix, outer_fold=2, condition="self"
    )

    assert isinstance(artifact, PredictionArtifact)
    assert artifact.outer_fold == 2
    assert artifact.condition == "self"
    assert artifact.subject_id == matrix.subject_id
    assert artifact.region_id == matrix.region_id
    assert artifact.target_dimensions == matrix.target_dimensions
    assert np.array_equal(artifact.forecast_origin, matrix.forecast_origin)
    assert np.array_equal(artifact.target_time, matrix.target_time)
    assert np.array_equal(artifact.y_true, matrix.y)
    assert artifact.y_pred.shape == matrix.y.shape
    assert np.all(np.isfinite(artifact.y_pred))


def test_invalid_rows_are_not_predicted_and_remain_nan() -> None:
    matrix = _matrix(
        np.array([[1.5, 1.5], [np.nan, np.nan]]),
        np.array([[1.5, 3.0], [np.nan, np.nan]]),
        valid_mask=np.array([True, False]),
    )

    artifact = predict_ridge_forecaster(
        _fitted(), matrix, outer_fold=0, condition="self"
    )

    assert np.all(np.isfinite(artifact.y_pred[0]))
    assert np.all(np.isnan(artifact.y_pred[1]))
    assert np.array_equal(artifact.valid_mask, np.array([True, False]))


def test_rejects_feature_name_provenance_mismatch() -> None:
    matrix = _matrix(
        np.array([[1.0, 1.0]]),
        np.array([[1.0, 2.0]]),
        valid_mask=np.array([True]),
        feature_names=("jaw.vx", "mouth.vy"),
    )

    with pytest.raises(ValueError, match="feature_names"):
        predict_ridge_forecaster(_fitted(), matrix, outer_fold=0, condition="self")


def test_rejects_feature_lag_provenance_mismatch() -> None:
    matrix = _matrix(
        np.array([[1.0, 1.0]]),
        np.array([[1.0, 2.0]]),
        valid_mask=np.array([True]),
        feature_lags=(1, 2),
    )

    with pytest.raises(ValueError, match="feature_lags"):
        predict_ridge_forecaster(_fitted(), matrix, outer_fold=0, condition="self")
