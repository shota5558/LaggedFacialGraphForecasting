from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting import (
    DesignMatrix,
    fit_ridge_forecaster,
    predict_ridge_forecaster,
)


def _matrix(
    *,
    feature_names: tuple[str, ...] = ("mouth.vx", "mouth.vy"),
    feature_lags: tuple[int, ...] = (1, 1),
    valid_mask: np.ndarray | None = None,
) -> DesignMatrix:
    if valid_mask is None:
        valid_mask = np.array([True, True, False, True], dtype=bool)
    return DesignMatrix(
        X=np.array(
            [[0.0, 1.0], [1.0, 2.0], [1000.0, 1000.0], [3.0, 4.0]],
            dtype=float,
        ),
        y=np.array([[0.0], [1.0], [1000.0], [3.0]], dtype=float),
        subject_id=("s01", "s01", "s01", "s01"),
        region_id=("mouth", "mouth", "mouth", "mouth"),
        forecast_origin=np.array([1.0, 2.0, 3.0, 4.0]),
        target_time=np.array([2.0, 3.0, 4.0, 5.0]),
        feature_names=feature_names,
        feature_lags=feature_lags,
        valid_mask=valid_mask,
    )


def test_prediction_preserves_row_provenance_and_masks_invalid_rows() -> None:
    train = _matrix(valid_mask=np.array([True, True, False, True], dtype=bool))
    fitted = fit_ridge_forecaster(train, alpha=1.0)

    artifact = predict_ridge_forecaster(
        fitted,
        _matrix(valid_mask=np.array([True, False, True, True], dtype=bool)),
        outer_fold=2,
        condition="self",
    )

    assert artifact.outer_fold == 2
    assert artifact.condition == "self"
    assert artifact.subject_id == ("s01",) * 4
    assert artifact.region_id == ("mouth",) * 4
    assert np.array_equal(artifact.forecast_origin, np.array([1.0, 2.0, 3.0, 4.0]))
    assert np.array_equal(artifact.target_time, np.array([2.0, 3.0, 4.0, 5.0]))
    assert np.array_equal(artifact.valid_mask, np.array([True, False, True, True]))
    assert np.isnan(artifact.y_pred[1]).all()
    assert np.isfinite(artifact.y_pred[artifact.valid_mask]).all()
    assert np.array_equal(artifact.y_true, _matrix().y)


def test_prediction_rejects_feature_name_drift() -> None:
    fitted = fit_ridge_forecaster(_matrix(), alpha=1.0)
    changed = _matrix(feature_names=("jaw.vx", "mouth.vy"))

    with pytest.raises(ValueError, match="feature_names"):
        predict_ridge_forecaster(fitted, changed, outer_fold=0, condition="self")


def test_prediction_rejects_feature_lag_drift() -> None:
    fitted = fit_ridge_forecaster(_matrix(), alpha=1.0)
    changed = _matrix(feature_lags=(1, 2))

    with pytest.raises(ValueError, match="feature_lags"):
        predict_ridge_forecaster(fitted, changed, outer_fold=0, condition="self")


def test_prediction_with_no_valid_rows_returns_masked_artifact() -> None:
    fitted = fit_ridge_forecaster(_matrix(), alpha=1.0)
    matrix = _matrix(valid_mask=np.array([False, False, False, False], dtype=bool))

    artifact = predict_ridge_forecaster(fitted, matrix, outer_fold=0, condition="self")

    assert not artifact.valid_mask.any()
    assert np.isnan(artifact.y_pred).all()
