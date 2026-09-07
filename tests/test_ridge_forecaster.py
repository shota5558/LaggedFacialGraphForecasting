from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting import DesignMatrix, fit_ridge_forecaster


def _matrix(*, valid_mask: np.ndarray | None = None) -> DesignMatrix:
    X = np.array(
        [
            [0.0, 10.0],
            [1.0, 11.0],
            [2.0, 12.0],
            [1000.0, 1000.0],
        ]
    )
    y = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
            [1000.0, 1000.0],
        ]
    )
    if valid_mask is None:
        valid_mask = np.array([True, True, True, False])
    return DesignMatrix(
        X=X,
        y=y,
        subject_id=("s01",) * 4,
        region_id=("mouth",) * 4,
        forecast_origin=np.array([0.0, 1.0, 2.0, 3.0]),
        target_time=np.array([1.0, 2.0, 3.0, 4.0]),
        feature_names=("mouth.vx", "mouth.vy"),
        feature_lags=(1, 1),
        valid_mask=valid_mask,
    )


def test_fit_ridge_forecaster_uses_valid_training_rows_only() -> None:
    fitted = fit_ridge_forecaster(_matrix(), alpha=2.5)

    scaler = fitted.pipeline.named_steps["scaler"]
    ridge = fitted.pipeline.named_steps["ridge"]

    assert np.allclose(scaler.mean_, np.array([1.0, 11.0]))
    assert fitted.training_row_count == 3
    assert fitted.alpha == 2.5
    assert ridge.alpha == 2.5


def test_fit_ridge_forecaster_preserves_feature_provenance() -> None:
    matrix = _matrix()
    fitted = fit_ridge_forecaster(matrix)

    assert fitted.feature_names == matrix.feature_names
    assert fitted.feature_lags == matrix.feature_lags


def test_fit_ridge_forecaster_is_deterministic_for_same_input() -> None:
    matrix = _matrix()
    first = fit_ridge_forecaster(matrix, alpha=1.0)
    second = fit_ridge_forecaster(matrix, alpha=1.0)

    assert np.allclose(
        first.pipeline.named_steps["scaler"].mean_,
        second.pipeline.named_steps["scaler"].mean_,
    )
    assert np.allclose(
        first.pipeline.named_steps["ridge"].coef_,
        second.pipeline.named_steps["ridge"].coef_,
    )
    assert np.allclose(
        first.pipeline.named_steps["ridge"].intercept_,
        second.pipeline.named_steps["ridge"].intercept_,
    )


@pytest.mark.parametrize("alpha", [0.0, -1.0, np.nan, np.inf])
def test_fit_ridge_forecaster_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must be finite and > 0"):
        fit_ridge_forecaster(_matrix(), alpha=alpha)


def test_fit_ridge_forecaster_rejects_no_valid_training_rows() -> None:
    with pytest.raises(ValueError, match="at least one valid training row"):
        fit_ridge_forecaster(_matrix(valid_mask=np.zeros(4, dtype=bool)))
