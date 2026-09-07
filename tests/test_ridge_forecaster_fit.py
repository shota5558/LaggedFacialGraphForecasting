from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from lagged_facial_graph_forecasting import DesignMatrix, fit_ridge_forecaster


def _matrix(valid_mask: np.ndarray | None = None) -> DesignMatrix:
    if valid_mask is None:
        valid_mask = np.array([True, False, True])
    return DesignMatrix(
        X=np.array([[0.0, 1.0], [100.0, 100.0], [2.0, 3.0]]),
        y=np.array([[0.0], [100.0], [2.0]]),
        subject_id=("s01", "s01", "s01"),
        region_id=("mouth", "mouth", "mouth"),
        target_dimensions=("vx",),
        forecast_origin=np.array([1.0, 2.0, 3.0]),
        target_time=np.array([2.0, 3.0, 4.0]),
        feature_names=("mouth.vx", "mouth.vy"),
        feature_lags=(1, 1),
        valid_mask=valid_mask,
    )


def test_fit_uses_sklearn_scaler_and_ridge() -> None:
    fitted = fit_ridge_forecaster(_matrix(), alpha=2.0)

    assert isinstance(fitted.pipeline.named_steps["scaler"], StandardScaler)
    assert isinstance(fitted.pipeline.named_steps["ridge"], Ridge)
    assert fitted.pipeline.named_steps["ridge"].alpha == 2.0
    assert fitted.alpha == 2.0


def test_scaler_is_fit_on_valid_rows_only() -> None:
    fitted = fit_ridge_forecaster(_matrix(), alpha=1.0)
    scaler = fitted.pipeline.named_steps["scaler"]

    assert fitted.training_row_count == 2
    assert np.allclose(scaler.mean_, np.array([1.0, 2.0]))
    assert fitted.feature_names == ("mouth.vx", "mouth.vy")
    assert fitted.feature_lags == (1, 1)
    assert fitted.target_dimensions == ("vx",)


def test_fit_is_deterministic_for_same_matrix_and_alpha() -> None:
    first = fit_ridge_forecaster(_matrix(), alpha=1.0)
    second = fit_ridge_forecaster(_matrix(), alpha=1.0)

    assert np.allclose(
        first.pipeline.named_steps["ridge"].coef_,
        second.pipeline.named_steps["ridge"].coef_,
    )
    assert np.allclose(
        first.pipeline.named_steps["ridge"].intercept_,
        second.pipeline.named_steps["ridge"].intercept_,
    )


@pytest.mark.parametrize("alpha", [0.0, -1.0, np.inf, np.nan])
def test_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        fit_ridge_forecaster(_matrix(), alpha=alpha)


def test_rejects_matrix_without_valid_training_rows() -> None:
    with pytest.raises(ValueError, match="at least one valid training row"):
        fit_ridge_forecaster(
            _matrix(valid_mask=np.array([False, False, False])), alpha=1.0
        )
