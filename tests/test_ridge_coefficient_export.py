from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting.contracts import DesignMatrix
from lagged_facial_graph_forecasting.forecaster import fit_ridge_forecaster
from lagged_facial_graph_forecasting.ridge_coefficients import export_ridge_coefficients


def _matrix(*, multi_output: bool) -> DesignMatrix:
    x = np.array(
        [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0], [4.0, 1.0]],
        dtype=float,
    )
    first = 2.0 * x[:, 0] - x[:, 1]
    y: np.ndarray
    if multi_output:
        second = -0.5 * x[:, 0] + 3.0 * x[:, 1]
        y = np.column_stack((first, second))
    else:
        y = first
    n_rows = x.shape[0]
    return DesignMatrix(
        X=x,
        y=y,
        subject_id=("s01",) * n_rows,
        region_id=("mouth",) * n_rows,
        forecast_origin=np.arange(n_rows, dtype=float),
        target_time=np.arange(1, n_rows + 1, dtype=float),
        feature_names=("left_cheek.vx", "jaw.vx"),
        feature_lags=(2, 1),
        valid_mask=np.ones(n_rows, dtype=bool),
    )


def test_export_preserves_alpha_feature_and_lag_provenance() -> None:
    fitted = fit_ridge_forecaster(_matrix(multi_output=True), alpha=0.1)

    exported = export_ridge_coefficients(fitted)

    assert exported.alpha == 0.1
    assert exported.feature_names == ("left_cheek.vx", "jaw.vx")
    assert exported.feature_lags == (2, 1)
    assert exported.coefficient_space == "standardized_feature_space"
    assert exported.coefficient.shape == (2, 2)
    assert exported.intercept.shape == (2,)
    assert np.isfinite(exported.coefficient).all()
    assert np.isfinite(exported.intercept).all()


def test_single_output_coefficients_are_normalized_to_two_dimensions() -> None:
    fitted = fit_ridge_forecaster(_matrix(multi_output=False), alpha=1.0)

    exported = export_ridge_coefficients(fitted)

    assert exported.coefficient.shape == (1, 2)
    assert exported.intercept.shape == (1,)


def test_exported_arrays_are_read_only_snapshots() -> None:
    fitted = fit_ridge_forecaster(_matrix(multi_output=True), alpha=1.0)
    exported = export_ridge_coefficients(fitted)

    with pytest.raises(ValueError):
        exported.coefficient[0, 0] = 0.0
    with pytest.raises(ValueError):
        exported.intercept[0] = 0.0
