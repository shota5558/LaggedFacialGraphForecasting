"""Coefficient export for a frozen Primary Ridge forecaster."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .forecaster import FittedRidgeForecaster


@dataclass(frozen=True, slots=True)
class RidgeCoefficientExport:
    """Read-only Ridge coefficients with frozen feature/lag provenance.

    Coefficients are those of the Ridge estimator after ``StandardScaler`` and
    therefore live in standardized-feature space.  ``coefficient`` is always
    normalized to shape ``(output_dim, feature_count)`` and ``intercept`` to
    ``(output_dim,)``.
    """

    alpha: float
    feature_names: tuple[str, ...]
    feature_lags: tuple[int, ...]
    coefficient: np.ndarray
    intercept: np.ndarray
    coefficient_space: str = "standardized_feature_space"


def export_ridge_coefficients(fitted: FittedRidgeForecaster) -> RidgeCoefficientExport:
    """Export fitted Ridge parameters without refitting or transforming data."""

    ridge = fitted.pipeline.named_steps.get("ridge")
    scaler = fitted.pipeline.named_steps.get("scaler")
    if ridge is None or scaler is None:
        raise RuntimeError("fitted Ridge pipeline must contain scaler and ridge steps")
    if not hasattr(ridge, "coef_") or not hasattr(ridge, "intercept_"):
        raise RuntimeError("Ridge estimator is not fitted")

    coefficient = np.asarray(ridge.coef_, dtype=float)
    if coefficient.ndim == 1:
        coefficient = coefficient.reshape(1, -1)
    if coefficient.ndim != 2:
        raise RuntimeError("Ridge coefficient array must be one- or two-dimensional")

    feature_count = len(fitted.feature_names)
    if coefficient.shape[1] != feature_count:
        raise RuntimeError(
            "Ridge coefficient feature dimension does not match frozen provenance"
        )
    if len(fitted.feature_lags) != feature_count:
        raise RuntimeError("feature_lags do not match frozen feature provenance")

    intercept = np.asarray(ridge.intercept_, dtype=float)
    if intercept.ndim == 0:
        intercept = intercept.reshape(1)
    else:
        intercept = intercept.reshape(-1)
    if intercept.shape != (coefficient.shape[0],):
        raise RuntimeError("Ridge intercept output dimension does not match coefficients")

    coefficient = coefficient.copy()
    intercept = intercept.copy()
    coefficient.setflags(write=False)
    intercept.setflags(write=False)

    return RidgeCoefficientExport(
        alpha=float(fitted.alpha),
        feature_names=fitted.feature_names,
        feature_lags=fitted.feature_lags,
        coefficient=coefficient,
        intercept=intercept,
    )
