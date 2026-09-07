"""Forecasting wrappers for the V0 vertical slice."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .contracts import DesignMatrix, PredictionArtifact


@dataclass(slots=True)
class FittedRidgeForecaster:
    """A fitted StandardScaler + Ridge pipeline with frozen feature provenance."""

    pipeline: Pipeline
    feature_names: tuple[str, ...]
    feature_lags: tuple[int, ...]
    alpha: float
    training_row_count: int


def fit_ridge_forecaster(
    matrix: DesignMatrix,
    *,
    alpha: float = 1.0,
) -> FittedRidgeForecaster:
    """Fit StandardScaler and Ridge on valid rows only.

    V0 uses a fixed ``alpha``. Hyperparameter selection and nested-CV isolation are
    intentionally deferred to Lane E; no tuning occurs in this function.
    """

    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and > 0")

    valid_rows = np.asarray(matrix.valid_mask, dtype=bool)
    if not np.any(valid_rows):
        raise ValueError("cannot fit Ridge without at least one valid training row")

    X_train = matrix.X[valid_rows]
    y_train = matrix.y[valid_rows]

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    pipeline.fit(X_train, y_train)

    return FittedRidgeForecaster(
        pipeline=pipeline,
        feature_names=matrix.feature_names,
        feature_lags=matrix.feature_lags,
        alpha=alpha,
        training_row_count=int(np.count_nonzero(valid_rows)),
    )


def predict_ridge_forecaster(
    fitted: FittedRidgeForecaster,
    matrix: DesignMatrix,
    *,
    outer_fold: int,
    condition: str,
) -> PredictionArtifact:
    """Predict valid rows and preserve all row provenance in an artifact."""

    if matrix.feature_names != fitted.feature_names:
        raise ValueError("prediction feature_names do not match fitted model provenance")
    if matrix.feature_lags != fitted.feature_lags:
        raise ValueError("prediction feature_lags do not match fitted model provenance")

    valid_rows = np.asarray(matrix.valid_mask, dtype=bool)
    y_pred = np.full(matrix.y.shape, np.nan, dtype=float)
    if np.any(valid_rows):
        prediction = np.asarray(fitted.pipeline.predict(matrix.X[valid_rows]))
        expected_shape = matrix.y[valid_rows].shape
        if prediction.shape != expected_shape:
            raise RuntimeError(
                f"Ridge prediction shape mismatch: expected {expected_shape}, "
                f"got {prediction.shape}"
            )
        y_pred[valid_rows] = prediction

    return PredictionArtifact(
        outer_fold=outer_fold,
        subject_id=matrix.subject_id,
        region_id=matrix.region_id,
        condition=condition,
        forecast_origin=matrix.forecast_origin,
        target_time=matrix.target_time,
        y_true=matrix.y,
        y_pred=y_pred,
        valid_mask=matrix.valid_mask,
    )
