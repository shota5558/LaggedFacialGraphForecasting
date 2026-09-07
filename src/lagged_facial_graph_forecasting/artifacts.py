"""Artifact serialization for the V0 vertical slice."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import DesignMatrix, PredictionArtifact
from .forecaster import FittedRidgeForecaster


def _jsonable_numeric_array(values: np.ndarray) -> list[Any]:
    array = np.asarray(values)

    def convert(value: Any) -> Any:
        if isinstance(value, (np.floating, float)):
            number = float(value)
            return number if np.isfinite(number) else None
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        return value

    if array.ndim == 1:
        return [convert(value) for value in array.tolist()]
    return [[convert(value) for value in row] for row in array.tolist()]


def build_v0_result_payload(
    *,
    matrix: DesignMatrix,
    prediction: PredictionArtifact,
    fitted: FittedRidgeForecaster,
    velocity_rmse_value: float,
) -> dict[str, Any]:
    """Build the deterministic, provenance-complete V0 result payload."""

    if prediction.subject_id != matrix.subject_id:
        raise ValueError("prediction subject provenance does not match DesignMatrix")
    if prediction.region_id != matrix.region_id:
        raise ValueError("prediction region provenance does not match DesignMatrix")
    if not np.array_equal(prediction.forecast_origin, matrix.forecast_origin):
        raise ValueError("prediction forecast_origin does not match DesignMatrix")
    if not np.array_equal(prediction.target_time, matrix.target_time):
        raise ValueError("prediction target_time does not match DesignMatrix")
    if matrix.feature_names != fitted.feature_names or matrix.feature_lags != fitted.feature_lags:
        raise ValueError("fitted feature provenance does not match DesignMatrix")

    velocity_rmse_value = float(velocity_rmse_value)
    if not np.isfinite(velocity_rmse_value) or velocity_rmse_value < 0:
        raise ValueError("velocity_rmse_value must be finite and >= 0")

    return {
        "schema_version": 1,
        "outer_fold": prediction.outer_fold,
        "condition": prediction.condition,
        "feature_provenance": {
            "feature_names": list(matrix.feature_names),
            "feature_lags": list(matrix.feature_lags),
        },
        "model": {
            "forecaster": "ridge",
            "alpha": fitted.alpha,
            "training_row_count": fitted.training_row_count,
        },
        "metric": {
            "velocity_rmse": velocity_rmse_value,
        },
        "rows": {
            "subject_id": list(prediction.subject_id),
            "region_id": list(prediction.region_id),
            "forecast_origin": _jsonable_numeric_array(prediction.forecast_origin),
            "target_time": _jsonable_numeric_array(prediction.target_time),
            "y_true": _jsonable_numeric_array(prediction.y_true),
            "y_pred": _jsonable_numeric_array(prediction.y_pred),
            "valid_mask": _jsonable_numeric_array(prediction.valid_mask),
        },
    }


def write_v0_result_json(
    path: str | Path,
    *,
    matrix: DesignMatrix,
    prediction: PredictionArtifact,
    fitted: FittedRidgeForecaster,
    velocity_rmse_value: float,
) -> Path:
    """Write a deterministic UTF-8 JSON result and return its path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_v0_result_payload(
        matrix=matrix,
        prediction=prediction,
        fitted=fitted,
        velocity_rmse_value=velocity_rmse_value,
    )
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path
