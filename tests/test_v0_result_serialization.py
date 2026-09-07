from __future__ import annotations

import json

import numpy as np

from lagged_facial_graph_forecasting import (
    DesignMatrix,
    fit_ridge_forecaster,
    predict_ridge_forecaster,
    velocity_rmse,
    write_v0_result_json,
)


def _matrix() -> DesignMatrix:
    return DesignMatrix(
        X=np.array([[0.0, 1.0], [1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]]),
        y=np.array([[1.0, 2.0], [2.0, 3.0], [np.nan, np.nan], [4.0, 5.0]]),
        subject_id=("s01", "s01", "s01", "s01"),
        region_id=("mouth", "mouth", "mouth", "mouth"),
        target_dimensions=("vx", "vy"),
        forecast_origin=np.array([0.0, 1.0, 2.0, 3.0]),
        target_time=np.array([1.0, 2.0, 3.0, 4.0]),
        feature_names=("mouth.vx", "mouth.vy"),
        feature_lags=(1, 1),
        valid_mask=np.array([True, True, False, True]),
    )


def test_result_serialization_preserves_provenance_and_uses_json_null(tmp_path) -> None:
    matrix = _matrix()
    fitted = fit_ridge_forecaster(matrix, alpha=1.0)
    prediction = predict_ridge_forecaster(
        fitted, matrix, outer_fold=3, condition="self"
    )
    metric = velocity_rmse(prediction)

    path = write_v0_result_json(
        tmp_path / "nested" / "result.json",
        matrix=matrix,
        prediction=prediction,
        fitted=fitted,
        velocity_rmse_value=metric,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 2
    assert payload["outer_fold"] == 3
    assert payload["condition"] == "self"
    assert payload["feature_provenance"] == {
        "feature_names": ["mouth.vx", "mouth.vy"],
        "feature_lags": [1, 1],
    }
    assert payload["target_provenance"] == {
        "target_dimensions": ["vx", "vy"],
    }
    assert payload["model"]["forecaster"] == "ridge"
    assert payload["model"]["alpha"] == 1.0
    assert payload["metric"]["velocity_rmse"] == metric
    assert payload["rows"]["valid_mask"] == [True, True, False, True]
    assert payload["rows"]["y_true"][2] == [None, None]
    assert payload["rows"]["y_pred"][2] == [None, None]
    assert "NaN" not in path.read_text(encoding="utf-8")


def test_serialization_is_byte_deterministic(tmp_path) -> None:
    matrix = _matrix()
    fitted = fit_ridge_forecaster(matrix, alpha=1.0)
    prediction = predict_ridge_forecaster(
        fitted, matrix, outer_fold=0, condition="self"
    )
    metric = velocity_rmse(prediction)

    first = write_v0_result_json(
        tmp_path / "first.json",
        matrix=matrix,
        prediction=prediction,
        fitted=fitted,
        velocity_rmse_value=metric,
    )
    second = write_v0_result_json(
        tmp_path / "second.json",
        matrix=matrix,
        prediction=prediction,
        fitted=fitted,
        velocity_rmse_value=metric,
    )

    assert first.read_bytes() == second.read_bytes()
