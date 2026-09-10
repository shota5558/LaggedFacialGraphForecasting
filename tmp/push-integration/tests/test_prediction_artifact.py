from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting import ContractError, PredictionArtifact


def _valid_kwargs() -> dict:
    return {
        "outer_fold": 0,
        "subject_id": ("s01", "s01", "s02", "s02"),
        "region_id": ("mouth", "mouth", "jaw", "jaw"),
        "target_dimensions": ("vx", "vy"),
        "condition": "self",
        "forecast_origin": np.array([10.0, 11.0, 10.0, 11.0]),
        "target_time": np.array([11.0, 12.0, 11.0, 12.0]),
        "y_true": np.array(
            [[0.2, 0.1], [0.3, 0.2], [0.4, 0.3], [0.5, 0.4]], dtype=float
        ),
        "y_pred": np.array(
            [[0.19, 0.11], [0.29, 0.21], [0.41, 0.31], [0.49, 0.39]], dtype=float
        ),
        "valid_mask": np.array([True, True, True, True]),
    }


def test_prediction_artifact_accepts_canonical_provenance() -> None:
    artifact = PredictionArtifact(**_valid_kwargs())

    assert artifact.outer_fold == 0
    assert artifact.condition == "self"
    assert artifact.y_true.shape == artifact.y_pred.shape == (4, 2)
    assert len(artifact.subject_id) == 4
    assert len(artifact.region_id) == 4
    assert artifact.target_dimensions == ("vx", "vy")
    assert artifact.valid_mask.dtype == np.bool_


def test_rejects_prediction_shape_mismatch() -> None:
    kwargs = _valid_kwargs()
    kwargs["y_pred"] = np.zeros((4,), dtype=float)

    with pytest.raises(ContractError, match="y_pred must match y_true.shape"):
        PredictionArtifact(**kwargs)


def test_rejects_target_dimension_count_mismatch() -> None:
    kwargs = _valid_kwargs()
    kwargs["target_dimensions"] = ("vx",)

    with pytest.raises(ContractError, match="target_dimensions must contain 2 entries"):
        PredictionArtifact(**kwargs)


def test_rejects_row_provenance_length_mismatch() -> None:
    kwargs = _valid_kwargs()
    kwargs["region_id"] = ("mouth", "mouth", "jaw")

    with pytest.raises(ContractError, match="region_id must contain 4 entries"):
        PredictionArtifact(**kwargs)


def test_rejects_target_not_after_forecast_origin() -> None:
    kwargs = _valid_kwargs()
    kwargs["target_time"] = np.array([11.0, 11.0, 11.0, 12.0])

    with pytest.raises(ContractError, match="strictly after forecast_origin"):
        PredictionArtifact(**kwargs)


def test_rejects_non_boolean_valid_mask() -> None:
    kwargs = _valid_kwargs()
    kwargs["valid_mask"] = np.ones(4, dtype=np.int8)

    with pytest.raises(ContractError, match="boolean dtype"):
        PredictionArtifact(**kwargs)


def test_valid_rows_must_be_finite_but_invalid_rows_may_be_missing() -> None:
    kwargs = _valid_kwargs()
    kwargs["y_pred"] = kwargs["y_pred"].copy()
    kwargs["y_pred"][3, 0] = np.nan
    kwargs["valid_mask"] = np.array([True, True, True, False])

    artifact = PredictionArtifact(**kwargs)
    assert np.isnan(artifact.y_pred[3, 0])
    assert not artifact.valid_mask[3]

    kwargs["valid_mask"] = np.array([True, True, True, True])
    with pytest.raises(ContractError, match="valid rows of y_pred must be finite"):
        PredictionArtifact(**kwargs)


@pytest.mark.parametrize("condition", ["", "   "])
def test_rejects_empty_condition(condition: str) -> None:
    kwargs = _valid_kwargs()
    kwargs["condition"] = condition

    with pytest.raises(ContractError, match="condition must be a non-empty string"):
        PredictionArtifact(**kwargs)
