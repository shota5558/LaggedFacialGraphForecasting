from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting.contracts import PredictionArtifact
from lagged_facial_graph_forecasting.metrics import (
    VELOCITY_RMSE_NAME,
    velocity_rmse_by_subject_region,
)


def _artifact() -> PredictionArtifact:
    return PredictionArtifact(
        outer_fold=2,
        subject_id=("subject_b", "subject_a", "subject_a", "subject_b", "subject_a"),
        region_id=("mouth", "mouth", "eye", "mouth", "mouth"),
        target_dimensions=("vx", "vy"),
        condition="pcmci",
        forecast_origin=np.asarray([0, 1, 2, 3, 4], dtype=float),
        target_time=np.asarray([1, 2, 3, 4, 5], dtype=float),
        y_true=np.asarray(
            [
                [1.0, 2.0],
                [2.0, 4.0],
                [5.0, 6.0],
                [np.nan, np.nan],
                [4.0, 8.0],
            ]
        ),
        y_pred=np.asarray(
            [
                [2.0, 2.0],
                [1.0, 5.0],
                [4.0, 8.0],
                [np.nan, np.nan],
                [5.0, 6.0],
            ]
        ),
        valid_mask=np.asarray([True, True, True, False, True]),
    )


def test_subject_region_velocity_rmse_matches_frozen_rmse_definition() -> None:
    results = velocity_rmse_by_subject_region(_artifact())

    assert [(r.subject_id, r.region_id) for r in results] == [
        ("subject_a", "eye"),
        ("subject_a", "mouth"),
        ("subject_b", "mouth"),
    ]
    assert [r.value for r in results] == pytest.approx(
        [np.sqrt(2.5), np.sqrt(1.75), np.sqrt(0.5)]
    )
    assert [r.n_valid for r in results] == [1, 2, 1]
    assert all(r.metric_name == VELOCITY_RMSE_NAME for r in results)
    assert all(r.condition == "pcmci" for r in results)
    assert all(r.outer_fold == 2 for r in results)


def test_subject_region_velocity_rmse_fails_closed_without_valid_rows() -> None:
    artifact = PredictionArtifact(
        outer_fold=0,
        subject_id=("subject_a", "subject_a"),
        region_id=("mouth", "mouth"),
        target_dimensions=("vx",),
        condition="self",
        forecast_origin=np.asarray([0.0, 1.0]),
        target_time=np.asarray([1.0, 2.0]),
        y_true=np.asarray([np.nan, np.nan]),
        y_pred=np.asarray([np.nan, np.nan]),
        valid_mask=np.asarray([False, False]),
    )

    with pytest.raises(ValueError, match="at least one valid prediction row"):
        velocity_rmse_by_subject_region(artifact)
