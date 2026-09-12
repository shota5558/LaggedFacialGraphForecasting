from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting.contracts import PredictionArtifact
from lagged_facial_graph_forecasting.core_contracts import MetricsResult
from lagged_facial_graph_forecasting.metrics import evaluate_metric_by_subject_region


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


def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    assert np.all(np.isfinite(y_true))
    assert np.all(np.isfinite(y_pred))
    return float(np.mean(np.abs(y_pred - y_true)))


def test_metric_interface_returns_subject_region_metrics_with_full_provenance() -> None:
    results = evaluate_metric_by_subject_region(
        _artifact(),
        metric_name="fixture_mae",
        metric=_mean_absolute_error,
    )

    assert all(isinstance(result, MetricsResult) for result in results)
    assert [(r.subject_id, r.region_id) for r in results] == [
        ("subject_a", "eye"),
        ("subject_a", "mouth"),
        ("subject_b", "mouth"),
    ]
    assert [r.n_valid for r in results] == [1, 2, 1]
    assert [r.value for r in results] == pytest.approx([1.5, 1.25, 0.5])

    for result in results:
        assert result.outer_fold == 2
        assert result.condition == "pcmci"
        assert result.metric_name == "fixture_mae"


def test_metric_interface_fails_closed_when_no_valid_prediction_rows_exist() -> None:
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
        evaluate_metric_by_subject_region(
            artifact,
            metric_name="fixture_mae",
            metric=_mean_absolute_error,
        )
