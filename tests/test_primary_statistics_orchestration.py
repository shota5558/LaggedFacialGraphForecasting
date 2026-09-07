from __future__ import annotations

import numpy as np

from lagged_facial_graph_forecasting.contracts import PredictionArtifact
from lagged_facial_graph_forecasting.primary_statistics import (
    PrimaryPairedStatistics,
    compute_primary_velocity_paired_statistics,
)


def _prediction(*, fold: int, subject: str, condition: str, error: float) -> PredictionArtifact:
    return PredictionArtifact(
        outer_fold=fold,
        subject_id=(subject,),
        region_id=("mouth",),
        target_dimensions=("vx", "vy"),
        condition=condition,
        forecast_origin=np.asarray([10.0 + fold]),
        target_time=np.asarray([11.0 + fold]),
        y_true=np.asarray([[0.0, 0.0]]),
        y_pred=np.asarray([[error, error]]),
        valid_mask=np.asarray([True]),
    )


def _predictions(condition: str, errors: tuple[float, ...]) -> tuple[PredictionArtifact, ...]:
    return tuple(
        _prediction(fold=fold, subject=f"subject_{fold}", condition=condition, error=error)
        for fold, error in enumerate(errors)
    )


def test_primary_statistics_composes_g00_through_g04_with_frozen_bootstrap() -> None:
    result = compute_primary_velocity_paired_statistics(
        _predictions("self", (2.0, 3.0, 4.0)),
        _predictions("pcmci", (1.0, 1.0, 2.0)),
        experiment_seed=20260908,
    )

    assert isinstance(result, PrimaryPairedStatistics)
    assert [row.subject_id for row in result.paired_differences] == [
        "subject_0",
        "subject_1",
        "subject_2",
    ]
    assert [row.difference for row in result.paired_differences] == [1.0, 2.0, 2.0]
    assert len(result.median_differences) == 1
    assert result.median_differences[0].median_difference == 2.0

    assert len(result.bootstrap_cis) == 1
    ci = result.bootstrap_cis[0]
    assert ci.confidence_level == 0.95
    assert ci.n_resamples == 10000
    assert ci.method == "percentile"
    assert ci.seed == 20260908
    assert ci.subject_ids == ("subject_0", "subject_1", "subject_2")
    assert ci.outer_folds == (0, 1, 2)


def test_primary_statistics_is_reproducible_for_same_experiment_seed() -> None:
    reference = _predictions("self", (2.0, 3.0, 4.0, 5.0))
    comparison = _predictions("pcmci", (1.0, 1.5, 2.0, 2.5))

    first = compute_primary_velocity_paired_statistics(
        reference,
        comparison,
        experiment_seed=77,
    )
    second = compute_primary_velocity_paired_statistics(
        reference,
        comparison,
        experiment_seed=77,
    )

    assert first == second
