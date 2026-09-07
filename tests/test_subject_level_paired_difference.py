from __future__ import annotations

import pytest

from lagged_facial_graph_forecasting.core_contracts import MetricsResult
from lagged_facial_graph_forecasting.statistics import (
    PairedMetricDifference,
    subject_level_paired_difference,
)


def _metric(
    *,
    fold: int,
    subject: str,
    region: str,
    condition: str,
    value: float,
    n_valid: int = 10,
    metric_name: str = "velocity_rmse",
) -> MetricsResult:
    return MetricsResult(
        outer_fold=fold,
        subject_id=subject,
        region_id=region,
        condition=condition,
        metric_name=metric_name,
        value=value,
        n_valid=n_valid,
    )


def test_subject_level_paired_difference_matches_incremental_gain_sign() -> None:
    self_results = (
        _metric(fold=1, subject="s2", region="mouth", condition="self", value=1.8),
        _metric(fold=0, subject="s1", region="eye", condition="self", value=2.0),
        _metric(fold=0, subject="s1", region="mouth", condition="self", value=1.5),
    )
    pcmci_results = (
        _metric(fold=0, subject="s1", region="mouth", condition="pcmci", value=1.1),
        _metric(fold=1, subject="s2", region="mouth", condition="pcmci", value=1.6),
        _metric(fold=0, subject="s1", region="eye", condition="pcmci", value=2.2),
    )

    paired = subject_level_paired_difference(self_results, pcmci_results)

    assert all(isinstance(result, PairedMetricDifference) for result in paired)
    assert [(p.outer_fold, p.subject_id, p.region_id) for p in paired] == [
        (0, "s1", "eye"),
        (0, "s1", "mouth"),
        (1, "s2", "mouth"),
    ]
    assert [p.difference for p in paired] == pytest.approx([-0.2, 0.4, 0.2])
    assert all(p.reference_condition == "self" for p in paired)
    assert all(p.comparison_condition == "pcmci" for p in paired)
    assert all(p.metric_name == "velocity_rmse" for p in paired)
    assert all(p.n_valid == 10 for p in paired)


def test_subject_level_paired_difference_rejects_different_evaluation_support() -> None:
    reference = (
        _metric(fold=0, subject="s1", region="mouth", condition="self", value=1.5),
    )
    comparison = (
        _metric(fold=0, subject="s2", region="mouth", condition="pcmci", value=1.2),
    )

    with pytest.raises(ValueError, match="paired metric supports differ"):
        subject_level_paired_difference(reference, comparison)


def test_subject_level_paired_difference_rejects_n_valid_mismatch() -> None:
    reference = (
        _metric(
            fold=0,
            subject="s1",
            region="mouth",
            condition="self",
            value=1.5,
            n_valid=10,
        ),
    )
    comparison = (
        _metric(
            fold=0,
            subject="s1",
            region="mouth",
            condition="pcmci",
            value=1.2,
            n_valid=9,
        ),
    )

    with pytest.raises(ValueError, match="n_valid differs"):
        subject_level_paired_difference(reference, comparison)


def test_subject_level_paired_difference_rejects_duplicate_keys() -> None:
    duplicate_reference = (
        _metric(fold=0, subject="s1", region="mouth", condition="self", value=1.5),
        _metric(fold=0, subject="s1", region="mouth", condition="self", value=1.6),
    )
    comparison = (
        _metric(fold=0, subject="s1", region="mouth", condition="pcmci", value=1.2),
    )

    with pytest.raises(ValueError, match="duplicate reference metric result"):
        subject_level_paired_difference(duplicate_reference, comparison)


def test_subject_level_paired_difference_requires_one_distinct_condition_per_side() -> None:
    mixed_reference = (
        _metric(fold=0, subject="s1", region="mouth", condition="self", value=1.5),
        _metric(fold=0, subject="s2", region="mouth", condition="full", value=1.4),
    )
    comparison = (
        _metric(fold=0, subject="s1", region="mouth", condition="pcmci", value=1.2),
        _metric(fold=0, subject="s2", region="mouth", condition="pcmci", value=1.1),
    )

    with pytest.raises(ValueError, match="exactly one condition"):
        subject_level_paired_difference(mixed_reference, comparison)

    same_condition_reference = (
        _metric(fold=0, subject="s1", region="mouth", condition="self", value=1.5),
    )
    same_condition_comparison = (
        _metric(fold=0, subject="s1", region="mouth", condition="self", value=1.2),
    )
    with pytest.raises(ValueError, match="must be different"):
        subject_level_paired_difference(
            same_condition_reference,
            same_condition_comparison,
        )
