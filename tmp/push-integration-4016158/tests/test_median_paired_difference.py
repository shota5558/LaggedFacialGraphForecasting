from __future__ import annotations

import pytest

from lagged_facial_graph_forecasting.statistics import (
    MedianPairedDifference,
    PairedMetricDifference,
    median_paired_difference,
)


def _paired(
    *,
    fold: int,
    subject: str,
    region: str,
    difference: float,
    metric_name: str = "velocity_rmse",
    reference: str = "self",
    comparison: str = "pcmci",
) -> PairedMetricDifference:
    comparison_value = 1.0
    return PairedMetricDifference(
        outer_fold=fold,
        subject_id=subject,
        region_id=region,
        metric_name=metric_name,
        reference_condition=reference,
        comparison_condition=comparison,
        reference_value=comparison_value + difference,
        comparison_value=comparison_value,
        difference=difference,
        n_valid=10,
    )


def test_median_aggregation_uses_one_unweighted_value_per_subject() -> None:
    summaries = median_paired_difference(
        (
            _paired(fold=2, subject="s3", region="mouth", difference=0.8),
            _paired(fold=0, subject="s1", region="mouth", difference=0.1),
            _paired(fold=1, subject="s2", region="mouth", difference=0.4),
        )
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert isinstance(summary, MedianPairedDifference)
    assert summary.region_id == "mouth"
    assert summary.metric_name == "velocity_rmse"
    assert summary.reference_condition == "self"
    assert summary.comparison_condition == "pcmci"
    assert summary.median_difference == pytest.approx(0.4)
    assert summary.n_subjects == 3
    assert summary.subject_ids == ("s1", "s2", "s3")
    assert summary.outer_folds == (0, 1, 2)


def test_median_aggregation_keeps_regions_and_comparisons_separate() -> None:
    summaries = median_paired_difference(
        (
            _paired(fold=0, subject="s1", region="eye", difference=-0.2),
            _paired(fold=1, subject="s2", region="eye", difference=0.2),
            _paired(fold=0, subject="s1", region="mouth", difference=0.3),
            _paired(
                fold=0,
                subject="s1",
                region="mouth",
                difference=0.1,
                comparison="full",
            ),
        )
    )

    assert [
        (s.region_id, s.reference_condition, s.comparison_condition)
        for s in summaries
    ] == [
        ("eye", "self", "pcmci"),
        ("mouth", "self", "full"),
        ("mouth", "self", "pcmci"),
    ]
    assert [s.median_difference for s in summaries] == pytest.approx([0.0, 0.1, 0.3])


def test_median_aggregation_handles_even_subject_count_by_standard_median() -> None:
    summary = median_paired_difference(
        (
            _paired(fold=0, subject="s1", region="mouth", difference=0.0),
            _paired(fold=1, subject="s2", region="mouth", difference=1.0),
            _paired(fold=2, subject="s3", region="mouth", difference=2.0),
            _paired(fold=3, subject="s4", region="mouth", difference=7.0),
        )
    )[0]

    assert summary.median_difference == pytest.approx(1.5)


def test_median_aggregation_rejects_duplicate_subject_contributions() -> None:
    with pytest.raises(ValueError, match="each subject must contribute exactly one"):
        median_paired_difference(
            (
                _paired(fold=0, subject="s1", region="mouth", difference=0.1),
                _paired(fold=1, subject="s1", region="mouth", difference=0.2),
            )
        )


def test_median_aggregation_rejects_empty_or_nonpaired_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        median_paired_difference(())

    with pytest.raises(TypeError, match="PairedMetricDifference"):
        median_paired_difference((object(),))  # type: ignore[arg-type]
