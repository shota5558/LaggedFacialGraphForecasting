"""Primary subject-level paired statistics.

Pairing, median aggregation, and bootstrap inference are kept as separate stages so
subject-level observations remain inspectable and reproducible.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .core_contracts import MetricsResult


@dataclass(frozen=True, slots=True)
class PairedMetricDifference:
    """One matched subject-region error difference.

    ``difference`` is always ``reference_value - comparison_value``. Therefore,
    for the Primary Self-vs-PCMCI error comparison, a positive value is exactly the
    preregistered Incremental Gain ``Error_Self - Error_PCMCI``.
    """

    outer_fold: int
    subject_id: str
    region_id: str
    metric_name: str
    reference_condition: str
    comparison_condition: str
    reference_value: float
    comparison_value: float
    difference: float
    n_valid: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.outer_fold, int)
            or isinstance(self.outer_fold, bool)
            or self.outer_fold < 0
        ):
            raise ValueError("outer_fold must be a non-negative integer")
        for name in (
            "subject_id",
            "region_id",
            "metric_name",
            "reference_condition",
            "comparison_condition",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.reference_condition == self.comparison_condition:
            raise ValueError("paired conditions must be different")
        for name in ("reference_value", "comparison_value", "difference"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if (
            not isinstance(self.n_valid, int)
            or isinstance(self.n_valid, bool)
            or self.n_valid < 1
        ):
            raise ValueError("n_valid must be a positive integer")


@dataclass(frozen=True, slots=True)
class MedianPairedDifference:
    """Unweighted median of subject-level paired differences for one region."""

    region_id: str
    metric_name: str
    reference_condition: str
    comparison_condition: str
    median_difference: float
    n_subjects: int
    subject_ids: tuple[str, ...]
    outer_folds: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in (
            "region_id",
            "metric_name",
            "reference_condition",
            "comparison_condition",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.reference_condition == self.comparison_condition:
            raise ValueError("paired conditions must be different")
        median_difference = float(self.median_difference)
        if not np.isfinite(median_difference):
            raise ValueError("median_difference must be finite")
        object.__setattr__(self, "median_difference", median_difference)
        if (
            not isinstance(self.n_subjects, int)
            or isinstance(self.n_subjects, bool)
            or self.n_subjects < 1
        ):
            raise ValueError("n_subjects must be a positive integer")
        subject_ids = tuple(self.subject_ids)
        outer_folds = tuple(self.outer_folds)
        if len(subject_ids) != self.n_subjects or len(outer_folds) != self.n_subjects:
            raise ValueError("subject_ids and outer_folds must match n_subjects")
        if any(not isinstance(subject, str) or not subject.strip() for subject in subject_ids):
            raise ValueError("subject_ids entries must be non-empty strings")
        if len(set(subject_ids)) != len(subject_ids):
            raise ValueError("subject_ids must be unique within an aggregation")
        if any(
            not isinstance(fold, int) or isinstance(fold, bool) or fold < 0
            for fold in outer_folds
        ):
            raise ValueError("outer_folds entries must be non-negative integers")
        object.__setattr__(self, "subject_ids", subject_ids)
        object.__setattr__(self, "outer_folds", outer_folds)


def _index_metric_results(
    results: Sequence[MetricsResult],
    *,
    label: str,
) -> tuple[str, dict[tuple[int, str, str, str], MetricsResult]]:
    normalized = tuple(results)
    if not normalized:
        raise ValueError(f"{label} metric results must not be empty")
    if any(not isinstance(result, MetricsResult) for result in normalized):
        raise TypeError(f"{label} entries must be MetricsResult instances")

    conditions = {result.condition for result in normalized}
    if len(conditions) != 1:
        raise ValueError(f"{label} metric results must contain exactly one condition")
    condition = next(iter(conditions))

    indexed: dict[tuple[int, str, str, str], MetricsResult] = {}
    for result in normalized:
        key = (
            result.outer_fold,
            result.subject_id,
            result.region_id,
            result.metric_name,
        )
        if key in indexed:
            raise ValueError(f"duplicate {label} metric result for paired key {key!r}")
        indexed[key] = result
    return condition, indexed


def subject_level_paired_difference(
    reference_results: Sequence[MetricsResult],
    comparison_results: Sequence[MetricsResult],
) -> tuple[PairedMetricDifference, ...]:
    """Pair metric results exactly and return ``reference - comparison`` values.

    Pairing is fail-closed on ``outer_fold × subject × region × metric`` support.
    The two conditions must also have the same ``n_valid`` for every pair so a
    performance difference cannot silently compare different evaluation supports.
    """

    reference_condition, reference = _index_metric_results(
        reference_results, label="reference"
    )
    comparison_condition, comparison = _index_metric_results(
        comparison_results, label="comparison"
    )
    if reference_condition == comparison_condition:
        raise ValueError("reference and comparison conditions must be different")

    reference_keys = set(reference)
    comparison_keys = set(comparison)
    if reference_keys != comparison_keys:
        missing_in_comparison = sorted(reference_keys - comparison_keys)
        missing_in_reference = sorted(comparison_keys - reference_keys)
        raise ValueError(
            "paired metric supports differ; "
            f"missing_in_comparison={missing_in_comparison}, "
            f"missing_in_reference={missing_in_reference}"
        )

    paired: list[PairedMetricDifference] = []
    for key in sorted(reference_keys):
        reference_result = reference[key]
        comparison_result = comparison[key]
        if reference_result.n_valid != comparison_result.n_valid:
            raise ValueError(
                "paired metric n_valid differs for "
                f"{key!r}: reference={reference_result.n_valid}, "
                f"comparison={comparison_result.n_valid}"
            )
        difference = reference_result.value - comparison_result.value
        paired.append(
            PairedMetricDifference(
                outer_fold=reference_result.outer_fold,
                subject_id=reference_result.subject_id,
                region_id=reference_result.region_id,
                metric_name=reference_result.metric_name,
                reference_condition=reference_condition,
                comparison_condition=comparison_condition,
                reference_value=reference_result.value,
                comparison_value=comparison_result.value,
                difference=difference,
                n_valid=reference_result.n_valid,
            )
        )

    return tuple(paired)


def median_paired_difference(
    paired_results: Sequence[PairedMetricDifference],
) -> tuple[MedianPairedDifference, ...]:
    """Aggregate paired differences by region/metric/condition using the median.

    Each subject contributes exactly one unweighted paired difference per group.
    The returned subject/fold tuples preserve which held-out units support each
    median; duplicate subject contributions are rejected rather than averaged.
    """

    normalized = tuple(paired_results)
    if not normalized:
        raise ValueError("paired_results must not be empty")
    if any(not isinstance(result, PairedMetricDifference) for result in normalized):
        raise TypeError("paired_results entries must be PairedMetricDifference instances")

    grouped: dict[
        tuple[str, str, str, str], list[PairedMetricDifference]
    ] = defaultdict(list)
    for result in normalized:
        key = (
            result.region_id,
            result.metric_name,
            result.reference_condition,
            result.comparison_condition,
        )
        grouped[key].append(result)

    summaries: list[MedianPairedDifference] = []
    for key in sorted(grouped):
        rows = sorted(
            grouped[key],
            key=lambda result: (result.subject_id, result.outer_fold),
        )
        subjects = tuple(result.subject_id for result in rows)
        if len(set(subjects)) != len(subjects):
            raise ValueError(
                "each subject must contribute exactly one paired difference per "
                f"aggregation group; group={key!r}"
            )
        folds = tuple(result.outer_fold for result in rows)
        differences = np.asarray([result.difference for result in rows], dtype=float)
        summaries.append(
            MedianPairedDifference(
                region_id=key[0],
                metric_name=key[1],
                reference_condition=key[2],
                comparison_condition=key[3],
                median_difference=float(np.median(differences)),
                n_subjects=len(rows),
                subject_ids=subjects,
                outer_folds=folds,
            )
        )

    return tuple(summaries)
