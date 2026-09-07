"""Primary subject-level paired statistics.

This module starts with the G-02 pairing boundary only.  Aggregation and bootstrap
inference are deliberately separate later tasks so the subject-level observations
remain inspectable and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from .core_contracts import MetricsResult


@dataclass(frozen=True, slots=True)
class PairedMetricDifference:
    """One matched subject-region error difference.

    ``difference`` is always ``reference_value - comparison_value``.  Therefore,
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
        if not isinstance(self.outer_fold, int) or isinstance(self.outer_fold, bool) or self.outer_fold < 0:
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
        if not isinstance(self.n_valid, int) or isinstance(self.n_valid, bool) or self.n_valid < 1:
            raise ValueError("n_valid must be a positive integer")


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
