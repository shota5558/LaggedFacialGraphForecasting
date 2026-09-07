"""Bootstrap inference for Primary subject-level paired statistics.

Generic resampling is delegated to ``scipy.stats.bootstrap``.  The scientific
choices that are not yet frozen (resample count and interval method) are mandatory
arguments rather than hidden software defaults.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import bootstrap as scipy_bootstrap

from .statistics import (
    PairedMetricDifference,
    median_paired_difference,
)


PRIMARY_BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
_BOOTSTRAP_METHODS = {"percentile", "basic", "BCa"}


@dataclass(frozen=True, slots=True)
class BootstrapMedianCI:
    """95% bootstrap CI for one region-level median paired difference."""

    region_id: str
    metric_name: str
    reference_condition: str
    comparison_condition: str
    median_difference: float
    ci_low: float
    ci_high: float
    confidence_level: float
    n_subjects: int
    n_resamples: int
    seed: int
    method: str
    subject_ids: tuple[str, ...]
    outer_folds: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in (
            "region_id",
            "metric_name",
            "reference_condition",
            "comparison_condition",
            "method",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.reference_condition == self.comparison_condition:
            raise ValueError("paired conditions must be different")
        for name in ("median_difference", "ci_low", "ci_high", "confidence_level"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.confidence_level != PRIMARY_BOOTSTRAP_CONFIDENCE_LEVEL:
            raise ValueError("Primary bootstrap confidence_level must be 0.95")
        if self.ci_low > self.ci_high:
            raise ValueError("ci_low must not exceed ci_high")
        for name in ("n_subjects", "n_resamples"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.method not in _BOOTSTRAP_METHODS:
            raise ValueError(f"method must be one of {sorted(_BOOTSTRAP_METHODS)!r}")
        subject_ids = tuple(self.subject_ids)
        outer_folds = tuple(self.outer_folds)
        if len(subject_ids) != self.n_subjects or len(outer_folds) != self.n_subjects:
            raise ValueError("subject_ids and outer_folds must match n_subjects")
        object.__setattr__(self, "subject_ids", subject_ids)
        object.__setattr__(self, "outer_folds", outer_folds)


def bootstrap_median_paired_difference_ci(
    paired_results: Sequence[PairedMetricDifference],
    *,
    n_resamples: int,
    seed: int,
    method: str,
) -> tuple[BootstrapMedianCI, ...]:
    """Compute reproducible 95% CIs for subject-level median differences.

    ``n_resamples`` and ``method`` are intentionally required because the current
    Scientific Freeze specifies a 95% bootstrap CI but does not yet select those
    two implementation-sensitive analysis settings.

    ``random_state`` is used rather than SciPy's newer ``rng`` keyword because the
    project declares SciPy >=1.10; ``random_state`` accepts a NumPy Generator in
    that supported range while ``rng`` was introduced only in SciPy 1.15.
    """

    if not isinstance(n_resamples, int) or isinstance(n_resamples, bool) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if method not in _BOOTSTRAP_METHODS:
        raise ValueError(f"method must be one of {sorted(_BOOTSTRAP_METHODS)!r}")

    normalized = tuple(paired_results)
    summaries = median_paired_difference(normalized)

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

    rng = np.random.default_rng(seed)
    results: list[BootstrapMedianCI] = []
    for summary in summaries:
        key = (
            summary.region_id,
            summary.metric_name,
            summary.reference_condition,
            summary.comparison_condition,
        )
        rows = grouped[key]
        if len(rows) < 2:
            raise ValueError(
                "bootstrap CI requires at least two held-out subjects per aggregation group"
            )
        differences = np.asarray([row.difference for row in rows], dtype=float)
        bootstrap_result = scipy_bootstrap(
            (differences,),
            np.median,
            confidence_level=PRIMARY_BOOTSTRAP_CONFIDENCE_LEVEL,
            n_resamples=n_resamples,
            method=method,
            vectorized=False,
            random_state=rng,
        )
        low = float(bootstrap_result.confidence_interval.low)
        high = float(bootstrap_result.confidence_interval.high)
        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError(
                "SciPy bootstrap returned a non-finite confidence interval; "
                "the subject-level sample may be degenerate for the requested method"
            )
        results.append(
            BootstrapMedianCI(
                region_id=summary.region_id,
                metric_name=summary.metric_name,
                reference_condition=summary.reference_condition,
                comparison_condition=summary.comparison_condition,
                median_difference=summary.median_difference,
                ci_low=low,
                ci_high=high,
                confidence_level=PRIMARY_BOOTSTRAP_CONFIDENCE_LEVEL,
                n_subjects=summary.n_subjects,
                n_resamples=n_resamples,
                seed=seed,
                method=method,
                subject_ids=summary.subject_ids,
                outer_folds=summary.outer_folds,
            )
        )

    return tuple(results)
