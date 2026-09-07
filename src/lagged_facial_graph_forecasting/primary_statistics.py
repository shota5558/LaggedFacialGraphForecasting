"""Primary statistical orchestration from frozen prediction artifacts.

This module is the production composition boundary for Issue #15 G-00..G-04.  It
keeps the individual metric/pairing/bootstrap primitives independently testable,
while providing one auditable call path from held-out ``PredictionArtifact``
objects to subject-level paired differences, median aggregation, and the frozen
95% bootstrap confidence interval.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .bootstrap_statistics import BootstrapMedianCI, bootstrap_median_paired_difference_ci
from .contracts import PredictionArtifact
from .core_contracts import MetricsResult
from .metrics import velocity_rmse_by_subject_region
from .scientific_config import load_scientific_config
from .statistics import (
    MedianPairedDifference,
    PairedMetricDifference,
    median_paired_difference,
    subject_level_paired_difference,
)


@dataclass(frozen=True, slots=True)
class PrimaryPairedStatistics:
    """Complete Primary paired-statistic evidence for one condition comparison."""

    reference_metrics: tuple[MetricsResult, ...]
    comparison_metrics: tuple[MetricsResult, ...]
    paired_differences: tuple[PairedMetricDifference, ...]
    median_differences: tuple[MedianPairedDifference, ...]
    bootstrap_cis: tuple[BootstrapMedianCI, ...]


def _velocity_metrics(
    predictions: Sequence[PredictionArtifact],
    *,
    label: str,
) -> tuple[MetricsResult, ...]:
    normalized = tuple(predictions)
    if not normalized:
        raise ValueError(f"{label} predictions must not be empty")
    if any(not isinstance(artifact, PredictionArtifact) for artifact in normalized):
        raise TypeError(f"{label} predictions must contain PredictionArtifact instances")

    conditions = {artifact.condition for artifact in normalized}
    if len(conditions) != 1:
        raise ValueError(f"{label} predictions must contain exactly one condition")

    seen_folds: set[int] = set()
    results: list[MetricsResult] = []
    for artifact in sorted(normalized, key=lambda item: item.outer_fold):
        if artifact.outer_fold in seen_folds:
            raise ValueError(
                f"{label} predictions contain duplicate outer_fold={artifact.outer_fold}"
            )
        seen_folds.add(artifact.outer_fold)
        results.extend(velocity_rmse_by_subject_region(artifact))
    return tuple(results)


def compute_primary_velocity_paired_statistics(
    reference_predictions: Sequence[PredictionArtifact],
    comparison_predictions: Sequence[PredictionArtifact],
    *,
    experiment_seed: int,
    scientific_config_path: str | Path = Path("configs/scientific_freeze.yaml"),
) -> PrimaryPairedStatistics:
    """Compute the frozen Primary Velocity-RMSE paired statistic end to end.

    The function consumes only already-frozen held-out predictions.  It performs no
    preprocessing fitting, discovery, feature/lag selection, Ridge tuning, or Null
    construction.  Bootstrap parameters are read from the strict Scientific Freeze
    and passed explicitly to the low-level SciPy-backed implementation.

    For ``reference_condition='self'`` and ``comparison_condition='pcmci'``, each
    paired difference is the preregistered Incremental Gain
    ``Error_Self - Error_PCMCI``.
    """

    if not isinstance(experiment_seed, int) or isinstance(experiment_seed, bool):
        raise ValueError("experiment_seed must be an integer")
    if experiment_seed < 0:
        raise ValueError("experiment_seed must be non-negative")

    scientific_config = load_scientific_config(scientific_config_path)
    statistics_config = scientific_config["primary"]["statistics"]
    if statistics_config["paired_unit"] != "subject":
        raise ValueError("Primary paired_unit must remain 'subject'")
    if statistics_config["point_aggregation"] != "median":
        raise ValueError("Primary point_aggregation must remain 'median'")

    bootstrap_config = statistics_config["bootstrap"]
    if bootstrap_config["seed_source"] != "experiment_seed":
        raise ValueError("Primary bootstrap seed_source must remain 'experiment_seed'")

    reference_metrics = _velocity_metrics(reference_predictions, label="reference")
    comparison_metrics = _velocity_metrics(comparison_predictions, label="comparison")
    paired = subject_level_paired_difference(reference_metrics, comparison_metrics)
    medians = median_paired_difference(paired)
    cis = bootstrap_median_paired_difference_ci(
        paired,
        n_resamples=int(bootstrap_config["n_resamples"]),
        seed=experiment_seed,
        method=str(bootstrap_config["method"]),
    )

    return PrimaryPairedStatistics(
        reference_metrics=reference_metrics,
        comparison_metrics=comparison_metrics,
        paired_differences=paired,
        median_differences=medians,
        bootstrap_cis=cis,
    )
