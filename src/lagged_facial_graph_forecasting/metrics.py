"""Primary metric interfaces and primitives.

Metrics consume only frozen ``PredictionArtifact`` values.  The generic G-00
interface groups valid prediction rows by held-out subject and target region and
returns the versioned ``MetricsResult`` contract required by downstream paired
statistics.  Metric-specific numerical definitions remain separate tasks.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .contracts import PredictionArtifact
from .core_contracts import MetricsResult


MetricFunction = Callable[[np.ndarray, np.ndarray], float]
VELOCITY_RMSE_NAME = "velocity_rmse"
DISPLACEMENT_EUCLIDEAN_NAME = "displacement_mean_euclidean_error"


def displacement_mean_euclidean_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean pointwise 2D distance on an already matched (time, point, xy) support.

    Geometric normalization must precede this calculation. Missing points/rows
    must be resolved by the common-support policy, never silently skipped here.
    This is neither coordinate RMSE nor the norm of a region centroid.
    """
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if truth.shape != prediction.shape or truth.ndim != 3 or truth.shape[-1] != 2:
        raise ValueError("displacement arrays must share shape (time, point, 2)")
    if truth.size == 0 or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("displacement support must be nonempty and finite")
    return float(np.linalg.norm(prediction - truth, axis=-1).mean())


def evaluate_metric_by_subject_region(
    artifact: PredictionArtifact,
    *,
    metric_name: str,
    metric: MetricFunction,
) -> tuple[MetricsResult, ...]:
    """Evaluate one metric on each valid subject-region support in ``artifact``.

    The interface deliberately preserves the outer-fold, subject, region, and
    condition provenance needed for subject-level paired comparisons.  Invalid rows
    are excluded before grouping and never become metric inputs.  No discovery,
    tuning, feature selection, or Null construction is performed here.
    """

    if not callable(metric):
        raise TypeError("metric must be callable")
    if not isinstance(metric_name, str) or not metric_name.strip():
        raise ValueError("metric_name must be a non-empty string")

    valid_rows = np.asarray(artifact.valid_mask, dtype=bool)
    valid_indices = np.flatnonzero(valid_rows)
    if valid_indices.size == 0:
        raise ValueError("metric evaluation requires at least one valid prediction row")

    groups = sorted(
        {
            (artifact.subject_id[int(index)], artifact.region_id[int(index)])
            for index in valid_indices
        }
    )

    results: list[MetricsResult] = []
    subject_array = np.asarray(artifact.subject_id, dtype=object)
    region_array = np.asarray(artifact.region_id, dtype=object)

    for subject_id, region_id in groups:
        group_mask = (
            valid_rows
            & (subject_array == subject_id)
            & (region_array == region_id)
        )
        n_valid = int(np.count_nonzero(group_mask))
        value = float(metric(artifact.y_true[group_mask], artifact.y_pred[group_mask]))
        results.append(
            MetricsResult(
                outer_fold=artifact.outer_fold,
                subject_id=subject_id,
                region_id=region_id,
                condition=artifact.condition,
                metric_name=metric_name.strip(),
                value=value,
                n_valid=n_valid,
            )
        )

    return tuple(results)


def _root_mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sqrt(np.mean(np.square(residual))))


def velocity_rmse(artifact: PredictionArtifact) -> float:
    """Compute RMSE over all valid prediction rows and velocity dimensions.

    This V0-compatible scalar helper is retained unchanged in semantics.  Primary
    statistics should use :func:`velocity_rmse_by_subject_region` so subject-level
    pairing provenance is not lost.
    """

    valid_rows = np.asarray(artifact.valid_mask, dtype=bool)
    if not np.any(valid_rows):
        raise ValueError("Velocity RMSE requires at least one valid prediction row")

    return _root_mean_square_error(
        artifact.y_true[valid_rows],
        artifact.y_pred[valid_rows],
    )


def velocity_rmse_by_subject_region(
    artifact: PredictionArtifact,
) -> tuple[MetricsResult, ...]:
    """Return subject-region Velocity RMSE results with paired-test provenance."""

    return evaluate_metric_by_subject_region(
        artifact,
        metric_name=VELOCITY_RMSE_NAME,
        metric=_root_mean_square_error,
    )
