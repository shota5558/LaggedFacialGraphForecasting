"""Ridge hyperparameter-grid and inner-CV utilities for Primary tuning."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .contracts import DesignMatrix, SplitManifest
from .forecaster import fit_ridge_forecaster, predict_ridge_forecaster

# Scientific tuning candidate set used by the Primary Ridge path.  The values are
# explicit rather than generated at runtime so configuration/artifact provenance is
# stable across NumPy versions and platforms.  scikit-learn calls Ridge's lambda
# parameter ``alpha``.
DEFAULT_RIDGE_ALPHA_GRID: tuple[float, ...] = (
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    1e1,
    1e2,
    1e3,
    1e4,
)


@dataclass(frozen=True, slots=True)
class RidgeInnerFoldScore:
    """One alpha/fold validation score produced entirely inside outer-train."""

    alpha: float
    inner_fold: int
    rmse: float
    n_train_valid: int
    n_val_valid: int


def normalize_ridge_alpha_grid(
    values: Iterable[float] = DEFAULT_RIDGE_ALPHA_GRID,
) -> tuple[float, ...]:
    """Validate and deterministically normalize a Ridge lambda/alpha candidate grid.

    Candidates must be finite, strictly positive, and unique.  The returned tuple
    is sorted in ascending numeric order so downstream tie-breaking and artifact
    serialization do not depend on caller iteration order.
    """

    normalized: list[float] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("Ridge alpha grid values must be numeric, not boolean")
        try:
            alpha = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Ridge alpha grid values must be numeric") from exc
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Ridge alpha grid values must be finite and > 0")
        normalized.append(alpha)

    if not normalized:
        raise ValueError("Ridge alpha grid must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Ridge alpha grid values must be unique")

    return tuple(sorted(normalized))


def _slice_design_matrix(matrix: DesignMatrix, row_mask: np.ndarray) -> DesignMatrix:
    row_mask = np.asarray(row_mask, dtype=bool)
    if row_mask.shape != (matrix.X.shape[0],):
        raise ValueError("row_mask must match DesignMatrix row count")
    if not np.any(row_mask):
        raise ValueError("inner fold contains no DesignMatrix rows")
    indices = np.flatnonzero(row_mask)
    return DesignMatrix(
        X=matrix.X[indices],
        y=matrix.y[indices],
        subject_id=tuple(matrix.subject_id[index] for index in indices),
        region_id=tuple(matrix.region_id[index] for index in indices),
        forecast_origin=matrix.forecast_origin[indices],
        target_time=matrix.target_time[indices],
        feature_names=matrix.feature_names,
        feature_lags=matrix.feature_lags,
        valid_mask=matrix.valid_mask[indices],
    )


def evaluate_ridge_inner_cv(
    matrix: DesignMatrix,
    *,
    split_manifest: SplitManifest,
    alpha_grid: Iterable[float] = DEFAULT_RIDGE_ALPHA_GRID,
) -> tuple[RidgeInnerFoldScore, ...]:
    """Evaluate Ridge candidates with subject-isolated inner folds.

    The input matrix must contain outer-train subjects only.  For every inner fold
    and alpha, ``fit_ridge_forecaster`` receives only rows belonging to that fold's
    ``inner_train_subject_ids``; StandardScaler and Ridge are therefore fitted
    together on inner-train.  Validation rows are only transformed/predicted after
    the fitted pipeline is frozen for that fold.
    """

    alphas = normalize_ridge_alpha_grid(alpha_grid)
    matrix_subjects = set(matrix.subject_id)
    outer_train_subjects = set(split_manifest.train_subject_ids)
    outer_test_subjects = set(split_manifest.test_subject_ids)

    leaked_test_subjects = matrix_subjects & outer_test_subjects
    if leaked_test_subjects:
        raise ValueError(
            "Ridge tuning matrix contains outer-test subjects: "
            f"{sorted(leaked_test_subjects)}"
        )
    unexpected_subjects = matrix_subjects - outer_train_subjects
    if unexpected_subjects:
        raise ValueError(
            "Ridge tuning matrix contains subjects outside outer-train: "
            f"{sorted(unexpected_subjects)}"
        )

    subject_array = np.asarray(matrix.subject_id, dtype=object)
    scores: list[RidgeInnerFoldScore] = []

    for alpha in alphas:
        for fold_index, inner_fold in enumerate(split_manifest.inner_folds):
            train_mask = np.isin(
                subject_array, np.asarray(inner_fold.inner_train_subject_ids, dtype=object)
            )
            val_mask = np.isin(
                subject_array, np.asarray(inner_fold.inner_val_subject_ids, dtype=object)
            )
            train_matrix = _slice_design_matrix(matrix, train_mask)
            val_matrix = _slice_design_matrix(matrix, val_mask)

            fitted = fit_ridge_forecaster(train_matrix, alpha=alpha)
            prediction = predict_ridge_forecaster(
                fitted,
                val_matrix,
                outer_fold=split_manifest.outer_fold,
                condition="ridge_inner_cv",
            )
            valid = np.asarray(prediction.valid_mask, dtype=bool)
            n_val_valid = int(np.count_nonzero(valid))
            if n_val_valid == 0:
                raise ValueError("inner validation fold contains no valid rows")
            residual = prediction.y_pred[valid] - prediction.y_true[valid]
            rmse = float(np.sqrt(np.mean(np.square(residual))))
            if not np.isfinite(rmse):
                raise RuntimeError("inner validation RMSE must be finite")

            scores.append(
                RidgeInnerFoldScore(
                    alpha=alpha,
                    inner_fold=fold_index,
                    rmse=rmse,
                    n_train_valid=fitted.training_row_count,
                    n_val_valid=n_val_valid,
                )
            )

    return tuple(scores)
