"""Ridge hyperparameter-grid and inner-CV utilities for Primary tuning."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .contracts import DesignMatrix, SplitManifest
from .forecaster import FittedRidgeForecaster, fit_ridge_forecaster, predict_ridge_forecaster

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


@dataclass(frozen=True, slots=True)
class RidgeAlphaScore:
    """Deterministic inner-fold aggregate for one Ridge alpha candidate."""

    alpha: float
    mean_rmse: float
    fold_count: int
    total_val_valid: int


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
        target_dimensions=matrix.target_dimensions,
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


def aggregate_ridge_inner_cv_scores(
    scores: Iterable[RidgeInnerFoldScore],
) -> tuple[RidgeAlphaScore, ...]:
    """Aggregate fold RMSE by alpha using an equal-weight mean across inner folds.

    The aggregation is deliberately independent of outer-test data and does not
    select an alpha.  Each inner fold contributes one RMSE regardless of its row
    count; ``total_val_valid`` is retained as audit metadata rather than used as a
    weight.  Every alpha must contain exactly the same unique inner-fold indices so
    candidates are compared on an identical validation partition set.
    """

    score_rows = tuple(scores)
    if not score_rows:
        raise ValueError("Ridge inner-CV scores must not be empty")

    by_alpha: dict[float, list[RidgeInnerFoldScore]] = {}
    for score in score_rows:
        alpha = float(score.alpha)
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Ridge inner-CV score alpha must be finite and > 0")
        if not isinstance(score.inner_fold, int) or isinstance(score.inner_fold, bool):
            raise ValueError("inner_fold must be an integer")
        if score.inner_fold < 0:
            raise ValueError("inner_fold must be >= 0")
        if not np.isfinite(score.rmse) or score.rmse < 0:
            raise ValueError("inner-CV RMSE must be finite and >= 0")
        if score.n_val_valid <= 0:
            raise ValueError("n_val_valid must be > 0")
        by_alpha.setdefault(alpha, []).append(score)

    expected_folds: tuple[int, ...] | None = None
    aggregates: list[RidgeAlphaScore] = []
    for alpha in sorted(by_alpha):
        rows = sorted(by_alpha[alpha], key=lambda row: row.inner_fold)
        folds = tuple(row.inner_fold for row in rows)
        if len(set(folds)) != len(folds):
            raise ValueError(f"duplicate inner-fold score for alpha={alpha}")
        if expected_folds is None:
            expected_folds = folds
        elif folds != expected_folds:
            raise ValueError("all Ridge alpha candidates must cover identical inner folds")

        aggregates.append(
            RidgeAlphaScore(
                alpha=alpha,
                mean_rmse=float(np.mean([row.rmse for row in rows])),
                fold_count=len(rows),
                total_val_valid=int(sum(row.n_val_valid for row in rows)),
            )
        )

    return tuple(aggregates)


def select_ridge_alpha(
    aggregates: Iterable[RidgeAlphaScore],
) -> RidgeAlphaScore:
    """Select the minimum mean-RMSE Ridge alpha with a frozen deterministic tie rule.

    Selection uses only the E-04 inner-CV aggregate.  Exact ties in ``mean_rmse``
    are resolved by choosing the numerically smallest alpha.  This is an explicit
    deterministic rule rather than a result-dependent secondary criterion.
    """

    candidates = tuple(aggregates)
    if not candidates:
        raise ValueError("Ridge alpha aggregates must not be empty")

    seen_alpha: set[float] = set()
    validated: list[RidgeAlphaScore] = []
    for candidate in candidates:
        alpha = float(candidate.alpha)
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Ridge aggregate alpha must be finite and > 0")
        if alpha in seen_alpha:
            raise ValueError(f"duplicate Ridge aggregate alpha={alpha}")
        seen_alpha.add(alpha)
        if not np.isfinite(candidate.mean_rmse) or candidate.mean_rmse < 0:
            raise ValueError("Ridge aggregate mean_rmse must be finite and >= 0")
        if candidate.fold_count <= 0 or candidate.total_val_valid <= 0:
            raise ValueError("Ridge aggregate counts must be > 0")
        validated.append(candidate)

    return min(validated, key=lambda candidate: (candidate.mean_rmse, candidate.alpha))


def refit_selected_ridge(
    matrix: DesignMatrix,
    *,
    split_manifest: SplitManifest,
    selection: RidgeAlphaScore,
) -> FittedRidgeForecaster:
    """Refit scaler + Ridge once on complete outer-train after lambda selection.

    This is the canonical E-06 implementation.  The DesignMatrix must contain
    exactly the outer-train subject set and no outer-test/unknown subject.  The
    already-selected alpha is revalidated and then passed to the common
    StandardScaler + scikit-learn Ridge pipeline; no tuning occurs here.
    """

    matrix_subjects = set(matrix.subject_id)
    outer_train_subjects = set(split_manifest.train_subject_ids)
    outer_test_subjects = set(split_manifest.test_subject_ids)

    leaked_test_subjects = matrix_subjects & outer_test_subjects
    if leaked_test_subjects:
        raise ValueError(
            "Ridge frozen-refit matrix contains outer-test subjects: "
            f"{sorted(leaked_test_subjects)}"
        )
    unexpected_subjects = matrix_subjects - outer_train_subjects
    if unexpected_subjects:
        raise ValueError(
            "Ridge frozen-refit matrix contains subjects outside outer-train: "
            f"{sorted(unexpected_subjects)}"
        )
    missing_subjects = outer_train_subjects - matrix_subjects
    if missing_subjects:
        raise ValueError(
            "Ridge frozen-refit matrix is missing outer-train subjects: "
            f"{sorted(missing_subjects)}"
        )

    selected = select_ridge_alpha((selection,))
    return fit_ridge_forecaster(matrix, alpha=selected.alpha)
