"""Frozen outer-train Ridge refit after inner-CV lambda selection."""

from __future__ import annotations

from .contracts import DesignMatrix, SplitManifest
from .forecaster import FittedRidgeForecaster, fit_ridge_forecaster
from .ridge_tuning import RidgeAlphaScore, select_ridge_alpha


def refit_frozen_ridge(
    matrix: DesignMatrix,
    *,
    split_manifest: SplitManifest,
    selected: RidgeAlphaScore,
) -> FittedRidgeForecaster:
    """Refit scaler + Ridge once on complete outer-train using the frozen alpha.

    E-05 owns model selection. This function only validates that the supplied
    selection metadata is well-formed, verifies the refit matrix contains exactly
    the outer-train subject set, and delegates the single final fit to the common
    scikit-learn StandardScaler + Ridge pipeline.
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

    # Validation only: a singleton cannot change the E-05 choice.  This rejects
    # malformed/non-finite selection metadata before consuming the frozen alpha.
    validated_selection = select_ridge_alpha((selected,))
    return fit_ridge_forecaster(matrix, alpha=validated_selection.alpha)
