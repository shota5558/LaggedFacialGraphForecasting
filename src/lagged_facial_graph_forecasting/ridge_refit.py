"""Frozen outer-train Ridge refit after inner-CV lambda selection."""

from __future__ import annotations

from .contracts import DesignMatrix, SplitManifest
from .forecaster import FittedRidgeForecaster, fit_ridge_forecaster
from .ridge_tuning import RidgeAlphaScore


def refit_frozen_ridge(
    matrix: DesignMatrix,
    *,
    split_manifest: SplitManifest,
    selected: RidgeAlphaScore,
) -> FittedRidgeForecaster:
    """Refit scaler + Ridge on the complete outer-train matrix using frozen alpha.

    This function is valid only after E-05 selection.  It accepts no outer-test rows
    and performs no tuning or selection; ``selected.alpha`` is used verbatim.
    """

    matrix_subjects = set(matrix.subject_id)
    outer_train_subjects = set(split_manifest.train_subject_ids)
    outer_test_subjects = set(split_manifest.test_subject_ids)

    leaked = matrix_subjects & outer_test_subjects
    if leaked:
        raise ValueError(
            "frozen Ridge refit matrix contains outer-test subjects: "
            f"{sorted(leaked)}"
        )
    unexpected = matrix_subjects - outer_train_subjects
    if unexpected:
        raise ValueError(
            "frozen Ridge refit matrix contains subjects outside outer-train: "
            f"{sorted(unexpected)}"
        )
    missing = outer_train_subjects - matrix_subjects
    if missing:
        raise ValueError(
            "frozen Ridge refit matrix is missing outer-train subjects: "
            f"{sorted(missing)}"
        )

    return fit_ridge_forecaster(matrix, alpha=float(selected.alpha))
