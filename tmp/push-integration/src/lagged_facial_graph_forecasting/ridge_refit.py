"""Compatibility entry point for the canonical frozen outer-train Ridge refit."""

from __future__ import annotations

from .contracts import DesignMatrix, SplitManifest
from .forecaster import FittedRidgeForecaster
from .ridge_tuning import RidgeAlphaScore, refit_selected_ridge


def refit_frozen_ridge(
    matrix: DesignMatrix,
    *,
    split_manifest: SplitManifest,
    selected: RidgeAlphaScore,
) -> FittedRidgeForecaster:
    """Delegate E-06 refit to the single canonical implementation.

    ``ridge_tuning.refit_selected_ridge`` owns all scope checks, selected-alpha
    validation, and the final StandardScaler + Ridge fit.  This wrapper retains the
    task-oriented import path without duplicating fitting or leakage logic.
    """

    return refit_selected_ridge(
        matrix,
        split_manifest=split_manifest,
        selection=selected,
    )
