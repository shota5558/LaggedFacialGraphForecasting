"""Deterministic Ridge lambda selection from outer-train inner-CV aggregates."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .ridge_tuning import RidgeAlphaScore


def select_ridge_alpha(
    aggregates: Iterable[RidgeAlphaScore],
) -> RidgeAlphaScore:
    """Select the minimum mean-RMSE candidate without consulting outer-test data.

    Exact score ties are resolved by the smaller alpha.  The tie-break is fixed here
    so selection is deterministic and cannot be changed after observing outer-test
    performance.  The full selected aggregate is returned for audit provenance.
    """

    rows = tuple(aggregates)
    if not rows:
        raise ValueError("Ridge alpha aggregates must not be empty")

    seen: set[float] = set()
    validated: list[RidgeAlphaScore] = []
    for row in rows:
        alpha = float(row.alpha)
        mean_rmse = float(row.mean_rmse)
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Ridge alpha must be finite and > 0")
        if alpha in seen:
            raise ValueError(f"duplicate Ridge alpha aggregate: {alpha}")
        if not np.isfinite(mean_rmse) or mean_rmse < 0:
            raise ValueError("Ridge mean_rmse must be finite and >= 0")
        if not isinstance(row.fold_count, int) or isinstance(row.fold_count, bool):
            raise ValueError("Ridge fold_count must be an integer")
        if row.fold_count <= 0:
            raise ValueError("Ridge fold_count must be > 0")
        if not isinstance(row.total_val_valid, int) or isinstance(
            row.total_val_valid, bool
        ):
            raise ValueError("Ridge total_val_valid must be an integer")
        if row.total_val_valid <= 0:
            raise ValueError("Ridge total_val_valid must be > 0")
        seen.add(alpha)
        validated.append(row)

    return min(validated, key=lambda row: (float(row.mean_rmse), float(row.alpha)))
