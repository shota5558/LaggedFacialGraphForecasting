"""Ridge hyperparameter-grid utilities for Primary inner-CV tuning."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

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
