from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting.ridge_tuning import (
    DEFAULT_RIDGE_ALPHA_GRID,
    normalize_ridge_alpha_grid,
)


def test_default_alpha_grid_is_explicit_sorted_positive_and_unique() -> None:
    assert DEFAULT_RIDGE_ALPHA_GRID == (
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
    assert normalize_ridge_alpha_grid() == DEFAULT_RIDGE_ALPHA_GRID


def test_alpha_grid_normalization_is_order_independent() -> None:
    assert normalize_ridge_alpha_grid((10.0, 0.1, 1.0)) == (0.1, 1.0, 10.0)


@pytest.mark.parametrize(
    "grid",
    [
        (),
        (1.0, 1.0),
        (0.0, 1.0),
        (-1.0, 1.0),
        (np.inf, 1.0),
        (np.nan, 1.0),
        (True, 1.0),
    ],
)
def test_rejects_invalid_alpha_grids(grid: tuple[object, ...]) -> None:
    with pytest.raises(ValueError, match="Ridge alpha grid"):
        normalize_ridge_alpha_grid(grid)  # type: ignore[arg-type]
