from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting.ridge_tuning import RidgeAlphaScore, select_ridge_alpha


def _aggregate(alpha: float, rmse: float) -> RidgeAlphaScore:
    return RidgeAlphaScore(
        alpha=alpha,
        mean_rmse=rmse,
        fold_count=2,
        total_val_valid=20,
    )


def test_selects_minimum_mean_rmse_independent_of_input_order() -> None:
    candidates = (
        _aggregate(10.0, 2.0),
        _aggregate(0.1, 3.0),
        _aggregate(1.0, 1.0),
    )

    assert select_ridge_alpha(candidates) == _aggregate(1.0, 1.0)
    assert select_ridge_alpha(reversed(candidates)) == _aggregate(1.0, 1.0)


def test_exact_rmse_tie_chooses_smallest_alpha() -> None:
    selected = select_ridge_alpha(
        (_aggregate(10.0, 1.0), _aggregate(0.1, 1.0), _aggregate(1.0, 1.0))
    )

    assert selected.alpha == 0.1


def test_rejects_duplicate_alpha_aggregates() -> None:
    with pytest.raises(ValueError, match="duplicate Ridge aggregate alpha"):
        select_ridge_alpha((_aggregate(1.0, 1.0), _aggregate(1.0, 2.0)))


@pytest.mark.parametrize(
    "candidate",
    [
        RidgeAlphaScore(alpha=0.0, mean_rmse=1.0, fold_count=2, total_val_valid=20),
        RidgeAlphaScore(alpha=np.inf, mean_rmse=1.0, fold_count=2, total_val_valid=20),
        RidgeAlphaScore(alpha=1.0, mean_rmse=np.nan, fold_count=2, total_val_valid=20),
        RidgeAlphaScore(alpha=1.0, mean_rmse=-1.0, fold_count=2, total_val_valid=20),
        RidgeAlphaScore(alpha=1.0, mean_rmse=1.0, fold_count=0, total_val_valid=20),
        RidgeAlphaScore(alpha=1.0, mean_rmse=1.0, fold_count=2, total_val_valid=0),
    ],
)
def test_rejects_invalid_aggregate(candidate: RidgeAlphaScore) -> None:
    with pytest.raises(ValueError):
        select_ridge_alpha((candidate,))


def test_rejects_empty_aggregate_set() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        select_ridge_alpha(())
