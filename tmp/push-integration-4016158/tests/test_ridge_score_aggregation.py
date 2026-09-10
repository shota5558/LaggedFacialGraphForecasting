from __future__ import annotations

import pytest

from lagged_facial_graph_forecasting.ridge_tuning import (
    RidgeInnerFoldScore,
    aggregate_ridge_inner_cv_scores,
)


def _score(alpha: float, fold: int, rmse: float, n_val: int = 5) -> RidgeInnerFoldScore:
    return RidgeInnerFoldScore(
        alpha=alpha,
        inner_fold=fold,
        rmse=rmse,
        n_train_valid=10,
        n_val_valid=n_val,
    )


def test_aggregation_is_equal_weight_mean_per_fold_and_sorted_by_alpha() -> None:
    aggregates = aggregate_ridge_inner_cv_scores(
        (
            _score(1.0, 1, 4.0, n_val=100),
            _score(0.1, 0, 1.0, n_val=2),
            _score(1.0, 0, 2.0, n_val=1),
            _score(0.1, 1, 3.0, n_val=20),
        )
    )

    assert [(row.alpha, row.mean_rmse) for row in aggregates] == [
        (0.1, 2.0),
        (1.0, 3.0),
    ]
    assert [row.fold_count for row in aggregates] == [2, 2]
    assert [row.total_val_valid for row in aggregates] == [22, 101]


def test_aggregation_is_independent_of_input_order() -> None:
    forward = (
        _score(0.1, 0, 1.0),
        _score(0.1, 1, 2.0),
        _score(1.0, 0, 3.0),
        _score(1.0, 1, 4.0),
    )

    assert aggregate_ridge_inner_cv_scores(forward) == aggregate_ridge_inner_cv_scores(
        reversed(forward)
    )


def test_rejects_duplicate_alpha_fold_score() -> None:
    with pytest.raises(ValueError, match="duplicate inner-fold score"):
        aggregate_ridge_inner_cv_scores(
            (_score(1.0, 0, 1.0), _score(1.0, 0, 2.0))
        )


def test_rejects_candidates_with_different_fold_coverage() -> None:
    with pytest.raises(ValueError, match="identical inner folds"):
        aggregate_ridge_inner_cv_scores(
            (
                _score(0.1, 0, 1.0),
                _score(0.1, 1, 2.0),
                _score(1.0, 0, 3.0),
            )
        )


def test_rejects_empty_scores() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        aggregate_ridge_inner_cv_scores(())
