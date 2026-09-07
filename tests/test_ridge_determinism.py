from __future__ import annotations

import numpy as np

from lagged_facial_graph_forecasting.contracts import DesignMatrix, InnerFold, SplitManifest
from lagged_facial_graph_forecasting.forecaster import predict_ridge_forecaster
from lagged_facial_graph_forecasting.ridge_coefficients import export_ridge_coefficients
from lagged_facial_graph_forecasting.ridge_refit import refit_frozen_ridge
from lagged_facial_graph_forecasting.ridge_tuning import (
    aggregate_ridge_inner_cv_scores,
    evaluate_ridge_inner_cv,
    select_ridge_alpha,
)


def _manifest() -> SplitManifest:
    return SplitManifest(
        outer_fold=0,
        train_subject_ids=("s01", "s02", "s03", "s04"),
        test_subject_ids=("s05",),
        inner_folds=(
            InnerFold(
                inner_train_subject_ids=("s01", "s02"),
                inner_val_subject_ids=("s03", "s04"),
            ),
            InnerFold(
                inner_train_subject_ids=("s03", "s04"),
                inner_val_subject_ids=("s01", "s02"),
            ),
        ),
        seed=123,
    )


def _matrix() -> DesignMatrix:
    subjects = ("s01", "s01", "s02", "s02", "s03", "s03", "s04", "s04")
    x = np.array(
        [
            [0.0, 1.0],
            [1.0, 1.5],
            [2.0, 0.5],
            [3.0, 1.0],
            [4.0, 2.0],
            [5.0, 2.5],
            [6.0, 1.5],
            [7.0, 2.0],
        ],
        dtype=float,
    )
    y = (1.5 * x[:, 0] - 0.75 * x[:, 1] + 0.25).reshape(-1, 1)
    n_rows = x.shape[0]
    return DesignMatrix(
        X=x,
        y=y,
        subject_id=subjects,
        region_id=("mouth",) * n_rows,
        forecast_origin=np.arange(n_rows, dtype=float),
        target_time=np.arange(1, n_rows + 1, dtype=float),
        feature_names=("left_cheek.vx", "jaw.vx"),
        feature_lags=(2, 1),
        valid_mask=np.ones(n_rows, dtype=bool),
    )


def _run_once():
    matrix = _matrix()
    manifest = _manifest()
    fold_scores = evaluate_ridge_inner_cv(
        matrix,
        split_manifest=manifest,
        alpha_grid=(0.1, 1.0, 10.0),
    )
    aggregates = aggregate_ridge_inner_cv_scores(fold_scores)
    selected = select_ridge_alpha(aggregates)
    fitted = refit_frozen_ridge(
        matrix,
        split_manifest=manifest,
        selected=selected,
    )
    prediction = predict_ridge_forecaster(
        fitted,
        matrix,
        outer_fold=manifest.outer_fold,
        condition="self",
    )
    coefficients = export_ridge_coefficients(fitted)
    return fold_scores, aggregates, selected, fitted, prediction, coefficients


def test_primary_ridge_path_is_deterministic_for_identical_inputs() -> None:
    first = _run_once()
    second = _run_once()

    first_scores, first_aggregates, first_selected, first_fit, first_prediction, first_coef = first
    second_scores, second_aggregates, second_selected, second_fit, second_prediction, second_coef = second

    assert first_scores == second_scores
    assert first_aggregates == second_aggregates
    assert first_selected == second_selected
    assert first_fit.alpha == second_fit.alpha
    assert first_fit.training_row_count == second_fit.training_row_count
    assert np.array_equal(
        first_fit.pipeline.named_steps["scaler"].mean_,
        second_fit.pipeline.named_steps["scaler"].mean_,
    )
    assert np.array_equal(
        first_fit.pipeline.named_steps["scaler"].scale_,
        second_fit.pipeline.named_steps["scaler"].scale_,
    )
    assert np.array_equal(first_prediction.y_pred, second_prediction.y_pred)
    assert np.array_equal(first_coef.coefficient, second_coef.coefficient)
    assert np.array_equal(first_coef.intercept, second_coef.intercept)
