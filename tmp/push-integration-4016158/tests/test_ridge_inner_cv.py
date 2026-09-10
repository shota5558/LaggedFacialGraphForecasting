from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting.contracts import DesignMatrix, InnerFold, SplitManifest
from lagged_facial_graph_forecasting import ridge_tuning


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


def _matrix(*, include_outer_test: bool = False) -> DesignMatrix:
    subjects = ["s01", "s02", "s03", "s04"]
    x = [0.0, 2.0, 100.0, 102.0]
    y = [1.0, 3.0, 101.0, 103.0]
    if include_outer_test:
        subjects.append("s05")
        x.append(10000.0)
        y.append(10001.0)

    n_rows = len(subjects)
    return DesignMatrix(
        X=np.asarray(x, dtype=float).reshape(-1, 1),
        y=np.asarray(y, dtype=float).reshape(-1, 1),
        subject_id=tuple(subjects),
        region_id=tuple("mouth" for _ in subjects),
        target_dimensions=("vx",),
        forecast_origin=np.arange(n_rows, dtype=float),
        target_time=np.arange(1, n_rows + 1, dtype=float),
        feature_names=("mouth.vx",),
        feature_lags=(1,),
        valid_mask=np.ones(n_rows, dtype=bool),
    )


def test_inner_cv_fits_scaler_and_ridge_on_inner_train_only(monkeypatch) -> None:
    observed: list[tuple[tuple[str, ...], float]] = []
    original_fit = ridge_tuning.fit_ridge_forecaster

    def spy_fit(matrix: DesignMatrix, *, alpha: float):
        fitted = original_fit(matrix, alpha=alpha)
        observed.append(
            (
                tuple(dict.fromkeys(matrix.subject_id)),
                float(fitted.pipeline.named_steps["scaler"].mean_[0]),
            )
        )
        return fitted

    monkeypatch.setattr(ridge_tuning, "fit_ridge_forecaster", spy_fit)

    scores = ridge_tuning.evaluate_ridge_inner_cv(
        _matrix(), split_manifest=_manifest(), alpha_grid=(1.0,)
    )

    assert len(scores) == 2
    assert observed == [
        (("s01", "s02"), 1.0),
        (("s03", "s04"), 101.0),
    ]
    assert all(score.n_train_valid == 2 for score in scores)
    assert all(score.n_val_valid == 2 for score in scores)
    assert all(np.isfinite(score.rmse) for score in scores)


def test_inner_cv_rejects_outer_test_subject_before_any_fit(monkeypatch) -> None:
    fit_called = False
    original_fit = ridge_tuning.fit_ridge_forecaster

    def spy_fit(matrix: DesignMatrix, *, alpha: float):
        nonlocal fit_called
        fit_called = True
        return original_fit(matrix, alpha=alpha)

    monkeypatch.setattr(ridge_tuning, "fit_ridge_forecaster", spy_fit)

    with pytest.raises(ValueError, match="outer-test"):
        ridge_tuning.evaluate_ridge_inner_cv(
            _matrix(include_outer_test=True),
            split_manifest=_manifest(),
            alpha_grid=(1.0,),
        )

    assert fit_called is False


def test_inner_cv_rejects_subject_outside_outer_train_manifest() -> None:
    matrix = _matrix()
    subjects = list(matrix.subject_id)
    subjects[0] = "unknown"
    invalid = DesignMatrix(
        X=matrix.X,
        y=matrix.y,
        subject_id=tuple(subjects),
        region_id=matrix.region_id,
        target_dimensions=matrix.target_dimensions,
        forecast_origin=matrix.forecast_origin,
        target_time=matrix.target_time,
        feature_names=matrix.feature_names,
        feature_lags=matrix.feature_lags,
        valid_mask=matrix.valid_mask,
    )

    with pytest.raises(ValueError, match="outside outer-train"):
        ridge_tuning.evaluate_ridge_inner_cv(
            invalid, split_manifest=_manifest(), alpha_grid=(1.0,)
        )
