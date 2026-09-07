from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting.contracts import DesignMatrix, InnerFold, SplitManifest
from lagged_facial_graph_forecasting.ridge_refit import refit_frozen_ridge
from lagged_facial_graph_forecasting.ridge_tuning import RidgeAlphaScore


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


def _matrix(subjects: tuple[str, ...]) -> DesignMatrix:
    values = np.arange(len(subjects), dtype=float)
    return DesignMatrix(
        X=values.reshape(-1, 1),
        y=(2.0 * values + 1.0).reshape(-1, 1),
        subject_id=subjects,
        region_id=("mouth",) * len(subjects),
        forecast_origin=np.arange(len(subjects), dtype=float),
        target_time=np.arange(1, len(subjects) + 1, dtype=float),
        feature_names=("mouth.vx",),
        feature_lags=(1,),
        valid_mask=np.ones(len(subjects), dtype=bool),
    )


def _selected(alpha: float = 0.1) -> RidgeAlphaScore:
    return RidgeAlphaScore(alpha=alpha, mean_rmse=0.5, fold_count=2, total_val_valid=4)


def test_frozen_refit_uses_selected_alpha_and_all_outer_train_rows() -> None:
    fitted = refit_frozen_ridge(
        _matrix(("s01", "s02", "s03", "s04")),
        split_manifest=_manifest(),
        selected=_selected(0.1),
    )

    assert fitted.alpha == 0.1
    assert fitted.training_row_count == 4
    assert fitted.pipeline.named_steps["ridge"].alpha == 0.1
    assert np.allclose(fitted.pipeline.named_steps["scaler"].mean_, np.array([1.5]))


def test_frozen_refit_rejects_outer_test_subject() -> None:
    with pytest.raises(ValueError, match="outer-test"):
        refit_frozen_ridge(
            _matrix(("s01", "s02", "s03", "s04", "s05")),
            split_manifest=_manifest(),
            selected=_selected(),
        )


def test_frozen_refit_requires_complete_outer_train_subject_set() -> None:
    with pytest.raises(ValueError, match="missing outer-train"):
        refit_frozen_ridge(
            _matrix(("s01", "s02", "s03")),
            split_manifest=_manifest(),
            selected=_selected(),
        )
