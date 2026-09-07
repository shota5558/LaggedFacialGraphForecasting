from __future__ import annotations

import numpy as np

from lagged_facial_graph_forecasting.contracts import FaceTimeSeries
from lagged_facial_graph_forecasting.core_contracts import ParentSet
from lagged_facial_graph_forecasting.design_matrix import (
    build_pcmci_parent_design_matrix,
    build_self_history_design_matrix,
)
from lagged_facial_graph_forecasting.forecaster import (
    fit_ridge_forecaster,
    predict_ridge_forecaster,
)


def _series() -> FaceTimeSeries:
    time = np.arange(10, dtype=float)
    x = np.zeros((10, 2, 1), dtype=float)
    x[:, 0, 0] = 0.5 * time + 1.0
    x[:, 1, 0] = np.sin(time)
    return FaceTimeSeries(
        X=x,
        subject_id="s01",
        time_index=time,
        region_id=("mouth", "left_cheek"),
        dimension=("vx",),
        valid_mask=np.ones_like(x, dtype=bool),
        sampling_rate=30.0,
    )


def test_empty_pcmci_parent_set_reduces_exactly_to_self_and_ridge_runs() -> None:
    series = _series()
    parent_set = ParentSet(
        outer_fold=0,
        target_region="mouth",
        parents=(),
    )

    self_matrix = build_self_history_design_matrix(
        series,
        target_region="mouth",
        lags=(1,),
        horizon=1,
        alignment_lag=3,
    )
    pcmci_matrix = build_pcmci_parent_design_matrix(
        series,
        parent_set=parent_set,
        self_lags=(1,),
        horizon=1,
        alignment_lag=3,
    )

    assert np.array_equal(pcmci_matrix.X, self_matrix.X)
    assert np.array_equal(pcmci_matrix.y, self_matrix.y)
    assert np.array_equal(pcmci_matrix.forecast_origin, self_matrix.forecast_origin)
    assert np.array_equal(pcmci_matrix.target_time, self_matrix.target_time)
    assert np.array_equal(pcmci_matrix.valid_mask, self_matrix.valid_mask)
    assert pcmci_matrix.feature_names == self_matrix.feature_names
    assert pcmci_matrix.feature_lags == self_matrix.feature_lags

    fitted = fit_ridge_forecaster(pcmci_matrix, alpha=1.0)
    prediction = predict_ridge_forecaster(
        fitted,
        pcmci_matrix,
        outer_fold=0,
        condition="pcmci",
    )

    assert fitted.training_row_count == int(np.count_nonzero(pcmci_matrix.valid_mask))
    assert np.isfinite(prediction.y_pred[prediction.valid_mask]).all()
