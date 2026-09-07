from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting import FaceTimeSeries, ParentLink, ParentSet
from lagged_facial_graph_forecasting.design_matrix import (
    build_full_history_design_matrix,
    build_pcmci_parent_design_matrix,
    build_persistence_design_matrix,
    build_self_history_design_matrix,
)


def _series() -> FaceTimeSeries:
    X = np.arange(8 * 3 * 2, dtype=float).reshape(8, 3, 2)
    return FaceTimeSeries(
        X=X,
        subject_id="s01",
        time_index=np.arange(8, dtype=float),
        region_id=("mouth", "jaw", "eye"),
        dimension=("vx", "vy"),
        valid_mask=np.ones_like(X, dtype=bool),
        sampling_rate=30.0,
    )


def test_primary_conditions_can_share_exact_frozen_row_support() -> None:
    series = _series()
    alignment_lag = 3

    persistence = build_persistence_design_matrix(
        series,
        target_region="mouth",
        alignment_lag=alignment_lag,
    )
    self_history = build_self_history_design_matrix(
        series,
        target_region="mouth",
        lags=(1, 2),
        alignment_lag=alignment_lag,
    )
    full_history = build_full_history_design_matrix(
        series,
        target_region="mouth",
        lags=(1, 2, 3),
        alignment_lag=alignment_lag,
    )
    pcmci = build_pcmci_parent_design_matrix(
        series,
        parent_set=ParentSet(
            outer_fold=0,
            target_region="mouth",
            parents=(ParentLink(source_region="jaw", lag=3),),
        ),
        self_lags=(1, 2),
        alignment_lag=alignment_lag,
    )

    matrices = (persistence, self_history, full_history, pcmci)
    for matrix in matrices[1:]:
        assert np.array_equal(matrix.forecast_origin, persistence.forecast_origin)
        assert np.array_equal(matrix.target_time, persistence.target_time)
        assert np.array_equal(matrix.y, persistence.y)
        assert matrix.subject_id == persistence.subject_id
        assert matrix.region_id == persistence.region_id

    assert np.array_equal(persistence.target_time, np.arange(3, 8, dtype=float))
    assert persistence.feature_lags == (1, 1)
    assert self_history.feature_lags == (1, 1, 2, 2)
    assert max(pcmci.feature_lags) == 3


def test_alignment_lag_does_not_add_a_predictor() -> None:
    matrix = build_self_history_design_matrix(
        _series(),
        target_region="mouth",
        lags=(1,),
        alignment_lag=3,
    )

    assert matrix.feature_names == ("mouth.vx", "mouth.vy")
    assert matrix.feature_lags == (1, 1)
    assert matrix.X.shape[0] == 5


@pytest.mark.parametrize(
    "builder",
    [
        lambda series: build_self_history_design_matrix(
            series, target_region="mouth", lags=(1, 3), alignment_lag=2
        ),
        lambda series: build_full_history_design_matrix(
            series, target_region="mouth", lags=(1, 3), alignment_lag=2
        ),
        lambda series: build_pcmci_parent_design_matrix(
            series,
            parent_set=ParentSet(
                outer_fold=0,
                target_region="mouth",
                parents=(ParentLink(source_region="jaw", lag=3),),
            ),
            self_lags=(1,),
            alignment_lag=2,
        ),
    ],
)
def test_alignment_lag_cannot_be_smaller_than_a_used_feature_lag(builder) -> None:
    with pytest.raises(ValueError, match="alignment_lag"):
        builder(_series())


def test_alignment_lag_rejects_non_integer_values() -> None:
    with pytest.raises(ValueError, match="alignment_lag"):
        build_persistence_design_matrix(
            _series(), target_region="mouth", alignment_lag=2.5  # type: ignore[arg-type]
        )
