from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting import FaceTimeSeries, ParentLink, ParentSet
from lagged_facial_graph_forecasting.design_matrix import (
    build_pcmci_parent_design_matrix,
)


def _series() -> FaceTimeSeries:
    X = np.zeros((6, 3, 2), dtype=float)
    X[:, 0, 0] = np.array([10, 11, 12, 13, 14, 15], dtype=float)
    X[:, 0, 1] = np.array([20, 21, 22, 23, 24, 25], dtype=float)
    X[:, 1, 0] = np.array([100, 101, 102, 103, 104, 105], dtype=float)
    X[:, 1, 1] = np.array([200, 201, 202, 203, 204, 205], dtype=float)
    X[:, 2, 0] = np.array([300, 301, 302, 303, 304, 305], dtype=float)
    X[:, 2, 1] = np.array([400, 401, 402, 403, 404, 405], dtype=float)
    return FaceTimeSeries(
        X=X,
        subject_id="s01",
        time_index=np.arange(6, dtype=float),
        region_id=("mouth", "jaw", "eye"),
        dimension=("vx", "vy"),
        valid_mask=np.ones_like(X, dtype=bool),
        sampling_rate=30.0,
    )


def _parent_set() -> ParentSet:
    return ParentSet(
        outer_fold=0,
        target_region="mouth",
        parents=(
            ParentLink(source_region="jaw", lag=2),
            ParentLink(source_region="mouth", lag=1),
        ),
    )


def test_pcmci_parent_matrix_uses_only_parent_set_links_in_parent_order() -> None:
    matrix = build_pcmci_parent_design_matrix(
        _series(), parent_set=_parent_set(), horizon=1
    )

    assert np.array_equal(
        matrix.X,
        np.array(
            [
                [100, 200, 11, 21],
                [101, 201, 12, 22],
                [102, 202, 13, 23],
                [103, 203, 14, 24],
            ],
            dtype=float,
        ),
    )
    assert np.array_equal(
        matrix.y,
        np.array(
            [[12, 22], [13, 23], [14, 24], [15, 25]], dtype=float
        ),
    )
    assert matrix.feature_names == ("jaw.vx", "jaw.vy", "mouth.vx", "mouth.vy")
    assert matrix.feature_lags == (2, 2, 1, 1)
    assert np.array_equal(matrix.forecast_origin, np.array([1, 2, 3, 4], dtype=float))
    assert np.array_equal(matrix.target_time, np.array([2, 3, 4, 5], dtype=float))
    assert matrix.region_id == ("mouth",) * 4


def test_pcmci_parent_matrix_propagates_parent_and_target_missingness() -> None:
    series = _series()
    mask = series.valid_mask.copy()
    mask[0, 1, 0] = False
    mask[4, 0, 1] = False
    masked = FaceTimeSeries(
        X=series.X,
        subject_id=series.subject_id,
        time_index=series.time_index,
        region_id=series.region_id,
        dimension=series.dimension,
        valid_mask=mask,
        sampling_rate=series.sampling_rate,
    )

    matrix = build_pcmci_parent_design_matrix(masked, parent_set=_parent_set())

    assert np.array_equal(matrix.valid_mask, np.array([False, True, False, False]))


def test_pcmci_parent_matrix_rejects_unknown_source_region() -> None:
    parent_set = ParentSet(
        outer_fold=0,
        target_region="mouth",
        parents=(ParentLink(source_region="cheek", lag=1),),
    )

    with pytest.raises(ValueError, match="unknown source_region"):
        build_pcmci_parent_design_matrix(_series(), parent_set=parent_set)


def test_pcmci_parent_matrix_rejects_empty_parent_set() -> None:
    parent_set = ParentSet(outer_fold=0, target_region="mouth", parents=())

    with pytest.raises(ValueError, match="at least one parent"):
        build_pcmci_parent_design_matrix(_series(), parent_set=parent_set)


def test_pcmci_parent_matrix_rejects_non_primary_horizon() -> None:
    with pytest.raises(ValueError, match="frozen to h=1"):
        build_pcmci_parent_design_matrix(
            _series(), parent_set=_parent_set(), horizon=2
        )
