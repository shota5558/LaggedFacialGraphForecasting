from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting import FaceTimeSeries, ParentLink, ParentSet
from lagged_facial_graph_forecasting.design_matrix import (
    build_full_history_design_matrix,
    build_pcmci_parent_design_matrix,
    build_self_history_design_matrix,
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
            # A discovered self link must not replace or duplicate the fixed Self block.
            ParentLink(source_region="mouth", lag=1),
        ),
    )


def _feature_keys(matrix) -> tuple[tuple[str, int], ...]:
    return tuple(zip(matrix.feature_names, matrix.feature_lags, strict=True))


def test_pcmci_matrix_is_fixed_self_plus_interregional_parent_links() -> None:
    matrix = build_pcmci_parent_design_matrix(
        _series(), parent_set=_parent_set(), self_lags=(1,), horizon=1
    )

    assert np.array_equal(
        matrix.X,
        np.array(
            [
                [11, 21, 100, 200],
                [12, 22, 101, 201],
                [13, 23, 102, 202],
                [14, 24, 103, 203],
            ],
            dtype=float,
        ),
    )
    assert np.array_equal(
        matrix.y,
        np.array([[12, 22], [13, 23], [14, 24], [15, 25]], dtype=float),
    )
    assert matrix.feature_names == ("mouth.vx", "mouth.vy", "jaw.vx", "jaw.vy")
    assert matrix.feature_lags == (1, 1, 2, 2)
    assert np.array_equal(matrix.forecast_origin, np.array([1, 2, 3, 4], dtype=float))
    assert np.array_equal(matrix.target_time, np.array([2, 3, 4, 5], dtype=float))
    assert matrix.region_id == ("mouth",) * 4


def test_self_is_subset_of_pcmci_and_pcmci_is_subset_of_full() -> None:
    series = _series()
    self_matrix = build_self_history_design_matrix(
        series, target_region="mouth", lags=(1, 2)
    )
    pcmci_matrix = build_pcmci_parent_design_matrix(
        series,
        parent_set=ParentSet(
            outer_fold=0,
            target_region="mouth",
            parents=(ParentLink(source_region="jaw", lag=2),),
        ),
        self_lags=(1, 2),
    )
    full_matrix = build_full_history_design_matrix(
        series, target_region="mouth", lags=(1, 2)
    )

    self_keys = set(_feature_keys(self_matrix))
    pcmci_keys = set(_feature_keys(pcmci_matrix))
    full_keys = set(_feature_keys(full_matrix))
    assert self_keys < pcmci_keys
    assert pcmci_keys <= full_keys


def test_pcmci_uses_exact_same_self_feature_block_as_self_condition() -> None:
    series = _series()
    self_matrix = build_self_history_design_matrix(
        series, target_region="mouth", lags=(1, 2)
    )
    pcmci_matrix = build_pcmci_parent_design_matrix(
        series,
        parent_set=ParentSet(
            outer_fold=0,
            target_region="mouth",
            parents=(ParentLink(source_region="eye", lag=2),),
        ),
        self_lags=(1, 2),
    )

    n_self_features = self_matrix.X.shape[1]
    assert pcmci_matrix.feature_names[:n_self_features] == self_matrix.feature_names
    assert pcmci_matrix.feature_lags[:n_self_features] == self_matrix.feature_lags
    assert np.array_equal(pcmci_matrix.X[:, :n_self_features], self_matrix.X)
    assert np.array_equal(pcmci_matrix.y, self_matrix.y)
    assert np.array_equal(pcmci_matrix.forecast_origin, self_matrix.forecast_origin)
    assert np.array_equal(pcmci_matrix.target_time, self_matrix.target_time)


def test_empty_parent_set_deterministically_reduces_pcmci_to_self() -> None:
    series = _series()
    parent_set = ParentSet(outer_fold=0, target_region="mouth", parents=())
    self_matrix = build_self_history_design_matrix(
        series, target_region="mouth", lags=(1, 2)
    )
    pcmci_matrix = build_pcmci_parent_design_matrix(
        series, parent_set=parent_set, self_lags=(1, 2)
    )

    assert np.array_equal(pcmci_matrix.X, self_matrix.X)
    assert np.array_equal(pcmci_matrix.y, self_matrix.y)
    assert pcmci_matrix.feature_names == self_matrix.feature_names
    assert pcmci_matrix.feature_lags == self_matrix.feature_lags
    assert np.array_equal(pcmci_matrix.valid_mask, self_matrix.valid_mask)


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


def test_pcmci_parent_matrix_rejects_non_primary_horizon() -> None:
    with pytest.raises(ValueError, match="frozen to h=1"):
        build_pcmci_parent_design_matrix(
            _series(), parent_set=_parent_set(), horizon=2
        )
