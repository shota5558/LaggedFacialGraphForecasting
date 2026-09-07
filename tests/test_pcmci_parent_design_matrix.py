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
            ParentLink("jaw", 2, "vx", "vx"),
            ParentLink("eye", 2, "vy", "vy"),
            # Self-region discovery is already covered by the frozen Self block.
            ParentLink("mouth", 1, "vy", "vx"),
        ),
    )


def _feature_keys(matrix) -> tuple[tuple[str, int], ...]:
    return tuple(zip(matrix.feature_names, matrix.feature_lags, strict=True))


def test_pcmci_matrix_uses_exact_selected_source_component_for_target_component() -> None:
    matrix = build_pcmci_parent_design_matrix(
        _series(),
        parent_set=_parent_set(),
        self_lags=(1,),
        horizon=1,
        target_dimension="vx",
    )

    assert np.array_equal(
        matrix.X,
        np.array(
            [
                [11, 21, 100],
                [12, 22, 101],
                [13, 23, 102],
                [14, 24, 103],
            ],
            dtype=float,
        ),
    )
    assert np.array_equal(matrix.y, np.array([12, 13, 14, 15], dtype=float))
    assert matrix.feature_names == ("mouth.vx", "mouth.vy", "jaw.vx")
    assert matrix.feature_lags == (1, 1, 2)
    assert np.array_equal(matrix.forecast_origin, np.array([1, 2, 3, 4], dtype=float))
    assert np.array_equal(matrix.target_time, np.array([2, 3, 4, 5], dtype=float))


def test_target_component_filters_parent_links_without_cross_target_union() -> None:
    vx = build_pcmci_parent_design_matrix(
        _series(), parent_set=_parent_set(), target_dimension="vx"
    )
    vy = build_pcmci_parent_design_matrix(
        _series(), parent_set=_parent_set(), target_dimension="vy"
    )

    assert "jaw.vx" in vx.feature_names
    assert "eye.vy" not in vx.feature_names
    assert "eye.vy" in vy.feature_names
    assert "jaw.vx" not in vy.feature_names
    assert vx.y.ndim == 1
    assert vy.y.ndim == 1


def test_mixed_target_parentset_requires_explicit_target_dimension() -> None:
    with pytest.raises(ValueError, match="target_dimension is required"):
        build_pcmci_parent_design_matrix(_series(), parent_set=_parent_set())


def test_self_pcmci_full_share_same_scalar_target_and_self_feature_block() -> None:
    series = _series()
    self_matrix = build_self_history_design_matrix(
        series,
        target_region="mouth",
        target_dimension="vx",
        lags=(1, 2),
    )
    pcmci_matrix = build_pcmci_parent_design_matrix(
        series,
        parent_set=ParentSet(
            outer_fold=0,
            target_region="mouth",
            parents=(ParentLink("jaw", 2, "vy", "vx"),),
        ),
        target_dimension="vx",
        self_lags=(1, 2),
    )
    full_matrix = build_full_history_design_matrix(
        series,
        target_region="mouth",
        target_dimension="vx",
        lags=(1, 2),
    )

    n_self_features = self_matrix.X.shape[1]
    assert pcmci_matrix.feature_names[:n_self_features] == self_matrix.feature_names
    assert pcmci_matrix.feature_lags[:n_self_features] == self_matrix.feature_lags
    assert np.array_equal(pcmci_matrix.X[:, :n_self_features], self_matrix.X)
    assert np.array_equal(self_matrix.y, pcmci_matrix.y)
    assert np.array_equal(full_matrix.y, pcmci_matrix.y)

    self_keys = set(_feature_keys(self_matrix))
    pcmci_keys = set(_feature_keys(pcmci_matrix))
    full_keys = set(_feature_keys(full_matrix))
    assert self_keys < pcmci_keys
    assert pcmci_keys <= full_keys


def test_empty_parent_set_requires_component_for_d2_and_reduces_to_component_self() -> None:
    series = _series()
    parent_set = ParentSet(outer_fold=0, target_region="mouth", parents=())
    with pytest.raises(ValueError, match="target_dimension is required"):
        build_pcmci_parent_design_matrix(series, parent_set=parent_set, self_lags=(1, 2))

    self_matrix = build_self_history_design_matrix(
        series,
        target_region="mouth",
        target_dimension="vy",
        lags=(1, 2),
    )
    pcmci_matrix = build_pcmci_parent_design_matrix(
        series,
        parent_set=parent_set,
        target_dimension="vy",
        self_lags=(1, 2),
    )
    assert np.array_equal(pcmci_matrix.X, self_matrix.X)
    assert np.array_equal(pcmci_matrix.y, self_matrix.y)
    assert pcmci_matrix.feature_names == self_matrix.feature_names
    assert pcmci_matrix.feature_lags == self_matrix.feature_lags
    assert np.array_equal(pcmci_matrix.valid_mask, self_matrix.valid_mask)


def test_pcmci_missingness_uses_exact_selected_source_and_target_component() -> None:
    series = _series()
    mask = series.valid_mask.copy()
    # First vx-target row uses jaw.vx at source index 0.
    mask[0, 1, 0] = False
    # Last vx-target row does not depend on mouth.vy target validity.
    mask[5, 0, 1] = False
    masked = FaceTimeSeries(
        X=series.X,
        subject_id=series.subject_id,
        time_index=series.time_index,
        region_id=series.region_id,
        dimension=series.dimension,
        valid_mask=mask,
        sampling_rate=series.sampling_rate,
    )

    matrix = build_pcmci_parent_design_matrix(
        masked,
        parent_set=ParentSet(
            outer_fold=0,
            target_region="mouth",
            parents=(ParentLink("jaw", 2, "vx", "vx"),),
        ),
        target_dimension="vx",
    )
    assert np.array_equal(matrix.valid_mask, np.array([False, True, True, True]))


def test_pcmci_parent_matrix_rejects_ambiguous_legacy_component_on_d2() -> None:
    parent_set = ParentSet(
        outer_fold=0,
        target_region="mouth",
        parents=(ParentLink(source_region="jaw", lag=1),),
    )
    with pytest.raises(ValueError, match="ambiguous"):
        build_pcmci_parent_design_matrix(
            _series(), parent_set=parent_set, target_dimension="vx"
        )


def test_pcmci_parent_matrix_rejects_unknown_source_component() -> None:
    with pytest.raises(ValueError, match="unknown source_region"):
        build_pcmci_parent_design_matrix(
            _series(),
            parent_set=ParentSet(
                outer_fold=0,
                target_region="mouth",
                parents=(ParentLink("cheek", 1, "vx", "vx"),),
            ),
            target_dimension="vx",
        )

    with pytest.raises(ValueError, match="unknown source_dimension"):
        build_pcmci_parent_design_matrix(
            _series(),
            parent_set=ParentSet(
                outer_fold=0,
                target_region="mouth",
                parents=(ParentLink("jaw", 1, "vz", "vx"),),
            ),
            target_dimension="vx",
        )


def test_pcmci_parent_matrix_rejects_non_primary_horizon() -> None:
    with pytest.raises(ValueError, match="frozen to h=1"):
        build_pcmci_parent_design_matrix(
            _series(),
            parent_set=_parent_set(),
            horizon=2,
            target_dimension="vx",
        )
