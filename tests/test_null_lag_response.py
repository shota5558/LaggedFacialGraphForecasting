from __future__ import annotations

import pytest

from lagged_facial_graph_forecasting.contracts import InnerFold, SplitManifest
from lagged_facial_graph_forecasting.core_contracts import ParentLink, ParentSet
from lagged_facial_graph_forecasting.leakage_guard import LeakageGuardError
from lagged_facial_graph_forecasting.null_lag_response import (
    construct_lag_response_grid,
    feasible_common_lag_deltas,
)


def _manifest() -> SplitManifest:
    return SplitManifest(
        outer_fold=0,
        train_subject_ids=("train_a", "train_b"),
        test_subject_ids=("test_a",),
        inner_folds=(InnerFold(("train_a",), ("train_b",)),),
        seed=303,
    )


def test_feasible_grid_is_exhaustive_common_shift_interval_and_contains_zero() -> None:
    parents = ParentSet(
        outer_fold=0,
        target_region="mouth",
        parents=(
            ParentLink("left_cheek", 3, "vx", "vy"),
            ParentLink("jaw", 7, "vy", "vy"),
        ),
        discovery_method="pcmci_plus",
    )

    assert feasible_common_lag_deltas(parents) == tuple(range(-2, 4))

    grid = construct_lag_response_grid(
        _manifest(),
        parents,
        construction_subject_ids=("train_a", "train_b"),
    )

    assert tuple(point.delta for point in grid) == (-2, -1, 0, 1, 2, 3)
    center = next(point for point in grid if point.delta == 0)
    assert center.mapping is None

    for point in grid:
        if point.delta == 0:
            continue
        assert point.mapping is not None
        assert point.mapping.condition == "lag-shift"
        assert len(point.mapping.mapped_parents) == len(parents.parents)
        for source, mapped in zip(
            point.mapping.source_parents,
            point.mapping.mapped_parents,
            strict=True,
        ):
            assert mapped.source_region == source.source_region
            assert mapped.source_dimension == source.source_dimension
            assert mapped.target_dimension == source.target_dimension
            assert mapped.lag == source.lag + point.delta
            assert 1 <= mapped.lag <= 10


def test_grid_is_boundary_determined_not_clipped_or_performance_selected() -> None:
    parents = ParentSet(
        outer_fold=0,
        target_region="mouth",
        parents=(
            ParentLink("left_cheek", 1, "vx", "vy"),
            ParentLink("jaw", 9, "vy", "vy"),
        ),
        discovery_method="pcmci_plus",
    )

    assert feasible_common_lag_deltas(parents) == (0, 1)


def test_empty_parent_set_has_only_unmodified_center() -> None:
    parents = ParentSet(
        outer_fold=0,
        target_region="mouth",
        parents=(),
        discovery_method="pcmci_plus",
    )

    grid = construct_lag_response_grid(
        _manifest(),
        parents,
        construction_subject_ids=("train_a", "train_b"),
    )

    assert len(grid) == 1
    assert grid[0].delta == 0
    assert grid[0].mapping is None


def test_lag_response_grid_rejects_outer_test_construction_scope() -> None:
    parents = ParentSet(
        outer_fold=0,
        target_region="mouth",
        parents=(ParentLink("left_cheek", 3, "vx", "vy"),),
        discovery_method="pcmci_plus",
    )

    with pytest.raises(LeakageGuardError, match="outer-test"):
        construct_lag_response_grid(
            _manifest(),
            parents,
            construction_subject_ids=("train_a", "test_a"),
        )
