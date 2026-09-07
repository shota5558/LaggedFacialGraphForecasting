from __future__ import annotations

import pytest

from lagged_facial_graph_forecasting.contracts import InnerFold, SplitManifest
from lagged_facial_graph_forecasting.core_contracts import ParentLink, ParentSet
from lagged_facial_graph_forecasting.leakage_guard import LeakageGuardError
from lagged_facial_graph_forecasting.null_lag_response import (
    LAG_RESPONSE_EVALUABLE,
    LAG_RESPONSE_UNEVALUABLE_INCOMPLETE_GRID,
    LAG_RESPONSE_UNEVALUABLE_NO_PARENTS,
    PRIMARY_LAG_RESPONSE_DELTAS,
    construct_lag_response_grid,
)


def _manifest() -> SplitManifest:
    return SplitManifest(
        outer_fold=0,
        train_subject_ids=("train_a", "train_b"),
        test_subject_ids=("test_a",),
        inner_folds=(InnerFold(("train_a",), ("train_b",)),),
        seed=303,
    )


def test_complete_symmetric_grid_uses_frozen_common_shifts() -> None:
    parents = ParentSet(
        outer_fold=0,
        target_region="mouth",
        parents=(
            ParentLink("left_cheek", 3, "vx", "vy"),
            ParentLink("jaw", 7, "vy", "vy"),
        ),
        discovery_method="pcmci_plus",
    )

    grid = construct_lag_response_grid(
        _manifest(),
        parents,
        construction_subject_ids=("train_a", "train_b"),
    )

    assert grid.status == LAG_RESPONSE_EVALUABLE
    assert grid.is_evaluable is True
    assert grid.infeasible_deltas == ()
    assert tuple(point.delta for point in grid.points) == PRIMARY_LAG_RESPONSE_DELTAS
    assert PRIMARY_LAG_RESPONSE_DELTAS == (-2, -1, 0, 1, 2)

    center = next(point for point in grid.points if point.delta == 0)
    assert center.mapping is None

    for point in grid.points:
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


def test_boundary_violation_marks_entire_target_fold_unevaluable() -> None:
    parents = ParentSet(
        outer_fold=0,
        target_region="mouth",
        parents=(
            ParentLink("left_cheek", 1, "vx", "vy"),
            ParentLink("jaw", 9, "vy", "vy"),
        ),
        discovery_method="pcmci_plus",
    )

    grid = construct_lag_response_grid(
        _manifest(),
        parents,
        construction_subject_ids=("train_a", "train_b"),
    )

    assert grid.status == LAG_RESPONSE_UNEVALUABLE_INCOMPLETE_GRID
    assert grid.is_evaluable is False
    assert grid.points == ()
    assert grid.infeasible_deltas == (-2, -1, 2)


def test_boundary_policy_never_returns_one_sided_or_feature_dropped_grid() -> None:
    parents = ParentSet(
        outer_fold=0,
        target_region="mouth",
        parents=(ParentLink("left_cheek", 2, "vx", "vy"),),
        discovery_method="pcmci_plus",
    )

    grid = construct_lag_response_grid(
        _manifest(),
        parents,
        construction_subject_ids=("train_a", "train_b"),
    )

    assert grid.status == LAG_RESPONSE_UNEVALUABLE_INCOMPLETE_GRID
    assert grid.points == ()
    assert grid.infeasible_deltas == (-2,)


def test_empty_parent_set_is_unevaluable_no_parents() -> None:
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

    assert grid.status == LAG_RESPONSE_UNEVALUABLE_NO_PARENTS
    assert grid.is_evaluable is False
    assert grid.points == ()
    assert grid.infeasible_deltas == ()


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
