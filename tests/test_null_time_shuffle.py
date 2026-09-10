from __future__ import annotations

import pytest

from lagged_facial_graph_forecasting.contracts import InnerFold, SplitManifest
from lagged_facial_graph_forecasting.core_contracts import ParentLink, ParentSet
from lagged_facial_graph_forecasting.leakage_guard import LeakageGuardError
from lagged_facial_graph_forecasting.null_time_shuffle import (
    TimeShuffleMappingError,
    construct_time_shuffle_mapping,
)


def _manifest() -> SplitManifest:
    return SplitManifest(
        outer_fold=3,
        train_subject_ids=("train_a", "train_b"),
        test_subject_ids=("test_a",),
        inner_folds=(InnerFold(("train_a",), ("train_b",)),),
        seed=404,
    )


def _parent_set() -> ParentSet:
    return ParentSet(
        outer_fold=3,
        target_region="mouth",
        parents=(
            ParentLink("jaw", 2, "vx", "vx"),
            ParentLink("eye", 5, "vy", "vx"),
        ),
        discovery_method="pcmci_plus",
    )


def test_time_shuffle_freezes_rule_without_changing_parent_identity() -> None:
    mapping = construct_time_shuffle_mapping(
        _manifest(),
        _parent_set(),
        construction_subject_ids=("train_a", "train_b"),
    )

    assert mapping.condition == "time-shuffle"
    assert mapping.seed == _manifest().seed
    assert mapping.outer_fold == _manifest().outer_fold
    assert mapping.target_region == _parent_set().target_region
    assert mapping.source_parents == _parent_set().parents
    assert mapping.mapped_parents == _parent_set().parents
    assert mapping.permutation == ()


def test_time_shuffle_accepts_empty_parentset_without_inventing_features() -> None:
    empty = ParentSet(
        outer_fold=3,
        target_region="mouth",
        parents=(),
        discovery_method="pcmci_plus",
    )
    mapping = construct_time_shuffle_mapping(
        _manifest(),
        empty,
        construction_subject_ids=("train_a", "train_b"),
    )
    assert mapping.source_parents == ()
    assert mapping.mapped_parents == ()
    assert mapping.permutation == ()


def test_time_shuffle_rejects_outer_test_null_construction_scope() -> None:
    with pytest.raises(LeakageGuardError, match="outer-test"):
        construct_time_shuffle_mapping(
            _manifest(),
            _parent_set(),
            construction_subject_ids=("train_a", "test_a"),
        )


def test_time_shuffle_rejects_parentset_fold_mismatch() -> None:
    wrong_fold = ParentSet(
        outer_fold=4,
        target_region="mouth",
        parents=_parent_set().parents,
        discovery_method="pcmci_plus",
    )
    with pytest.raises(TimeShuffleMappingError, match="outer_fold"):
        construct_time_shuffle_mapping(
            _manifest(),
            wrong_fold,
            construction_subject_ids=("train_a", "train_b"),
        )
