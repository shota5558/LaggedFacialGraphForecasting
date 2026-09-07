from __future__ import annotations

import pytest

from lagged_facial_graph_forecasting.contracts import InnerFold, SplitManifest
from lagged_facial_graph_forecasting.core_contracts import ParentLink, ParentSet
from lagged_facial_graph_forecasting.leakage_guard import LeakageGuardError
from lagged_facial_graph_forecasting.null_lag_shift import (
    LagShiftMappingError,
    construct_lag_shift_mapping,
)


def _manifest() -> SplitManifest:
    return SplitManifest(
        outer_fold=1,
        train_subject_ids=("train_a", "train_b"),
        test_subject_ids=("test_a",),
        inner_folds=(InnerFold(("train_a",), ("train_b",)),),
        seed=73,
    )


def _parents() -> ParentSet:
    return ParentSet(
        outer_fold=1,
        target_region="mouth",
        parents=(
            ParentLink("left_cheek", 2, "vx", "vy"),
            ParentLink("jaw", 4, "vy", "vx"),
        ),
        discovery_method="pcmci_plus",
    )


def test_lag_shift_changes_only_lag_and_preserves_feature_count_and_components() -> None:
    mapping = construct_lag_shift_mapping(
        _manifest(),
        _parents(),
        construction_subject_ids=("train_a", "train_b"),
        lag_delta=2,
    )

    assert mapping.condition == "lag-shift"
    assert mapping.seed == 73
    assert mapping.source_parents == _parents().parents
    assert mapping.mapped_parents == (
        ParentLink("left_cheek", 4, "vx", "vy"),
        ParentLink("jaw", 6, "vy", "vx"),
    )
    assert len(mapping.source_parents) == len(mapping.mapped_parents)
    for source, mapped in zip(mapping.source_parents, mapping.mapped_parents, strict=True):
        assert mapped.source_region == source.source_region
        assert mapped.source_dimension == source.source_dimension
        assert mapped.target_dimension == source.target_dimension
        assert mapped.lag - source.lag == 2


def test_lag_shift_rejects_outer_test_scope_before_mapping() -> None:
    with pytest.raises(LeakageGuardError, match="outer-test"):
        construct_lag_shift_mapping(
            _manifest(),
            _parents(),
            construction_subject_ids=("train_a", "test_a"),
            lag_delta=1,
        )


def test_lag_shift_rejects_zero_delta_and_fold_mismatch() -> None:
    with pytest.raises(LagShiftMappingError, match="non-zero"):
        construct_lag_shift_mapping(
            _manifest(),
            _parents(),
            construction_subject_ids=("train_a", "train_b"),
            lag_delta=0,
        )

    wrong_fold = ParentSet(
        outer_fold=0,
        target_region="mouth",
        parents=_parents().parents,
        discovery_method="pcmci_plus",
    )
    with pytest.raises(LagShiftMappingError, match="outer_fold"):
        construct_lag_shift_mapping(
            _manifest(),
            wrong_fold,
            construction_subject_ids=("train_a", "train_b"),
            lag_delta=1,
        )
