from __future__ import annotations

import pytest

from lagged_facial_graph_forecasting.contracts import InnerFold, SplitManifest
from lagged_facial_graph_forecasting.core_contracts import ParentLink, ParentSet
from lagged_facial_graph_forecasting.leakage_guard import LeakageGuardError
from lagged_facial_graph_forecasting.null_matched_sparsity import (
    MatchedSparsityMappingError,
    construct_matched_sparsity_mapping,
)


def _manifest() -> SplitManifest:
    return SplitManifest(
        outer_fold=4,
        train_subject_ids=("train_a", "train_b"),
        test_subject_ids=("test_a",),
        inner_folds=(InnerFold(("train_a",), ("train_b",)),),
        seed=930,
    )


def _parent_set() -> ParentSet:
    return ParentSet(
        outer_fold=4,
        target_region="mouth",
        parents=(
            ParentLink("left_cheek", 2, "vx", "vy"),
            ParentLink("jaw", 6, "vy", "vx"),
        ),
        discovery_method="pcmci_plus",
    )


def test_matched_sparsity_freezes_preselected_equal_cardinality_mapping() -> None:
    sampled = (
        ParentLink("right_eye", 3, "vx", "vy"),
        ParentLink("left_brow", 7, "vy", "vx"),
    )

    mapping = construct_matched_sparsity_mapping(
        _manifest(),
        _parent_set(),
        construction_subject_ids=("train_a", "train_b"),
        sampled_parents=sampled,
    )

    assert mapping.condition == "matched-sparsity"
    assert mapping.seed == _manifest().seed
    assert mapping.source_parents == _parent_set().parents
    assert mapping.mapped_parents == sampled
    assert len(mapping.mapped_parents) == len(mapping.source_parents) == 2


def test_matched_sparsity_rejects_cardinality_mismatch() -> None:
    with pytest.raises(MatchedSparsityMappingError, match="cardinality"):
        construct_matched_sparsity_mapping(
            _manifest(),
            _parent_set(),
            construction_subject_ids=("train_a", "train_b"),
            sampled_parents=(ParentLink("right_eye", 3, "vx", "vy"),),
        )


def test_matched_sparsity_rejects_duplicate_sampled_parents() -> None:
    duplicate = ParentLink("right_eye", 3, "vx", "vy")
    with pytest.raises(MatchedSparsityMappingError, match="duplicates"):
        construct_matched_sparsity_mapping(
            _manifest(),
            _parent_set(),
            construction_subject_ids=("train_a", "train_b"),
            sampled_parents=(duplicate, duplicate),
        )


def test_matched_sparsity_rejects_outer_test_construction_scope() -> None:
    with pytest.raises(LeakageGuardError, match="outer-test"):
        construct_matched_sparsity_mapping(
            _manifest(),
            _parent_set(),
            construction_subject_ids=("train_a", "test_a"),
            sampled_parents=(
                ParentLink("right_eye", 3, "vx", "vy"),
                ParentLink("left_brow", 7, "vy", "vx"),
            ),
        )


def test_matched_sparsity_rejects_fold_mismatch() -> None:
    wrong_fold = ParentSet(
        outer_fold=5,
        target_region="mouth",
        parents=_parent_set().parents,
        discovery_method="pcmci_plus",
    )
    with pytest.raises(MatchedSparsityMappingError, match="outer_fold"):
        construct_matched_sparsity_mapping(
            _manifest(),
            wrong_fold,
            construction_subject_ids=("train_a", "train_b"),
            sampled_parents=(
                ParentLink("right_eye", 3, "vx", "vy"),
                ParentLink("left_brow", 7, "vy", "vx"),
            ),
        )
