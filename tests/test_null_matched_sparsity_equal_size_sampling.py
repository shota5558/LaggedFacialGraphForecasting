from __future__ import annotations

from collections import Counter

import pytest

from lagged_facial_graph_forecasting.contracts import InnerFold, SplitManifest
from lagged_facial_graph_forecasting.core_contracts import ParentLink, ParentSet
from lagged_facial_graph_forecasting.leakage_guard import LeakageGuardError
from lagged_facial_graph_forecasting.null_matched_sparsity import (
    MatchedSparsityMappingError,
    build_matched_sparsity_candidate_space,
    construct_matched_sparsity_mapping,
    sample_matched_sparsity_parents,
)


def _manifest(seed: int = 930) -> SplitManifest:
    return SplitManifest(
        outer_fold=4,
        train_subject_ids=("train_a", "train_b"),
        test_subject_ids=("test_a",),
        inner_folds=(InnerFold(("train_a",), ("train_b",)),),
        seed=seed,
    )


def _parent_set() -> ParentSet:
    return ParentSet(
        outer_fold=4,
        target_region="mouth",
        parents=(
            ParentLink("jaw", 2, "vx", "vx"),
            ParentLink("eye", 4, "vy", "vx"),
            ParentLink("jaw", 7, "vy", "vy"),
        ),
        discovery_method="pcmci_plus",
    )


def _candidates(manifest: SplitManifest | None = None):
    manifest = _manifest() if manifest is None else manifest
    return build_matched_sparsity_candidate_space(
        manifest,
        _parent_set(),
        construction_subject_ids=("train_a", "train_b"),
        region_ids=("mouth", "jaw", "eye", "cheek"),
        dimension_ids=("vx", "vy"),
    )


def test_equal_size_sampling_matches_pcmci_count_per_target_component_without_replacement() -> None:
    candidates = _candidates()
    sampled = sample_matched_sparsity_parents(
        _manifest(),
        _parent_set(),
        construction_subject_ids=("train_a", "train_b"),
        candidates=candidates,
    )

    assert len(sampled) == len(_parent_set().parents) == 3
    assert len(set(sampled)) == len(sampled)
    assert set(sampled) <= set(candidates)
    assert Counter(parent.target_dimension for parent in sampled) == Counter(
        parent.target_dimension for parent in _parent_set().parents
    )


def test_equal_size_sampling_is_deterministic_and_candidate_order_invariant() -> None:
    candidates = _candidates()
    forward = sample_matched_sparsity_parents(
        _manifest(),
        _parent_set(),
        construction_subject_ids=("train_a", "train_b"),
        candidates=candidates,
    )
    reversed_input = sample_matched_sparsity_parents(
        _manifest(),
        _parent_set(),
        construction_subject_ids=("train_a", "train_b"),
        candidates=tuple(reversed(candidates)),
    )
    repeat = sample_matched_sparsity_parents(
        _manifest(),
        _parent_set(),
        construction_subject_ids=("train_a", "train_b"),
        candidates=candidates,
    )

    assert forward == reversed_input == repeat


def test_equal_size_sampling_fails_closed_when_one_target_component_stratum_is_too_small() -> None:
    # vx requires two links, but this supplied candidate space contains only one vx link.
    candidates = (
        ParentLink("jaw", 1, "vx", "vx"),
        ParentLink("eye", 1, "vy", "vy"),
    )
    with pytest.raises(MatchedSparsityMappingError, match="smaller than required"):
        sample_matched_sparsity_parents(
            _manifest(),
            _parent_set(),
            construction_subject_ids=("train_a", "train_b"),
            candidates=candidates,
        )


def test_equal_size_sampling_rejects_candidate_target_component_not_in_pcmci_parentset() -> None:
    candidates = _candidates() + (ParentLink("jaw", 1, "vx", "vz"),)
    with pytest.raises(MatchedSparsityMappingError, match="absent from PCMCI ParentSet"):
        sample_matched_sparsity_parents(
            _manifest(),
            _parent_set(),
            construction_subject_ids=("train_a", "train_b"),
            candidates=candidates,
        )


def test_mapping_rejects_equal_total_count_with_wrong_per_target_component_capacity() -> None:
    wrong_component_balance = (
        ParentLink("jaw", 1, "vx", "vx"),
        ParentLink("eye", 2, "vx", "vx"),
        ParentLink("cheek", 3, "vy", "vx"),
    )
    with pytest.raises(MatchedSparsityMappingError, match="per target_dimension"):
        construct_matched_sparsity_mapping(
            _manifest(),
            _parent_set(),
            construction_subject_ids=("train_a", "train_b"),
            sampled_parents=wrong_component_balance,
        )


def test_equal_size_sampling_rejects_outer_test_construction_scope() -> None:
    with pytest.raises(LeakageGuardError, match="outer-test"):
        sample_matched_sparsity_parents(
            _manifest(),
            _parent_set(),
            construction_subject_ids=("train_a", "test_a"),
            candidates=_candidates(),
        )


def test_equal_size_sampling_rejects_fold_mismatch() -> None:
    wrong_fold = ParentSet(
        outer_fold=5,
        target_region="mouth",
        parents=_parent_set().parents,
        discovery_method="pcmci_plus",
    )
    with pytest.raises(MatchedSparsityMappingError, match="outer_fold"):
        sample_matched_sparsity_parents(
            _manifest(),
            wrong_fold,
            construction_subject_ids=("train_a", "train_b"),
            candidates=_candidates(),
        )
