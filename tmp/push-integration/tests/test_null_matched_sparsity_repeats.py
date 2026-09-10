from __future__ import annotations

from collections import Counter

import pytest

from lagged_facial_graph_forecasting.contracts import InnerFold, SplitManifest
from lagged_facial_graph_forecasting.core_contracts import ParentLink, ParentSet
from lagged_facial_graph_forecasting.leakage_guard import LeakageGuardError
from lagged_facial_graph_forecasting.null_matched_sparsity import (
    MatchedSparsityMappingError,
    build_matched_sparsity_candidate_space,
    sample_matched_sparsity_parents,
    sample_matched_sparsity_repeats,
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


def _candidates():
    return build_matched_sparsity_candidate_space(
        _manifest(),
        _parent_set(),
        construction_subject_ids=("train_a", "train_b"),
        region_ids=("mouth", "jaw", "eye", "cheek"),
        dimension_ids=("vx", "vy"),
    )


def test_repeated_sampling_returns_requested_number_with_matched_capacity_each_time() -> None:
    candidates = _candidates()
    repeats = sample_matched_sparsity_repeats(
        _manifest(),
        _parent_set(),
        construction_subject_ids=("train_a", "train_b"),
        candidates=candidates,
        repeat_count=5,
    )

    required_counts = Counter(
        parent.target_dimension for parent in _parent_set().parents
    )
    assert len(repeats) == 5
    for sampled in repeats:
        assert len(sampled) == len(_parent_set().parents)
        assert len(set(sampled)) == len(sampled)
        assert set(sampled) <= set(candidates)
        assert Counter(parent.target_dimension for parent in sampled) == required_counts


def test_repeat_zero_is_exactly_the_f32_single_sample() -> None:
    candidates = _candidates()
    single = sample_matched_sparsity_parents(
        _manifest(),
        _parent_set(),
        construction_subject_ids=("train_a", "train_b"),
        candidates=candidates,
    )
    repeats = sample_matched_sparsity_repeats(
        _manifest(),
        _parent_set(),
        construction_subject_ids=("train_a", "train_b"),
        candidates=candidates,
        repeat_count=3,
    )
    assert repeats[0] == single


def test_repeated_sampling_is_deterministic_candidate_order_invariant_and_prefix_stable() -> None:
    candidates = _candidates()
    three = sample_matched_sparsity_repeats(
        _manifest(),
        _parent_set(),
        construction_subject_ids=("train_a", "train_b"),
        candidates=candidates,
        repeat_count=3,
    )
    five = sample_matched_sparsity_repeats(
        _manifest(),
        _parent_set(),
        construction_subject_ids=("train_a", "train_b"),
        candidates=tuple(reversed(candidates)),
        repeat_count=5,
    )
    repeat_three = sample_matched_sparsity_repeats(
        _manifest(),
        _parent_set(),
        construction_subject_ids=("train_a", "train_b"),
        candidates=candidates,
        repeat_count=3,
    )

    assert three == repeat_three
    assert five[:3] == three


def test_repeated_sampling_does_not_force_cross_repeat_uniqueness() -> None:
    # One feasible candidate for one required feature means every independent repeat
    # must coincide. The contract must allow this rather than biasing the null through
    # rejection sampling.
    parent_set = ParentSet(
        outer_fold=4,
        target_region="mouth",
        parents=(ParentLink("jaw", 1, "vx", "vx"),),
        discovery_method="pcmci_plus",
    )
    only_candidate = (ParentLink("jaw", 1, "vx", "vx"),)
    repeats = sample_matched_sparsity_repeats(
        _manifest(),
        parent_set,
        construction_subject_ids=("train_a", "train_b"),
        candidates=only_candidate,
        repeat_count=4,
    )
    assert repeats == (only_candidate,) * 4


@pytest.mark.parametrize("repeat_count", [0, -1, True, 1.5])
def test_repeated_sampling_requires_explicit_positive_integer_count(repeat_count) -> None:
    with pytest.raises(MatchedSparsityMappingError, match="positive integer"):
        sample_matched_sparsity_repeats(
            _manifest(),
            _parent_set(),
            construction_subject_ids=("train_a", "train_b"),
            candidates=_candidates(),
            repeat_count=repeat_count,
        )


def test_repeated_sampling_rejects_outer_test_construction_scope() -> None:
    with pytest.raises(LeakageGuardError, match="outer-test"):
        sample_matched_sparsity_repeats(
            _manifest(),
            _parent_set(),
            construction_subject_ids=("train_a", "test_a"),
            candidates=_candidates(),
            repeat_count=2,
        )


def test_repeated_sampling_rejects_fold_mismatch() -> None:
    wrong_fold = ParentSet(
        outer_fold=5,
        target_region="mouth",
        parents=_parent_set().parents,
        discovery_method="pcmci_plus",
    )
    with pytest.raises(MatchedSparsityMappingError, match="outer_fold"):
        sample_matched_sparsity_repeats(
            _manifest(),
            wrong_fold,
            construction_subject_ids=("train_a", "train_b"),
            candidates=_candidates(),
            repeat_count=2,
        )
