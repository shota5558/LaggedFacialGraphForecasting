from __future__ import annotations

import pytest

from lagged_facial_graph_forecasting.contracts import InnerFold, SplitManifest
from lagged_facial_graph_forecasting.core_contracts import NullMapping, ParentLink, ParentSet
from lagged_facial_graph_forecasting.null_time_shuffle import (
    TimeShuffleMappingError,
    construct_time_shuffle_mapping,
    generate_time_shuffle_permutation,
)


def _manifest(seed: int = 404) -> SplitManifest:
    return SplitManifest(
        outer_fold=3,
        train_subject_ids=("train_a", "train_b"),
        test_subject_ids=("test_a",),
        inner_folds=(InnerFold(("train_a",), ("train_b",)),),
        seed=seed,
    )


def _parent_set() -> ParentSet:
    return ParentSet(
        outer_fold=3,
        target_region="mouth",
        parents=(ParentLink("jaw", 2, "vx", "vx"),),
        discovery_method="pcmci_plus",
    )


def _mapping() -> NullMapping:
    return construct_time_shuffle_mapping(
        _manifest(),
        _parent_set(),
        construction_subject_ids=("train_a", "train_b"),
    )


def test_permutation_contains_every_row_once_and_is_not_identity() -> None:
    permutation = generate_time_shuffle_permutation(_mapping(), row_count=12)
    assert len(permutation) == 12
    assert set(permutation) == set(range(12))
    assert permutation != tuple(range(12))


def test_permutation_is_deterministic_for_same_frozen_seed_and_row_count() -> None:
    first = generate_time_shuffle_permutation(_mapping(), row_count=15)
    second = generate_time_shuffle_permutation(_mapping(), row_count=15)
    assert first == second


def test_two_rows_always_produce_the_only_non_identity_permutation() -> None:
    assert generate_time_shuffle_permutation(_mapping(), row_count=2) == (1, 0)


@pytest.mark.parametrize("row_count", [0, 1, -1, True, 2.5])
def test_permutation_rejects_unevaluable_or_invalid_row_count(row_count) -> None:
    with pytest.raises(TimeShuffleMappingError, match="row_count"):
        generate_time_shuffle_permutation(_mapping(), row_count=row_count)


def test_permutation_rejects_non_time_shuffle_mapping() -> None:
    wrong = NullMapping(
        outer_fold=3,
        target_region="mouth",
        condition="random-region",
        seed=404,
        source_parents=_parent_set().parents,
        mapped_parents=_parent_set().parents,
    )
    with pytest.raises(TimeShuffleMappingError, match="condition='time-shuffle'"):
        generate_time_shuffle_permutation(wrong, row_count=8)


def test_permutation_rejects_parent_identity_change_before_temporal_shuffle() -> None:
    changed = NullMapping(
        outer_fold=3,
        target_region="mouth",
        condition="time-shuffle",
        seed=404,
        source_parents=(ParentLink("jaw", 2, "vx", "vx"),),
        mapped_parents=(ParentLink("eye", 2, "vx", "vx"),),
    )
    with pytest.raises(TimeShuffleMappingError, match="preserve ParentLink identity"):
        generate_time_shuffle_permutation(changed, row_count=8)


def test_permutation_rejects_already_realized_mapping() -> None:
    realized = NullMapping(
        outer_fold=3,
        target_region="mouth",
        condition="time-shuffle",
        seed=404,
        source_parents=_parent_set().parents,
        mapped_parents=_parent_set().parents,
        permutation=(1, 0),
    )
    with pytest.raises(TimeShuffleMappingError, match="unrealized"):
        generate_time_shuffle_permutation(realized, row_count=2)
