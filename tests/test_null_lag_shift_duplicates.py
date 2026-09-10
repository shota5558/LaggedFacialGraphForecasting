from __future__ import annotations

import pytest

from lagged_facial_graph_forecasting.contracts import ContractError, InnerFold, SplitManifest
from lagged_facial_graph_forecasting.core_contracts import NullMapping, ParentLink, ParentSet
from lagged_facial_graph_forecasting.null_lag_shift import construct_lag_shift_mapping


def _manifest() -> SplitManifest:
    return SplitManifest(
        outer_fold=0,
        train_subject_ids=("train_a", "train_b"),
        test_subject_ids=("test_a",),
        inner_folds=(InnerFold(("train_a",), ("train_b",)),),
        seed=101,
    )


def test_uniform_lag_shift_preserves_uniqueness_for_distinct_source_parents() -> None:
    parent_set = ParentSet(
        outer_fold=0,
        target_region="mouth",
        parents=(
            ParentLink("left_cheek", 2, "vx", "vy"),
            ParentLink("left_cheek", 3, "vx", "vy"),
            ParentLink("left_cheek", 2, "vy", "vy"),
        ),
        discovery_method="pcmci_plus",
    )

    mapping = construct_lag_shift_mapping(
        _manifest(),
        parent_set,
        construction_subject_ids=("train_a", "train_b"),
        lag_delta=2,
    )

    assert len(mapping.mapped_parents) == len(parent_set.parents)
    assert len(set(mapping.mapped_parents)) == len(mapping.mapped_parents)
    assert mapping.mapped_parents == (
        ParentLink("left_cheek", 4, "vx", "vy"),
        ParentLink("left_cheek", 5, "vx", "vy"),
        ParentLink("left_cheek", 4, "vy", "vy"),
    )


def test_null_mapping_fails_closed_instead_of_deduplicating_mapped_parents() -> None:
    source = (
        ParentLink("left_cheek", 2, "vx", "vy"),
        ParentLink("jaw", 3, "vy", "vy"),
    )
    duplicate = ParentLink("right_cheek", 4, "vx", "vy")

    with pytest.raises(ContractError, match="mapped_parents must not contain duplicates"):
        NullMapping(
            outer_fold=0,
            target_region="mouth",
            condition="lag-shift",
            seed=101,
            source_parents=source,
            mapped_parents=(duplicate, duplicate),
        )
