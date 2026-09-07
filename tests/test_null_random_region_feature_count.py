from __future__ import annotations

import pytest

from lagged_facial_graph_forecasting.contracts import InnerFold, SplitManifest
from lagged_facial_graph_forecasting.core_contracts import ParentLink, ParentSet
from lagged_facial_graph_forecasting.null_random_region import (
    RandomRegionMappingError,
    construct_random_region_mapping,
)


def _manifest() -> SplitManifest:
    return SplitManifest(
        outer_fold=0,
        train_subject_ids=("train_a", "train_b"),
        test_subject_ids=("test_a",),
        inner_folds=(InnerFold(("train_a",), ("train_b",)),),
        seed=515,
    )


def test_random_region_preserves_feature_count_one_for_one() -> None:
    parent_set = ParentSet(
        outer_fold=0,
        target_region="mouth",
        parents=(
            ParentLink("left_cheek", 1, "vx", "vx"),
            ParentLink("jaw", 5, "vy", "vx"),
            ParentLink("left_eye", 10, "vx", "vy"),
        ),
        discovery_method="pcmci_plus",
    )

    mapping = construct_random_region_mapping(
        _manifest(),
        parent_set,
        construction_subject_ids=("train_a", "train_b"),
        candidate_regions=(
            "left_cheek",
            "right_cheek",
            "jaw",
            "left_eye",
            "right_eye",
            "mouth",
        ),
    )

    assert len(mapping.source_parents) == len(parent_set.parents)
    assert len(mapping.mapped_parents) == len(parent_set.parents)
    assert len(set(mapping.mapped_parents)) == len(parent_set.parents)


def test_random_region_fails_closed_instead_of_dropping_feature_when_unique_replacements_are_exhausted() -> None:
    parent_set = ParentSet(
        outer_fold=0,
        target_region="mouth",
        parents=(
            ParentLink("left_cheek", 2, "vx", "vy"),
            ParentLink("jaw", 2, "vx", "vy"),
        ),
        discovery_method="pcmci_plus",
    )

    with pytest.raises(RandomRegionMappingError, match="no eligible replacement"):
        construct_random_region_mapping(
            _manifest(),
            parent_set,
            construction_subject_ids=("train_a", "train_b"),
            candidate_regions=("right_cheek", "mouth"),
        )
