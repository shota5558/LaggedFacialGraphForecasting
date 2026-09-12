from __future__ import annotations

import pytest

from lagged_facial_graph_forecasting.contracts import InnerFold, SplitManifest
from lagged_facial_graph_forecasting.core_contracts import ParentLink, ParentSet
from lagged_facial_graph_forecasting.leakage_guard import LeakageGuardError
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


def _parents() -> ParentSet:
    return ParentSet(
        outer_fold=0,
        target_region="mouth",
        parents=(
            ParentLink("left_cheek", 2, "vx", "vy"),
            ParentLink("jaw", 3, "vy", "vx"),
        ),
        discovery_method="pcmci_plus",
    )


def test_random_region_mapping_is_deterministic_from_frozen_seed_and_candidates() -> None:
    kwargs = dict(
        construction_subject_ids=("train_a", "train_b"),
        candidate_regions=("left_cheek", "right_cheek", "jaw", "left_eye", "mouth"),
    )

    first = construct_random_region_mapping(_manifest(), _parents(), **kwargs)
    second = construct_random_region_mapping(_manifest(), _parents(), **kwargs)

    assert first == second
    assert first.condition == "random-region"
    assert first.seed == _manifest().seed
    assert first.source_parents == _parents().parents
    assert len(first.mapped_parents) == len(first.source_parents)
    assert len(set(first.mapped_parents)) == len(first.mapped_parents)


def test_random_region_mapping_rejects_outer_test_construction_scope() -> None:
    with pytest.raises(LeakageGuardError, match="outer-test"):
        construct_random_region_mapping(
            _manifest(),
            _parents(),
            construction_subject_ids=("train_a", "test_a"),
            candidate_regions=("left_cheek", "right_cheek", "jaw", "left_eye"),
        )


def test_random_region_mapping_requires_frozen_nonempty_unique_candidate_universe() -> None:
    for candidates in ((), ("left_cheek", "left_cheek"), ("left_cheek", "")):
        with pytest.raises(RandomRegionMappingError):
            construct_random_region_mapping(
                _manifest(),
                _parents(),
                construction_subject_ids=("train_a", "train_b"),
                candidate_regions=candidates,
            )


def test_random_region_mapping_fails_when_no_replacement_exists() -> None:
    with pytest.raises(RandomRegionMappingError, match="no eligible replacement"):
        construct_random_region_mapping(
            _manifest(),
            _parents(),
            construction_subject_ids=("train_a", "train_b"),
            candidate_regions=("left_cheek", "mouth"),
        )
