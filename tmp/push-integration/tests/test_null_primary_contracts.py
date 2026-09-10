from __future__ import annotations

from collections import Counter

import pytest

from lagged_facial_graph_forecasting.contracts import InnerFold, SplitManifest
from lagged_facial_graph_forecasting.core_contracts import ParentLink, ParentSet
from lagged_facial_graph_forecasting.leakage_guard import LeakageGuardError
from lagged_facial_graph_forecasting.null_lag_shift import construct_lag_shift_mapping
from lagged_facial_graph_forecasting.null_matched_sparsity import (
    build_matched_sparsity_candidate_space,
    construct_matched_sparsity_mapping,
    sample_matched_sparsity_parents,
)
from lagged_facial_graph_forecasting.null_random_region import construct_random_region_mapping
from lagged_facial_graph_forecasting.null_time_shuffle import construct_time_shuffle_mapping


def _manifest() -> SplitManifest:
    return SplitManifest(
        outer_fold=6,
        train_subject_ids=("train_a", "train_b"),
        test_subject_ids=("test_a",),
        inner_folds=(InnerFold(("train_a",), ("train_b",)),),
        seed=2609,
    )


def _parent_set() -> ParentSet:
    return ParentSet(
        outer_fold=6,
        target_region="mouth",
        parents=(
            ParentLink("jaw", 2, "vx", "vx"),
            ParentLink("eye", 4, "vy", "vy"),
        ),
        discovery_method="pcmci_plus",
    )


def _matched_mapping():
    manifest = _manifest()
    parent_set = _parent_set()
    scope = manifest.train_subject_ids
    candidates = build_matched_sparsity_candidate_space(
        manifest,
        parent_set,
        construction_subject_ids=scope,
        region_ids=("mouth", "jaw", "eye", "cheek", "brow"),
        dimension_ids=("vx", "vy"),
    )
    sampled = sample_matched_sparsity_parents(
        manifest,
        parent_set,
        construction_subject_ids=scope,
        candidates=candidates,
    )
    return construct_matched_sparsity_mapping(
        manifest,
        parent_set,
        construction_subject_ids=scope,
        sampled_parents=sampled,
    )


def _primary_mappings():
    manifest = _manifest()
    parent_set = _parent_set()
    scope = manifest.train_subject_ids
    return (
        construct_lag_shift_mapping(
            manifest,
            parent_set,
            construction_subject_ids=scope,
            lag_delta=1,
        ),
        construct_random_region_mapping(
            manifest,
            parent_set,
            construction_subject_ids=scope,
            candidate_regions=("mouth", "jaw", "eye", "cheek", "brow"),
        ),
        _matched_mapping(),
        construct_time_shuffle_mapping(
            manifest,
            parent_set,
            construction_subject_ids=scope,
        ),
    )


def test_all_primary_null_mappings_preserve_common_contract_provenance() -> None:
    manifest = _manifest()
    parent_set = _parent_set()
    mappings = _primary_mappings()

    assert {mapping.condition for mapping in mappings} == {
        "lag-shift",
        "random-region",
        "matched-sparsity",
        "time-shuffle",
    }
    for mapping in mappings:
        assert mapping.outer_fold == manifest.outer_fold
        assert mapping.target_region == parent_set.target_region
        assert mapping.seed == manifest.seed
        assert mapping.source_parents == parent_set.parents
        assert len(mapping.mapped_parents) == len(parent_set.parents)


def test_lag_shift_changes_only_lag_identity() -> None:
    mapping = _primary_mappings()[0]
    for source, mapped in zip(mapping.source_parents, mapping.mapped_parents, strict=True):
        assert mapped.source_region == source.source_region
        assert mapped.source_dimension == source.source_dimension
        assert mapped.target_dimension == source.target_dimension
        assert mapped.lag == source.lag + 1


def test_random_region_changes_only_source_region_and_excludes_source_and_target() -> None:
    mapping = _primary_mappings()[1]
    for source, mapped in zip(mapping.source_parents, mapping.mapped_parents, strict=True):
        assert mapped.source_region != source.source_region
        assert mapped.source_region != mapping.target_region
        assert mapped.lag == source.lag
        assert mapped.source_dimension == source.source_dimension
        assert mapped.target_dimension == source.target_dimension


def test_matched_sparsity_preserves_exact_per_target_component_capacity() -> None:
    mapping = _primary_mappings()[2]
    assert Counter(parent.target_dimension for parent in mapping.mapped_parents) == Counter(
        parent.target_dimension for parent in mapping.source_parents
    )
    assert all(parent.source_region != mapping.target_region for parent in mapping.mapped_parents)
    assert all(1 <= parent.lag <= 10 for parent in mapping.mapped_parents)


def test_time_shuffle_freezes_parent_identity_and_defers_row_permutation() -> None:
    mapping = _primary_mappings()[3]
    assert mapping.mapped_parents == mapping.source_parents
    assert mapping.permutation == ()


@pytest.mark.parametrize(
    "constructor_name",
    ("lag-shift", "random-region", "matched-candidate", "matched-mapping", "time-shuffle"),
)
def test_every_primary_null_construction_boundary_rejects_outer_test_subjects(
    constructor_name: str,
) -> None:
    manifest = _manifest()
    parent_set = _parent_set()
    leaked_scope = ("train_a", "test_a")

    with pytest.raises(LeakageGuardError, match="outer-test"):
        if constructor_name == "lag-shift":
            construct_lag_shift_mapping(
                manifest,
                parent_set,
                construction_subject_ids=leaked_scope,
                lag_delta=1,
            )
        elif constructor_name == "random-region":
            construct_random_region_mapping(
                manifest,
                parent_set,
                construction_subject_ids=leaked_scope,
                candidate_regions=("mouth", "jaw", "eye", "cheek", "brow"),
            )
        elif constructor_name == "matched-candidate":
            build_matched_sparsity_candidate_space(
                manifest,
                parent_set,
                construction_subject_ids=leaked_scope,
                region_ids=("mouth", "jaw", "eye", "cheek", "brow"),
                dimension_ids=("vx", "vy"),
            )
        elif constructor_name == "matched-mapping":
            construct_matched_sparsity_mapping(
                manifest,
                parent_set,
                construction_subject_ids=leaked_scope,
                sampled_parents=parent_set.parents,
            )
        else:
            construct_time_shuffle_mapping(
                manifest,
                parent_set,
                construction_subject_ids=leaked_scope,
            )
