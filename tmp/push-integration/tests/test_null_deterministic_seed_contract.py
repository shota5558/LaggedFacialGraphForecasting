from __future__ import annotations

from lagged_facial_graph_forecasting.contracts import InnerFold, SplitManifest
from lagged_facial_graph_forecasting.core_contracts import ParentLink, ParentSet
from lagged_facial_graph_forecasting.null_lag_shift import construct_lag_shift_mapping
from lagged_facial_graph_forecasting.null_matched_sparsity import (
    build_matched_sparsity_candidate_space,
    construct_matched_sparsity_mapping,
    sample_matched_sparsity_parents,
    sample_matched_sparsity_repeats,
)
from lagged_facial_graph_forecasting.null_random_region import (
    construct_random_region_mapping,
)
from lagged_facial_graph_forecasting.null_time_shuffle import (
    construct_time_shuffle_mapping,
    generate_time_shuffle_permutation,
)


def _manifest(seed: int = 2026) -> SplitManifest:
    return SplitManifest(
        outer_fold=2,
        train_subject_ids=("train_a", "train_b"),
        test_subject_ids=("test_a",),
        inner_folds=(InnerFold(("train_a",), ("train_b",)),),
        seed=seed,
    )


def _parent_set() -> ParentSet:
    return ParentSet(
        outer_fold=2,
        target_region="mouth",
        parents=(
            ParentLink("jaw", 2, "vx", "vx"),
            ParentLink("eye", 4, "vy", "vy"),
        ),
        discovery_method="pcmci_plus",
    )


def test_all_primary_null_mappings_record_splitmanifest_seed_as_single_source_of_truth() -> None:
    manifest = _manifest()
    parent_set = _parent_set()
    scope = ("train_a", "train_b")

    lag_shift = construct_lag_shift_mapping(
        manifest, parent_set, construction_subject_ids=scope, lag_delta=1
    )
    random_region = construct_random_region_mapping(
        manifest,
        parent_set,
        construction_subject_ids=scope,
        candidate_regions=("mouth", "jaw", "eye", "cheek", "brow"),
    )
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
    matched = construct_matched_sparsity_mapping(
        manifest,
        parent_set,
        construction_subject_ids=scope,
        sampled_parents=sampled,
    )
    time_shuffle = construct_time_shuffle_mapping(
        manifest, parent_set, construction_subject_ids=scope
    )

    assert {
        lag_shift.seed,
        random_region.seed,
        matched.seed,
        time_shuffle.seed,
    } == {manifest.seed}


def test_stochastic_primary_nulls_are_exactly_reproducible_for_same_manifest_seed() -> None:
    manifest = _manifest()
    parent_set = _parent_set()
    scope = ("train_a", "train_b")

    random_a = construct_random_region_mapping(
        manifest,
        parent_set,
        construction_subject_ids=scope,
        candidate_regions=("mouth", "jaw", "eye", "cheek", "brow"),
    )
    random_b = construct_random_region_mapping(
        manifest,
        parent_set,
        construction_subject_ids=scope,
        candidate_regions=("mouth", "jaw", "eye", "cheek", "brow"),
    )
    assert random_a == random_b

    candidates = build_matched_sparsity_candidate_space(
        manifest,
        parent_set,
        construction_subject_ids=scope,
        region_ids=("mouth", "jaw", "eye", "cheek", "brow"),
        dimension_ids=("vx", "vy"),
    )
    sample_a = sample_matched_sparsity_parents(
        manifest,
        parent_set,
        construction_subject_ids=scope,
        candidates=candidates,
    )
    sample_b = sample_matched_sparsity_parents(
        manifest,
        parent_set,
        construction_subject_ids=scope,
        candidates=candidates,
    )
    assert sample_a == sample_b

    repeats_a = sample_matched_sparsity_repeats(
        manifest,
        parent_set,
        construction_subject_ids=scope,
        candidates=candidates,
        repeat_count=4,
    )
    repeats_b = sample_matched_sparsity_repeats(
        manifest,
        parent_set,
        construction_subject_ids=scope,
        candidates=candidates,
        repeat_count=4,
    )
    assert repeats_a == repeats_b

    time_mapping = construct_time_shuffle_mapping(
        manifest, parent_set, construction_subject_ids=scope
    )
    permutation_a = generate_time_shuffle_permutation(time_mapping, row_count=12)
    permutation_b = generate_time_shuffle_permutation(time_mapping, row_count=12)
    assert permutation_a == permutation_b


def test_repeat_zero_remains_the_same_seeded_matched_sparsity_draw() -> None:
    manifest = _manifest()
    parent_set = _parent_set()
    scope = ("train_a", "train_b")
    candidates = build_matched_sparsity_candidate_space(
        manifest,
        parent_set,
        construction_subject_ids=scope,
        region_ids=("mouth", "jaw", "eye", "cheek", "brow"),
        dimension_ids=("vx", "vy"),
    )
    single = sample_matched_sparsity_parents(
        manifest,
        parent_set,
        construction_subject_ids=scope,
        candidates=candidates,
    )
    repeats = sample_matched_sparsity_repeats(
        manifest,
        parent_set,
        construction_subject_ids=scope,
        candidates=candidates,
        repeat_count=3,
    )
    assert repeats[0] == single
