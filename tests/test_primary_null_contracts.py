from __future__ import annotations

import inspect
from collections import Counter
from pathlib import Path

import pytest

from lagged_facial_graph_forecasting.contracts import InnerFold, SplitManifest
from lagged_facial_graph_forecasting.core_contract_io import (
    dumps_core_contract,
    loads_core_contract,
)
from lagged_facial_graph_forecasting.core_contracts import NullMapping, ParentLink, ParentSet
from lagged_facial_graph_forecasting.leakage_guard import LeakageGuardError
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
)
from lagged_facial_graph_forecasting.scientific_config import load_scientific_config


PRIMARY_NULL_CONDITIONS = (
    "lag-shift",
    "random-region",
    "matched-sparsity",
    "time-shuffle",
)


def _manifest() -> SplitManifest:
    return SplitManifest(
        outer_fold=6,
        train_subject_ids=("train_a", "train_b"),
        test_subject_ids=("test_a",),
        inner_folds=(InnerFold(("train_a",), ("train_b",)),),
        seed=20260908,
    )


def _parent_set() -> ParentSet:
    return ParentSet(
        outer_fold=6,
        target_region="mouth",
        parents=(
            ParentLink("jaw", 3, "vx", "vx"),
            ParentLink("eye", 6, "vy", "vy"),
        ),
        discovery_method="pcmci_plus",
    )


def _matched_candidates(
    manifest: SplitManifest,
    parent_set: ParentSet,
    *,
    scope: tuple[str, ...],
) -> tuple[ParentLink, ...]:
    return build_matched_sparsity_candidate_space(
        manifest,
        parent_set,
        construction_subject_ids=scope,
        region_ids=("mouth", "jaw", "eye", "cheek", "brow"),
        dimension_ids=("vx", "vy"),
    )


def _all_primary_mappings() -> tuple[NullMapping, ...]:
    manifest = _manifest()
    parent_set = _parent_set()
    scope = manifest.train_subject_ids

    candidates = _matched_candidates(manifest, parent_set, scope=scope)
    sampled = sample_matched_sparsity_parents(
        manifest,
        parent_set,
        construction_subject_ids=scope,
        candidates=candidates,
    )

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
        construct_matched_sparsity_mapping(
            manifest,
            parent_set,
            construction_subject_ids=scope,
            sampled_parents=sampled,
        ),
        construct_time_shuffle_mapping(
            manifest,
            parent_set,
            construction_subject_ids=scope,
        ),
    )


def test_runtime_primary_null_conditions_match_scientific_freeze() -> None:
    config = load_scientific_config(Path("configs/scientific_freeze.yaml"))
    frozen = {
        condition.replace("_", "-")
        for condition in config["primary"]["conditions"]
        if condition in {
            "lag_shift",
            "random_region",
            "matched_sparsity",
            "time_shuffle",
        }
    }

    assert frozen == set(PRIMARY_NULL_CONDITIONS)
    assert {mapping.condition for mapping in _all_primary_mappings()} == frozen

    # Stronger autocorrelation-preserving surrogates remain Sensitivity-only.
    sensitivity = set(config["sensitivity"]["methods"])
    assert "phase_shuffled_surrogate" in sensitivity
    assert "circular_shift_surrogate" in sensitivity
    assert "phase-shuffled-surrogate" not in frozen
    assert "circular-shift-surrogate" not in frozen


def test_all_primary_null_mappings_share_fold_target_seed_and_source_provenance() -> None:
    manifest = _manifest()
    parent_set = _parent_set()

    for mapping in _all_primary_mappings():
        assert mapping.outer_fold == manifest.outer_fold
        assert mapping.target_region == parent_set.target_region
        assert mapping.seed == manifest.seed
        assert mapping.source_parents == parent_set.parents
        assert len(mapping.mapped_parents) == len(parent_set.parents)

        # Every control remains component-aware for D=2 scalar-component forecasting.
        assert Counter(parent.target_dimension for parent in mapping.mapped_parents) == Counter(
            parent.target_dimension for parent in parent_set.parents
        )


def test_condition_specific_null_invariants_hold_under_common_contract() -> None:
    lag_shift, random_region, matched_sparsity, time_shuffle = _all_primary_mappings()

    for source, mapped in zip(
        lag_shift.source_parents, lag_shift.mapped_parents, strict=True
    ):
        assert mapped.source_region == source.source_region
        assert mapped.source_dimension == source.source_dimension
        assert mapped.target_dimension == source.target_dimension
        assert mapped.lag == source.lag + 1

    for source, mapped in zip(
        random_region.source_parents, random_region.mapped_parents, strict=True
    ):
        assert mapped.source_region not in {
            source.source_region,
            random_region.target_region,
        }
        assert mapped.lag == source.lag
        assert mapped.source_dimension == source.source_dimension
        assert mapped.target_dimension == source.target_dimension

    assert len(set(matched_sparsity.mapped_parents)) == len(
        matched_sparsity.mapped_parents
    )
    assert Counter(
        parent.target_dimension for parent in matched_sparsity.mapped_parents
    ) == Counter(parent.target_dimension for parent in matched_sparsity.source_parents)

    assert time_shuffle.mapped_parents == time_shuffle.source_parents
    assert time_shuffle.permutation == ()


def test_every_primary_null_construction_rejects_outer_test_scope() -> None:
    manifest = _manifest()
    parent_set = _parent_set()
    bad_scope = ("train_a", "test_a")
    good_scope = manifest.train_subject_ids

    candidates = _matched_candidates(manifest, parent_set, scope=good_scope)
    sampled = sample_matched_sparsity_parents(
        manifest,
        parent_set,
        construction_subject_ids=good_scope,
        candidates=candidates,
    )

    calls = (
        lambda: construct_lag_shift_mapping(
            manifest,
            parent_set,
            construction_subject_ids=bad_scope,
            lag_delta=1,
        ),
        lambda: construct_random_region_mapping(
            manifest,
            parent_set,
            construction_subject_ids=bad_scope,
            candidate_regions=("mouth", "jaw", "eye", "cheek", "brow"),
        ),
        lambda: construct_matched_sparsity_mapping(
            manifest,
            parent_set,
            construction_subject_ids=bad_scope,
            sampled_parents=sampled,
        ),
        lambda: construct_time_shuffle_mapping(
            manifest,
            parent_set,
            construction_subject_ids=bad_scope,
        ),
    )

    for call in calls:
        with pytest.raises(LeakageGuardError, match="outer-test"):
            call()


def test_all_primary_null_mappings_roundtrip_and_repeat_count_is_not_implicitly_invented() -> None:
    for mapping in _all_primary_mappings():
        restored = loads_core_contract(dumps_core_contract(mapping))
        assert restored == mapping

    # F-33 supports repeated controls, but the scientific freeze has not yet selected
    # the actual Primary repeat count. The implementation must therefore stay
    # fail-closed instead of silently introducing a default experimental setting.
    repeat_count = inspect.signature(sample_matched_sparsity_repeats).parameters[
        "repeat_count"
    ]
    assert repeat_count.default is inspect.Parameter.empty
