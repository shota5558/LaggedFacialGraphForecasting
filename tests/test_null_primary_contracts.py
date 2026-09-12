from __future__ import annotations

import pytest

from lagged_facial_graph_forecasting.contracts import InnerFold, SplitManifest
from lagged_facial_graph_forecasting.core_contracts import ParentLink, ParentSet
from lagged_facial_graph_forecasting.leakage_guard import LeakageGuardError
from lagged_facial_graph_forecasting.null_lag_shift import construct_lag_shift_mapping
from lagged_facial_graph_forecasting.null_matched_sparsity import (
    build_matched_sparsity_candidate_space,
    construct_matched_sparsity_mapping,
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
