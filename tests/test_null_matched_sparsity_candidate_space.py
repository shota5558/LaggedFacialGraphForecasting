from __future__ import annotations

import pytest

from lagged_facial_graph_forecasting.contracts import InnerFold, SplitManifest
from lagged_facial_graph_forecasting.core_contracts import ParentLink, ParentSet
from lagged_facial_graph_forecasting.leakage_guard import LeakageGuardError
from lagged_facial_graph_forecasting.null_matched_sparsity import (
    MatchedSparsityMappingError,
    build_matched_sparsity_candidate_space,
)


def _manifest() -> SplitManifest:
    return SplitManifest(
        outer_fold=4,
        train_subject_ids=("train_a", "train_b"),
        test_subject_ids=("test_a",),
        inner_folds=(InnerFold(("train_a",), ("train_b",)),),
        seed=930,
    )


def _parent_set() -> ParentSet:
    return ParentSet(
        outer_fold=4,
        target_region="mouth",
        parents=(
            ParentLink("jaw", 2, "vx", "vx"),
            ParentLink("eye", 4, "vy", "vy"),
        ),
        discovery_method="pcmci_plus",
    )


def _candidate_space(
    *,
    parent_set: ParentSet | None = None,
    region_ids: tuple[str, ...] = ("mouth", "jaw", "eye"),
    dimension_ids: tuple[str, ...] = ("vx", "vy"),
):
    return build_matched_sparsity_candidate_space(
        _manifest(),
        _parent_set() if parent_set is None else parent_set,
        construction_subject_ids=("train_a", "train_b"),
        region_ids=region_ids,
        dimension_ids=dimension_ids,
    )


def test_candidate_space_is_full_cross_region_primary_lag_component_universe() -> None:
    candidates = _candidate_space()

    # 2 selected target-component strata x 2 non-target regions x 2 source
    # components x 10 Primary lags.
    assert len(candidates) == 80
    assert len(set(candidates)) == 80
    assert {parent.source_region for parent in candidates} == {"jaw", "eye"}
    assert {parent.source_dimension for parent in candidates} == {"vx", "vy"}
    assert {parent.target_dimension for parent in candidates} == {"vx", "vy"}
    assert {parent.lag for parent in candidates} == set(range(1, 11))
    assert all(parent.source_region != "mouth" for parent in candidates)


def test_candidate_space_does_not_adversarially_exclude_pcmci_selected_links() -> None:
    candidates = set(_candidate_space())
    assert set(_parent_set().parents) <= candidates


def test_candidate_space_is_canonical_under_region_and_dimension_input_order() -> None:
    forward = _candidate_space()
    reversed_order = _candidate_space(
        region_ids=("eye", "jaw", "mouth"),
        dimension_ids=("vy", "vx"),
    )
    assert forward == reversed_order


def test_empty_parent_set_has_empty_matched_sparsity_candidate_space() -> None:
    empty = ParentSet(
        outer_fold=4,
        target_region="mouth",
        parents=(),
        discovery_method="pcmci_plus",
    )
    assert _candidate_space(parent_set=empty) == ()


def test_candidate_space_rejects_target_region_outside_frozen_region_universe() -> None:
    with pytest.raises(MatchedSparsityMappingError, match="target_region"):
        _candidate_space(region_ids=("jaw", "eye"))


def test_candidate_space_rejects_unknown_selected_target_component() -> None:
    parent_set = ParentSet(
        outer_fold=4,
        target_region="mouth",
        parents=(ParentLink("jaw", 2, "vx", "vz"),),
        discovery_method="pcmci_plus",
    )
    with pytest.raises(MatchedSparsityMappingError, match="target dimensions"):
        _candidate_space(parent_set=parent_set)


def test_candidate_space_rejects_outer_test_construction_scope() -> None:
    with pytest.raises(LeakageGuardError, match="outer-test"):
        build_matched_sparsity_candidate_space(
            _manifest(),
            _parent_set(),
            construction_subject_ids=("train_a", "test_a"),
            region_ids=("mouth", "jaw", "eye"),
            dimension_ids=("vx", "vy"),
        )


def test_candidate_space_rejects_parentset_fold_mismatch() -> None:
    wrong_fold = ParentSet(
        outer_fold=5,
        target_region="mouth",
        parents=_parent_set().parents,
        discovery_method="pcmci_plus",
    )
    with pytest.raises(MatchedSparsityMappingError, match="outer_fold"):
        _candidate_space(parent_set=wrong_fold)
