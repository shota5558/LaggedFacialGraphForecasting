from __future__ import annotations

import pytest

from lagged_facial_graph_forecasting.core_contracts import ParentLink
from lagged_facial_graph_forecasting.pcmci_link_extraction import SignificantPCMCIPlusLink
from lagged_facial_graph_forecasting.pcmci_parent_set import (
    ParentSetConversionError,
    convert_primary_lagged_links_to_parent_set,
)


def _link(
    *,
    source_region: str = "left_cheek",
    source_dimension: str = "vx",
    target_region: str = "mouth",
    target_dimension: str = "vy",
    lag: int = 2,
    mark: str = "-->",
    source_node: int = 0,
    target_node: int = 1,
) -> SignificantPCMCIPlusLink:
    return SignificantPCMCIPlusLink(
        source_node_index=source_node,
        target_node_index=target_node,
        lag=lag,
        link_mark=mark,
        p_value=0.005,
        test_statistic=0.4,
        source_variable_name=f"{source_region}::{source_dimension}",
        target_variable_name=f"{target_region}::{target_dimension}",
        source_region=source_region,
        source_dimension=source_dimension,
        target_region=target_region,
        target_dimension=target_dimension,
    )


def test_converts_cross_region_links_preserving_component_and_lag_provenance() -> None:
    links = (
        _link(source_region="jaw", source_dimension="vy", target_dimension="vx", lag=3),
        _link(source_region="left_cheek", source_dimension="vx", target_dimension="vy", lag=2),
    )

    parent_set = convert_primary_lagged_links_to_parent_set(
        links, outer_fold=4, target_region="mouth"
    )

    assert parent_set.outer_fold == 4
    assert parent_set.target_region == "mouth"
    assert parent_set.discovery_method == "pcmci_plus"
    assert parent_set.parents == (
        ParentLink("jaw", 3, "vy", "vx"),
        ParentLink("left_cheek", 2, "vx", "vy"),
    )


def test_same_region_links_are_not_duplicated_beyond_fixed_self_history() -> None:
    same_region = _link(source_region="mouth", source_dimension="vx", target_dimension="vy")
    cross_region = _link(source_region="jaw", source_dimension="vy", target_dimension="vy")

    parent_set = convert_primary_lagged_links_to_parent_set(
        (same_region, cross_region), outer_fold=0, target_region="mouth"
    )

    assert parent_set.parents == (ParentLink("jaw", 2, "vy", "vy"),)


def test_links_for_other_target_regions_are_not_mixed_into_target_parent_set() -> None:
    mouth = _link(target_region="mouth", source_region="jaw", lag=2)
    eye = _link(target_region="left_eye", source_region="left_cheek", lag=1)

    parent_set = convert_primary_lagged_links_to_parent_set(
        (eye, mouth), outer_fold=1, target_region="mouth"
    )

    assert parent_set.parents == (ParentLink("jaw", 2, "vx", "vy"),)


def test_empty_selected_parent_set_is_valid_for_target_region() -> None:
    parent_set = convert_primary_lagged_links_to_parent_set(
        (), outer_fold=2, target_region="mouth"
    )

    assert parent_set.parents == ()


def test_parent_order_is_canonical_and_independent_of_raw_input_order() -> None:
    a = _link(source_region="jaw", source_dimension="vy", target_dimension="vx", lag=4)
    b = _link(source_region="left_cheek", source_dimension="vx", target_dimension="vx", lag=2)

    forward = convert_primary_lagged_links_to_parent_set(
        (a, b), outer_fold=0, target_region="mouth"
    )
    reverse = convert_primary_lagged_links_to_parent_set(
        (b, a), outer_fold=0, target_region="mouth"
    )

    assert forward == reverse


@pytest.mark.parametrize(
    ("lag", "mark", "message"),
    [
        (0, "-->", "1 <= tau <= 10"),
        (11, "-->", "1 <= tau <= 10"),
        (2, "<--", "only D-09 validated"),
        (2, "o-o", "only D-09 validated"),
    ],
)
def test_rejects_links_outside_d09_primary_domain(lag: int, mark: str, message: str) -> None:
    with pytest.raises(ParentSetConversionError, match=message):
        convert_primary_lagged_links_to_parent_set(
            (_link(lag=lag, mark=mark),), outer_fold=0, target_region="mouth"
        )


def test_rejects_noncanonical_raw_link_items() -> None:
    with pytest.raises(TypeError, match="SignificantPCMCIPlusLink"):
        convert_primary_lagged_links_to_parent_set(
            (_link(), object()),  # type: ignore[arg-type]
            outer_fold=0,
            target_region="mouth",
        )
