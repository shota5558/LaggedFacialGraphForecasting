from __future__ import annotations

from lagged_facial_graph_forecasting.core_contract_io import (
    dumps_core_contract,
    loads_core_contract,
    serialize_core_contract,
)
from lagged_facial_graph_forecasting.core_contracts import ParentSet
from lagged_facial_graph_forecasting.pcmci_link_extraction import SignificantPCMCIPlusLink
from lagged_facial_graph_forecasting.pcmci_parent_set import (
    convert_primary_lagged_links_to_parent_set,
)


def _link(
    *,
    source_region: str,
    source_dimension: str,
    target_dimension: str,
    lag: int,
    source_node: int,
) -> SignificantPCMCIPlusLink:
    return SignificantPCMCIPlusLink(
        source_node_index=source_node,
        target_node_index=4,
        lag=lag,
        link_mark="-->",
        p_value=0.004,
        test_statistic=0.42,
        source_variable_name=f"{source_region}::{source_dimension}",
        target_variable_name=f"mouth::{target_dimension}",
        source_region=source_region,
        source_dimension=source_dimension,
        target_region="mouth",
        target_dimension=target_dimension,
    )


def test_d10_parent_set_round_trips_through_existing_core_contract_json() -> None:
    parent_set = convert_primary_lagged_links_to_parent_set(
        (
            _link(
                source_region="right_cheek",
                source_dimension="vy",
                target_dimension="vx",
                lag=3,
                source_node=3,
            ),
            _link(
                source_region="left_cheek",
                source_dimension="vx",
                target_dimension="vy",
                lag=2,
                source_node=2,
            ),
        ),
        outer_fold=2,
        target_region="mouth",
    )

    text = dumps_core_contract(parent_set)
    restored = loads_core_contract(text)

    assert isinstance(restored, ParentSet)
    assert restored == parent_set
    assert dumps_core_contract(restored) == text


def test_parent_set_wire_format_contains_only_canonical_forecast_provenance() -> None:
    parent_set = convert_primary_lagged_links_to_parent_set(
        (
            _link(
                source_region="left_cheek",
                source_dimension="vx",
                target_dimension="vy",
                lag=2,
                source_node=2,
            ),
        ),
        outer_fold=1,
        target_region="mouth",
    )

    envelope = serialize_core_contract(parent_set)

    assert envelope["contract_type"] == "ParentSet"
    assert envelope["payload"] == {
        "outer_fold": 1,
        "target_region": "mouth",
        "parents": [
            {
                "source_region": "left_cheek",
                "lag": 2,
                "source_dimension": "vx",
                "target_dimension": "vy",
            }
        ],
        "discovery_method": "pcmci_plus",
    }
    serialized = dumps_core_contract(parent_set)
    assert "p_value" not in serialized
    assert "test_statistic" not in serialized
    assert "node_index" not in serialized
    assert "link_mark" not in serialized


def test_empty_d10_parent_set_round_trips_without_special_case() -> None:
    parent_set = convert_primary_lagged_links_to_parent_set(
        (), outer_fold=3, target_region="mouth"
    )

    assert loads_core_contract(dumps_core_contract(parent_set)) == parent_set
