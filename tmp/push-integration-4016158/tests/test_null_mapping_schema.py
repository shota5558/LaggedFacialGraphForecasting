from __future__ import annotations

from lagged_facial_graph_forecasting.core_contract_io import dumps_core_contract, loads_core_contract
from lagged_facial_graph_forecasting.core_contracts import NullMapping, ParentLink


def test_null_mapping_schema_preserves_component_aware_parent_mapping_and_seed() -> None:
    mapping = NullMapping(
        outer_fold=2,
        target_region="mouth",
        condition="lag-shift",
        seed=20260908,
        source_parents=(
            ParentLink("left_cheek", 2, "vx", "vy"),
            ParentLink("jaw", 3, "vy", "vx"),
        ),
        mapped_parents=(
            ParentLink("left_cheek", 4, "vx", "vy"),
            ParentLink("jaw", 5, "vy", "vx"),
        ),
    )

    restored = loads_core_contract(dumps_core_contract(mapping))

    assert restored == mapping
    assert restored.seed == 20260908
    assert restored.source_parents[0].source_dimension == "vx"
    assert restored.source_parents[0].target_dimension == "vy"
    assert restored.mapped_parents[0].source_region == "left_cheek"
    assert restored.mapped_parents[0].lag == 4
    assert restored.permutation == ()
