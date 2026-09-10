from __future__ import annotations

import pytest

from lagged_facial_graph_forecasting.pcmci_pc_alpha import (
    PRIMARY_PC_ALPHA,
    PRIMARY_PC_ALPHA_SCHEMA_VERSION,
    PrimaryPCAlpha,
    freeze_primary_pc_alpha,
)


def test_primary_pc_alpha_is_finally_frozen_to_001() -> None:
    frozen = freeze_primary_pc_alpha()

    assert PRIMARY_PC_ALPHA == 0.01
    assert frozen.pc_alpha == 0.01
    assert frozen.schema_version == PRIMARY_PC_ALPHA_SCHEMA_VERSION == 1
    assert dict(frozen.tigramite_kwargs()) == {"pc_alpha": 0.01}


def test_explicit_matching_pc_alpha_is_accepted() -> None:
    frozen = freeze_primary_pc_alpha(0.01)

    assert frozen.pc_alpha == 0.01


@pytest.mark.parametrize("bad_value", [0.0, 0.001, 0.005, 0.02, 0.05, 0.1, 1.0])
def test_primary_pc_alpha_rejects_any_numeric_drift(bad_value: float) -> None:
    with pytest.raises(ValueError, match="scientifically frozen to 0.01"):
        freeze_primary_pc_alpha(bad_value)


@pytest.mark.parametrize("bad_value", [True, False, "0.01", None, [0.01], (0.01,)])
def test_primary_pc_alpha_rejects_non_scalar_or_selection_modes(bad_value: object) -> None:
    with pytest.raises(TypeError, match="must be a real scalar"):
        freeze_primary_pc_alpha(bad_value)  # type: ignore[arg-type]


def test_primary_pc_alpha_rejects_schema_drift() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        PrimaryPCAlpha(pc_alpha=0.01, schema_version=2)


def test_pc_alpha_tigramite_mapping_is_immutable() -> None:
    kwargs = freeze_primary_pc_alpha().tigramite_kwargs()

    with pytest.raises(TypeError):
        kwargs["pc_alpha"] = 0.05  # type: ignore[index]


def test_pc_alpha_contract_exposes_no_data_or_selection_inputs() -> None:
    # D-05 must not offer a path that can infer pc_alpha from observed outcomes.
    assert set(PrimaryPCAlpha.__dataclass_fields__) == {"pc_alpha", "schema_version"}
