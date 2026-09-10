from __future__ import annotations

import pytest

from lagged_facial_graph_forecasting.pcmci_tau_max import (
    PRIMARY_FORECAST_HORIZON,
    PRIMARY_TAU_MAX,
    PRIMARY_TAU_MAX_SCHEMA_VERSION,
    PrimaryTauMax,
    freeze_primary_tau_max,
)


def test_primary_tau_max_is_finally_frozen_to_10() -> None:
    frozen = freeze_primary_tau_max()

    assert PRIMARY_FORECAST_HORIZON == 1
    assert PRIMARY_TAU_MAX == 10
    assert frozen.tau_max == 10
    assert frozen.schema_version == PRIMARY_TAU_MAX_SCHEMA_VERSION == 2
    assert dict(frozen.tigramite_kwargs()) == {"tau_max": 10}


def test_explicit_matching_tau_max_is_accepted() -> None:
    frozen = freeze_primary_tau_max(10)

    assert frozen.tau_max == 10


@pytest.mark.parametrize("bad_value", [0, 1, 5, 9, 11, 20, -1])
def test_primary_tau_max_rejects_any_numeric_drift(bad_value: int) -> None:
    with pytest.raises(ValueError, match="scientifically frozen to 10"):
        freeze_primary_tau_max(bad_value)


@pytest.mark.parametrize("bad_value", [True, False, 10.0, "10", None])
def test_primary_tau_max_rejects_non_integer_values(bad_value: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        freeze_primary_tau_max(bad_value)  # type: ignore[arg-type]


def test_primary_tau_max_rejects_schema_drift() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        PrimaryTauMax(tau_max=10, schema_version=1)


def test_tau_max_tigramite_mapping_is_immutable() -> None:
    kwargs = freeze_primary_tau_max().tigramite_kwargs()

    with pytest.raises(TypeError):
        kwargs["tau_max"] = 3  # type: ignore[index]


def test_tau_max_contract_exposes_no_data_or_selection_inputs() -> None:
    # D-04 must not offer a path that can infer tau_max from observed outcomes.
    assert set(PrimaryTauMax.__dataclass_fields__) == {"tau_max", "schema_version"}
