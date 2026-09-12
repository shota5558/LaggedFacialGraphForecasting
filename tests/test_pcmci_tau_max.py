from __future__ import annotations

import pytest

from lagged_facial_graph_forecasting.pcmci_tau_max import (
    PRIMARY_FORECAST_HORIZON,
    PRIMARY_TAU_MAX,
    PRIMARY_TAU_MAX_SCHEMA_VERSION,
    freeze_primary_tau_max,
)


def test_primary_tau_max_is_frozen_to_10() -> None:
    frozen = freeze_primary_tau_max()

    assert PRIMARY_FORECAST_HORIZON == 1
    assert PRIMARY_TAU_MAX == 10
    assert frozen.tau_max == 10
    assert frozen.schema_version == PRIMARY_TAU_MAX_SCHEMA_VERSION == 2
    assert dict(frozen.tigramite_kwargs()) == {"tau_max": 10}


def test_primary_tau_max_rejects_scientific_drift() -> None:
    with pytest.raises(ValueError, match="scientifically frozen to 10"):
        freeze_primary_tau_max(11)


def test_primary_tau_max_rejects_non_integer_input() -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        freeze_primary_tau_max("10")  # type: ignore[arg-type]
