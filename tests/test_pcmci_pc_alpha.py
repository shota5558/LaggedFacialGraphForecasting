from __future__ import annotations

import pytest

from lagged_facial_graph_forecasting.pcmci_pc_alpha import (
    PRIMARY_PC_ALPHA,
    PRIMARY_PC_ALPHA_SCHEMA_VERSION,
    freeze_primary_pc_alpha,
)


def test_primary_pc_alpha_is_frozen_to_001() -> None:
    frozen = freeze_primary_pc_alpha()

    assert PRIMARY_PC_ALPHA == 0.01
    assert frozen.pc_alpha == 0.01
    assert frozen.schema_version == PRIMARY_PC_ALPHA_SCHEMA_VERSION == 1
    assert dict(frozen.tigramite_kwargs()) == {"pc_alpha": 0.01}


def test_primary_pc_alpha_rejects_scientific_drift() -> None:
    with pytest.raises(ValueError, match="scientifically frozen to 0.01"):
        freeze_primary_pc_alpha(0.05)


def test_primary_pc_alpha_rejects_non_scalar_selection_mode() -> None:
    with pytest.raises(TypeError, match="must be a real scalar"):
        freeze_primary_pc_alpha([0.01])  # type: ignore[arg-type]
