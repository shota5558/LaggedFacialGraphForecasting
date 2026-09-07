from __future__ import annotations

import pytest

from lagged_facial_graph_forecasting.pcmci_plus_config import (
    PCMCI_PLUS_CONFIG_SCHEMA_VERSION,
    PrimaryPCMCIPlusConfig,
    make_primary_pcmci_plus_config,
)


def test_primary_pcmci_plus_static_config_is_explicit_and_excludes_free_parameters() -> None:
    config = make_primary_pcmci_plus_config()
    kwargs = config.tigramite_kwargs()

    assert config.schema_version == PCMCI_PLUS_CONFIG_SCHEMA_VERSION == 1
    assert dict(kwargs) == {
        "tau_min": 0,
        "contemp_collider_rule": "majority",
        "conflict_resolution": True,
        "reset_lagged_links": False,
        "max_combinations": 1,
        "fdr_method": "none",
    }
    assert "tau_max" not in kwargs
    assert "pc_alpha" not in kwargs


def test_primary_pcmci_plus_static_config_rejects_primary_drift() -> None:
    with pytest.raises(ValueError, match="tau_min"):
        PrimaryPCMCIPlusConfig(tau_min=1)
    with pytest.raises(ValueError, match="collider rule"):
        PrimaryPCMCIPlusConfig(contemp_collider_rule="conservative")
    with pytest.raises(ValueError, match="conflict_resolution"):
        PrimaryPCMCIPlusConfig(conflict_resolution=False)
    with pytest.raises(ValueError, match="reset_lagged_links"):
        PrimaryPCMCIPlusConfig(reset_lagged_links=True)
    with pytest.raises(ValueError, match="max_combinations"):
        PrimaryPCMCIPlusConfig(max_combinations=2)
    with pytest.raises(ValueError, match="fdr_method"):
        PrimaryPCMCIPlusConfig(fdr_method="fdr_bh")
    with pytest.raises(ValueError, match="schema_version"):
        PrimaryPCMCIPlusConfig(schema_version=2)


def test_tigramite_kwargs_mapping_is_immutable() -> None:
    kwargs = make_primary_pcmci_plus_config().tigramite_kwargs()

    with pytest.raises(TypeError):
        kwargs["tau_min"] = 1  # type: ignore[index]
