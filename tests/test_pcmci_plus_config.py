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
        "link_assumptions": None,
        "tau_min": 0,
        "contemp_collider_rule": "majority",
        "conflict_resolution": True,
        "reset_lagged_links": False,
        "max_conds_dim": None,
        "max_combinations": 1,
        "max_conds_py": None,
        "max_conds_px": None,
        "max_conds_px_lagged": None,
        "fdr_method": "none",
    }
    assert "tau_max" not in kwargs
    assert "pc_alpha" not in kwargs


def test_primary_pcmci_plus_config_rejects_runtime_drift() -> None:
    with pytest.raises(ValueError, match="collider rule"):
        PrimaryPCMCIPlusConfig(contemp_collider_rule="conservative")
