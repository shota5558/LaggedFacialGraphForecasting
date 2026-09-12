from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from lagged_facial_graph_forecasting import (
    ScientificConfigError,
    load_scientific_config,
)


CONFIG_PATH = Path("configs/scientific_freeze.yaml")
SCHEMA_PATH = Path("schemas/scientific_freeze.schema.json")


def _load_raw() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "scientific_freeze.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_adopted_scientific_config_loads_in_explicit_preflight_state() -> None:
    config = load_scientific_config(CONFIG_PATH)

    assert config["schema_version"] == 7
    assert config["protocol_id"] == "primary-2026-09-12-adopted"
    assert config["protocol_status"] == "preflight_required"
    assert config["primary"]["horizon"] == 1
    assert config["primary"]["discovery"]["method"] == "pcmci_plus"
    assert config["primary"]["discovery"]["ci_test"] == "parcorr"
    assert config["primary"]["discovery"]["tau_max"] is None
    assert config["primary"]["discovery"]["tau_max_rule"] == "floor_half_fps"
    assert config["primary"]["matched_sparsity"]["repeat_count"] == 1000
    assert config["primary"]["lag_response"]["delta_frames"] is None
    assert config["primary"]["lag_response"]["delta_rule"] == "symmetric_floor_tenth_fps"
    assert config["primary"]["representation"]["forecast_projection"] == "region_block_or"
    assert config["primary"]["statistics"]["bootstrap_unit"] == "dependency_group"


def test_json_schema_names_adopted_protocol() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert schema["properties"]["schema_version"]["const"] == 7
    assert schema["properties"]["protocol_id"]["const"] == "primary-2026-09-12-adopted"
    assert schema["properties"]["compatibility"]["properties"]["legacy_artifacts"]["const"] == "reject"
    primary = schema["properties"]["primary"]["properties"]
    assert primary["matched_sparsity"]["properties"]["repeat_count"]["const"] == 1000
    assert primary["discovery"]["properties"]["tau_max_rule"]["const"] == "floor_half_fps"
    assert primary["lag_response"]["properties"]["delta_rule"]["const"] == "symmetric_floor_tenth_fps"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("protocol_id",), "scientific-freeze-v6"),
        (("primary", "horizon"), 2),
        (("primary", "discovery", "ci_test"), "gpdc"),
        (("primary", "matched_sparsity", "repeat_count"), 100),
        (("primary", "representation", "forecast_projection"), "identity"),
        (("split", "unit"), "subject"),
    ],
)
def test_rejects_adopted_protocol_drift(tmp_path: Path, path: tuple[str, ...], value: object) -> None:
    config = deepcopy(_load_raw())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ScientificConfigError, match="adopted protocol"):
        load_scientific_config(_write_config(tmp_path, config), repository_root=tmp_path)


def test_rejects_legacy_schema_version(tmp_path: Path) -> None:
    config = deepcopy(_load_raw())
    config["schema_version"] = 6

    with pytest.raises(ScientificConfigError, match="schema_version"):
        load_scientific_config(_write_config(tmp_path, config), repository_root=tmp_path)


def test_rejects_unreviewed_schema_extension(tmp_path: Path) -> None:
    config = deepcopy(_load_raw())
    config["primary"]["unreviewed_option"] = True

    with pytest.raises(ScientificConfigError, match="keys differ"):
        load_scientific_config(_write_config(tmp_path, config), repository_root=tmp_path)


def test_real_materialization_is_not_implied_by_preflight_config(tmp_path: Path) -> None:
    with pytest.raises(ScientificConfigError, match="preflight facts"):
        load_scientific_config(
            CONFIG_PATH.resolve(),
            require_materialized=True,
        )
