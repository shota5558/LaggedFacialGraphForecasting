from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from lagged_facial_graph_forecasting import ScientificConfigError, load_scientific_config


CONFIG_PATH = Path("configs/scientific_freeze.yaml")


def _load_raw() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "scientific_freeze.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_scientific_freeze_loads() -> None:
    config = load_scientific_config(CONFIG_PATH)

    assert config["primary"]["discovery"] == "pcmci_plus"
    assert config["primary"]["ci_test"] == "parcorr"
    assert config["primary"]["forecaster"] == "ridge"
    assert config["primary"]["horizon"] == 1
    assert config["evaluation"]["split_unit"] == "subject"
    assert config["evaluation"]["outer_test_usage"] == "frozen_final_evaluation_only"
    assert config["sensitivity"]["execute_after_primary_freeze"] is True


def test_rejects_primary_drift(tmp_path: Path) -> None:
    config = deepcopy(_load_raw())
    config["primary"]["ci_test"] = "gpdc"

    with pytest.raises(ScientificConfigError, match="primary.ci_test"):
        load_scientific_config(_write_config(tmp_path, config))


def test_rejects_outer_test_selection_leakage(tmp_path: Path) -> None:
    config = deepcopy(_load_raw())
    config["selection_scope"]["discovery"] = "all_data"

    with pytest.raises(ScientificConfigError, match="selection_scope.discovery"):
        load_scientific_config(_write_config(tmp_path, config))


def test_rejects_outer_test_usage_drift(tmp_path: Path) -> None:
    config = deepcopy(_load_raw())
    config["evaluation"]["outer_test_usage"] = "model_selection"

    with pytest.raises(ScientificConfigError, match="evaluation.outer_test_usage"):
        load_scientific_config(_write_config(tmp_path, config))


def test_rejects_sensitivity_before_primary_freeze(tmp_path: Path) -> None:
    config = deepcopy(_load_raw())
    config["sensitivity"]["execute_after_primary_freeze"] = False

    with pytest.raises(ScientificConfigError, match="sensitivity.execute_after_primary_freeze"):
        load_scientific_config(_write_config(tmp_path, config))


def test_rejects_unreviewed_schema_extension(tmp_path: Path) -> None:
    config = deepcopy(_load_raw())
    config["primary"]["experimental_model"] = "gru"

    with pytest.raises(ScientificConfigError, match="keys differ from scientific freeze"):
        load_scientific_config(_write_config(tmp_path, config))
