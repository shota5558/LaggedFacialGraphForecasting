from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from lagged_facial_graph_forecasting import ScientificConfigError, load_scientific_config


CONFIG_PATH = Path("configs/scientific_freeze.yaml")
SCHEMA_PATH = Path("schemas/scientific_freeze.schema.json")


def _load_raw() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "scientific_freeze.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_scientific_freeze_loads() -> None:
    config = load_scientific_config(CONFIG_PATH)

    assert config["schema_version"] == 5
    assert config["primary"]["discovery"] == "pcmci_plus"
    assert config["primary"]["ci_test"] == "parcorr"
    assert config["primary"]["forecaster"] == "ridge"
    assert config["primary"]["horizon"] == 1
    assert config["primary"]["tau_max"] == 10
    assert config["primary"]["pc_alpha"] == 0.01
    assert config["primary"]["lag_response"] == {
        "delta_frames": [-2, -1, 0, 1, 2],
        "reference_delta": 0,
        "shift_mode": "common_shift_all_selected_parents",
        "valid_lag_min": 1,
        "valid_lag_max": 10,
        "boundary_policy": "mark_target_fold_unevaluable",
        "clipping": "forbidden",
        "wrapping": "forbidden",
        "feature_dropping": "forbidden",
        "one_sided_grid": "forbidden",
        "evaluability": {
            "require_complete_symmetric_grid": True,
            "empty_parent_set": "unevaluable_no_parents",
        },
        "aggregation": {
            "support": "complete_grid_target_folds_only",
            "same_units_across_all_deltas": True,
            "report_unevaluable_counts": True,
        },
    }
    assert config["discovery_representation"]["node_unit"] == "region_dimension"
    assert config["discovery_representation"]["component_mapping"] == "identity"
    assert (
        config["discovery_representation"]["forecasting_feature_rule"]
        == "exact_selected_component"
    )
    assert config["discovery_representation"]["reporting_projection"] == "region_lag"
    assert config["evaluation"]["split_unit"] == "subject"
    assert config["evaluation"]["outer_test_usage"] == "frozen_final_evaluation_only"
    assert config["sensitivity"]["execute_after_primary_freeze"] is True


def test_json_schema_freezes_lag_response_contract() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert schema["properties"]["schema_version"]["const"] == 5
    lag_response = schema["properties"]["primary"]["properties"]["lag_response"]["properties"]
    assert lag_response["delta_frames"]["const"] == [-2, -1, 0, 1, 2]
    assert lag_response["reference_delta"]["const"] == 0
    assert lag_response["shift_mode"]["const"] == "common_shift_all_selected_parents"
    assert lag_response["valid_lag_min"]["const"] == 1
    assert lag_response["valid_lag_max"]["const"] == 10
    assert lag_response["boundary_policy"]["const"] == "mark_target_fold_unevaluable"
    assert lag_response["one_sided_grid"]["const"] == "forbidden"
    assert (
        lag_response["evaluability"]["properties"]["empty_parent_set"]["const"]
        == "unevaluable_no_parents"
    )
    assert (
        lag_response["aggregation"]["properties"]["support"]["const"]
        == "complete_grid_target_folds_only"
    )


def test_rejects_primary_drift(tmp_path: Path) -> None:
    config = deepcopy(_load_raw())
    config["primary"]["ci_test"] = "gpdc"

    with pytest.raises(ScientificConfigError, match="primary.ci_test"):
        load_scientific_config(_write_config(tmp_path, config))


def test_rejects_tau_max_drift(tmp_path: Path) -> None:
    config = deepcopy(_load_raw())
    config["primary"]["tau_max"] = 9

    with pytest.raises(ScientificConfigError, match="primary.tau_max"):
        load_scientific_config(_write_config(tmp_path, config))


def test_rejects_pc_alpha_drift(tmp_path: Path) -> None:
    config = deepcopy(_load_raw())
    config["primary"]["pc_alpha"] = 0.05

    with pytest.raises(ScientificConfigError, match="primary.pc_alpha"):
        load_scientific_config(_write_config(tmp_path, config))


def test_rejects_pc_alpha_selection_mode(tmp_path: Path) -> None:
    config = deepcopy(_load_raw())
    config["primary"]["pc_alpha"] = None

    with pytest.raises(ScientificConfigError, match="primary.pc_alpha"):
        load_scientific_config(_write_config(tmp_path, config))


def test_rejects_lag_response_delta_drift(tmp_path: Path) -> None:
    config = deepcopy(_load_raw())
    config["primary"]["lag_response"]["delta_frames"] = [-1, 0, 1]

    with pytest.raises(ScientificConfigError, match="primary.lag_response.delta_frames"):
        load_scientific_config(_write_config(tmp_path, config))


def test_rejects_lag_response_boundary_policy_drift(tmp_path: Path) -> None:
    config = deepcopy(_load_raw())
    config["primary"]["lag_response"]["boundary_policy"] = "clip"

    with pytest.raises(ScientificConfigError, match="primary.lag_response.boundary_policy"):
        load_scientific_config(_write_config(tmp_path, config))


def test_rejects_one_sided_lag_response_policy(tmp_path: Path) -> None:
    config = deepcopy(_load_raw())
    config["primary"]["lag_response"]["one_sided_grid"] = "allowed"

    with pytest.raises(ScientificConfigError, match="primary.lag_response.one_sided_grid"):
        load_scientific_config(_write_config(tmp_path, config))


def test_rejects_lag_response_aggregation_support_drift(tmp_path: Path) -> None:
    config = deepcopy(_load_raw())
    config["primary"]["lag_response"]["aggregation"]["support"] = "available_points"

    with pytest.raises(
        ScientificConfigError,
        match="primary.lag_response.aggregation.support",
    ):
        load_scientific_config(_write_config(tmp_path, config))


def test_rejects_discovery_node_scalarization_drift(tmp_path: Path) -> None:
    config = deepcopy(_load_raw())
    config["discovery_representation"]["node_unit"] = "region"

    with pytest.raises(ScientificConfigError, match="discovery_representation.node_unit"):
        load_scientific_config(_write_config(tmp_path, config))


def test_rejects_component_feature_rule_drift(tmp_path: Path) -> None:
    config = deepcopy(_load_raw())
    config["discovery_representation"]["forecasting_feature_rule"] = "all_region_dimensions"

    with pytest.raises(
        ScientificConfigError,
        match="discovery_representation.forecasting_feature_rule",
    ):
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
