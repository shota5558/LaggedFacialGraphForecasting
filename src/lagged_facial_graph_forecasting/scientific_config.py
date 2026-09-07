"""Strict loader for the Phase P0A scientific freeze.

This module intentionally validates the scientific freeze against exact values.
Changing these values is a scientific-specification change, not a routine runtime
configuration change, and therefore requires an explicit migration and test update.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ScientificConfigError(ValueError):
    """Raised when the scientific freeze is missing, malformed, or altered."""


_EXPECTED: dict[str, Any] = {
    "schema_version": 3,
    "primary": {
        "discovery": "pcmci_plus",
        "ci_test": "parcorr",
        "forecaster": "ridge",
        "horizon": 1,
        "tau_max": 10,
        "conditions": [
            "persistence",
            "self",
            "full",
            "pcmci",
            "lag_shift",
            "random_region",
            "matched_sparsity",
            "time_shuffle",
        ],
    },
    "discovery_representation": {
        "node_unit": "region_dimension",
        "component_mapping": "identity",
        "forecasting_feature_rule": "exact_selected_component",
        "reporting_projection": "region_lag",
    },
    "evaluation": {
        "split_unit": "subject",
        "outer_test_usage": "frozen_final_evaluation_only",
    },
    "selection_scope": {
        "preprocessing_fit": "outer_train_only",
        "discovery": "outer_train_only",
        "ridge_tuning": "outer_train_only",
        "null_mapping": "outer_train_only",
        "feature_selection": "outer_train_only",
        "lag_selection": "outer_train_only",
    },
    "sensitivity": {
        "execute_after_primary_freeze": True,
        "methods": [
            "gpdc",
            "lpcmci",
            "phase_shuffled_surrogate",
            "circular_shift_surrogate",
            "horizon_gt_1",
            "gru",
        ],
    },
    "artifacts": {
        "primary_root": "artifacts/primary",
        "sensitivity_root": "artifacts/sensitivity",
    },
}


def _validate_exact(actual: Any, expected: Any, path: str = "root") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ScientificConfigError(f"{path} must be a mapping")
        expected_keys = set(expected)
        actual_keys = set(actual)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ScientificConfigError(
                f"{path} keys differ from scientific freeze; missing={missing}, extra={extra}"
            )
        for key, expected_value in expected.items():
            _validate_exact(actual[key], expected_value, f"{path}.{key}")
        return

    if isinstance(expected, list):
        if not isinstance(actual, list):
            raise ScientificConfigError(f"{path} must be a list")
        if actual != expected:
            raise ScientificConfigError(
                f"{path} differs from scientific freeze: expected={expected!r}, actual={actual!r}"
            )
        return

    if actual != expected:
        raise ScientificConfigError(
            f"{path} differs from scientific freeze: expected={expected!r}, actual={actual!r}"
        )


def load_scientific_config(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the P0A scientific freeze YAML file."""

    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ScientificConfigError(f"scientific config not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ScientificConfigError(f"invalid YAML in scientific config: {config_path}") from exc

    if not isinstance(raw, dict):
        raise ScientificConfigError("scientific config root must be a mapping")

    _validate_exact(raw, _EXPECTED)

    if raw["artifacts"]["primary_root"] == raw["artifacts"]["sensitivity_root"]:
        raise ScientificConfigError("Primary and Sensitivity artifact roots must be separate")

    return raw
