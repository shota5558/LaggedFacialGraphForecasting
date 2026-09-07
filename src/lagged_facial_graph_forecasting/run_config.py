"""Strict loader for the minimal Primary runtime configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from .core_contracts import CORE_CONTRACT_SCHEMA_VERSION, ExperimentConfig
from .scientific_config import ScientificConfigError, load_scientific_config


class RunConfigError(ValueError):
    """Raised when the Primary runtime configuration is invalid or unsafe."""


_REQUIRED_KEYS = {
    "schema_version",
    "experiment_id",
    "scientific_config_path",
    "artifact_root",
    "seed",
}


def _strict_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RunConfigError("run config root must be a mapping")
    actual = set(value)
    if actual != _REQUIRED_KEYS:
        missing = sorted(_REQUIRED_KEYS - actual)
        extra = sorted(actual - _REQUIRED_KEYS)
        raise RunConfigError(
            f"run config fields are incompatible; missing={missing}, extra={extra}"
        )
    return value


def load_primary_run_config(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> ExperimentConfig:
    """Load the minimal Primary run envelope and validate scientific separation.

    The scientific freeze remains authoritative.  This runtime config may choose
    only identity, seed, and the Primary artifact root; it cannot redefine the
    Primary method, horizon, split scope, or Sensitivity policy.
    """

    root = Path(repository_root) if repository_root is not None else Path.cwd()
    root = root.resolve()
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()
    try:
        relative_config_path = config_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise RunConfigError("run config must be inside repository_root") from exc

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RunConfigError(f"run config not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise RunConfigError(f"invalid YAML in run config: {config_path}") from exc

    values = _strict_mapping(raw)
    if values["schema_version"] != CORE_CONTRACT_SCHEMA_VERSION:
        raise RunConfigError(
            "run config schema_version must match the frozen core contract version"
        )
    if values["experiment_id"] != "primary":
        raise RunConfigError("H-00 accepts only the Primary experiment configuration")

    scientific_path = values["scientific_config_path"]
    if not isinstance(scientific_path, str) or not scientific_path.strip():
        raise RunConfigError("scientific_config_path must be a non-empty string")
    scientific_file = root / scientific_path
    try:
        scientific = load_scientific_config(scientific_file)
    except ScientificConfigError as exc:
        raise RunConfigError("referenced scientific freeze is invalid") from exc

    artifact_root = values["artifact_root"]
    if artifact_root != scientific["artifacts"]["primary_root"]:
        raise RunConfigError(
            "Primary run artifact_root must equal the scientific freeze primary_root"
        )

    seed = values["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise RunConfigError("seed must be a non-negative integer")

    try:
        return ExperimentConfig(
            experiment_id=values["experiment_id"],
            scientific_config_path=scientific_path,
            run_config_path=relative_config_path,
            artifact_root=artifact_root,
            seed=seed,
        )
    except Exception as exc:
        raise RunConfigError("run config does not satisfy ExperimentConfig contract") from exc
