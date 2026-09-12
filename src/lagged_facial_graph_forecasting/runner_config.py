"""Primary runner descriptor and adopted-protocol execution gate."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from .contracts import ContractError
from .core_contracts import ExperimentConfig
from .scientific_config import (
    LEGACY_PRIMARY_ARTIFACT_MANIFEST,
    PRIMARY_PROTOCOL_ID,
    ScientificConfigError,
    load_scientific_config,
)


class RunnerConfigError(ValueError):
    """Raised when a Primary run descriptor violates the adopted protocol."""


class PrimaryExecutionMode(str, Enum):
    """Execution states; only ``real`` requires materialized preflight facts."""

    PREFLIGHT = "preflight"
    SYNTHETIC = "synthetic"
    REAL = "real"


@dataclass(frozen=True, slots=True)
class PrimaryRunConfig:
    """Run descriptor plus the protocol identity that governs the run."""

    experiment: ExperimentConfig
    protocol_id: str
    execution_mode: PrimaryExecutionMode
    preflight_manifest_path: str | None
    executable_freeze_path: str | None
    run_schema_version: int

    def __getattr__(self, name: str) -> Any:
        # Preserve the old loader's convenient ExperimentConfig attribute API.
        return getattr(self.experiment, name)


_REQUIRED_KEYS = {
    "schema_version",
    "experiment_id",
    "scientific_config_path",
    "run_config_path",
    "artifact_root",
    "seed",
    "protocol_id",
    "execution_mode",
    "preflight_manifest_path",
    "executable_freeze_path",
}
_LEGACY_REQUIRED_KEYS = {
    "schema_version",
    "experiment_id",
    "scientific_config_path",
    "run_config_path",
    "artifact_root",
    "seed",
}
RUN_DESCRIPTOR_SCHEMA_VERSION = 4
CORE_EXPERIMENT_SCHEMA_VERSION = 3


def _within_repository(path: Path, repository_root: Path, field_name: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(repository_root)
    except ValueError as exc:
        raise RunnerConfigError(f"{field_name} must resolve inside repository_root") from exc
    return resolved


def _optional_relative_path(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RunnerConfigError(f"{field_name} must be null or a non-empty path")
    path = Path(value.strip())
    if path.is_absolute() or ".." in path.parts:
        raise RunnerConfigError(f"{field_name} must be repository-relative")
    return path.as_posix()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RunnerConfigError(f"run config not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise RunnerConfigError(f"invalid YAML in run config: {path}") from exc
    if not isinstance(raw, dict):
        raise RunnerConfigError("run config root must be a mapping")
    return raw


def _experiment(raw: dict[str, Any]) -> ExperimentConfig:
    payload = {
        key: raw[key]
        for key in (
            "schema_version",
            "experiment_id",
            "scientific_config_path",
            "run_config_path",
            "artifact_root",
            "seed",
        )
    }
    # The run descriptor has its own schema version; the embedded cross-lane
    # ExperimentConfig remains on core schema v3.
    payload["schema_version"] = CORE_EXPERIMENT_SCHEMA_VERSION
    try:
        return ExperimentConfig(**payload)
    except (ContractError, TypeError, ValueError) as exc:
        raise RunnerConfigError(f"invalid ExperimentConfig: {exc}") from exc


def load_primary_experiment_config(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> PrimaryRunConfig:
    """Load the adopted Primary descriptor and enforce its real-run gate.

    A legacy six-field descriptor is accepted only as a preflight API
    compatibility shim. It cannot activate real execution and is never treated
    as a migrated run descriptor.
    """

    root = Path(repository_root or Path.cwd()).resolve()
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = _within_repository(config_path, root, "config path")
    raw = _load_yaml(config_path)

    actual_keys = set(raw)
    legacy = actual_keys == _LEGACY_REQUIRED_KEYS
    if not legacy and actual_keys != _REQUIRED_KEYS:
        missing = sorted(_REQUIRED_KEYS - actual_keys)
        extra = sorted(actual_keys - _REQUIRED_KEYS)
        raise RunnerConfigError(
            f"run config keys differ from adopted descriptor; missing={missing}, extra={extra}"
        )
    if not legacy and raw["schema_version"] != RUN_DESCRIPTOR_SCHEMA_VERSION:
        raise RunnerConfigError(
            f"run descriptor schema_version must be {RUN_DESCRIPTOR_SCHEMA_VERSION}"
        )

    experiment = _experiment(raw)
    expected_run_path = config_path.relative_to(root).as_posix()
    if experiment.run_config_path != expected_run_path:
        raise RunnerConfigError(
            "run_config_path must identify the loaded run descriptor exactly: "
            f"expected={expected_run_path!r}, actual={experiment.run_config_path!r}"
        )

    scientific_path = _within_repository(
        root / experiment.scientific_config_path,
        root,
        "scientific_config_path",
    )
    try:
        scientific = load_scientific_config(
            scientific_path,
            require_materialized=(not legacy and raw["execution_mode"] == PrimaryExecutionMode.REAL.value),
            repository_root=root,
        )
    except ScientificConfigError as exc:
        raise RunnerConfigError(f"invalid adopted scientific config: {exc}") from exc

    protocol_id = scientific["protocol_id"] if legacy else raw["protocol_id"]
    if protocol_id != PRIMARY_PROTOCOL_ID or protocol_id != scientific["protocol_id"]:
        raise RunnerConfigError(
            f"protocol_id must equal adopted Primary protocol {PRIMARY_PROTOCOL_ID!r}"
        )
    if experiment.artifact_root != scientific["artifacts"]["primary_root"]:
        raise RunnerConfigError(
            "Primary runner artifact_root must equal the adopted Primary artifact root: "
            f"expected={scientific['artifacts']['primary_root']!r}, actual={experiment.artifact_root!r}"
        )

    if legacy:
        mode = PrimaryExecutionMode.PREFLIGHT
        preflight_path = scientific["preflight"]["dataset_manifest_path"]
        freeze_path = scientific["preflight"]["executable_freeze_path"]
        descriptor_version = int(raw["schema_version"])
    else:
        try:
            mode = PrimaryExecutionMode(raw["execution_mode"])
        except (TypeError, ValueError) as exc:
            raise RunnerConfigError(
                "execution_mode must be explicitly 'preflight', 'synthetic', or 'real'"
            ) from exc
        preflight_path = _optional_relative_path(
            raw["preflight_manifest_path"], "preflight_manifest_path"
        )
        freeze_path = _optional_relative_path(
            raw["executable_freeze_path"], "executable_freeze_path"
        )
        if preflight_path != scientific["preflight"]["dataset_manifest_path"]:
            raise RunnerConfigError("run and scientific preflight manifest paths differ")
        if freeze_path != scientific["preflight"]["executable_freeze_path"]:
            raise RunnerConfigError("run and scientific executable freeze paths differ")
        descriptor_version = int(raw["schema_version"])

    if mode is PrimaryExecutionMode.REAL:
        if (root / LEGACY_PRIMARY_ARTIFACT_MANIFEST).is_file():
            raise RunnerConfigError(
                "legacy v6 PRIMARY FREEZE artifact is not valid for the adopted protocol"
            )
        if preflight_path is None or freeze_path is None:
            raise RunnerConfigError("real execution requires preflight and executable-freeze paths")
        for field_name, relative in (
            ("preflight_manifest_path", preflight_path),
            ("executable_freeze_path", freeze_path),
        ):
            if not _within_repository(root / relative, root, field_name).is_file():
                raise RunnerConfigError(f"{field_name} does not exist: {relative}")

    return PrimaryRunConfig(
        experiment=experiment,
        protocol_id=protocol_id,
        execution_mode=mode,
        preflight_manifest_path=preflight_path,
        executable_freeze_path=freeze_path,
        run_schema_version=descriptor_version,
    )
