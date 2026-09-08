"""Sensitivity execution boundary for pre-Primary-Freeze implementation work.

Issue #22 permits Sensitivity code and synthetic evidence to be implemented before
PRIMARY FREEZE, while real-data Sensitivity execution remains blocked.  This
module keeps that boundary independent of the Primary runner so Primary semantics
cannot be relaxed as a side effect of preparing Sensitivity code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from .contracts import ContractError
from .core_contracts import ExperimentConfig
from .scientific_config import ScientificConfigError, load_scientific_config


class SensitivityExecutionError(ValueError):
    """Raised when a Sensitivity run violates the implementation boundary."""


class SensitivityExecutionMode(str, Enum):
    """Explicitly distinguish implementation-only synthetic runs from real runs."""

    SYNTHETIC = "synthetic"
    REAL = "real"


@dataclass(frozen=True, slots=True)
class SensitivityExperimentConfig:
    """Sensitivity-only wrapper around the frozen cross-lane ExperimentConfig."""

    experiment: ExperimentConfig
    namespace: str
    execution_mode: SensitivityExecutionMode
    primary_freeze_manifest: str

    @property
    def artifact_root(self) -> str:
        return self.experiment.artifact_root


_REQUIRED_KEYS = {
    "schema_version",
    "namespace",
    "execution_mode",
    "experiment_id",
    "scientific_config_path",
    "run_config_path",
    "artifact_root",
    "seed",
    "primary_freeze_manifest",
}
_EXPERIMENT_KEYS = {
    "schema_version",
    "experiment_id",
    "scientific_config_path",
    "run_config_path",
    "artifact_root",
    "seed",
}


def _inside(path: Path, parent: Path, field_name: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(parent.resolve())
    except ValueError as exc:
        raise SensitivityExecutionError(
            f"{field_name} must resolve inside {parent.resolve()}"
        ) from exc
    return resolved


def _relative_posix(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SensitivityExecutionError(f"{field_name} must be a non-empty string")
    path = PurePosixPath(value.strip())
    if path.is_absolute() or ".." in path.parts:
        raise SensitivityExecutionError(
            f"{field_name} must be a repository-relative POSIX path"
        )
    return path.as_posix()


def load_sensitivity_experiment_config(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> SensitivityExperimentConfig:
    """Load a Sensitivity descriptor without weakening the Primary runner contract."""

    root = Path(repository_root or Path.cwd()).resolve()
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = _inside(config_path, root, "config path")

    try:
        raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SensitivityExecutionError(
            f"Sensitivity run config not found: {config_path}"
        ) from exc
    except yaml.YAMLError as exc:
        raise SensitivityExecutionError(
            f"invalid YAML in Sensitivity run config: {config_path}"
        ) from exc

    if not isinstance(raw, dict):
        raise SensitivityExecutionError("Sensitivity run config root must be a mapping")
    actual_keys = set(raw)
    if actual_keys != _REQUIRED_KEYS:
        missing = sorted(_REQUIRED_KEYS - actual_keys)
        extra = sorted(actual_keys - _REQUIRED_KEYS)
        raise SensitivityExecutionError(
            "Sensitivity run config keys differ from the frozen boundary; "
            f"missing={missing}, extra={extra}"
        )

    if raw["namespace"] != "sensitivity":
        raise SensitivityExecutionError("Sensitivity namespace must equal 'sensitivity'")
    try:
        execution_mode = SensitivityExecutionMode(raw["execution_mode"])
    except (TypeError, ValueError) as exc:
        raise SensitivityExecutionError(
            "execution_mode must be explicitly 'synthetic' or 'real'"
        ) from exc

    experiment_payload = {key: raw[key] for key in _EXPERIMENT_KEYS}
    try:
        experiment = ExperimentConfig(**experiment_payload)
    except (ContractError, TypeError, ValueError) as exc:
        raise SensitivityExecutionError(f"invalid ExperimentConfig: {exc}") from exc

    expected_run_path = config_path.relative_to(root).as_posix()
    if experiment.run_config_path != expected_run_path:
        raise SensitivityExecutionError(
            "run_config_path must identify the loaded Sensitivity descriptor exactly: "
            f"expected={expected_run_path!r}, actual={experiment.run_config_path!r}"
        )

    scientific_path = _inside(
        root / experiment.scientific_config_path, root, "scientific_config_path"
    )
    try:
        scientific = load_scientific_config(scientific_path)
    except ScientificConfigError as exc:
        raise SensitivityExecutionError(
            f"invalid frozen scientific config: {exc}"
        ) from exc

    sensitivity_root = scientific["artifacts"]["sensitivity_root"]
    primary_root = scientific["artifacts"]["primary_root"]
    if experiment.artifact_root != sensitivity_root:
        raise SensitivityExecutionError(
            "Sensitivity artifact_root must equal the frozen Sensitivity artifact root: "
            f"expected={sensitivity_root!r}, actual={experiment.artifact_root!r}"
        )
    if experiment.artifact_root == primary_root:
        raise SensitivityExecutionError("Sensitivity cannot use the Primary artifact root")

    freeze_manifest = _relative_posix(
        raw["primary_freeze_manifest"], "primary_freeze_manifest"
    )
    freeze_path = _inside(root / freeze_manifest, root, "primary_freeze_manifest")
    primary_root_path = _inside(root / primary_root, root, "Primary artifact root")
    _inside(freeze_path, primary_root_path, "primary_freeze_manifest")

    return SensitivityExperimentConfig(
        experiment=experiment,
        namespace="sensitivity",
        execution_mode=execution_mode,
        primary_freeze_manifest=freeze_manifest,
    )


def assert_sensitivity_execution_allowed(
    config: SensitivityExperimentConfig,
    *,
    repository_root: str | Path | None = None,
) -> None:
    """Fail closed for real execution until a Primary Freeze manifest exists.

    S-IMPL-01 establishes the execution barrier.  S-IMPL-10 hardens this check by
    validating the manifest's complete frozen-artifact contents.  Synthetic mode
    deliberately requires no Primary Freeze manifest so implementation tests can
    run before Primary execution is complete.
    """

    if config.execution_mode is SensitivityExecutionMode.SYNTHETIC:
        return

    root = Path(repository_root or Path.cwd()).resolve()
    manifest_path = _inside(
        root / config.primary_freeze_manifest, root, "primary_freeze_manifest"
    )
    if not manifest_path.is_file():
        raise SensitivityExecutionError(
            "real-data Sensitivity execution is blocked until PRIMARY FREEZE manifest exists"
        )


def resolve_sensitivity_output_path(
    config: SensitivityExperimentConfig,
    relative_path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> Path:
    """Resolve an output path and structurally reject Primary/out-of-namespace writes."""

    root = Path(repository_root or Path.cwd()).resolve()
    sensitivity_root = _inside(root / config.artifact_root, root, "artifact_root")
    candidate = Path(relative_path)
    if not candidate.is_absolute():
        candidate = sensitivity_root / candidate
    return _inside(candidate, sensitivity_root, "Sensitivity output path")
