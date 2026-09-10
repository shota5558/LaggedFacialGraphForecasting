"""Primary runner configuration loader.

H-00 deliberately reuses the frozen P0A scientific-config loader and the versioned
``ExperimentConfig`` contract.  It only binds a run descriptor to those existing
contracts; it does not redefine Primary scientific settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .contracts import ContractError
from .core_contracts import ExperimentConfig
from .scientific_config import ScientificConfigError, load_scientific_config


class RunnerConfigError(ValueError):
    """Raised when a Primary run descriptor is malformed or violates the freeze."""


_REQUIRED_KEYS = {
    "schema_version",
    "experiment_id",
    "scientific_config_path",
    "run_config_path",
    "artifact_root",
    "seed",
}


def _within_repository(path: Path, repository_root: Path, field_name: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(repository_root)
    except ValueError as exc:
        raise RunnerConfigError(f"{field_name} must resolve inside repository_root") from exc
    return resolved


def load_primary_experiment_config(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> ExperimentConfig:
    """Load one Primary run descriptor and validate its scientific provenance.

    The descriptor contains only the already-frozen ``ExperimentConfig`` fields.
    ``scientific_config_path`` is loaded through the strict P0A loader, the run
    descriptor must identify its own repository-relative path, and its artifact
    root must equal the frozen Primary artifact root.  This prevents the Runner
    Core from silently selecting Sensitivity output space or a modified scientific
    configuration.
    """

    root = Path(repository_root or Path.cwd()).resolve()
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = _within_repository(config_path, root, "config path")

    try:
        raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RunnerConfigError(f"run config not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise RunnerConfigError(f"invalid YAML in run config: {config_path}") from exc

    if not isinstance(raw, dict):
        raise RunnerConfigError("run config root must be a mapping")
    actual_keys = set(raw)
    if actual_keys != _REQUIRED_KEYS:
        missing = sorted(_REQUIRED_KEYS - actual_keys)
        extra = sorted(actual_keys - _REQUIRED_KEYS)
        raise RunnerConfigError(
            f"run config keys differ from ExperimentConfig; missing={missing}, extra={extra}"
        )

    try:
        experiment = ExperimentConfig(**raw)
    except (ContractError, TypeError, ValueError) as exc:
        raise RunnerConfigError(f"invalid ExperimentConfig: {exc}") from exc

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
        scientific = load_scientific_config(scientific_path)
    except ScientificConfigError as exc:
        raise RunnerConfigError(f"invalid frozen scientific config: {exc}") from exc

    primary_root = scientific["artifacts"]["primary_root"]
    if experiment.artifact_root != primary_root:
        raise RunnerConfigError(
            "Primary runner artifact_root must equal the frozen Primary artifact root: "
            f"expected={primary_root!r}, actual={experiment.artifact_root!r}"
        )

    return experiment
