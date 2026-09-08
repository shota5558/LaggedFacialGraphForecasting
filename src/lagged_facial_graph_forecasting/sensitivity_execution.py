"""Sensitivity execution boundary and Primary-Freeze barrier.

Issue #22 permits Sensitivity code and synthetic evidence before PRIMARY FREEZE,
while real-data Sensitivity execution remains blocked.  S-IMPL-10 upgrades the
initial existence check into a strict completeness/integrity audit of the future
Issue #20 freeze manifest without generating or claiming that freeze here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

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

PRIMARY_FREEZE_MANIFEST_SCHEMA_VERSION = 1
PRIMARY_FREEZE_STATUS = "PASS"
PRIMARY_ARTIFACT_ROOT = "artifacts/primary"
PRIMARY_FREEZE_MANIFEST_PATH = "artifacts/primary/freeze_manifest.json"
PRIMARY_FREEZE_REQUIRED_ARTIFACT_TYPES = frozenset(
    {
        "primary_config",
        "split_manifest",
        "preprocessing_rules",
        "parent_set",
        "ridge_lambda",
        "null_mapping",
        "predictions",
        "metrics",
        "paired_statistics",
        "lag_response",
        "edge_stability",
        "seeds",
        "software_versions",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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


def _strict_mapping(value: Any, required: set[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SensitivityExecutionError(f"{context} must be a mapping")
    actual = set(value)
    if actual != required:
        raise SensitivityExecutionError(
            f"{context} fields differ from the frozen schema; "
            f"missing={sorted(required - actual)}, extra={sorted(actual - required)}"
        )
    return value


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
    if primary_root != PRIMARY_ARTIFACT_ROOT:
        raise SensitivityExecutionError(
            f"Primary artifact root must remain {PRIMARY_ARTIFACT_ROOT!r}"
        )

    freeze_manifest = _relative_posix(
        raw["primary_freeze_manifest"], "primary_freeze_manifest"
    )
    if freeze_manifest != PRIMARY_FREEZE_MANIFEST_PATH:
        raise SensitivityExecutionError(
            f"primary_freeze_manifest must equal {PRIMARY_FREEZE_MANIFEST_PATH!r}"
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_primary_freeze_manifest(
    config: SensitivityExperimentConfig,
    *,
    repository_root: str | Path | None = None,
) -> Mapping[str, Any]:
    """Validate Issue #20 completeness, path confinement, existence, and hashes.

    This reader does not create Primary artifacts.  A valid manifest must explicitly
    claim PASS and reference every required Issue #20 freeze target with a SHA-256
    that matches an existing file beneath ``artifacts/primary``.
    """

    if not isinstance(config, SensitivityExperimentConfig):
        raise SensitivityExecutionError("config must be SensitivityExperimentConfig")
    if config.primary_freeze_manifest != PRIMARY_FREEZE_MANIFEST_PATH:
        raise SensitivityExecutionError(
            f"primary_freeze_manifest must equal {PRIMARY_FREEZE_MANIFEST_PATH!r}"
        )

    root = Path(repository_root or Path.cwd()).resolve()
    primary_root = _inside(root / PRIMARY_ARTIFACT_ROOT, root, "Primary artifact root")
    manifest_path = _inside(
        root / config.primary_freeze_manifest, primary_root, "primary_freeze_manifest"
    )
    if not manifest_path.is_file():
        raise SensitivityExecutionError(
            "real-data Sensitivity execution is blocked until PRIMARY FREEZE manifest exists"
        )

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SensitivityExecutionError("PRIMARY FREEZE manifest is not valid JSON") from exc

    manifest = _strict_mapping(
        raw,
        {"schema_version", "freeze_status", "artifact_root", "artifacts"},
        "PRIMARY FREEZE manifest",
    )
    if manifest["schema_version"] != PRIMARY_FREEZE_MANIFEST_SCHEMA_VERSION:
        raise SensitivityExecutionError("unsupported PRIMARY FREEZE manifest schema_version")
    if manifest["freeze_status"] != PRIMARY_FREEZE_STATUS:
        raise SensitivityExecutionError("PRIMARY FREEZE manifest must declare freeze_status='PASS'")
    if manifest["artifact_root"] != PRIMARY_ARTIFACT_ROOT:
        raise SensitivityExecutionError(
            f"PRIMARY FREEZE artifact_root must equal {PRIMARY_ARTIFACT_ROOT!r}"
        )

    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise SensitivityExecutionError("PRIMARY FREEZE artifacts must be a non-empty list")

    seen_paths: set[str] = set()
    seen_types: set[str] = set()
    for index, raw_entry in enumerate(artifacts):
        entry = _strict_mapping(
            raw_entry,
            {"artifact_type", "relative_path", "sha256"},
            f"PRIMARY FREEZE artifacts[{index}]",
        )
        artifact_type = entry["artifact_type"]
        if not isinstance(artifact_type, str) or not artifact_type.strip():
            raise SensitivityExecutionError("PRIMARY FREEZE artifact_type must be non-empty")
        artifact_type = artifact_type.strip()
        relative_path = _relative_posix(
            entry["relative_path"], f"PRIMARY FREEZE artifacts[{index}].relative_path"
        )
        if relative_path == PRIMARY_FREEZE_MANIFEST_PATH:
            raise SensitivityExecutionError("freeze manifest cannot hash-reference itself")
        if relative_path in seen_paths:
            raise SensitivityExecutionError(
                f"duplicate PRIMARY FREEZE artifact path: {relative_path}"
            )
        seen_paths.add(relative_path)
        seen_types.add(artifact_type)

        expected_sha = entry["sha256"]
        if not isinstance(expected_sha, str) or not _SHA256_RE.fullmatch(expected_sha):
            raise SensitivityExecutionError(
                f"PRIMARY FREEZE artifacts[{index}].sha256 must be 64 lowercase hex characters"
            )
        artifact_path = _inside(
            root / relative_path,
            primary_root,
            f"PRIMARY FREEZE artifacts[{index}].relative_path",
        )
        if not artifact_path.is_file():
            raise SensitivityExecutionError(
                f"PRIMARY FREEZE referenced artifact does not exist: {relative_path}"
            )
        actual_sha = _sha256_file(artifact_path)
        if actual_sha != expected_sha:
            raise SensitivityExecutionError(
                f"PRIMARY FREEZE artifact hash mismatch: {relative_path}"
            )

    missing_types = PRIMARY_FREEZE_REQUIRED_ARTIFACT_TYPES - seen_types
    if missing_types:
        raise SensitivityExecutionError(
            "PRIMARY FREEZE manifest is incomplete; missing artifact types: "
            f"{sorted(missing_types)}"
        )
    return manifest


def assert_sensitivity_execution_allowed(
    config: SensitivityExperimentConfig,
    *,
    repository_root: str | Path | None = None,
) -> None:
    """Allow synthetic implementation tests; fully audit Primary Freeze for real runs."""

    if not isinstance(config, SensitivityExperimentConfig):
        raise SensitivityExecutionError("config must be SensitivityExperimentConfig")
    if config.execution_mode is SensitivityExecutionMode.SYNTHETIC:
        return
    validate_primary_freeze_manifest(config, repository_root=repository_root)


def assert_real_sensitivity_artifact_provenance(
    config: SensitivityExperimentConfig,
    artifact: Any,
    *,
    repository_root: str | Path | None = None,
) -> None:
    """Bind every real Sensitivity result to the exact validated freeze manifest."""

    if config.execution_mode is SensitivityExecutionMode.SYNTHETIC:
        return
    validate_primary_freeze_manifest(config, repository_root=repository_root)
    actual = getattr(artifact, "source_primary_freeze_manifest", None)
    if actual != config.primary_freeze_manifest:
        raise SensitivityExecutionError(
            "real Sensitivity artifact must reference the exact validated PRIMARY FREEZE manifest"
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
