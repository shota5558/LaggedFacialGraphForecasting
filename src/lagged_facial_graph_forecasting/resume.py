"""Strict resume verification for deterministic Runner Core execution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from .core_contract_io import CoreContractIOError, deserialize_core_contract
from .core_contracts import ExperimentArtifact, ExperimentConfig


class ResumeError(ValueError):
    """Raised when an existing experiment cannot be safely resumed."""


@dataclass(frozen=True, slots=True)
class VerifiedResumeState:
    config: ExperimentConfig
    artifacts: tuple[ExperimentArtifact, ...]
    software_versions: tuple[tuple[str, str], ...]


def _inside_artifact_root(path: str, artifact_root: str) -> bool:
    candidate = PurePosixPath(path)
    root = PurePosixPath(artifact_root)
    return candidate == root or root in candidate.parents


def load_verified_resume_state(
    manifest_path: str | Path,
    *,
    expected_config: ExperimentConfig,
    repository_root: str | Path = Path("."),
) -> VerifiedResumeState:
    """Return resume state only when config and every registered artifact still match."""

    root = Path(repository_root).resolve()
    manifest = Path(manifest_path)
    if not manifest.is_absolute():
        manifest = root / manifest
    manifest = manifest.resolve()
    try:
        manifest.relative_to(root)
    except ValueError as exc:
        raise ResumeError("manifest_path must resolve inside repository_root") from exc

    try:
        payload: Any = json.loads(manifest.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ResumeError(f"resume manifest not found: {manifest}") from exc
    except json.JSONDecodeError as exc:
        raise ResumeError("resume manifest is not valid JSON") from exc

    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "experiment_config",
        "artifacts",
        "software_versions",
    }:
        raise ResumeError("resume manifest fields are incompatible")
    if payload["schema_version"] != 1:
        raise ResumeError("resume manifest schema_version must be 1")

    try:
        config = deserialize_core_contract(payload["experiment_config"])
    except CoreContractIOError as exc:
        raise ResumeError("resume ExperimentConfig is invalid") from exc
    if not isinstance(config, ExperimentConfig):
        raise ResumeError("resume experiment_config must contain ExperimentConfig")
    if config != expected_config:
        raise ResumeError("resume ExperimentConfig does not match the current run")

    artifact_payloads = payload["artifacts"]
    if not isinstance(artifact_payloads, list):
        raise ResumeError("resume artifacts must be a list")
    artifacts: list[ExperimentArtifact] = []
    seen_paths: set[str] = set()
    for index, envelope in enumerate(artifact_payloads):
        try:
            artifact = deserialize_core_contract(envelope)
        except CoreContractIOError as exc:
            raise ResumeError(f"resume artifact[{index}] is invalid") from exc
        if not isinstance(artifact, ExperimentArtifact):
            raise ResumeError(f"resume artifact[{index}] is not ExperimentArtifact")
        if artifact.experiment_id != expected_config.experiment_id:
            raise ResumeError("resume artifact experiment_id does not match current run")
        if artifact.relative_path in seen_paths:
            raise ResumeError("resume artifact paths must be unique")
        seen_paths.add(artifact.relative_path)
        if not _inside_artifact_root(artifact.relative_path, expected_config.artifact_root):
            raise ResumeError("resume artifact lies outside the configured artifact_root")

        artifact_file = (root / artifact.relative_path).resolve()
        try:
            artifact_file.relative_to(root)
        except ValueError as exc:
            raise ResumeError("resume artifact resolves outside repository_root") from exc
        if not artifact_file.is_file():
            raise ResumeError(f"resume artifact is missing: {artifact.relative_path}")
        digest = hashlib.sha256(artifact_file.read_bytes()).hexdigest()
        if digest != artifact.sha256:
            raise ResumeError(f"resume artifact hash mismatch: {artifact.relative_path}")
        artifacts.append(artifact)

    versions = payload["software_versions"]
    if not isinstance(versions, dict) or not versions:
        raise ResumeError("resume software_versions must be a non-empty mapping")
    normalized_versions: list[tuple[str, str]] = []
    for name, version in versions.items():
        if not isinstance(name, str) or not name.strip():
            raise ResumeError("resume software version names must be non-empty strings")
        if not isinstance(version, str) or not version.strip():
            raise ResumeError("resume software version values must be non-empty strings")
        normalized_versions.append((name.strip(), version.strip()))

    return VerifiedResumeState(
        config=config,
        artifacts=tuple(sorted(artifacts, key=lambda item: item.relative_path)),
        software_versions=tuple(sorted(normalized_versions)),
    )
