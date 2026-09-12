"""Atomic writer for the validated PRIMARY FREEZE manifest."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from .core_contracts import ExperimentArtifact, ExperimentConfig
from .failure_handling import atomic_artifact_output
from .sensitivity_execution import (
    PRIMARY_ARTIFACT_ROOT,
    PRIMARY_FREEZE_MANIFEST_PATH,
    PRIMARY_FREEZE_MANIFEST_SCHEMA_VERSION,
    PRIMARY_FREEZE_REQUIRED_ARTIFACT_TYPES,
    PRIMARY_FREEZE_STATUS,
)


class PrimaryFreezeError(ValueError):
    """Raised when a PRIMARY FREEZE cannot be created safely."""


def _inside_primary(relative_path: str) -> bool:
    path = PurePosixPath(relative_path)
    root = PurePosixPath(PRIMARY_ARTIFACT_ROOT)
    return path != root and root in path.parents


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_primary_freeze_manifest(
    artifacts: Sequence[ExperimentArtifact],
    *,
    config: ExperimentConfig,
    repository_root: str | Path = ".",
) -> dict[str, Any]:
    """Validate registered Primary artifacts and build a deterministic freeze payload."""

    if config.artifact_root != PRIMARY_ARTIFACT_ROOT:
        raise PrimaryFreezeError(
            f"Primary freeze requires artifact_root={PRIMARY_ARTIFACT_ROOT!r}"
        )

    root = Path(repository_root).resolve()
    entries = tuple(artifacts)
    if not entries:
        raise PrimaryFreezeError("Primary freeze requires at least one artifact")

    seen_paths: set[str] = set()
    payload_entries: list[dict[str, str]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, ExperimentArtifact):
            raise PrimaryFreezeError(
                f"artifacts[{index}] must be an ExperimentArtifact"
            )
        if entry.experiment_id != config.experiment_id:
            raise PrimaryFreezeError(
                f"artifacts[{index}] belongs to a different experiment"
            )
        if not _inside_primary(entry.relative_path):
            raise PrimaryFreezeError(
                f"artifacts[{index}].relative_path must remain inside {PRIMARY_ARTIFACT_ROOT}"
            )
        if entry.relative_path == PRIMARY_FREEZE_MANIFEST_PATH:
            raise PrimaryFreezeError("freeze manifest cannot reference itself")
        if entry.relative_path in seen_paths:
            raise PrimaryFreezeError(
                f"duplicate Primary artifact path: {entry.relative_path}"
            )
        seen_paths.add(entry.relative_path)
        path = root / Path(*PurePosixPath(entry.relative_path).parts)
        if not path.is_file():
            raise PrimaryFreezeError(
                f"Primary artifact file not found: {entry.relative_path}"
            )
        if path.stat().st_size == 0:
            raise PrimaryFreezeError(
                f"Primary artifact must not be empty: {entry.relative_path}"
            )
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != entry.sha256:
            raise PrimaryFreezeError(
                f"Primary artifact hash mismatch: {entry.relative_path}"
            )
        payload_entries.append(
            {
                "artifact_type": entry.artifact_type,
                "relative_path": entry.relative_path,
                "sha256": actual_sha256,
            }
        )

    actual_types = {entry["artifact_type"] for entry in payload_entries}
    missing_types = PRIMARY_FREEZE_REQUIRED_ARTIFACT_TYPES - actual_types
    if missing_types:
        raise PrimaryFreezeError(
            f"Primary freeze is incomplete; missing artifact types: {sorted(missing_types)}"
        )

    payload_entries.sort(key=lambda entry: (entry["relative_path"], entry["artifact_type"]))
    return {
        "schema_version": PRIMARY_FREEZE_MANIFEST_SCHEMA_VERSION,
        "freeze_status": PRIMARY_FREEZE_STATUS,
        "artifact_root": PRIMARY_ARTIFACT_ROOT,
        "artifacts": payload_entries,
    }


def write_primary_freeze_manifest(
    artifacts: Sequence[ExperimentArtifact],
    *,
    config: ExperimentConfig,
    repository_root: str | Path = ".",
) -> Path:
    """Write the Primary freeze manifest once, using atomic publication."""

    root = Path(repository_root).resolve()
    payload = build_primary_freeze_manifest(
        artifacts, config=config, repository_root=root
    )
    output_path = root / Path(*PurePosixPath(PRIMARY_FREEZE_MANIFEST_PATH).parts)
    try:
        with atomic_artifact_output(
            output_path, config=config, repository_root=root
        ) as staging:
            staging.write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
    except Exception as exc:
        if isinstance(exc, PrimaryFreezeError):
            raise
        raise PrimaryFreezeError(f"could not write Primary freeze manifest: {exc}") from exc
    return output_path
