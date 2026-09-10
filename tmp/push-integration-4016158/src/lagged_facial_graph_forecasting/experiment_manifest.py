"""Deterministic experiment manifest for Runner Core provenance."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

from .core_contract_io import serialize_core_contract
from .core_contracts import ExperimentArtifact, ExperimentConfig
from .preprocessing_provenance import PreprocessingProvenance


EXPERIMENT_MANIFEST_SCHEMA_VERSION = 2


class ExperimentManifestError(ValueError):
    """Raised when manifest provenance is incomplete or inconsistent."""


def _artifact_is_within_root(relative_path: str, artifact_root: str) -> bool:
    path = PurePosixPath(relative_path)
    root = PurePosixPath(artifact_root)
    return path == root or root in path.parents


def build_experiment_manifest(
    *,
    config: ExperimentConfig,
    artifacts: Sequence[ExperimentArtifact],
    software_versions: Mapping[str, str],
    preprocessing_provenance: PreprocessingProvenance | None = None,
) -> dict[str, object]:
    """Build a stable manifest from frozen config and registered artifact entries.

    Lane A scientific provenance is mandatory: every experiment manifest records
    the ordered region-to-landmark definition and the exact feature semantics.
    This prevents a run from being reproducible at the file/hash level while
    remaining ambiguous about what each facial node and feature dimension meant.
    """

    if not software_versions:
        raise ExperimentManifestError("software_versions must not be empty")
    normalized_versions: dict[str, str] = {}
    for name, version in software_versions.items():
        if not isinstance(name, str) or not name.strip():
            raise ExperimentManifestError(
                "software version names must be non-empty strings"
            )
        if not isinstance(version, str) or not version.strip():
            raise ExperimentManifestError(
                "software version values must be non-empty strings"
            )
        normalized_versions[name.strip()] = version.strip()

    if preprocessing_provenance is None:
        raise ExperimentManifestError(
            "preprocessing_provenance is required for every experiment manifest"
        )
    if not isinstance(preprocessing_provenance, PreprocessingProvenance):
        raise ExperimentManifestError(
            "preprocessing_provenance must be PreprocessingProvenance"
        )

    entries = tuple(artifacts)
    paths = [entry.relative_path for entry in entries]
    if len(paths) != len(set(paths)):
        raise ExperimentManifestError("artifact relative paths must be unique")
    for entry in entries:
        if entry.experiment_id != config.experiment_id:
            raise ExperimentManifestError(
                "artifact experiment_id must match ExperimentConfig"
            )
        if not _artifact_is_within_root(entry.relative_path, config.artifact_root):
            raise ExperimentManifestError(
                "artifact relative_path must remain inside ExperimentConfig artifact_root"
            )

    ordered_entries = sorted(entries, key=lambda entry: entry.relative_path)
    return {
        "schema_version": EXPERIMENT_MANIFEST_SCHEMA_VERSION,
        "experiment_config": serialize_core_contract(config),
        "preprocessing_provenance": preprocessing_provenance.to_payload(),
        "artifacts": [serialize_core_contract(entry) for entry in ordered_entries],
        "software_versions": {
            name: normalized_versions[name] for name in sorted(normalized_versions)
        },
    }


def write_experiment_manifest(
    path: str | Path,
    *,
    config: ExperimentConfig,
    artifacts: Sequence[ExperimentArtifact],
    software_versions: Mapping[str, str],
    preprocessing_provenance: PreprocessingProvenance | None = None,
) -> Path:
    """Write a deterministic UTF-8 JSON manifest."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_experiment_manifest(
        config=config,
        artifacts=artifacts,
        software_versions=software_versions,
        preprocessing_provenance=preprocessing_provenance,
    )
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path
