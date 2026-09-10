from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from lagged_facial_graph_forecasting.artifact_registry import (
    ArtifactRegistry,
    ArtifactRegistryError,
)
from lagged_facial_graph_forecasting.core_contracts import ExperimentConfig


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="primary-test",
        scientific_config_path="configs/scientific_freeze.yaml",
        run_config_path="configs/primary_run.yaml",
        artifact_root="artifacts/primary",
        seed=17,
    )


def test_register_file_records_content_hash_and_provenance(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts" / "primary" / "fold_00" / "prediction.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"prediction\n")
    registry = ArtifactRegistry(_config(), repository_root=tmp_path)

    entry = registry.register_file(
        artifact,
        artifact_type="prediction",
        outer_fold=0,
        condition="self",
    )

    assert entry.experiment_id == "primary-test"
    assert entry.relative_path == "artifacts/primary/fold_00/prediction.json"
    assert entry.sha256 == hashlib.sha256(b"prediction\n").hexdigest()
    assert entry.outer_fold == 0
    assert entry.condition == "self"
    assert registry.entries == (entry,)


def test_registry_orders_entries_deterministically(tmp_path: Path) -> None:
    root = tmp_path / "artifacts" / "primary"
    root.mkdir(parents=True)
    (root / "z.json").write_text("z", encoding="utf-8")
    (root / "a.json").write_text("a", encoding="utf-8")
    registry = ArtifactRegistry(_config(), repository_root=tmp_path)

    registry.register_file(root / "z.json", artifact_type="metric")
    registry.register_file(root / "a.json", artifact_type="prediction")

    assert [entry.relative_path for entry in registry.entries] == [
        "artifacts/primary/a.json",
        "artifacts/primary/z.json",
    ]


def test_registry_rejects_file_outside_configured_artifact_root(tmp_path: Path) -> None:
    outside = tmp_path / "artifacts" / "sensitivity" / "result.json"
    outside.parent.mkdir(parents=True)
    outside.write_text("sensitivity", encoding="utf-8")
    registry = ArtifactRegistry(_config(), repository_root=tmp_path)

    with pytest.raises(ArtifactRegistryError, match="artifact path"):
        registry.register_file(outside, artifact_type="prediction")


def test_registry_rejects_changed_content_for_registered_path(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts" / "primary" / "result.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("first", encoding="utf-8")
    registry = ArtifactRegistry(_config(), repository_root=tmp_path)
    registry.register_file(artifact, artifact_type="metric")

    artifact.write_text("changed", encoding="utf-8")

    with pytest.raises(ArtifactRegistryError, match="different provenance"):
        registry.register_file(artifact, artifact_type="metric")
