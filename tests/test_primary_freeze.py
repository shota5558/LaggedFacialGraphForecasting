from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from lagged_facial_graph_forecasting.core_contracts import ExperimentArtifact, ExperimentConfig
from lagged_facial_graph_forecasting.primary_freeze import (
    PrimaryFreezeError,
    build_primary_freeze_manifest,
    write_primary_freeze_manifest,
)
from lagged_facial_graph_forecasting.sensitivity_execution import (
    PRIMARY_ARTIFACT_ROOT,
    PRIMARY_FREEZE_MANIFEST_PATH,
    PRIMARY_FREEZE_REQUIRED_ARTIFACT_TYPES,
    SensitivityExecutionMode,
    SensitivityExperimentConfig,
    validate_primary_freeze_manifest,
)


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="primary-freeze-test",
        scientific_config_path="configs/scientific_freeze.yaml",
        run_config_path="configs/primary_run.yaml",
        artifact_root=PRIMARY_ARTIFACT_ROOT,
        seed=11,
    )


def _artifacts(root: Path) -> tuple[ExperimentArtifact, ...]:
    result: list[ExperimentArtifact] = []
    for artifact_type in sorted(PRIMARY_FREEZE_REQUIRED_ARTIFACT_TYPES):
        relative_path = f"{PRIMARY_ARTIFACT_ROOT}/{artifact_type}.json"
        path = root / Path(*relative_path.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps({"type": artifact_type}, sort_keys=True).encode()
        path.write_bytes(content)
        result.append(
            ExperimentArtifact(
                experiment_id=_config().experiment_id,
                artifact_type=artifact_type,
                relative_path=relative_path,
                sha256=sha256(content).hexdigest(),
            )
        )
    return tuple(result)


def _sensitivity_config() -> SensitivityExperimentConfig:
    return SensitivityExperimentConfig(
        experiment=ExperimentConfig(
            experiment_id="sensitivity-test",
            scientific_config_path="configs/scientific_freeze.yaml",
            run_config_path="configs/sensitivity_preimplementation.yaml",
            artifact_root="artifacts/sensitivity",
            seed=11,
        ),
        namespace="sensitivity",
        execution_mode=SensitivityExecutionMode.REAL,
        primary_freeze_manifest=PRIMARY_FREEZE_MANIFEST_PATH,
    )


def test_primary_freeze_writer_round_trips_through_consumer(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)

    first = build_primary_freeze_manifest(
        artifacts, config=_config(), repository_root=tmp_path
    )
    second = build_primary_freeze_manifest(
        tuple(reversed(artifacts)), config=_config(), repository_root=tmp_path
    )
    assert first == second

    path = write_primary_freeze_manifest(
        artifacts, config=_config(), repository_root=tmp_path
    )
    assert path.relative_to(tmp_path).as_posix() == PRIMARY_FREEZE_MANIFEST_PATH
    assert validate_primary_freeze_manifest(
        _sensitivity_config(), repository_root=tmp_path
    ) == first


def test_primary_freeze_rejects_missing_type_and_hash_drift(tmp_path: Path) -> None:
    artifacts = list(_artifacts(tmp_path))
    artifacts.pop()
    with pytest.raises(PrimaryFreezeError, match="missing artifact types"):
        build_primary_freeze_manifest(
            artifacts, config=_config(), repository_root=tmp_path
        )

    artifacts = list(_artifacts(tmp_path))
    path = tmp_path / Path(*artifacts[0].relative_path.split("/"))
    path.write_text("tampered", encoding="utf-8")
    with pytest.raises(PrimaryFreezeError, match="hash mismatch"):
        build_primary_freeze_manifest(
            artifacts, config=_config(), repository_root=tmp_path
        )


def test_primary_freeze_rejects_duplicate_and_overwrite(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    with pytest.raises(PrimaryFreezeError, match="duplicate"):
        build_primary_freeze_manifest(
            artifacts + (artifacts[0],), config=_config(), repository_root=tmp_path
        )

    write_primary_freeze_manifest(artifacts, config=_config(), repository_root=tmp_path)
    with pytest.raises(PrimaryFreezeError, match="already exists"):
        write_primary_freeze_manifest(artifacts, config=_config(), repository_root=tmp_path)
