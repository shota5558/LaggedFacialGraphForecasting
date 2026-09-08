from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lagged_facial_graph_forecasting.sensitivity_execution import (
    SensitivityExecutionError,
    SensitivityExecutionMode,
    assert_sensitivity_execution_allowed,
    load_sensitivity_experiment_config,
    resolve_sensitivity_output_path,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FROZEN_SCIENTIFIC_CONFIG = REPOSITORY_ROOT / "configs" / "scientific_freeze.yaml"


def _write_fixture(
    tmp_path: Path,
    *,
    execution_mode: str = "synthetic",
    artifact_root: str = "artifacts/sensitivity",
    namespace: str = "sensitivity",
    freeze_manifest: str = "artifacts/primary/freeze_manifest.json",
) -> Path:
    configs = tmp_path / "configs"
    configs.mkdir(parents=True)
    scientific = yaml.safe_load(FROZEN_SCIENTIFIC_CONFIG.read_text(encoding="utf-8"))
    (configs / "scientific_freeze.yaml").write_text(
        yaml.safe_dump(scientific, sort_keys=False), encoding="utf-8"
    )
    run_config = {
        "schema_version": 3,
        "namespace": namespace,
        "execution_mode": execution_mode,
        "experiment_id": "sensitivity-test",
        "scientific_config_path": "configs/scientific_freeze.yaml",
        "run_config_path": "configs/sensitivity_preimplementation.yaml",
        "artifact_root": artifact_root,
        "seed": 20260908,
        "primary_freeze_manifest": freeze_manifest,
    }
    path = configs / "sensitivity_preimplementation.yaml"
    path.write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    return path


def test_repository_sensitivity_descriptor_is_synthetic_and_separate() -> None:
    loaded = load_sensitivity_experiment_config(
        "configs/sensitivity_preimplementation.yaml", repository_root=REPOSITORY_ROOT
    )
    assert loaded.namespace == "sensitivity"
    assert loaded.execution_mode is SensitivityExecutionMode.SYNTHETIC
    assert loaded.artifact_root == "artifacts/sensitivity"
    assert loaded.primary_freeze_manifest == "artifacts/primary/freeze_manifest.json"


def test_synthetic_mode_does_not_require_primary_freeze(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    loaded = load_sensitivity_experiment_config(
        "configs/sensitivity_preimplementation.yaml", repository_root=tmp_path
    )
    assert_sensitivity_execution_allowed(loaded, repository_root=tmp_path)


def test_real_mode_fails_closed_without_primary_freeze_manifest(tmp_path: Path) -> None:
    _write_fixture(tmp_path, execution_mode="real")
    loaded = load_sensitivity_experiment_config(
        "configs/sensitivity_preimplementation.yaml", repository_root=tmp_path
    )
    with pytest.raises(SensitivityExecutionError, match="PRIMARY FREEZE"):
        assert_sensitivity_execution_allowed(loaded, repository_root=tmp_path)


def test_real_mode_passes_stage_one_barrier_when_manifest_exists(tmp_path: Path) -> None:
    _write_fixture(tmp_path, execution_mode="real")
    manifest = tmp_path / "artifacts" / "primary" / "freeze_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    loaded = load_sensitivity_experiment_config(
        "configs/sensitivity_preimplementation.yaml", repository_root=tmp_path
    )
    assert_sensitivity_execution_allowed(loaded, repository_root=tmp_path)


def test_sensitivity_loader_rejects_primary_artifact_root(tmp_path: Path) -> None:
    _write_fixture(tmp_path, artifact_root="artifacts/primary")
    with pytest.raises(SensitivityExecutionError, match="Sensitivity artifact_root"):
        load_sensitivity_experiment_config(
            "configs/sensitivity_preimplementation.yaml", repository_root=tmp_path
        )


def test_sensitivity_loader_rejects_non_sensitivity_namespace(tmp_path: Path) -> None:
    _write_fixture(tmp_path, namespace="primary")
    with pytest.raises(SensitivityExecutionError, match="namespace"):
        load_sensitivity_experiment_config(
            "configs/sensitivity_preimplementation.yaml", repository_root=tmp_path
        )


def test_sensitivity_loader_requires_explicit_execution_mode(tmp_path: Path) -> None:
    _write_fixture(tmp_path, execution_mode="auto")
    with pytest.raises(SensitivityExecutionError, match="execution_mode"):
        load_sensitivity_experiment_config(
            "configs/sensitivity_preimplementation.yaml", repository_root=tmp_path
        )


def test_primary_freeze_manifest_must_live_under_primary_namespace(tmp_path: Path) -> None:
    _write_fixture(tmp_path, freeze_manifest="artifacts/sensitivity/freeze_manifest.json")
    with pytest.raises(SensitivityExecutionError, match="primary_freeze_manifest"):
        load_sensitivity_experiment_config(
            "configs/sensitivity_preimplementation.yaml", repository_root=tmp_path
        )


def test_output_path_is_confined_to_sensitivity_namespace(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    loaded = load_sensitivity_experiment_config(
        "configs/sensitivity_preimplementation.yaml", repository_root=tmp_path
    )
    resolved = resolve_sensitivity_output_path(
        loaded, "gpdc/result.json", repository_root=tmp_path
    )
    assert resolved == (tmp_path / "artifacts/sensitivity/gpdc/result.json").resolve()

    with pytest.raises(SensitivityExecutionError, match="Sensitivity output path"):
        resolve_sensitivity_output_path(
            loaded,
            tmp_path / "artifacts/primary/forbidden.json",
            repository_root=tmp_path,
        )
