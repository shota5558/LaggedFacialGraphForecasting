from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lagged_facial_graph_forecasting.runner_config import (
    PrimaryExecutionMode,
    RunnerConfigError,
    load_primary_experiment_config,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FROZEN_SCIENTIFIC_CONFIG = REPOSITORY_ROOT / "configs" / "scientific_freeze.yaml"


def _write_fixture(
    tmp_path: Path,
    *,
    artifact_root: str = "artifacts/primary",
    horizon: int = 1,
    run_config_path: str = "configs/primary_run.yaml",
    mode: str = "preflight",
) -> Path:
    configs = tmp_path / "configs"
    configs.mkdir()
    scientific = yaml.safe_load(FROZEN_SCIENTIFIC_CONFIG.read_text(encoding="utf-8"))
    scientific["primary"]["horizon"] = horizon
    (configs / "scientific_freeze.yaml").write_text(
        yaml.safe_dump(scientific, sort_keys=False), encoding="utf-8"
    )
    run_config = {
        "schema_version": 4,
        "experiment_id": "primary-test",
        "scientific_config_path": "configs/scientific_freeze.yaml",
        "run_config_path": run_config_path,
        "artifact_root": artifact_root,
        "seed": 20260911,
        "protocol_id": "primary-2026-09-12-adopted",
        "execution_mode": mode,
        "preflight_manifest_path": None,
        "executable_freeze_path": None,
    }
    path = configs / "primary_run.yaml"
    path.write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    return path


def test_repository_primary_run_is_explicitly_preflight_only() -> None:
    loaded = load_primary_experiment_config(
        "configs/primary_run.yaml", repository_root=REPOSITORY_ROOT
    )

    assert loaded.experiment_id == "primary"
    assert loaded.protocol_id == "primary-2026-09-12-adopted"
    assert loaded.execution_mode is PrimaryExecutionMode.PREFLIGHT
    assert loaded.run_schema_version == 4
    assert loaded.schema_version == 3
    assert loaded.seed == 20260911


def test_loader_reuses_adopted_scientific_config(tmp_path: Path) -> None:
    loaded = load_primary_experiment_config(
        _write_fixture(tmp_path), repository_root=tmp_path
    )

    assert loaded.experiment_id == "primary-test"
    assert loaded.artifact_root == "artifacts/primary"


def test_primary_loader_rejects_sensitivity_artifact_root(tmp_path: Path) -> None:
    with pytest.raises(RunnerConfigError, match="artifact_root"):
        load_primary_experiment_config(
            _write_fixture(tmp_path, artifact_root="artifacts/sensitivity"),
            repository_root=tmp_path,
        )


def test_primary_loader_rejects_modified_scientific_config(tmp_path: Path) -> None:
    with pytest.raises(RunnerConfigError, match="adopted scientific config"):
        load_primary_experiment_config(
            _write_fixture(tmp_path, horizon=2), repository_root=tmp_path
        )


def test_primary_loader_rejects_self_path_drift(tmp_path: Path) -> None:
    with pytest.raises(RunnerConfigError, match="run_config_path"):
        load_primary_experiment_config(
            _write_fixture(tmp_path, run_config_path="configs/other.yaml"),
            repository_root=tmp_path,
        )


def test_real_mode_fails_closed_without_materialized_facts(tmp_path: Path) -> None:
    with pytest.raises(RunnerConfigError, match="preflight facts"):
        load_primary_experiment_config(
            _write_fixture(tmp_path, mode="real"), repository_root=tmp_path
        )
