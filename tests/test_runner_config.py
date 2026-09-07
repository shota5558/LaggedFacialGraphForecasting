from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lagged_facial_graph_forecasting.runner_config import (
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
) -> Path:
    configs = tmp_path / "configs"
    configs.mkdir()

    scientific = yaml.safe_load(FROZEN_SCIENTIFIC_CONFIG.read_text(encoding="utf-8"))
    scientific["primary"]["horizon"] = horizon
    (configs / "scientific_freeze.yaml").write_text(
        yaml.safe_dump(scientific, sort_keys=False), encoding="utf-8"
    )

    run_config = {
        "schema_version": 1,
        "experiment_id": "primary-test",
        "scientific_config_path": "configs/scientific_freeze.yaml",
        "run_config_path": run_config_path,
        "artifact_root": artifact_root,
        "seed": 20260908,
    }
    path = configs / "primary_run.yaml"
    path.write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    return path


def test_load_primary_experiment_config_reuses_frozen_scientific_config(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path)

    loaded = load_primary_experiment_config(
        "configs/primary_run.yaml", repository_root=tmp_path
    )

    assert loaded.experiment_id == "primary-test"
    assert loaded.scientific_config_path == "configs/scientific_freeze.yaml"
    assert loaded.run_config_path == "configs/primary_run.yaml"
    assert loaded.artifact_root == "artifacts/primary"
    assert loaded.seed == 20260908
    assert loaded.schema_version == 1


def test_primary_loader_rejects_sensitivity_artifact_root(tmp_path: Path) -> None:
    _write_fixture(tmp_path, artifact_root="artifacts/sensitivity")

    with pytest.raises(RunnerConfigError, match="Primary artifact root"):
        load_primary_experiment_config(
            "configs/primary_run.yaml", repository_root=tmp_path
        )


def test_primary_loader_rejects_modified_scientific_freeze(tmp_path: Path) -> None:
    _write_fixture(tmp_path, horizon=2)

    with pytest.raises(RunnerConfigError, match="frozen scientific config"):
        load_primary_experiment_config(
            "configs/primary_run.yaml", repository_root=tmp_path
        )


def test_primary_loader_requires_self_identifying_run_config_path(tmp_path: Path) -> None:
    _write_fixture(tmp_path, run_config_path="configs/other.yaml")

    with pytest.raises(RunnerConfigError, match="run_config_path"):
        load_primary_experiment_config(
            "configs/primary_run.yaml", repository_root=tmp_path
        )
