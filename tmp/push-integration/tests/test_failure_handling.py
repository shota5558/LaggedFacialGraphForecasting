from __future__ import annotations

from pathlib import Path

import pytest

from lagged_facial_graph_forecasting.artifact_registry import (
    ArtifactRegistry,
    ArtifactRegistryError,
)
from lagged_facial_graph_forecasting.core_contracts import ExperimentConfig
from lagged_facial_graph_forecasting.failure_handling import (
    FailureHandlingError,
    atomic_artifact_output,
)


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="primary-test",
        scientific_config_path="configs/scientific_freeze.yaml",
        run_config_path="configs/primary_run.yaml",
        artifact_root="artifacts/primary",
        seed=17,
    )


def test_atomic_output_is_visible_only_after_success(tmp_path: Path) -> None:
    config = _config()
    target = tmp_path / "artifacts" / "primary" / "fold_00" / "prediction.json"
    registry = ArtifactRegistry(config, repository_root=tmp_path)

    with atomic_artifact_output(
        target,
        config=config,
        repository_root=tmp_path,
    ) as staging:
        assert not target.exists()
        assert staging.name.endswith(".partial")
        staging.write_text("complete\n", encoding="utf-8")
        with pytest.raises(ArtifactRegistryError, match="incomplete staging"):
            registry.register_file(staging, artifact_type="prediction")

    assert target.read_text(encoding="utf-8") == "complete\n"
    entry = registry.register_file(target, artifact_type="prediction", outer_fold=0)
    assert entry.relative_path == "artifacts/primary/fold_00/prediction.json"


def test_atomic_output_cleans_failed_write_and_preserves_exception(tmp_path: Path) -> None:
    config = _config()
    target = tmp_path / "artifacts" / "primary" / "fold_00" / "prediction.json"
    staging_path: Path | None = None

    with pytest.raises(RuntimeError, match="synthetic failure"):
        with atomic_artifact_output(
            target,
            config=config,
            repository_root=tmp_path,
        ) as staging:
            staging_path = staging
            staging.write_text("incomplete\n", encoding="utf-8")
            raise RuntimeError("synthetic failure")

    assert not target.exists()
    assert staging_path is not None
    assert not staging_path.exists()


def test_atomic_output_rejects_existing_final_artifact(tmp_path: Path) -> None:
    config = _config()
    target = tmp_path / "artifacts" / "primary" / "result.json"
    target.parent.mkdir(parents=True)
    target.write_text("frozen\n", encoding="utf-8")

    with pytest.raises(FailureHandlingError, match="already exists"):
        with atomic_artifact_output(
            target,
            config=config,
            repository_root=tmp_path,
        ):
            pass

    assert target.read_text(encoding="utf-8") == "frozen\n"


def test_atomic_output_rejects_path_outside_primary_artifact_root(tmp_path: Path) -> None:
    config = _config()
    target = tmp_path / "artifacts" / "sensitivity" / "result.json"

    with pytest.raises(FailureHandlingError, match="final_path"):
        with atomic_artifact_output(
            target,
            config=config,
            repository_root=tmp_path,
        ):
            pass


def test_atomic_output_rejects_missing_staging_file_on_success(tmp_path: Path) -> None:
    config = _config()
    target = tmp_path / "artifacts" / "primary" / "result.json"

    with pytest.raises(FailureHandlingError, match="did not leave"):
        with atomic_artifact_output(
            target,
            config=config,
            repository_root=tmp_path,
        ) as staging:
            staging.unlink()

    assert not target.exists()
