from __future__ import annotations

from pathlib import Path

import pytest

from lagged_facial_graph_forecasting.artifact_registry import ArtifactRegistry
from lagged_facial_graph_forecasting.core_contracts import ExperimentConfig
from lagged_facial_graph_forecasting.experiment_manifest import write_experiment_manifest
from lagged_facial_graph_forecasting.preprocessing_provenance import (
    FeatureSemantics,
    PreprocessingProvenance,
)
from lagged_facial_graph_forecasting.regions import build_region_definition
from lagged_facial_graph_forecasting.resume import ResumeError, load_verified_resume_state


def _config(*, seed: int = 17) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="primary-test",
        scientific_config_path="configs/scientific_freeze.yaml",
        run_config_path="configs/primary_run.yaml",
        artifact_root="artifacts/primary",
        seed=seed,
    )


def _preprocessing_provenance() -> PreprocessingProvenance:
    return PreprocessingProvenance(
        region_definition=build_region_definition(
            {
                "mouth": (0, 1),
                "left_cheek": (2, 3),
            }
        ),
        feature_semantics=FeatureSemantics(
            feature_name="velocity",
            dimensions=("v_x", "v_y"),
            method="displacement_over_observed_time_interval",
            units="normalized_coordinate_per_timestamp_unit",
            timestamp_basis="observed_frame_timestamps",
        ),
        region_aggregation_method="mean",
        translation_normalization="reference_centroid",
        scale_normalization="reference_pair_distance",
        rotation_normalization="reference_pair_axis_alignment",
        missing_frame_handling="preserve_invalidity",
        interpolation_policy="none",
        acceleration_primary=False,
    )


def _fixture(tmp_path: Path) -> tuple[ExperimentConfig, Path, Path]:
    config = _config()
    artifact = tmp_path / "artifacts" / "primary" / "fold_00" / "prediction.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("prediction\n", encoding="utf-8")
    registry = ArtifactRegistry(config, repository_root=tmp_path)
    registry.register_file(
        artifact,
        artifact_type="prediction",
        outer_fold=0,
        condition="self",
    )
    manifest = tmp_path / "artifacts" / "primary" / "experiment_manifest.json"
    write_experiment_manifest(
        manifest,
        config=config,
        artifacts=registry.entries,
        software_versions={"python": "3.11", "numpy": "2.0"},
        preprocessing_provenance=_preprocessing_provenance(),
    )
    return config, manifest, artifact


def test_resume_accepts_only_exact_config_and_artifact_hashes(tmp_path: Path) -> None:
    config, manifest, _ = _fixture(tmp_path)

    state = load_verified_resume_state(
        manifest,
        expected_config=config,
        repository_root=tmp_path,
    )

    assert state.config == config
    assert state.preprocessing_provenance == _preprocessing_provenance()
    assert state.preprocessing_provenance.feature_semantics.dimensions == ("v_x", "v_y")
    assert [entry.relative_path for entry in state.artifacts] == [
        "artifacts/primary/fold_00/prediction.json"
    ]
    assert state.software_versions == (("numpy", "2.0"), ("python", "3.11"))


def test_resume_rejects_config_drift(tmp_path: Path) -> None:
    _, manifest, _ = _fixture(tmp_path)

    with pytest.raises(ResumeError, match="ExperimentConfig"):
        load_verified_resume_state(
            manifest,
            expected_config=_config(seed=18),
            repository_root=tmp_path,
        )


def test_resume_rejects_changed_artifact_bytes(tmp_path: Path) -> None:
    config, manifest, artifact = _fixture(tmp_path)
    artifact.write_text("changed\n", encoding="utf-8")

    with pytest.raises(ResumeError, match="hash mismatch"):
        load_verified_resume_state(
            manifest,
            expected_config=config,
            repository_root=tmp_path,
        )


def test_resume_rejects_missing_artifact(tmp_path: Path) -> None:
    config, manifest, artifact = _fixture(tmp_path)
    artifact.unlink()

    with pytest.raises(ResumeError, match="is missing"):
        load_verified_resume_state(
            manifest,
            expected_config=config,
            repository_root=tmp_path,
        )
