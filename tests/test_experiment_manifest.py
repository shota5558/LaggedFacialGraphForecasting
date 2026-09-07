from __future__ import annotations

from pathlib import Path

import pytest

from lagged_facial_graph_forecasting.core_contracts import ExperimentArtifact, ExperimentConfig
from lagged_facial_graph_forecasting.experiment_manifest import (
    ExperimentManifestError,
    build_experiment_manifest,
    write_experiment_manifest,
)
from lagged_facial_graph_forecasting.preprocessing_provenance import (
    FeatureSemantics,
    PreprocessingProvenance,
)
from lagged_facial_graph_forecasting.regions import build_region_definition


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="primary-test",
        scientific_config_path="configs/scientific_freeze.yaml",
        run_config_path="configs/primary_run.yaml",
        artifact_root="artifacts/primary",
        seed=17,
    )


def _artifact(path: str, sha: str, *, experiment_id: str = "primary-test") -> ExperimentArtifact:
    return ExperimentArtifact(
        experiment_id=experiment_id,
        artifact_type="prediction",
        relative_path=path,
        sha256=sha,
        outer_fold=0,
        condition="self",
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
            units="normalized_coordinate_per_second",
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


def test_manifest_is_deterministic_across_input_order(tmp_path: Path) -> None:
    first = _artifact("artifacts/primary/fold_00/z.json", "a" * 64)
    second = _artifact("artifacts/primary/fold_00/a.json", "b" * 64)
    provenance = _preprocessing_provenance()

    path_a = write_experiment_manifest(
        tmp_path / "a.json",
        config=_config(),
        artifacts=(first, second),
        software_versions={"scikit-learn": "1.7.0", "numpy": "2.0.0"},
        preprocessing_provenance=provenance,
    )
    path_b = write_experiment_manifest(
        tmp_path / "b.json",
        config=_config(),
        artifacts=(second, first),
        software_versions={"numpy": "2.0.0", "scikit-learn": "1.7.0"},
        preprocessing_provenance=provenance,
    )

    assert path_a.read_bytes() == path_b.read_bytes()


def test_manifest_records_region_definition_and_feature_semantics() -> None:
    payload = build_experiment_manifest(
        config=_config(),
        artifacts=(
            _artifact("artifacts/primary/fold_00/prediction.json", "a" * 64),
        ),
        software_versions={"python": "3.11"},
        preprocessing_provenance=_preprocessing_provenance(),
    )

    provenance = payload["preprocessing_provenance"]
    assert provenance["region_definition"]["ordered_regions"] == [
        {"region_id": "mouth", "landmark_indices": [0, 1]},
        {"region_id": "left_cheek", "landmark_indices": [2, 3]},
    ]
    assert provenance["feature_semantics"] == {
        "feature_name": "velocity",
        "dimensions": ["v_x", "v_y"],
        "method": "displacement_over_observed_time_interval",
        "units": "normalized_coordinate_per_second",
        "timestamp_basis": "observed_frame_timestamps",
    }
    assert provenance["acceleration_primary"] is False


def test_manifest_requires_preprocessing_provenance() -> None:
    with pytest.raises(ExperimentManifestError, match="preprocessing_provenance"):
        build_experiment_manifest(
            config=_config(),
            artifacts=(
                _artifact("artifacts/primary/fold_00/prediction.json", "a" * 64),
            ),
            software_versions={"python": "3.11"},
        )


def test_manifest_rejects_artifact_from_other_experiment() -> None:
    foreign = _artifact(
        "artifacts/primary/fold_00/prediction.json",
        "c" * 64,
        experiment_id="other",
    )

    with pytest.raises(ExperimentManifestError, match="experiment_id"):
        build_experiment_manifest(
            config=_config(),
            artifacts=(foreign,),
            software_versions={"python": "3.11"},
            preprocessing_provenance=_preprocessing_provenance(),
        )


def test_manifest_rejects_artifact_outside_primary_root() -> None:
    sensitivity = _artifact("artifacts/sensitivity/result.json", "d" * 64)

    with pytest.raises(ExperimentManifestError, match="artifact_root"):
        build_experiment_manifest(
            config=_config(),
            artifacts=(sensitivity,),
            software_versions={"python": "3.11"},
            preprocessing_provenance=_preprocessing_provenance(),
        )


def test_manifest_requires_software_provenance() -> None:
    with pytest.raises(ExperimentManifestError, match="software_versions"):
        build_experiment_manifest(
            config=_config(),
            artifacts=(),
            software_versions={},
            preprocessing_provenance=_preprocessing_provenance(),
        )
