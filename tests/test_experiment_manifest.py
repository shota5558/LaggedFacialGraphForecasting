from __future__ import annotations

from pathlib import Path

import pytest

from lagged_facial_graph_forecasting.core_contracts import ExperimentArtifact, ExperimentConfig
from lagged_facial_graph_forecasting.experiment_manifest import (
    ExperimentManifestError,
    build_experiment_manifest,
    write_experiment_manifest,
)


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


def test_manifest_is_deterministic_across_input_order(tmp_path: Path) -> None:
    first = _artifact("artifacts/primary/fold_00/z.json", "a" * 64)
    second = _artifact("artifacts/primary/fold_00/a.json", "b" * 64)

    path_a = write_experiment_manifest(
        tmp_path / "a.json",
        config=_config(),
        artifacts=(first, second),
        software_versions={"scikit-learn": "1.7.0", "numpy": "2.0.0"},
    )
    path_b = write_experiment_manifest(
        tmp_path / "b.json",
        config=_config(),
        artifacts=(second, first),
        software_versions={"numpy": "2.0.0", "scikit-learn": "1.7.0"},
    )

    assert path_a.read_bytes() == path_b.read_bytes()


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
        )


def test_manifest_rejects_artifact_outside_primary_root() -> None:
    sensitivity = _artifact("artifacts/sensitivity/result.json", "d" * 64)

    with pytest.raises(ExperimentManifestError, match="artifact_root"):
        build_experiment_manifest(
            config=_config(),
            artifacts=(sensitivity,),
            software_versions={"python": "3.11"},
        )


def test_manifest_requires_software_provenance() -> None:
    with pytest.raises(ExperimentManifestError, match="software_versions"):
        build_experiment_manifest(
            config=_config(),
            artifacts=(),
            software_versions={},
        )
