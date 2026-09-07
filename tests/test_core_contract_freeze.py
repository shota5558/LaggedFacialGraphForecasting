from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lagged_facial_graph_forecasting.core_contracts import (
    CORE_CONTRACT_SCHEMA_VERSION,
    ExperimentArtifact,
    ExperimentConfig,
    MetricsResult,
    NullMapping,
    ParentLink,
    ParentSet,
)
from lagged_facial_graph_forecasting.contracts import ContractError


def test_core_contract_registry_freezes_all_required_contracts_at_v1() -> None:
    registry = yaml.safe_load(Path("schemas/core_contracts_v1.yaml").read_text(encoding="utf-8"))

    assert registry["schema_version"] == 1
    assert registry["status"] == "frozen_after_v0"
    assert set(registry["contracts"]) == {
        "FaceTimeSeries",
        "SplitManifest",
        "ParentSet",
        "DesignMatrix",
        "PredictionArtifact",
        "NullMapping",
        "MetricsResult",
        "ExperimentArtifact",
        "ExperimentConfig",
    }
    assert all(
        contract["schema_version"] == 1
        for contract in registry["contracts"].values()
    )
    assert registry["migration_policy"]["required_on_change"] == [
        "reason",
        "impact_scope",
        "migration",
        "tests",
    ]


def test_parent_set_preserves_target_source_and_lag_without_tigramite_coupling() -> None:
    parents = ParentSet(
        outer_fold=2,
        target_region="mouth",
        parents=(ParentLink("left_cheek", 2), ParentLink("jaw", 1)),
    )

    assert parents.schema_version == CORE_CONTRACT_SCHEMA_VERSION == 1
    assert parents.target_region == "mouth"
    assert parents.parents == (
        ParentLink("left_cheek", 2),
        ParentLink("jaw", 1),
    )
    assert parents.discovery_method == "pcmci_plus"

    with pytest.raises(ContractError, match="duplicate"):
        ParentSet(
            outer_fold=0,
            target_region="mouth",
            parents=(ParentLink("jaw", 1), ParentLink("jaw", 1)),
        )


def test_null_mapping_is_frozen_reproducible_and_validates_permutation() -> None:
    source = (ParentLink("left_cheek", 2), ParentLink("jaw", 1))
    mapped = (ParentLink("right_cheek", 2), ParentLink("left_eye", 1))
    mapping = NullMapping(
        outer_fold=1,
        target_region="mouth",
        condition="random-region",
        seed=11,
        source_parents=source,
        mapped_parents=mapped,
        permutation=(2, 0, 1),
    )

    assert mapping.schema_version == 1
    assert mapping.seed == 11
    assert mapping.source_parents == source
    assert mapping.mapped_parents == mapped
    assert mapping.permutation == (2, 0, 1)

    with pytest.raises(ContractError, match="each row index exactly once"):
        NullMapping(
            outer_fold=1,
            target_region="mouth",
            condition="time-shuffle",
            seed=11,
            source_parents=source,
            mapped_parents=source,
            permutation=(0, 0, 1),
        )


def test_metrics_result_is_subject_region_condition_aligned() -> None:
    result = MetricsResult(
        outer_fold=3,
        subject_id="s01",
        region_id="mouth",
        condition="self",
        metric_name="velocity_rmse",
        value=0.125,
        n_valid=37,
    )

    assert result.schema_version == 1
    assert result.value == 0.125
    assert result.n_valid == 37

    with pytest.raises(ContractError, match="finite"):
        MetricsResult(
            outer_fold=3,
            subject_id="s01",
            region_id="mouth",
            condition="self",
            metric_name="velocity_rmse",
            value=float("nan"),
            n_valid=37,
        )


def test_experiment_artifact_requires_content_hash_and_relative_path() -> None:
    artifact = ExperimentArtifact(
        experiment_id="primary-001",
        artifact_type="prediction",
        relative_path="artifacts/primary/fold_00/prediction.json",
        sha256="a" * 64,
        outer_fold=0,
        condition="self",
    )

    assert artifact.schema_version == 1
    assert artifact.relative_path == "artifacts/primary/fold_00/prediction.json"

    with pytest.raises(ContractError, match="repository-relative"):
        ExperimentArtifact(
            experiment_id="primary-001",
            artifact_type="prediction",
            relative_path="../escape.json",
            sha256="a" * 64,
        )

    with pytest.raises(ContractError, match="64 lowercase hexadecimal"):
        ExperimentArtifact(
            experiment_id="primary-001",
            artifact_type="prediction",
            relative_path="artifacts/prediction.json",
            sha256="not-a-hash",
        )


def test_experiment_config_is_stable_path_based_envelope() -> None:
    config = ExperimentConfig(
        experiment_id="primary-001",
        scientific_config_path="configs/scientific_freeze.yaml",
        run_config_path="configs/primary_run.yaml",
        artifact_root="artifacts/primary",
        seed=20260908,
    )

    assert config.schema_version == 1
    assert config.scientific_config_path == "configs/scientific_freeze.yaml"
    assert config.run_config_path == "configs/primary_run.yaml"
    assert config.artifact_root == "artifacts/primary"
    assert config.seed == 20260908

    with pytest.raises(ContractError, match="schema_version must be 1"):
        ExperimentConfig(
            experiment_id="primary-001",
            scientific_config_path="configs/scientific_freeze.yaml",
            run_config_path="configs/primary_run.yaml",
            artifact_root="artifacts/primary",
            seed=0,
            schema_version=2,
        )
