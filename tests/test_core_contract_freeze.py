from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lagged_facial_graph_forecasting.core_contracts import (
    CORE_CONTRACT_SCHEMA_VERSION,
    ExperimentArtifact,
    ParentLink,
    ParentSet,
)
from lagged_facial_graph_forecasting.contracts import ContractError


def test_core_contract_registry_freezes_required_contracts_and_component_provenance() -> None:
    registry = yaml.safe_load(
        Path("schemas/core_contracts_v3.yaml").read_text(encoding="utf-8")
    )

    assert registry["schema_version"] == 3
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
        contract["schema_version"] == 3
        for contract in registry["contracts"].values()
    )
    assert registry["contracts"]["ParentSet"]["parent_link_fields"] == [
        "source_region",
        "lag",
        "source_dimension",
        "target_dimension",
    ]
    assert registry["contracts"]["DesignMatrix"]["target_provenance_fields"] == [
        "region_id",
        "target_dimensions",
    ]
    assert registry["contracts"]["PredictionArtifact"]["target_provenance_fields"] == [
        "region_id",
        "target_dimensions",
    ]


def test_parent_set_preserves_component_identity_and_rejects_exact_duplicates() -> None:
    parents = ParentSet(
        outer_fold=0,
        target_region="mouth",
        parents=(
            ParentLink("jaw", 1, "vx", "vx"),
            ParentLink("jaw", 1, "vy", "vx"),
            ParentLink("jaw", 1, "vx", "vy"),
        ),
    )

    assert parents.schema_version == CORE_CONTRACT_SCHEMA_VERSION == 3
    assert len(set(parents.parents)) == 3

    with pytest.raises(ContractError, match="duplicate"):
        ParentSet(
            outer_fold=0,
            target_region="mouth",
            parents=(
                ParentLink("jaw", 1, "vx", "vy"),
                ParentLink("jaw", 1, "vx", "vy"),
            ),
        )


def test_experiment_artifact_rejects_unsafe_path_and_invalid_hash() -> None:
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
