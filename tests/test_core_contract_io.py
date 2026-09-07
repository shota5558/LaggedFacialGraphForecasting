from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from lagged_facial_graph_forecasting import (
    DesignMatrix,
    ExperimentArtifact,
    ExperimentConfig,
    FaceTimeSeries,
    InnerFold,
    MetricsResult,
    NullMapping,
    ParentLink,
    ParentSet,
    PredictionArtifact,
    SplitManifest,
)
from lagged_facial_graph_forecasting.core_contract_io import (
    CoreContractIOError,
    deserialize_core_contract,
    dumps_core_contract,
    loads_core_contract,
    migrate_core_contract_v1_to_v2,
    serialize_core_contract,
)


def _samples():
    face = FaceTimeSeries(
        X=np.arange(24, dtype=np.float64).reshape(4, 3, 2),
        subject_id="s01",
        time_index=np.arange(4, dtype=np.int64),
        region_id=("mouth", "left_eye", "right_eye"),
        dimension=("vx", "vy"),
        valid_mask=np.ones((4, 3, 2), dtype=bool),
        sampling_rate=30.0,
    )
    manifest = SplitManifest(
        outer_fold=2,
        train_subject_ids=("s01", "s02", "s03"),
        test_subject_ids=("s04",),
        inner_folds=(
            InnerFold(("s01", "s02"), ("s03",)),
            InnerFold(("s02", "s03"), ("s01",)),
        ),
        seed=17,
    )
    parent_set = ParentSet(
        outer_fold=2,
        target_region="mouth",
        parents=(
            ParentLink("left_eye", 1, "vx", "vy"),
            ParentLink("right_eye", 2, "vy", "vx"),
        ),
    )
    design = DesignMatrix(
        X=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        y=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        subject_id=("s01", "s01"),
        region_id=("mouth", "mouth"),
        forecast_origin=np.array([5, 6], dtype=np.int64),
        target_time=np.array([6, 7], dtype=np.int64),
        feature_names=("left_eye.vx", "right_eye.vy"),
        feature_lags=(1, 2),
        valid_mask=np.array([True, True], dtype=bool),
    )
    prediction = PredictionArtifact(
        outer_fold=2,
        subject_id=("s04", "s04"),
        region_id=("mouth", "mouth"),
        condition="pcmci",
        forecast_origin=np.array([5, 6], dtype=np.int64),
        target_time=np.array([6, 7], dtype=np.int64),
        y_true=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        y_pred=np.array([[0.11, 0.19], [0.29, 0.41]], dtype=np.float64),
        valid_mask=np.array([True, True], dtype=bool),
    )
    null_mapping = NullMapping(
        outer_fold=2,
        target_region="mouth",
        condition="random-region",
        seed=19,
        source_parents=(
            ParentLink("left_eye", 1, "vx", "vy"),
            ParentLink("right_eye", 2, "vy", "vx"),
        ),
        mapped_parents=(
            ParentLink("left_cheek", 1, "vx", "vy"),
            ParentLink("right_cheek", 2, "vy", "vx"),
        ),
        permutation=(1, 0),
    )
    metrics = MetricsResult(
        outer_fold=2,
        subject_id="s04",
        region_id="mouth",
        condition="pcmci",
        metric_name="velocity_rmse",
        value=0.125,
        n_valid=20,
    )
    artifact = ExperimentArtifact(
        experiment_id="primary-001",
        artifact_type="prediction",
        relative_path="artifacts/primary/fold_02/prediction.json",
        sha256="a" * 64,
        outer_fold=2,
        condition="pcmci",
    )
    config = ExperimentConfig(
        experiment_id="primary-001",
        scientific_config_path="configs/scientific_freeze.yaml",
        run_config_path="configs/primary_run.yaml",
        artifact_root="artifacts/primary",
        seed=20260908,
    )
    return (
        face,
        manifest,
        parent_set,
        design,
        prediction,
        null_mapping,
        metrics,
        artifact,
        config,
    )


@pytest.mark.parametrize("sample", _samples())
def test_all_canonical_contracts_round_trip_through_deterministic_json(sample) -> None:
    encoded = dumps_core_contract(sample)
    restored = loads_core_contract(encoded)

    assert type(restored) is type(sample)
    assert dumps_core_contract(restored) == encoded
    assert serialize_core_contract(restored) == serialize_core_contract(sample)


def test_parent_component_provenance_round_trips_losslessly() -> None:
    sample = _samples()[2]
    envelope = serialize_core_contract(sample)

    assert envelope["schema_version"] == 2
    assert envelope["payload"]["parents"] == [
        {
            "source_region": "left_eye",
            "lag": 1,
            "source_dimension": "vx",
            "target_dimension": "vy",
        },
        {
            "source_region": "right_eye",
            "lag": 2,
            "source_dimension": "vy",
            "target_dimension": "vx",
        },
    ]
    assert deserialize_core_contract(envelope) == sample


@pytest.mark.parametrize("version", [0, 1, 3])
def test_rejects_noncurrent_schema_versions(version: int) -> None:
    envelope = serialize_core_contract(_samples()[2])
    envelope["schema_version"] = version

    with pytest.raises(CoreContractIOError, match="incompatible core contract schema_version"):
        deserialize_core_contract(envelope)


def test_v1_parentset_migration_requires_certified_scalar_dimension() -> None:
    v1 = {
        "schema_version": 1,
        "contract_type": "ParentSet",
        "payload": {
            "outer_fold": 2,
            "target_region": "mouth",
            "parents": [
                {"source_region": "left_eye", "lag": 1},
                {"source_region": "right_eye", "lag": 2},
            ],
            "discovery_method": "pcmci_plus",
        },
    }

    with pytest.raises(CoreContractIOError, match="certified_scalar_dimension"):
        migrate_core_contract_v1_to_v2(v1)

    migrated = migrate_core_contract_v1_to_v2(
        v1,
        certified_scalar_dimension="vx",
    )
    assert migrated["schema_version"] == 2
    assert migrated["payload"]["parents"][0] == {
        "source_region": "left_eye",
        "lag": 1,
        "source_dimension": "vx",
        "target_dimension": "vx",
    }
    restored = deserialize_core_contract(migrated)
    assert restored.parents[0] == ParentLink("left_eye", 1, "vx", "vx")


def test_v1_null_mapping_migration_preserves_mapping_with_certified_scalar_dimension() -> None:
    v1 = {
        "schema_version": 1,
        "contract_type": "NullMapping",
        "payload": {
            "outer_fold": 2,
            "target_region": "mouth",
            "condition": "random-region",
            "seed": 19,
            "source_parents": [{"source_region": "left_eye", "lag": 1}],
            "mapped_parents": [{"source_region": "left_cheek", "lag": 1}],
            "permutation": [],
        },
    }

    migrated = migrate_core_contract_v1_to_v2(
        v1,
        certified_scalar_dimension="value",
    )
    restored = deserialize_core_contract(migrated)

    assert restored.source_parents == (ParentLink("left_eye", 1),)
    assert restored.mapped_parents == (ParentLink("left_cheek", 1),)


def test_v1_non_parent_contract_migrates_without_payload_change() -> None:
    current = serialize_core_contract(_samples()[6])
    v1 = deepcopy(current)
    v1["schema_version"] = 1

    migrated = migrate_core_contract_v1_to_v2(v1)

    assert migrated["schema_version"] == 2
    assert migrated["payload"] == current["payload"]
    assert deserialize_core_contract(migrated) == _samples()[6]


def test_rejects_unknown_contract_type() -> None:
    envelope = serialize_core_contract(_samples()[2])
    envelope["contract_type"] = "FutureParentSet"

    with pytest.raises(CoreContractIOError, match="unknown core contract type"):
        deserialize_core_contract(envelope)


def test_rejects_forward_payload_extension() -> None:
    envelope = serialize_core_contract(_samples()[2])
    envelope["payload"]["future_field"] = "unreviewed"

    with pytest.raises(CoreContractIOError, match="fields are incompatible"):
        deserialize_core_contract(envelope)


def test_rejects_component_payload_missing_field() -> None:
    envelope = deepcopy(serialize_core_contract(_samples()[2]))
    del envelope["payload"]["parents"][0]["source_dimension"]

    with pytest.raises(CoreContractIOError, match="fields are incompatible"):
        deserialize_core_contract(envelope)
