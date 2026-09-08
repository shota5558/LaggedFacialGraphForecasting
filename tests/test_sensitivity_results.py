from __future__ import annotations

import json

import pytest

from lagged_facial_graph_forecasting.core_contract_io import serialize_core_contract
from lagged_facial_graph_forecasting.core_contracts import MetricsResult
from lagged_facial_graph_forecasting.sensitivity_results import (
    SensitivityMetricArtifact,
    SensitivityResultError,
    SoftwareVersion,
    collect_software_versions,
    dumps_sensitivity_metric,
    loads_sensitivity_metric,
    serialize_sensitivity_metric,
)


def _metric(condition: str = "gru") -> MetricsResult:
    return MetricsResult(
        outer_fold=2,
        subject_id="subject_07",
        region_id="mouth",
        condition=condition,
        metric_name="velocity_rmse",
        value=0.125,
        n_valid=42,
    )


def _artifact() -> SensitivityMetricArtifact:
    metric = _metric()
    return SensitivityMetricArtifact(
        method_id="gru",
        source_primary_freeze_manifest="artifacts/primary/freeze_manifest.json",
        outer_fold=metric.outer_fold,
        subject_id=metric.subject_id,
        region_id=metric.region_id,
        horizon=1,
        metric=metric,
        config_path="configs/sensitivity_preimplementation.yaml",
        seed=808,
        software_versions=(
            SoftwareVersion("numpy", "2.4.0"),
            SoftwareVersion("torch", "2.8.0+cpu"),
        ),
    )


def test_sensitivity_metric_round_trip_reuses_core_metric_for_paired_statistics() -> None:
    artifact = _artifact()
    restored = loads_sensitivity_metric(dumps_sensitivity_metric(artifact))

    assert restored == artifact
    assert restored.as_metrics_result() == artifact.metric
    assert serialize_core_contract(restored.metric) == serialize_core_contract(artifact.metric)
    assert restored.outer_fold == restored.metric.outer_fold
    assert restored.subject_id == restored.metric.subject_id
    assert restored.region_id == restored.metric.region_id
    assert restored.horizon == 1


def test_sensitivity_wire_format_is_explicitly_separate_from_primary_results() -> None:
    envelope = serialize_sensitivity_metric(_artifact())
    assert envelope["namespace"] == "sensitivity"
    assert envelope["artifact_type"] == "sensitivity_metric"
    assert envelope["payload"]["method_id"] == "gru"
    assert envelope["payload"]["source_primary_freeze_manifest"].startswith(
        "artifacts/primary/"
    )
    assert envelope["payload"]["metric"]["contract_type"] == "MetricsResult"

    contaminated = json.loads(json.dumps(envelope))
    contaminated["namespace"] = "primary"
    with pytest.raises(SensitivityResultError, match="namespace"):
        loads_sensitivity_metric(json.dumps(contaminated))


def test_sensitivity_metric_rejects_provenance_mismatch_or_primary_method() -> None:
    metric = _metric()
    with pytest.raises(SensitivityResultError, match="subject_id"):
        SensitivityMetricArtifact(
            method_id="gru",
            source_primary_freeze_manifest="artifacts/primary/freeze_manifest.json",
            outer_fold=metric.outer_fold,
            subject_id="different_subject",
            region_id=metric.region_id,
            horizon=1,
            metric=metric,
            config_path="configs/sensitivity_preimplementation.yaml",
            seed=1,
            software_versions=(SoftwareVersion("numpy", "2.4.0"),),
        )

    with pytest.raises(SensitivityResultError, match="Primary"):
        SensitivityMetricArtifact(
            method_id="primary",
            source_primary_freeze_manifest="artifacts/primary/freeze_manifest.json",
            outer_fold=metric.outer_fold,
            subject_id=metric.subject_id,
            region_id=metric.region_id,
            horizon=1,
            metric=metric,
            config_path="configs/sensitivity_preimplementation.yaml",
            seed=1,
            software_versions=(SoftwareVersion("numpy", "2.4.0"),),
        )


def test_software_version_provenance_can_be_collected_from_installed_packages() -> None:
    versions = collect_software_versions(("numpy", "scikit-learn"))
    assert tuple(item.package for item in versions) == ("numpy", "scikit-learn")
    assert all(item.version for item in versions)
