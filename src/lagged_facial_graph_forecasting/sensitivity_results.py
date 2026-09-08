"""Sensitivity-only result envelope reusing frozen Core metric contracts (S-IMPL-08)."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from .core_contract_io import deserialize_core_contract, serialize_core_contract
from .core_contracts import MetricsResult
from .sensitivity_execution import (
    PRIMARY_FREEZE_MANIFEST_PATH,
    SensitivityExperimentConfig,
    assert_real_sensitivity_artifact_provenance,
    assert_sensitivity_execution_allowed,
    resolve_sensitivity_output_path,
)


SENSITIVITY_RESULT_SCHEMA_VERSION = 1
SENSITIVITY_RESULT_NAMESPACE = "sensitivity"
SENSITIVITY_METRIC_ARTIFACT_TYPE = "sensitivity_metric"


class SensitivityResultError(ValueError):
    """Raised when Sensitivity result provenance or serialization is invalid."""


def _non_empty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SensitivityResultError(f"{field} must be a non-empty string")
    return value.strip()


def _relative_path(value: Any, field: str) -> str:
    text = _non_empty_text(value, field)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts:
        raise SensitivityResultError(f"{field} must be a repository-relative POSIX path")
    return path.as_posix()


def _strict_mapping(value: Any, keys: set[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SensitivityResultError(f"{context} must be a mapping")
    actual = set(value)
    if actual != keys:
        raise SensitivityResultError(
            f"{context} fields differ from schema; missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)}"
        )
    return value


@dataclass(frozen=True, slots=True, order=True)
class SoftwareVersion:
    package: str
    version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "package", _non_empty_text(self.package, "package"))
        object.__setattr__(self, "version", _non_empty_text(self.version, "version"))


@dataclass(frozen=True, slots=True)
class SensitivityMetricArtifact:
    method_id: str
    source_primary_freeze_manifest: str
    outer_fold: int
    subject_id: str
    region_id: str
    horizon: int
    metric: MetricsResult
    config_path: str
    seed: int
    software_versions: tuple[SoftwareVersion, ...]
    namespace: str = SENSITIVITY_RESULT_NAMESPACE
    artifact_type: str = SENSITIVITY_METRIC_ARTIFACT_TYPE
    schema_version: int = SENSITIVITY_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "method_id", _non_empty_text(self.method_id, "method_id"))
        if self.method_id.lower() == "primary":
            raise SensitivityResultError("method_id cannot identify a Primary result")

        freeze_path = _relative_path(
            self.source_primary_freeze_manifest, "source_primary_freeze_manifest"
        )
        if freeze_path != PRIMARY_FREEZE_MANIFEST_PATH:
            raise SensitivityResultError(
                "source_primary_freeze_manifest must reference the canonical Primary Freeze manifest"
            )
        object.__setattr__(self, "source_primary_freeze_manifest", freeze_path)
        object.__setattr__(self, "config_path", _relative_path(self.config_path, "config_path"))

        if not isinstance(self.outer_fold, int) or isinstance(self.outer_fold, bool) or self.outer_fold < 0:
            raise SensitivityResultError("outer_fold must be a non-negative integer")
        object.__setattr__(self, "subject_id", _non_empty_text(self.subject_id, "subject_id"))
        object.__setattr__(self, "region_id", _non_empty_text(self.region_id, "region_id"))
        if not isinstance(self.horizon, int) or isinstance(self.horizon, bool) or self.horizon < 1:
            raise SensitivityResultError("horizon must be a positive integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise SensitivityResultError("seed must be a non-negative integer")
        if not isinstance(self.metric, MetricsResult):
            raise SensitivityResultError("metric must be the frozen Core MetricsResult")
        if self.metric.outer_fold != self.outer_fold:
            raise SensitivityResultError("outer_fold must match metric.outer_fold")
        if self.metric.subject_id != self.subject_id:
            raise SensitivityResultError("subject_id must match metric.subject_id")
        if self.metric.region_id != self.region_id:
            raise SensitivityResultError("region_id must match metric.region_id")

        versions = tuple(self.software_versions)
        if not versions or any(not isinstance(item, SoftwareVersion) for item in versions):
            raise SensitivityResultError("software_versions must contain SoftwareVersion records")
        names = [item.package for item in versions]
        if len(set(names)) != len(names):
            raise SensitivityResultError("software_versions package names must be unique")
        object.__setattr__(self, "software_versions", tuple(sorted(versions)))

        if self.namespace != SENSITIVITY_RESULT_NAMESPACE:
            raise SensitivityResultError("namespace is frozen to 'sensitivity'")
        if self.artifact_type != SENSITIVITY_METRIC_ARTIFACT_TYPE:
            raise SensitivityResultError("artifact_type is frozen to 'sensitivity_metric'")
        if self.schema_version != SENSITIVITY_RESULT_SCHEMA_VERSION:
            raise SensitivityResultError("unsupported Sensitivity result schema_version")

    def as_metrics_result(self) -> MetricsResult:
        return self.metric


def collect_software_versions(packages: Iterable[str]) -> tuple[SoftwareVersion, ...]:
    names = tuple(_non_empty_text(name, "package name") for name in packages)
    if not names or len(set(names)) != len(names):
        raise SensitivityResultError("packages must be a non-empty unique sequence")
    versions: list[SoftwareVersion] = []
    for name in names:
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise SensitivityResultError(f"software package is not installed: {name}") from exc
        versions.append(SoftwareVersion(name, version))
    return tuple(sorted(versions))


def serialize_sensitivity_metric(value: SensitivityMetricArtifact) -> dict[str, Any]:
    if not isinstance(value, SensitivityMetricArtifact):
        raise SensitivityResultError("value must be SensitivityMetricArtifact")
    return {
        "schema_version": value.schema_version,
        "namespace": value.namespace,
        "artifact_type": value.artifact_type,
        "payload": {
            "method_id": value.method_id,
            "source_primary_freeze_manifest": value.source_primary_freeze_manifest,
            "outer_fold": value.outer_fold,
            "subject_id": value.subject_id,
            "region_id": value.region_id,
            "horizon": value.horizon,
            "metric": serialize_core_contract(value.metric),
            "config_path": value.config_path,
            "seed": value.seed,
            "software_versions": [
                {"package": item.package, "version": item.version}
                for item in value.software_versions
            ],
        },
    }


def deserialize_sensitivity_metric(envelope: Any) -> SensitivityMetricArtifact:
    root = _strict_mapping(
        envelope,
        {"schema_version", "namespace", "artifact_type", "payload"},
        "SensitivityMetricArtifact",
    )
    if root["schema_version"] != SENSITIVITY_RESULT_SCHEMA_VERSION:
        raise SensitivityResultError("incompatible Sensitivity result schema_version")
    if root["namespace"] != SENSITIVITY_RESULT_NAMESPACE:
        raise SensitivityResultError("Sensitivity result namespace must equal 'sensitivity'")
    if root["artifact_type"] != SENSITIVITY_METRIC_ARTIFACT_TYPE:
        raise SensitivityResultError("unexpected Sensitivity artifact_type")

    payload = _strict_mapping(
        root["payload"],
        {
            "method_id",
            "source_primary_freeze_manifest",
            "outer_fold",
            "subject_id",
            "region_id",
            "horizon",
            "metric",
            "config_path",
            "seed",
            "software_versions",
        },
        "SensitivityMetricArtifact.payload",
    )
    metric = deserialize_core_contract(payload["metric"])
    if not isinstance(metric, MetricsResult):
        raise SensitivityResultError("embedded core contract must be MetricsResult")
    raw_versions = payload["software_versions"]
    if not isinstance(raw_versions, list):
        raise SensitivityResultError("software_versions must be a list")
    versions = []
    for index, item in enumerate(raw_versions):
        record = _strict_mapping(item, {"package", "version"}, f"software_versions[{index}]")
        versions.append(SoftwareVersion(record["package"], record["version"]))

    return SensitivityMetricArtifact(
        method_id=payload["method_id"],
        source_primary_freeze_manifest=payload["source_primary_freeze_manifest"],
        outer_fold=payload["outer_fold"],
        subject_id=payload["subject_id"],
        region_id=payload["region_id"],
        horizon=payload["horizon"],
        metric=metric,
        config_path=payload["config_path"],
        seed=payload["seed"],
        software_versions=tuple(versions),
        namespace=root["namespace"],
        artifact_type=root["artifact_type"],
        schema_version=root["schema_version"],
    )


def dumps_sensitivity_metric(value: SensitivityMetricArtifact) -> str:
    return json.dumps(serialize_sensitivity_metric(value), sort_keys=True, separators=(",", ":"))


def loads_sensitivity_metric(text: str) -> SensitivityMetricArtifact:
    try:
        envelope = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise SensitivityResultError("invalid Sensitivity metric JSON") from exc
    return deserialize_sensitivity_metric(envelope)


def write_sensitivity_metric_json(
    execution: SensitivityExperimentConfig,
    value: SensitivityMetricArtifact,
    relative_path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> Path:
    """Canonical artifact writer: authorize, verify provenance, confine path, then write."""

    assert_sensitivity_execution_allowed(execution, repository_root=repository_root)
    assert_real_sensitivity_artifact_provenance(
        execution, value, repository_root=repository_root
    )
    output = resolve_sensitivity_output_path(
        execution, relative_path, repository_root=repository_root
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(dumps_sensitivity_metric(value) + "\n", encoding="utf-8")
    return output
