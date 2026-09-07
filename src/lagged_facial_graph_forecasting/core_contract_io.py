"""Strict JSON serialization for the frozen core contract registry.

Every canonical contract is serialized through a versioned envelope.  The
reader rejects unknown versions, contract types, missing fields, and additional
fields so backward/forward incompatibility is explicit instead of silently
accepted.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from .contracts import (
    ContractError,
    DesignMatrix,
    FaceTimeSeries,
    InnerFold,
    PredictionArtifact,
    SplitManifest,
)
from .core_contracts import (
    CORE_CONTRACT_SCHEMA_VERSION,
    ExperimentArtifact,
    ExperimentConfig,
    MetricsResult,
    NullMapping,
    ParentLink,
    ParentSet,
)


class CoreContractIOError(ContractError):
    """Raised when a serialized core-contract envelope is incompatible."""


_CANONICAL_TYPES = {
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


def _strict_mapping(
    value: Any, required_keys: set[str], context: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CoreContractIOError(f"{context} must be a mapping")
    actual_keys = set(value)
    if actual_keys != required_keys:
        missing = sorted(required_keys - actual_keys)
        extra = sorted(actual_keys - required_keys)
        raise CoreContractIOError(
            f"{context} fields are incompatible; missing={missing}, extra={extra}"
        )
    return value


def _encode_array(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "data": array.tolist(),
    }


def _decode_array(value: Any, context: str) -> np.ndarray:
    payload = _strict_mapping(value, {"dtype", "shape", "data"}, context)
    dtype_raw = payload["dtype"]
    shape_raw = payload["shape"]
    if not isinstance(dtype_raw, str):
        raise CoreContractIOError(f"{context}.dtype must be a string")
    if not isinstance(shape_raw, list) or any(
        not isinstance(size, int) or isinstance(size, bool) or size < 0
        for size in shape_raw
    ):
        raise CoreContractIOError(
            f"{context}.shape must be a list of non-negative integers"
        )
    try:
        dtype = np.dtype(dtype_raw)
        array = np.asarray(payload["data"], dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise CoreContractIOError(f"{context} contains invalid array data") from exc
    expected_shape = tuple(shape_raw)
    if array.shape != expected_shape:
        raise CoreContractIOError(
            f"{context}.shape mismatch: declared={expected_shape}, actual={array.shape}"
        )
    return array


def _encode_parent(parent: ParentLink) -> dict[str, Any]:
    return {"source_region": parent.source_region, "lag": parent.lag}


def _decode_parent(value: Any, context: str) -> ParentLink:
    payload = _strict_mapping(value, {"source_region", "lag"}, context)
    return ParentLink(
        source_region=payload["source_region"],
        lag=payload["lag"],
    )


def serialize_core_contract(value: Any) -> dict[str, Any]:
    """Serialize one canonical contract into the frozen v1 envelope."""

    contract_type = type(value).__name__
    if contract_type not in _CANONICAL_TYPES:
        raise CoreContractIOError(
            f"unsupported core contract type: {contract_type}"
        )

    if isinstance(value, FaceTimeSeries):
        payload = {
            "X": _encode_array(value.X),
            "subject_id": value.subject_id,
            "time_index": _encode_array(value.time_index),
            "region_id": list(value.region_id),
            "dimension": list(value.dimension),
            "valid_mask": _encode_array(value.valid_mask),
            "sampling_rate": value.sampling_rate,
        }
    elif isinstance(value, SplitManifest):
        payload = {
            "outer_fold": value.outer_fold,
            "train_subject_ids": list(value.train_subject_ids),
            "test_subject_ids": list(value.test_subject_ids),
            "inner_folds": [
                {
                    "inner_train_subject_ids": list(fold.inner_train_subject_ids),
                    "inner_val_subject_ids": list(fold.inner_val_subject_ids),
                }
                for fold in value.inner_folds
            ],
            "seed": value.seed,
        }
    elif isinstance(value, ParentSet):
        payload = {
            "outer_fold": value.outer_fold,
            "target_region": value.target_region,
            "parents": [_encode_parent(parent) for parent in value.parents],
            "discovery_method": value.discovery_method,
        }
    elif isinstance(value, DesignMatrix):
        payload = {
            "X": _encode_array(value.X),
            "y": _encode_array(value.y),
            "subject_id": list(value.subject_id),
            "region_id": list(value.region_id),
            "forecast_origin": _encode_array(value.forecast_origin),
            "target_time": _encode_array(value.target_time),
            "feature_names": list(value.feature_names),
            "feature_lags": list(value.feature_lags),
            "valid_mask": _encode_array(value.valid_mask),
        }
    elif isinstance(value, PredictionArtifact):
        payload = {
            "outer_fold": value.outer_fold,
            "subject_id": list(value.subject_id),
            "region_id": list(value.region_id),
            "condition": value.condition,
            "forecast_origin": _encode_array(value.forecast_origin),
            "target_time": _encode_array(value.target_time),
            "y_true": _encode_array(value.y_true),
            "y_pred": _encode_array(value.y_pred),
            "valid_mask": _encode_array(value.valid_mask),
        }
    elif isinstance(value, NullMapping):
        payload = {
            "outer_fold": value.outer_fold,
            "target_region": value.target_region,
            "condition": value.condition,
            "seed": value.seed,
            "source_parents": [
                _encode_parent(parent) for parent in value.source_parents
            ],
            "mapped_parents": [
                _encode_parent(parent) for parent in value.mapped_parents
            ],
            "permutation": list(value.permutation),
        }
    elif isinstance(value, MetricsResult):
        payload = {
            "outer_fold": value.outer_fold,
            "subject_id": value.subject_id,
            "region_id": value.region_id,
            "condition": value.condition,
            "metric_name": value.metric_name,
            "value": value.value,
            "n_valid": value.n_valid,
        }
    elif isinstance(value, ExperimentArtifact):
        payload = {
            "experiment_id": value.experiment_id,
            "artifact_type": value.artifact_type,
            "relative_path": value.relative_path,
            "sha256": value.sha256,
            "outer_fold": value.outer_fold,
            "condition": value.condition,
        }
    elif isinstance(value, ExperimentConfig):
        payload = {
            "experiment_id": value.experiment_id,
            "scientific_config_path": value.scientific_config_path,
            "run_config_path": value.run_config_path,
            "artifact_root": value.artifact_root,
            "seed": value.seed,
        }
    else:  # pragma: no cover - guarded by the canonical type-name check above.
        raise CoreContractIOError(f"unsupported core contract type: {contract_type}")

    return {
        "schema_version": CORE_CONTRACT_SCHEMA_VERSION,
        "contract_type": contract_type,
        "payload": payload,
    }


def deserialize_core_contract(envelope: Any) -> Any:
    """Deserialize a canonical contract and reject incompatible wire formats."""

    root = _strict_mapping(
        envelope, {"schema_version", "contract_type", "payload"}, "envelope"
    )
    version = root["schema_version"]
    if version != CORE_CONTRACT_SCHEMA_VERSION:
        raise CoreContractIOError(
            "incompatible core contract schema_version: "
            f"expected={CORE_CONTRACT_SCHEMA_VERSION}, actual={version!r}"
        )
    contract_type = root["contract_type"]
    if contract_type not in _CANONICAL_TYPES:
        raise CoreContractIOError(f"unknown core contract type: {contract_type!r}")
    payload = root["payload"]

    if contract_type == "FaceTimeSeries":
        values = _strict_mapping(
            payload,
            {
                "X",
                "subject_id",
                "time_index",
                "region_id",
                "dimension",
                "valid_mask",
                "sampling_rate",
            },
            "FaceTimeSeries.payload",
        )
        return FaceTimeSeries(
            X=_decode_array(values["X"], "FaceTimeSeries.X"),
            subject_id=values["subject_id"],
            time_index=_decode_array(
                values["time_index"], "FaceTimeSeries.time_index"
            ),
            region_id=tuple(values["region_id"]),
            dimension=tuple(values["dimension"]),
            valid_mask=_decode_array(
                values["valid_mask"], "FaceTimeSeries.valid_mask"
            ),
            sampling_rate=values["sampling_rate"],
        )

    if contract_type == "SplitManifest":
        values = _strict_mapping(
            payload,
            {
                "outer_fold",
                "train_subject_ids",
                "test_subject_ids",
                "inner_folds",
                "seed",
            },
            "SplitManifest.payload",
        )
        inner_raw = values["inner_folds"]
        if not isinstance(inner_raw, list):
            raise CoreContractIOError("SplitManifest.inner_folds must be a list")
        inner_folds = []
        for index, fold_raw in enumerate(inner_raw):
            fold = _strict_mapping(
                fold_raw,
                {"inner_train_subject_ids", "inner_val_subject_ids"},
                f"SplitManifest.inner_folds[{index}]",
            )
            inner_folds.append(
                InnerFold(
                    inner_train_subject_ids=tuple(fold["inner_train_subject_ids"]),
                    inner_val_subject_ids=tuple(fold["inner_val_subject_ids"]),
                )
            )
        return SplitManifest(
            outer_fold=values["outer_fold"],
            train_subject_ids=tuple(values["train_subject_ids"]),
            test_subject_ids=tuple(values["test_subject_ids"]),
            inner_folds=tuple(inner_folds),
            seed=values["seed"],
        )

    if contract_type == "ParentSet":
        values = _strict_mapping(
            payload,
            {"outer_fold", "target_region", "parents", "discovery_method"},
            "ParentSet.payload",
        )
        if not isinstance(values["parents"], list):
            raise CoreContractIOError("ParentSet.parents must be a list")
        return ParentSet(
            outer_fold=values["outer_fold"],
            target_region=values["target_region"],
            parents=tuple(
                _decode_parent(parent, f"ParentSet.parents[{index}]")
                for index, parent in enumerate(values["parents"])
            ),
            discovery_method=values["discovery_method"],
            schema_version=version,
        )

    if contract_type == "DesignMatrix":
        values = _strict_mapping(
            payload,
            {
                "X",
                "y",
                "subject_id",
                "region_id",
                "forecast_origin",
                "target_time",
                "feature_names",
                "feature_lags",
                "valid_mask",
            },
            "DesignMatrix.payload",
        )
        return DesignMatrix(
            X=_decode_array(values["X"], "DesignMatrix.X"),
            y=_decode_array(values["y"], "DesignMatrix.y"),
            subject_id=tuple(values["subject_id"]),
            region_id=tuple(values["region_id"]),
            forecast_origin=_decode_array(
                values["forecast_origin"], "DesignMatrix.forecast_origin"
            ),
            target_time=_decode_array(
                values["target_time"], "DesignMatrix.target_time"
            ),
            feature_names=tuple(values["feature_names"]),
            feature_lags=tuple(values["feature_lags"]),
            valid_mask=_decode_array(values["valid_mask"], "DesignMatrix.valid_mask"),
        )

    if contract_type == "PredictionArtifact":
        values = _strict_mapping(
            payload,
            {
                "outer_fold",
                "subject_id",
                "region_id",
                "condition",
                "forecast_origin",
                "target_time",
                "y_true",
                "y_pred",
                "valid_mask",
            },
            "PredictionArtifact.payload",
        )
        return PredictionArtifact(
            outer_fold=values["outer_fold"],
            subject_id=tuple(values["subject_id"]),
            region_id=tuple(values["region_id"]),
            condition=values["condition"],
            forecast_origin=_decode_array(
                values["forecast_origin"], "PredictionArtifact.forecast_origin"
            ),
            target_time=_decode_array(
                values["target_time"], "PredictionArtifact.target_time"
            ),
            y_true=_decode_array(values["y_true"], "PredictionArtifact.y_true"),
            y_pred=_decode_array(values["y_pred"], "PredictionArtifact.y_pred"),
            valid_mask=_decode_array(
                values["valid_mask"], "PredictionArtifact.valid_mask"
            ),
        )

    if contract_type == "NullMapping":
        values = _strict_mapping(
            payload,
            {
                "outer_fold",
                "target_region",
                "condition",
                "seed",
                "source_parents",
                "mapped_parents",
                "permutation",
            },
            "NullMapping.payload",
        )
        for key in ("source_parents", "mapped_parents"):
            if not isinstance(values[key], list):
                raise CoreContractIOError(f"NullMapping.{key} must be a list")
        return NullMapping(
            outer_fold=values["outer_fold"],
            target_region=values["target_region"],
            condition=values["condition"],
            seed=values["seed"],
            source_parents=tuple(
                _decode_parent(parent, f"NullMapping.source_parents[{index}]")
                for index, parent in enumerate(values["source_parents"])
            ),
            mapped_parents=tuple(
                _decode_parent(parent, f"NullMapping.mapped_parents[{index}]")
                for index, parent in enumerate(values["mapped_parents"])
            ),
            permutation=tuple(values["permutation"]),
            schema_version=version,
        )

    if contract_type == "MetricsResult":
        values = _strict_mapping(
            payload,
            {
                "outer_fold",
                "subject_id",
                "region_id",
                "condition",
                "metric_name",
                "value",
                "n_valid",
            },
            "MetricsResult.payload",
        )
        return MetricsResult(**values, schema_version=version)

    if contract_type == "ExperimentArtifact":
        values = _strict_mapping(
            payload,
            {
                "experiment_id",
                "artifact_type",
                "relative_path",
                "sha256",
                "outer_fold",
                "condition",
            },
            "ExperimentArtifact.payload",
        )
        return ExperimentArtifact(**values, schema_version=version)

    if contract_type == "ExperimentConfig":
        values = _strict_mapping(
            payload,
            {
                "experiment_id",
                "scientific_config_path",
                "run_config_path",
                "artifact_root",
                "seed",
            },
            "ExperimentConfig.payload",
        )
        return ExperimentConfig(**values, schema_version=version)

    raise CoreContractIOError(f"unhandled core contract type: {contract_type}")


def dumps_core_contract(value: Any) -> str:
    """Return a deterministic JSON representation of a canonical contract."""

    return json.dumps(
        serialize_core_contract(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def loads_core_contract(text: str) -> Any:
    """Load deterministic JSON and strictly validate the frozen v1 envelope."""

    if not isinstance(text, str):
        raise CoreContractIOError("serialized core contract must be JSON text")
    try:
        envelope = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CoreContractIOError("invalid core contract JSON") from exc
    return deserialize_core_contract(envelope)
