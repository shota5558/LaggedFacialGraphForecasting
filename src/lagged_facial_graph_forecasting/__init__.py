"""Lagged Facial Graph Forecasting research package."""

from .alignment import AlignedIndices, aligned_indices
from .artifacts import build_v0_result_payload, write_v0_result_json
from .contracts import (
    ContractError,
    DesignMatrix,
    FaceTimeSeries,
    InnerFold,
    PredictionArtifact,
    SplitManifest,
)
from .core_contract_io import (
    CoreContractIOError,
    deserialize_core_contract,
    dumps_core_contract,
    loads_core_contract,
    serialize_core_contract,
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
from .design_matrix import (
    build_full_history_design_matrix,
    build_pcmci_parent_design_matrix,
    build_persistence_design_matrix,
    build_self_history_design_matrix,
    decode_feature_name,
    encode_feature_name,
)
from .forecaster import (
    FittedRidgeForecaster,
    fit_ridge_forecaster,
    predict_ridge_forecaster,
)
from .lag_response_statistics import (
    LagResponseMetricAggregate,
    aggregate_lag_response_metrics,
)
from .leakage_guard import (
    LeakageGuardError,
    assert_discovery_fit_scope,
    assert_null_construction_scope,
    assert_preprocessing_fit_scope,
    assert_ridge_tuning_scope,
)
from .metrics import velocity_rmse
from .primary_statistics import (
    PrimaryPairedStatistics,
    compute_primary_velocity_paired_statistics,
)
from .runner import FoldRunResult, MinimalFoldRunner, OuterTestLockedError
from .scientific_config import ScientificConfigError, load_scientific_config
from .split_manifest_io import (
    SPLIT_MANIFEST_SCHEMA_VERSION,
    SplitManifestReadError,
    build_split_manifest_payload,
    read_split_manifest_json,
    write_split_manifest_json,
)
from .splits import (
    build_grouped_kfold_split_manifests,
    build_loso_split_manifests,
    build_subject_split_manifest,
)
from .subject_metadata import (
    SubjectMetadataLoadError,
    SubjectMetadataValidationError,
    load_subject_metadata_csv,
    validate_subject_ids,
)
from .synthetic import generate_synthetic_face_time_series

__all__ = [
    "AlignedIndices",
    "CORE_CONTRACT_SCHEMA_VERSION",
    "ContractError",
    "CoreContractIOError",
    "DesignMatrix",
    "ExperimentArtifact",
    "ExperimentConfig",
    "FaceTimeSeries",
    "FittedRidgeForecaster",
    "FoldRunResult",
    "InnerFold",
    "LagResponseMetricAggregate",
    "LeakageGuardError",
    "MetricsResult",
    "MinimalFoldRunner",
    "NullMapping",
    "OuterTestLockedError",
    "ParentLink",
    "ParentSet",
    "PredictionArtifact",
    "PrimaryPairedStatistics",
    "SPLIT_MANIFEST_SCHEMA_VERSION",
    "SplitManifest",
    "SplitManifestReadError",
    "ScientificConfigError",
    "SubjectMetadataLoadError",
    "SubjectMetadataValidationError",
    "aggregate_lag_response_metrics",
    "aligned_indices",
    "assert_discovery_fit_scope",
    "assert_null_construction_scope",
    "assert_preprocessing_fit_scope",
    "assert_ridge_tuning_scope",
    "build_full_history_design_matrix",
    "build_grouped_kfold_split_manifests",
    "build_loso_split_manifests",
    "build_pcmci_parent_design_matrix",
    "build_persistence_design_matrix",
    "build_self_history_design_matrix",
    "build_split_manifest_payload",
    "build_subject_split_manifest",
    "build_v0_result_payload",
    "compute_primary_velocity_paired_statistics",
    "decode_feature_name",
    "deserialize_core_contract",
    "dumps_core_contract",
    "encode_feature_name",
    "fit_ridge_forecaster",
    "generate_synthetic_face_time_series",
    "load_scientific_config",
    "load_subject_metadata_csv",
    "loads_core_contract",
    "predict_ridge_forecaster",
    "read_split_manifest_json",
    "serialize_core_contract",
    "validate_subject_ids",
    "velocity_rmse",
    "write_split_manifest_json",
    "write_v0_result_json",
]
