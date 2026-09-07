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
from .core_contracts import (
    CORE_CONTRACT_SCHEMA_VERSION,
    ExperimentArtifact,
    ExperimentConfig,
    MetricsResult,
    NullMapping,
    ParentLink,
    ParentSet,
)
from .design_matrix import build_self_history_design_matrix
from .forecaster import (
    FittedRidgeForecaster,
    fit_ridge_forecaster,
    predict_ridge_forecaster,
)
from .metrics import velocity_rmse
from .runner import FoldRunResult, MinimalFoldRunner, OuterTestLockedError
from .scientific_config import ScientificConfigError, load_scientific_config
from .split_manifest_io import (
    SPLIT_MANIFEST_SCHEMA_VERSION,
    build_split_manifest_payload,
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
    "DesignMatrix",
    "ExperimentArtifact",
    "ExperimentConfig",
    "FaceTimeSeries",
    "FittedRidgeForecaster",
    "FoldRunResult",
    "InnerFold",
    "MetricsResult",
    "MinimalFoldRunner",
    "NullMapping",
    "OuterTestLockedError",
    "ParentLink",
    "ParentSet",
    "PredictionArtifact",
    "SPLIT_MANIFEST_SCHEMA_VERSION",
    "SplitManifest",
    "ScientificConfigError",
    "SubjectMetadataLoadError",
    "SubjectMetadataValidationError",
    "aligned_indices",
    "build_grouped_kfold_split_manifests",
    "build_loso_split_manifests",
    "build_self_history_design_matrix",
    "build_split_manifest_payload",
    "build_subject_split_manifest",
    "build_v0_result_payload",
    "fit_ridge_forecaster",
    "generate_synthetic_face_time_series",
    "load_scientific_config",
    "load_subject_metadata_csv",
    "predict_ridge_forecaster",
    "validate_subject_ids",
    "velocity_rmse",
    "write_split_manifest_json",
    "write_v0_result_json",
]
