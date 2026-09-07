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
from .design_matrix import build_self_history_design_matrix
from .forecaster import (
    FittedRidgeForecaster,
    fit_ridge_forecaster,
    predict_ridge_forecaster,
)
from .metrics import velocity_rmse
from .runner import FoldRunResult, MinimalFoldRunner, OuterTestLockedError
from .scientific_config import ScientificConfigError, load_scientific_config
from .splits import build_subject_split_manifest
from .synthetic import generate_synthetic_face_time_series

__all__ = [
    "AlignedIndices",
    "ContractError",
    "DesignMatrix",
    "FaceTimeSeries",
    "FittedRidgeForecaster",
    "FoldRunResult",
    "InnerFold",
    "MinimalFoldRunner",
    "OuterTestLockedError",
    "PredictionArtifact",
    "SplitManifest",
    "ScientificConfigError",
    "aligned_indices",
    "build_self_history_design_matrix",
    "build_subject_split_manifest",
    "build_v0_result_payload",
    "fit_ridge_forecaster",
    "generate_synthetic_face_time_series",
    "load_scientific_config",
    "predict_ridge_forecaster",
    "velocity_rmse",
    "write_v0_result_json",
]
