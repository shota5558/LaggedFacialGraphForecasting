"""Lagged Facial Graph Forecasting research package."""

from .contracts import (
    ContractError,
    DesignMatrix,
    FaceTimeSeries,
    InnerFold,
    PredictionArtifact,
    SplitManifest,
)
from .scientific_config import ScientificConfigError, load_scientific_config
from .splits import build_subject_split_manifest
from .synthetic import generate_synthetic_face_time_series

__all__ = [
    "ContractError",
    "DesignMatrix",
    "FaceTimeSeries",
    "InnerFold",
    "PredictionArtifact",
    "SplitManifest",
    "ScientificConfigError",
    "build_subject_split_manifest",
    "generate_synthetic_face_time_series",
    "load_scientific_config",
]
