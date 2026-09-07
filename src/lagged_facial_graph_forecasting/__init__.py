"""Lagged Facial Graph Forecasting research package."""

from .contracts import (
    ContractError,
    DesignMatrix,
    FaceTimeSeries,
    InnerFold,
    SplitManifest,
)
from .scientific_config import ScientificConfigError, load_scientific_config

__all__ = [
    "ContractError",
    "DesignMatrix",
    "FaceTimeSeries",
    "InnerFold",
    "SplitManifest",
    "ScientificConfigError",
    "load_scientific_config",
]
