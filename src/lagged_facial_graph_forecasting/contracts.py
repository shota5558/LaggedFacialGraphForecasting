"""Core data contracts for the facial motion forecasting pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ContractError(ValueError):
    """Raised when a core data contract is internally inconsistent."""


@dataclass(frozen=True, slots=True)
class FaceTimeSeries:
    """Subject-level facial motion time series.

    The canonical tensor layout is ``(T, K, D)`` where ``T`` is time,
    ``K`` is facial region, and ``D`` is feature dimension.
    ``valid_mask`` is elementwise and therefore has the same shape as ``X``.
    """

    X: np.ndarray
    subject_id: str
    time_index: np.ndarray
    region_id: tuple[str, ...]
    dimension: tuple[str, ...]
    valid_mask: np.ndarray
    sampling_rate: float

    def __post_init__(self) -> None:
        X = np.asarray(self.X)
        time_index = np.asarray(self.time_index)
        valid_mask = np.asarray(self.valid_mask)

        if X.ndim != 3:
            raise ContractError(f"X must have shape (T, K, D); got ndim={X.ndim}")
        if not np.issubdtype(X.dtype, np.number):
            raise ContractError("X must contain numeric values")

        T, K, D = X.shape
        if T < 1 or K < 1 or D < 1:
            raise ContractError(f"X dimensions must be non-empty; got shape={X.shape}")

        if not isinstance(self.subject_id, str) or not self.subject_id.strip():
            raise ContractError("subject_id must be a non-empty string")

        if time_index.ndim != 1 or time_index.shape[0] != T:
            raise ContractError(
                f"time_index must have shape ({T},); got {time_index.shape}"
            )
        if not np.issubdtype(time_index.dtype, np.number):
            raise ContractError("time_index must be numeric")
        if not np.all(np.isfinite(time_index)):
            raise ContractError("time_index must be finite")
        if T > 1 and not np.all(np.diff(time_index) > 0):
            raise ContractError("time_index must be strictly increasing")

        region_id = tuple(self.region_id)
        if len(region_id) != K:
            raise ContractError(
                f"region_id must contain {K} entries; got {len(region_id)}"
            )
        if any(not isinstance(value, str) or not value.strip() for value in region_id):
            raise ContractError("region_id entries must be non-empty strings")
        if len(set(region_id)) != len(region_id):
            raise ContractError("region_id entries must be unique")

        dimension = tuple(self.dimension)
        if len(dimension) != D:
            raise ContractError(
                f"dimension must contain {D} entries; got {len(dimension)}"
            )
        if any(not isinstance(value, str) or not value.strip() for value in dimension):
            raise ContractError("dimension entries must be non-empty strings")
        if len(set(dimension)) != len(dimension):
            raise ContractError("dimension entries must be unique")

        if valid_mask.shape != X.shape:
            raise ContractError(
                f"valid_mask must match X.shape={X.shape}; got {valid_mask.shape}"
            )
        if valid_mask.dtype != np.bool_:
            raise ContractError("valid_mask must have boolean dtype")

        sampling_rate = float(self.sampling_rate)
        if not np.isfinite(sampling_rate) or sampling_rate <= 0:
            raise ContractError("sampling_rate must be finite and > 0")

        object.__setattr__(self, "X", X)
        object.__setattr__(self, "time_index", time_index)
        object.__setattr__(self, "region_id", region_id)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "valid_mask", valid_mask)
        object.__setattr__(self, "sampling_rate", sampling_rate)
