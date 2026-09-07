"""Prediction artifact contract with row-level scientific provenance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import ContractError, _normalize_row_labels


@dataclass(frozen=True, slots=True)
class PredictionArtifact:
    """Predictions for one outer fold and one experimental condition.

    Row-level metadata preserves the exact subject, target region, forecast origin,
    and target time corresponding to every prediction. ``valid_mask`` determines
    which rows are eligible for downstream evaluation.
    """

    outer_fold: int
    subject_id: tuple[str, ...]
    region_id: tuple[str, ...]
    condition: str
    forecast_origin: np.ndarray
    target_time: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    valid_mask: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.outer_fold, int) or isinstance(self.outer_fold, bool):
            raise ContractError("outer_fold must be an integer")
        if self.outer_fold < 0:
            raise ContractError("outer_fold must be >= 0")
        if not isinstance(self.condition, str) or not self.condition.strip():
            raise ContractError("condition must be a non-empty string")

        y_true = np.asarray(self.y_true)
        y_pred = np.asarray(self.y_pred)
        forecast_origin = np.asarray(self.forecast_origin)
        target_time = np.asarray(self.target_time)
        valid_mask = np.asarray(self.valid_mask)

        if y_true.ndim not in (1, 2):
            raise ContractError("y_true must be 1D or 2D")
        if y_true.shape[0] < 1:
            raise ContractError("y_true must contain at least one row")
        if y_true.ndim == 2 and y_true.shape[1] < 1:
            raise ContractError("y_true output dimension must be non-empty")
        if not np.issubdtype(y_true.dtype, np.number):
            raise ContractError("y_true must be numeric")

        if y_pred.shape != y_true.shape:
            raise ContractError(
                f"y_pred must match y_true.shape={y_true.shape}; got {y_pred.shape}"
            )
        if not np.issubdtype(y_pred.dtype, np.number):
            raise ContractError("y_pred must be numeric")

        N = y_true.shape[0]
        subject_id = _normalize_row_labels(self.subject_id, N, "subject_id")
        region_id = _normalize_row_labels(self.region_id, N, "region_id")

        for values, name in (
            (forecast_origin, "forecast_origin"),
            (target_time, "target_time"),
        ):
            if values.ndim != 1 or values.shape[0] != N:
                raise ContractError(f"{name} must have shape ({N},); got {values.shape}")
            if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
                raise ContractError(f"{name} must contain finite numeric values")

        if not np.all(target_time > forecast_origin):
            raise ContractError("target_time must be strictly after forecast_origin")

        if valid_mask.shape != (N,):
            raise ContractError(f"valid_mask must have shape ({N},); got {valid_mask.shape}")
        if valid_mask.dtype != np.bool_:
            raise ContractError("valid_mask must have boolean dtype")

        if np.any(valid_mask):
            if not np.all(np.isfinite(y_true[valid_mask])):
                raise ContractError("valid rows of y_true must be finite")
            if not np.all(np.isfinite(y_pred[valid_mask])):
                raise ContractError("valid rows of y_pred must be finite")

        object.__setattr__(self, "subject_id", subject_id)
        object.__setattr__(self, "region_id", region_id)
        object.__setattr__(self, "forecast_origin", forecast_origin)
        object.__setattr__(self, "target_time", target_time)
        object.__setattr__(self, "y_true", y_true)
        object.__setattr__(self, "y_pred", y_pred)
        object.__setattr__(self, "valid_mask", valid_mask)
