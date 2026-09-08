"""Multi-horizon DesignMatrix adapters for Sensitivity only (S-IMPL-06).

Primary remains frozen to h=1.  This module exposes explicit h>1 entry points and
adds the direct-observation availability guard ``tau >= h`` before delegating to
the existing canonical DesignMatrix builders.  The core DesignMatrix schema is not
changed; horizon is carried in a Sensitivity-only provenance wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .contracts import DesignMatrix, FaceTimeSeries
from .core_contracts import ParentSet
from .design_matrix import (
    build_full_history_design_matrix,
    build_pcmci_parent_design_matrix,
    build_self_history_design_matrix,
)


SENSITIVITY_HORIZON_SCHEMA_VERSION = 1


class SensitivityHorizonError(ValueError):
    """Raised when an h>1 Sensitivity matrix violates direct-observation semantics."""


@dataclass(frozen=True, slots=True)
class SensitivityHorizonDesignMatrix:
    """Canonical DesignMatrix plus explicit Sensitivity horizon provenance."""

    design_matrix: DesignMatrix
    horizon: int
    condition: str
    schema_version: int = SENSITIVITY_HORIZON_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.design_matrix, DesignMatrix):
            raise SensitivityHorizonError("design_matrix must be a DesignMatrix")
        _validate_sensitivity_horizon(self.horizon)
        if not isinstance(self.condition, str) or not self.condition.strip():
            raise SensitivityHorizonError("condition must be a non-empty string")
        if self.schema_version != SENSITIVITY_HORIZON_SCHEMA_VERSION:
            raise SensitivityHorizonError("unsupported multi-horizon schema_version")
        if any(lag < self.horizon for lag in self.design_matrix.feature_lags):
            raise SensitivityHorizonError(
                "Sensitivity matrix contains a feature unavailable at the forecast origin"
            )


def _validate_sensitivity_horizon(horizon: int) -> int:
    if not isinstance(horizon, (int, np.integer)) or isinstance(
        horizon, (bool, np.bool_)
    ):
        raise SensitivityHorizonError("horizon must be an integer")
    value = int(horizon)
    if value <= 1:
        raise SensitivityHorizonError(
            "this adapter is Sensitivity-only and requires horizon > 1; Primary remains h=1"
        )
    return value


def _validate_available_lags(lags: Iterable[int], *, horizon: int, field: str) -> tuple[int, ...]:
    normalized: list[int] = []
    for lag in lags:
        if not isinstance(lag, (int, np.integer)) or isinstance(lag, (bool, np.bool_)):
            raise SensitivityHorizonError(f"{field} must contain integers")
        normalized.append(int(lag))
    if not normalized:
        raise SensitivityHorizonError(f"{field} must not be empty")
    if any(lag < horizon for lag in normalized):
        raise SensitivityHorizonError(
            f"direct-observation h={horizon} requires every {field} value tau >= h"
        )
    return tuple(normalized)


def build_sensitivity_self_history_design_matrix(
    series: FaceTimeSeries,
    *,
    target_region: str,
    horizon: int,
    lags: Iterable[int],
    alignment_lag: int | None = None,
    target_dimension: str | None = None,
) -> SensitivityHorizonDesignMatrix:
    h = _validate_sensitivity_horizon(horizon)
    available_lags = _validate_available_lags(lags, horizon=h, field="lags")
    matrix = build_self_history_design_matrix(
        series,
        target_region=target_region,
        lags=available_lags,
        horizon=h,
        alignment_lag=alignment_lag,
        target_dimension=target_dimension,
    )
    return SensitivityHorizonDesignMatrix(matrix, h, "self")


def build_sensitivity_persistence_design_matrix(
    series: FaceTimeSeries,
    *,
    target_region: str,
    horizon: int,
    alignment_lag: int | None = None,
    target_dimension: str | None = None,
) -> SensitivityHorizonDesignMatrix:
    """Use the observation at forecast origin as the h-step persistence feature.

    In target-relative lag notation the forecast-origin observation has ``tau=h``.
    """

    h = _validate_sensitivity_horizon(horizon)
    matrix = build_self_history_design_matrix(
        series,
        target_region=target_region,
        lags=(h,),
        horizon=h,
        alignment_lag=alignment_lag,
        target_dimension=target_dimension,
    )
    return SensitivityHorizonDesignMatrix(matrix, h, "persistence")


def build_sensitivity_full_history_design_matrix(
    series: FaceTimeSeries,
    *,
    target_region: str,
    horizon: int,
    lags: Iterable[int],
    alignment_lag: int | None = None,
    target_dimension: str | None = None,
) -> SensitivityHorizonDesignMatrix:
    h = _validate_sensitivity_horizon(horizon)
    available_lags = _validate_available_lags(lags, horizon=h, field="lags")
    matrix = build_full_history_design_matrix(
        series,
        target_region=target_region,
        lags=available_lags,
        horizon=h,
        alignment_lag=alignment_lag,
        target_dimension=target_dimension,
    )
    return SensitivityHorizonDesignMatrix(matrix, h, "full")


def build_sensitivity_pcmci_parent_design_matrix(
    series: FaceTimeSeries,
    *,
    parent_set: ParentSet,
    horizon: int,
    self_lags: Iterable[int],
    alignment_lag: int | None = None,
    target_dimension: str | None = None,
) -> SensitivityHorizonDesignMatrix:
    h = _validate_sensitivity_horizon(horizon)
    available_self_lags = _validate_available_lags(
        self_lags, horizon=h, field="self_lags"
    )
    unavailable = tuple(parent for parent in parent_set.parents if parent.lag < h)
    if unavailable:
        raise SensitivityHorizonError(
            f"ParentSet contains {len(unavailable)} direct-observation parent(s) with tau < h={h}; "
            "construct an explicitly horizon-compatible ParentSet instead of silently dropping links"
        )
    matrix = build_pcmci_parent_design_matrix(
        series,
        parent_set=parent_set,
        self_lags=available_self_lags,
        horizon=h,
        alignment_lag=alignment_lag,
        target_dimension=target_dimension,
    )
    return SensitivityHorizonDesignMatrix(matrix, h, "pcmci")
