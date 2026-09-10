"""Multi-horizon DesignMatrix adapters for Sensitivity only (S-IMPL-06).

Primary remains frozen to h=1. Existing Primary DesignMatrix builders and the V0
``aligned_indices`` contract are intentionally left unchanged. Every public h>1
builder requires the common Sensitivity execution boundary before constructing data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .contracts import DesignMatrix, FaceTimeSeries
from .core_contracts import ParentSet
from .design_matrix import (
    build_full_history_design_matrix,
    build_pcmci_parent_design_matrix,
    build_self_history_design_matrix,
)
from .sensitivity_execution import (
    SensitivityExperimentConfig,
    assert_sensitivity_execution_allowed,
)


SENSITIVITY_HORIZON_SCHEMA_VERSION = 1


class SensitivityHorizonError(ValueError):
    """Raised when an h>1 Sensitivity matrix violates direct-observation semantics."""


@dataclass(frozen=True, slots=True)
class SensitivityHorizonDesignMatrix:
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


def _validate_available_lags(
    lags: Iterable[int], *, horizon: int, field: str
) -> tuple[int, ...]:
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


def _rebase_forecast_origin(
    matrix: DesignMatrix,
    series: FaceTimeSeries,
    *,
    horizon: int,
    alignment_lag: int | None,
) -> DesignMatrix:
    feature_support = max(matrix.feature_lags)
    support_lag = feature_support if alignment_lag is None else int(alignment_lag)
    if support_lag < feature_support:
        raise SensitivityHorizonError("alignment_lag must be >= every feature lag")
    if support_lag < horizon:
        raise SensitivityHorizonError("alignment support must be >= horizon")

    target_indices = np.arange(support_lag, len(series.time_index), dtype=np.int64)
    if len(target_indices) != matrix.X.shape[0]:
        raise SensitivityHorizonError(
            "Primary builder row support disagrees with multi-horizon provenance"
        )
    expected_target_time = series.time_index[target_indices]
    if not np.array_equal(matrix.target_time, expected_target_time):
        raise SensitivityHorizonError(
            "Primary builder target support disagrees with canonical time_index"
        )
    origin_indices = target_indices - horizon
    if np.any(origin_indices < 0):
        raise SensitivityHorizonError("multi-horizon forecast origin would be negative")

    return DesignMatrix(
        X=np.array(matrix.X, copy=True),
        y=np.array(matrix.y, copy=True),
        subject_id=matrix.subject_id,
        region_id=matrix.region_id,
        target_dimensions=matrix.target_dimensions,
        forecast_origin=np.array(series.time_index[origin_indices], copy=True),
        target_time=np.array(matrix.target_time, copy=True),
        feature_names=matrix.feature_names,
        feature_lags=matrix.feature_lags,
        valid_mask=np.array(matrix.valid_mask, copy=True),
    )


def _wrap(
    matrix: DesignMatrix,
    series: FaceTimeSeries,
    *,
    horizon: int,
    condition: str,
    alignment_lag: int | None,
) -> SensitivityHorizonDesignMatrix:
    rebased = _rebase_forecast_origin(
        matrix, series, horizon=horizon, alignment_lag=alignment_lag
    )
    return SensitivityHorizonDesignMatrix(rebased, horizon, condition)


def _authorize(
    execution: SensitivityExperimentConfig,
    repository_root: str | Path | None,
) -> None:
    assert_sensitivity_execution_allowed(execution, repository_root=repository_root)


def build_sensitivity_self_history_design_matrix(
    execution: SensitivityExperimentConfig,
    series: FaceTimeSeries,
    *,
    target_region: str,
    horizon: int,
    lags: Iterable[int],
    alignment_lag: int | None = None,
    target_dimension: str | None = None,
    repository_root: str | Path | None = None,
) -> SensitivityHorizonDesignMatrix:
    _authorize(execution, repository_root)
    h = _validate_sensitivity_horizon(horizon)
    available_lags = _validate_available_lags(lags, horizon=h, field="lags")
    matrix = build_self_history_design_matrix(
        series,
        target_region=target_region,
        lags=available_lags,
        horizon=1,
        alignment_lag=alignment_lag,
        target_dimension=target_dimension,
    )
    return _wrap(matrix, series, horizon=h, condition="self", alignment_lag=alignment_lag)


def build_sensitivity_persistence_design_matrix(
    execution: SensitivityExperimentConfig,
    series: FaceTimeSeries,
    *,
    target_region: str,
    horizon: int,
    alignment_lag: int | None = None,
    target_dimension: str | None = None,
    repository_root: str | Path | None = None,
) -> SensitivityHorizonDesignMatrix:
    _authorize(execution, repository_root)
    h = _validate_sensitivity_horizon(horizon)
    matrix = build_self_history_design_matrix(
        series,
        target_region=target_region,
        lags=(h,),
        horizon=1,
        alignment_lag=alignment_lag,
        target_dimension=target_dimension,
    )
    return _wrap(
        matrix,
        series,
        horizon=h,
        condition="persistence",
        alignment_lag=alignment_lag,
    )


def build_sensitivity_full_history_design_matrix(
    execution: SensitivityExperimentConfig,
    series: FaceTimeSeries,
    *,
    target_region: str,
    horizon: int,
    lags: Iterable[int],
    alignment_lag: int | None = None,
    target_dimension: str | None = None,
    repository_root: str | Path | None = None,
) -> SensitivityHorizonDesignMatrix:
    _authorize(execution, repository_root)
    h = _validate_sensitivity_horizon(horizon)
    available_lags = _validate_available_lags(lags, horizon=h, field="lags")
    matrix = build_full_history_design_matrix(
        series,
        target_region=target_region,
        lags=available_lags,
        horizon=1,
        alignment_lag=alignment_lag,
        target_dimension=target_dimension,
    )
    return _wrap(matrix, series, horizon=h, condition="full", alignment_lag=alignment_lag)


def build_sensitivity_pcmci_parent_design_matrix(
    execution: SensitivityExperimentConfig,
    series: FaceTimeSeries,
    *,
    parent_set: ParentSet,
    horizon: int,
    self_lags: Iterable[int],
    alignment_lag: int | None = None,
    target_dimension: str | None = None,
    repository_root: str | Path | None = None,
) -> SensitivityHorizonDesignMatrix:
    _authorize(execution, repository_root)
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
        horizon=1,
        alignment_lag=alignment_lag,
        target_dimension=target_dimension,
    )
    return _wrap(matrix, series, horizon=h, condition="pcmci", alignment_lag=alignment_lag)
