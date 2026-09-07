"""Design-matrix builders for frozen Primary forecasting conditions."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .alignment import aligned_indices
from .contracts import DesignMatrix, FaceTimeSeries


def _normalize_lags(lags: Iterable[int]) -> tuple[int, ...]:
    normalized: list[int] = []
    for lag in lags:
        if not isinstance(lag, (int, np.integer)) or isinstance(lag, (bool, np.bool_)):
            raise ValueError("lags must contain integers")
        normalized.append(int(lag))
    if not normalized:
        raise ValueError("lags must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("lags must be unique")
    if any(lag < 1 for lag in normalized):
        raise ValueError("lags must be positive")
    return tuple(sorted(normalized))


def build_self_history_design_matrix(
    series: FaceTimeSeries,
    *,
    target_region: str,
    lags: Iterable[int] = (1,),
    horizon: int = 1,
) -> DesignMatrix:
    """Build a self-history-only matrix for one subject and target region.

    Every feature is drawn from the target region itself. Lags are target-relative:
    feature ``tau`` for target index ``u`` is observed at ``u - tau``. The frozen
    Primary horizon is enforced by ``aligned_indices``.
    """

    normalized_lags = _normalize_lags(lags)
    if target_region not in series.region_id:
        raise ValueError(f"unknown target_region: {target_region!r}")

    max_lag = max(normalized_lags)
    common = aligned_indices(len(series.time_index), lag=max_lag, horizon=horizon)
    target_indices = common.target_index
    origin_indices = common.forecast_origin
    region_index = series.region_id.index(target_region)

    feature_columns: list[np.ndarray] = []
    feature_valid_columns: list[np.ndarray] = []
    feature_names: list[str] = []
    feature_lags: list[int] = []

    for lag in normalized_lags:
        source_indices = target_indices - lag
        for dimension_index, dimension_name in enumerate(series.dimension):
            feature_columns.append(
                series.X[source_indices, region_index, dimension_index]
            )
            feature_valid_columns.append(
                series.valid_mask[source_indices, region_index, dimension_index]
            )
            feature_names.append(f"{target_region}.{dimension_name}")
            feature_lags.append(lag)

    X = np.column_stack(feature_columns)
    feature_valid = np.column_stack(feature_valid_columns)
    y = series.X[target_indices, region_index, :]
    target_valid = series.valid_mask[target_indices, region_index, :]
    valid_mask = np.all(feature_valid, axis=1) & np.all(target_valid, axis=1)

    forecast_origin = series.time_index[origin_indices]
    target_time = series.time_index[target_indices]
    row_count = len(target_indices)

    return DesignMatrix(
        X=X,
        y=y,
        subject_id=(series.subject_id,) * row_count,
        region_id=(target_region,) * row_count,
        forecast_origin=forecast_origin,
        target_time=target_time,
        feature_names=tuple(feature_names),
        feature_lags=tuple(feature_lags),
        valid_mask=valid_mask.astype(bool, copy=False),
    )


def build_persistence_design_matrix(
    series: FaceTimeSeries,
    *,
    target_region: str,
    horizon: int = 1,
) -> DesignMatrix:
    """Build the frozen Persistence condition: current target-region value only.

    With Primary ``h=1``, the value available at forecast origin ``t`` is the
    target-relative lag-1 value for target ``t+1``.  Persistence therefore uses
    exactly the target region's dimensions at lag 1 and no other region/history.
    The returned matrix remains compatible with the common Ridge pipeline so that
    conditions differ by input information rather than estimator implementation.
    """

    return build_self_history_design_matrix(
        series,
        target_region=target_region,
        lags=(1,),
        horizon=horizon,
    )
