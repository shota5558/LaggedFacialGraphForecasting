"""Design-matrix construction from the frozen PCMCI+ ParentSet contract."""

from __future__ import annotations

import numpy as np

from .alignment import aligned_indices
from .contracts import DesignMatrix, FaceTimeSeries
from .core_contracts import ParentSet


def build_pcmci_parent_design_matrix(
    series: FaceTimeSeries,
    *,
    parent_set: ParentSet,
    horizon: int = 1,
) -> DesignMatrix:
    """Build the Primary PCMCI condition from one frozen ``ParentSet``.

    Parent lags are target-relative: for target index ``u`` and parent lag
    ``tau``, the predictor is read from ``u - tau``.  ``aligned_indices`` keeps
    every source at or before the forecast origin and hard-rejects non-Primary
    horizons. ParentSet order and FaceTimeSeries dimension order define the
    deterministic feature-column order.
    """

    if not isinstance(parent_set, ParentSet):
        raise TypeError("parent_set must be a ParentSet")
    if parent_set.discovery_method != "pcmci_plus":
        raise ValueError(
            "Primary PCMCI DesignMatrix requires discovery_method='pcmci_plus'"
        )
    if parent_set.target_region not in series.region_id:
        raise ValueError(
            f"unknown target_region from ParentSet: {parent_set.target_region!r}"
        )
    if not parent_set.parents:
        raise ValueError("ParentSet.parents must not be empty for a DesignMatrix")

    missing_sources = sorted(
        {
            parent.source_region
            for parent in parent_set.parents
            if parent.source_region not in series.region_id
        }
    )
    if missing_sources:
        raise ValueError(f"ParentSet contains unknown source regions: {missing_sources}")

    max_lag = max(parent.lag for parent in parent_set.parents)
    common = aligned_indices(
        len(series.time_index),
        lag=max_lag,
        horizon=horizon,
    )
    target_indices = common.target_index
    origin_indices = common.forecast_origin
    target_region_index = series.region_id.index(parent_set.target_region)

    feature_columns: list[np.ndarray] = []
    feature_valid_columns: list[np.ndarray] = []
    feature_names: list[str] = []
    feature_lags: list[int] = []

    for parent in parent_set.parents:
        # Validate the exact parent lag against the shared forecast-availability
        # contract as well, not only the maximum lag used for common row alignment.
        aligned_indices(
            len(series.time_index),
            lag=parent.lag,
            horizon=horizon,
        )
        source_indices = target_indices - parent.lag
        source_region_index = series.region_id.index(parent.source_region)
        if np.any(source_indices > origin_indices):
            raise RuntimeError(
                "forecast leakage invariant violated: parent source after forecast origin"
            )

        for dimension_index, dimension_name in enumerate(series.dimension):
            feature_columns.append(
                series.X[source_indices, source_region_index, dimension_index]
            )
            feature_valid_columns.append(
                series.valid_mask[source_indices, source_region_index, dimension_index]
            )
            feature_names.append(f"{parent.source_region}.{dimension_name}")
            feature_lags.append(parent.lag)

    X = np.column_stack(feature_columns)
    feature_valid = np.column_stack(feature_valid_columns)
    y = series.X[target_indices, target_region_index, :]
    target_valid = series.valid_mask[target_indices, target_region_index, :]
    valid_mask = np.all(feature_valid, axis=1) & np.all(target_valid, axis=1)

    row_count = len(target_indices)
    return DesignMatrix(
        X=X,
        y=y,
        subject_id=(series.subject_id,) * row_count,
        region_id=(parent_set.target_region,) * row_count,
        forecast_origin=series.time_index[origin_indices],
        target_time=series.time_index[target_indices],
        feature_names=tuple(feature_names),
        feature_lags=tuple(feature_lags),
        valid_mask=valid_mask.astype(bool, copy=False),
    )
