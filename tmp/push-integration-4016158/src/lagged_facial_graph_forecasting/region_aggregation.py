"""Deterministic facial landmark-to-region aggregation (A-04)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocessing import validate_landmark_shape
from .regions import RegionDefinition, validate_region_landmark_bounds


class RegionAggregationError(ValueError):
    """Raised when a requested region aggregation is not part of the frozen contract."""


@dataclass(frozen=True, slots=True)
class RegionAggregation:
    """Immutable region-level coordinate tensor with explicit aggregation provenance."""

    values: np.ndarray
    region_ids: tuple[str, ...]
    method: str

    def __post_init__(self) -> None:
        values = np.array(self.values, copy=True)
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "region_ids", tuple(self.region_ids))


def aggregate_regions(
    landmarks: np.ndarray,
    definition: RegionDefinition,
    *,
    method: str = "mean",
) -> RegionAggregation:
    """Aggregate raw ``(T, P, C)`` landmarks to ordered ``(T, K, C)`` regions.

    Primary A-04 freezes simple arithmetic-mean aggregation (region centroid).
    Missing numeric values are intentionally *not* ignored: ordinary ``mean``
    propagates NaN so A-12/A-13 retain responsibility for missingness and
    interpolation.  No normalization, weighting, or data-dependent fitting occurs.
    """

    if method != "mean":
        raise RegionAggregationError("A-04 supports only the frozen 'mean' method")

    values = validate_landmark_shape(landmarks)
    validate_region_landmark_bounds(
        definition,
        landmark_count=int(values.shape[1]),
    )

    aggregated = np.stack(
        [
            np.mean(values[:, indices, :], axis=1)
            for indices in definition.landmark_indices
        ],
        axis=1,
    )
    return RegionAggregation(
        values=aggregated,
        region_ids=definition.region_ids,
        method=method,
    )
