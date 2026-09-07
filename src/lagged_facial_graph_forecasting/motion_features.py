"""Deterministic motion-feature primitives for canonical facial time series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class MotionFeatureError(ValueError):
    """Raised when a motion-feature input violates the explicit contract."""


def _validate_coordinate_sequence(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 3:
        raise MotionFeatureError(
            "coordinate sequence must have shape (frames, points_or_regions, coordinates)"
        )
    if any(size < 1 for size in array.shape):
        raise MotionFeatureError("coordinate sequence axes must be non-empty")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise MotionFeatureError("coordinate sequence must be numeric")
    if array.shape[0] < 2:
        raise MotionFeatureError(
            "at least two frames are required to compute displacement"
        )
    return array


@dataclass(frozen=True, slots=True)
class DisplacementSeries:
    """Immutable consecutive-frame displacement with explicit frame provenance."""

    values: np.ndarray
    source_frame_indices: np.ndarray
    target_frame_indices: np.ndarray
    method: str = "consecutive_frame_difference"

    def __post_init__(self) -> None:
        values = np.array(self.values, copy=True)
        source = np.array(self.source_frame_indices, copy=True)
        target = np.array(self.target_frame_indices, copy=True)
        values.setflags(write=False)
        source.setflags(write=False)
        target.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "source_frame_indices", source)
        object.__setattr__(self, "target_frame_indices", target)


def compute_displacement(values: np.ndarray) -> DisplacementSeries:
    """Compute observed consecutive-frame displacement without temporal repair.

    For input coordinates ``x[0:T]``, output row ``r`` is exactly
    ``x[r + 1] - x[r]``. The source and target frame indices are returned as
    immutable provenance so downstream velocity/alignment code cannot silently
    reinterpret the temporal relation.

    No interpolation, smoothing, imputation, or population-level fitting is
    performed. NaN therefore propagates naturally when either endpoint is missing;
    missing-frame policy remains the responsibility of A-12/A-13.
    """

    array = _validate_coordinate_sequence(values)
    displacement = array[1:] - array[:-1]
    source = np.arange(array.shape[0] - 1, dtype=np.int64)
    target = source + 1
    return DisplacementSeries(
        values=displacement,
        source_frame_indices=source,
        target_frame_indices=target,
    )
