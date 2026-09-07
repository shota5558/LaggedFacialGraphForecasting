"""Deterministic temporal feature primitives for facial motion preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class TemporalFeatureError(ValueError):
    """Raised when temporal feature construction violates an explicit contract."""


def _validate_series(values: np.ndarray, *, minimum_frames: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 3:
        raise TemporalFeatureError(
            "series must have shape (frames, regions_or_points, dimensions)"
        )
    if array.shape[0] < minimum_frames or any(size < 1 for size in array.shape[1:]):
        raise TemporalFeatureError(
            f"series must contain at least {minimum_frames} frames and non-empty feature axes"
        )
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise TemporalFeatureError("series must be numeric")
    return array


@dataclass(frozen=True, slots=True)
class Displacement:
    """Immutable one-step displacement with explicit frame alignment provenance."""

    values: np.ndarray
    source_frame_indices: np.ndarray
    target_frame_indices: np.ndarray
    method: str = "consecutive_frame_difference"

    def __post_init__(self) -> None:
        values = np.array(self.values, copy=True)
        source = np.array(self.source_frame_indices, copy=True)
        target = np.array(self.target_frame_indices, copy=True)
        if values.ndim != 3 or values.shape[0] < 1:
            raise TemporalFeatureError("displacement values must have non-empty shape (T-1, K, D)")
        if source.shape != (values.shape[0],) or target.shape != (values.shape[0],):
            raise TemporalFeatureError(
                "source_frame_indices and target_frame_indices must align with displacement rows"
            )
        if self.method != "consecutive_frame_difference":
            raise TemporalFeatureError(
                "A-08 method must be 'consecutive_frame_difference'"
            )
        values.setflags(write=False)
        source.setflags(write=False)
        target.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "source_frame_indices", source)
        object.__setattr__(self, "target_frame_indices", target)


def compute_displacement(values: np.ndarray) -> Displacement:
    """Compute ``x[t] - x[t-1]`` and retain the exact frame correspondence.

    A-08 deliberately computes geometric displacement, not velocity: irregular
    timestamp spacing is handled by A-09.  The output row ``r`` always describes
    motion from source frame ``r`` to target frame ``r + 1``.  NumPy arithmetic
    intentionally propagates NaN so missingness remains an A-12/A-13 responsibility.
    No population statistic or learned preprocessing parameter is used.
    """

    array = _validate_series(values, minimum_frames=2)
    displacement = array[1:] - array[:-1]
    source = np.arange(array.shape[0] - 1, dtype=np.int64)
    target = source + 1
    return Displacement(
        values=displacement,
        source_frame_indices=source,
        target_frame_indices=target,
    )
