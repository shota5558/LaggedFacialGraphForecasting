"""Deterministic kinematic transforms for facial motion preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocessing import validate_timestamps


class KinematicsError(ValueError):
    """Raised when a kinematic transform cannot preserve its time alignment contract."""


def _validate_series(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 3:
        raise KinematicsError(
            "values must have shape (frames, points_or_regions, dimensions)"
        )
    if any(size < 1 for size in array.shape):
        raise KinematicsError("values axes must be non-empty")
    if array.shape[0] < 2:
        raise KinematicsError("at least two frames are required for displacement")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise KinematicsError("values must be numeric")
    return array


@dataclass(frozen=True, slots=True)
class DisplacementSeries:
    """First differences with explicit source/target timestamp provenance.

    ``values[n]`` equals the motion from ``source_time_index[n]`` to
    ``target_time_index[n]``. The output has one fewer frame than the input rather
    than inventing a synthetic zero/NaN displacement at the first observation.
    """

    values: np.ndarray
    source_time_index: np.ndarray
    target_time_index: np.ndarray
    method: str = "first_difference"

    def __post_init__(self) -> None:
        values = np.array(self.values, copy=True)
        source = np.array(self.source_time_index, copy=True)
        target = np.array(self.target_time_index, copy=True)
        values.setflags(write=False)
        source.setflags(write=False)
        target.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "source_time_index", source)
        object.__setattr__(self, "target_time_index", target)


def compute_displacement(
    values: np.ndarray,
    *,
    time_index: np.ndarray,
) -> DisplacementSeries:
    """Compute frame-to-frame displacement without converting it to velocity.

    The transform is the direct first difference ``x[t] - x[t-1]``. Observed
    timestamps are validated and retained only as provenance; displacement is not
    divided by elapsed time here because time-normalized motion belongs to A-09
    velocity. Irregular positive sampling intervals are therefore preserved rather
    than resampled or silently normalized.

    Missing numeric values are not imputed. NumPy difference semantics propagate
    them into the affected displacement rows so missingness/interpolation remains
    the responsibility of A-12/A-13.
    """

    array = _validate_series(values)
    timestamps = validate_timestamps(
        time_index,
        expected_frame_count=int(array.shape[0]),
    )

    displacement = np.diff(array, axis=0)
    return DisplacementSeries(
        values=displacement,
        source_time_index=timestamps[:-1],
        target_time_index=timestamps[1:],
    )
