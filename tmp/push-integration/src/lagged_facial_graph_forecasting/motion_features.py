"""Deterministic motion-feature primitives for canonical facial time series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocessing import PreprocessingValidationError, validate_timestamps


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


@dataclass(frozen=True, slots=True)
class VelocitySeries:
    """Immutable displacement-per-observed-time with temporal provenance."""

    values: np.ndarray
    time_intervals: np.ndarray
    source_frame_indices: np.ndarray
    target_frame_indices: np.ndarray
    method: str = "displacement_over_observed_time_interval"

    def __post_init__(self) -> None:
        values = np.array(self.values, copy=True)
        intervals = np.array(self.time_intervals, copy=True)
        source = np.array(self.source_frame_indices, copy=True)
        target = np.array(self.target_frame_indices, copy=True)
        values.setflags(write=False)
        intervals.setflags(write=False)
        source.setflags(write=False)
        target.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "time_intervals", intervals)
        object.__setattr__(self, "source_frame_indices", source)
        object.__setattr__(self, "target_frame_indices", target)


@dataclass(frozen=True, slots=True)
class AccelerationSeries:
    """Immutable consecutive velocity change per observed target-time interval."""

    values: np.ndarray
    time_intervals: np.ndarray
    source_frame_indices: np.ndarray
    target_frame_indices: np.ndarray
    method: str = "consecutive_velocity_difference_over_target_time_interval"

    def __post_init__(self) -> None:
        values = np.array(self.values, copy=True)
        intervals = np.array(self.time_intervals, copy=True)
        source = np.array(self.source_frame_indices, copy=True)
        target = np.array(self.target_frame_indices, copy=True)
        values.setflags(write=False)
        intervals.setflags(write=False)
        source.setflags(write=False)
        target.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "time_intervals", intervals)
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


def compute_velocity(
    displacement: DisplacementSeries,
    *,
    timestamps: np.ndarray,
) -> VelocitySeries:
    """Convert canonical displacement to velocity using observed time intervals.

    A-01 explicitly permits irregular positive frame intervals, so velocity uses
    the validated timestamp difference for each displacement endpoint rather than
    assuming a fixed frame rate. The numeric unit is therefore coordinate units per
    timestamp unit; no unstated seconds/frame conversion is introduced here.

    The canonical A-08 provenance must describe consecutive frame pairs. No
    resampling, interpolation, smoothing, imputation, or population-level fitting is
    performed. NaN in displacement values remains NaN in velocity values.
    """

    values = np.asarray(displacement.values)
    source = np.asarray(displacement.source_frame_indices)
    target = np.asarray(displacement.target_frame_indices)

    if values.ndim != 3 or values.shape[0] < 1:
        raise MotionFeatureError(
            "displacement values must have shape (transitions, points_or_regions, coordinates)"
        )
    if source.ndim != 1 or target.ndim != 1:
        raise MotionFeatureError("displacement frame provenance must be one-dimensional")
    if source.shape != target.shape or source.size != values.shape[0]:
        raise MotionFeatureError(
            "displacement frame provenance must align with displacement rows"
        )
    if not np.issubdtype(source.dtype, np.integer) or not np.issubdtype(
        target.dtype, np.integer
    ):
        raise MotionFeatureError("displacement frame provenance must be integer indices")
    if np.any(source < 0) or np.any(target != source + 1):
        raise MotionFeatureError(
            "velocity requires canonical consecutive source->target frame provenance"
        )

    try:
        validated_timestamps = validate_timestamps(timestamps)
    except PreprocessingValidationError as exc:
        raise MotionFeatureError(str(exc)) from exc

    if target.size == 0 or int(target[-1]) >= validated_timestamps.size:
        raise MotionFeatureError(
            "timestamps do not cover all displacement source/target frames"
        )
    if validated_timestamps.size != values.shape[0] + 1:
        raise MotionFeatureError(
            "canonical displacement requires exactly one more timestamp than displacement rows"
        )

    intervals = validated_timestamps[target] - validated_timestamps[source]
    velocity = values / intervals[:, np.newaxis, np.newaxis]
    return VelocitySeries(
        values=velocity,
        time_intervals=intervals,
        source_frame_indices=source,
        target_frame_indices=target,
    )


def compute_acceleration(
    velocity: VelocitySeries,
    *,
    timestamps: np.ndarray,
) -> AccelerationSeries:
    """Compute optional forward finite-difference acceleration from A-09 velocity.

    Each velocity row is associated with its target frame. Consecutive velocity
    rows are differenced and divided by the observed elapsed time between those
    target frames. This makes the convention explicit for irregular timestamps and
    avoids an unstated central-difference or fixed-FPS assumption.

    Acceleration is optional for Primary preprocessing. This helper performs no
    smoothing, interpolation, imputation, or population-level fitting and preserves
    missingness by ordinary NumPy propagation.
    """

    values = np.asarray(velocity.values)
    source = np.asarray(velocity.source_frame_indices)
    target = np.asarray(velocity.target_frame_indices)
    if values.ndim != 3 or values.shape[0] < 2:
        raise MotionFeatureError(
            "at least two velocity rows are required to compute acceleration"
        )
    if source.ndim != 1 or target.ndim != 1:
        raise MotionFeatureError("velocity frame provenance must be one-dimensional")
    if source.shape != target.shape or source.size != values.shape[0]:
        raise MotionFeatureError(
            "velocity frame provenance must align with velocity rows"
        )
    if not np.issubdtype(source.dtype, np.integer) or not np.issubdtype(
        target.dtype, np.integer
    ):
        raise MotionFeatureError("velocity frame provenance must be integer indices")
    if np.any(source < 0) or np.any(target != source + 1):
        raise MotionFeatureError(
            "acceleration requires canonical consecutive velocity provenance"
        )
    if np.any(source[1:] != source[:-1] + 1):
        raise MotionFeatureError(
            "acceleration requires consecutive velocity rows without temporal gaps"
        )

    try:
        validated_timestamps = validate_timestamps(timestamps)
    except PreprocessingValidationError as exc:
        raise MotionFeatureError(str(exc)) from exc
    if int(target[-1]) >= validated_timestamps.size:
        raise MotionFeatureError("timestamps do not cover all velocity target frames")
    if validated_timestamps.size != values.shape[0] + 1:
        raise MotionFeatureError(
            "canonical velocity requires exactly one more timestamp than velocity rows"
        )

    acceleration_source = target[:-1]
    acceleration_target = target[1:]
    intervals = (
        validated_timestamps[acceleration_target]
        - validated_timestamps[acceleration_source]
    )
    acceleration = (values[1:] - values[:-1]) / intervals[:, np.newaxis, np.newaxis]
    return AccelerationSeries(
        values=acceleration,
        time_intervals=intervals,
        source_frame_indices=acceleration_source,
        target_frame_indices=acceleration_target,
    )
