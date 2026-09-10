from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting.motion_features import (
    MotionFeatureError,
    VelocitySeries,
    compute_acceleration,
    compute_displacement,
    compute_velocity,
)


def test_acceleration_uses_elapsed_time_between_velocity_target_frames() -> None:
    coordinates = np.array([[[0.0]], [[1.0]], [[5.0]], [[14.0]]])
    timestamps = np.array([0.0, 1.0, 3.0, 6.0])
    displacement = compute_displacement(coordinates)
    velocity = compute_velocity(displacement, timestamps=timestamps)

    acceleration = compute_acceleration(velocity, timestamps=timestamps)

    # Velocity rows are [1, 2, 3] at target frames [1, 2, 3].
    # Acceleration therefore uses (2-1)/(3-1)=0.5 and (3-2)/(6-3)=1/3.
    assert np.allclose(acceleration.values[:, 0, 0], np.array([0.5, 1.0 / 3.0]))
    assert np.allclose(acceleration.time_intervals, np.array([2.0, 3.0]))
    assert np.array_equal(acceleration.source_frame_indices, np.array([1, 2]))
    assert np.array_equal(acceleration.target_frame_indices, np.array([2, 3]))
    assert (
        acceleration.method
        == "consecutive_velocity_difference_over_target_time_interval"
    )


def test_constant_velocity_has_zero_acceleration() -> None:
    coordinates = np.array([[[0.0]], [[2.0]], [[4.0]], [[6.0]]])
    timestamps = np.array([0.0, 1.0, 2.0, 3.0])
    velocity = compute_velocity(compute_displacement(coordinates), timestamps=timestamps)

    acceleration = compute_acceleration(velocity, timestamps=timestamps)

    assert np.array_equal(acceleration.values, np.zeros((2, 1, 1)))


def test_nan_velocity_propagates_without_imputation() -> None:
    velocity = VelocitySeries(
        values=np.array([[[1.0]], [[np.nan]], [[3.0]]]),
        time_intervals=np.ones(3),
        source_frame_indices=np.array([0, 1, 2]),
        target_frame_indices=np.array([1, 2, 3]),
    )

    acceleration = compute_acceleration(
        velocity,
        timestamps=np.array([0.0, 1.0, 2.0, 3.0]),
    )

    assert np.isnan(acceleration.values[:, 0, 0]).all()


def test_acceleration_requires_two_velocity_rows() -> None:
    velocity = VelocitySeries(
        values=np.ones((1, 1, 1)),
        time_intervals=np.ones(1),
        source_frame_indices=np.array([0]),
        target_frame_indices=np.array([1]),
    )

    with pytest.raises(MotionFeatureError, match="at least two velocity rows"):
        compute_acceleration(velocity, timestamps=np.array([0.0, 1.0]))


def test_nonconsecutive_velocity_provenance_is_rejected() -> None:
    velocity = VelocitySeries(
        values=np.ones((2, 1, 1)),
        time_intervals=np.ones(2),
        source_frame_indices=np.array([0, 2]),
        target_frame_indices=np.array([1, 3]),
    )

    with pytest.raises(MotionFeatureError, match="without temporal gaps"):
        compute_acceleration(
            velocity,
            timestamps=np.array([0.0, 1.0, 2.0, 3.0]),
        )


def test_acceleration_outputs_are_immutable_and_inputs_unchanged() -> None:
    coordinates = np.array([[[0.0]], [[1.0]], [[4.0]], [[9.0]]])
    timestamps = np.array([0.0, 1.0, 2.0, 3.0])
    timestamp_before = timestamps.copy()
    velocity = compute_velocity(compute_displacement(coordinates), timestamps=timestamps)
    velocity_before = velocity.values.copy()

    acceleration = compute_acceleration(velocity, timestamps=timestamps)

    assert np.array_equal(timestamps, timestamp_before)
    assert np.array_equal(velocity.values, velocity_before)
    assert not acceleration.values.flags.writeable
    assert not acceleration.time_intervals.flags.writeable
    assert not acceleration.source_frame_indices.flags.writeable
    assert not acceleration.target_frame_indices.flags.writeable
