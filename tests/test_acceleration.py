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

    assert np.allclose(acceleration.values[:, 0, 0], np.array([0.5, 1.0 / 3.0]))
    assert np.allclose(acceleration.time_intervals, np.array([2.0, 3.0]))
    assert np.array_equal(acceleration.source_frame_indices, np.array([1, 2]))
    assert np.array_equal(acceleration.target_frame_indices, np.array([2, 3]))
    assert (
        acceleration.method
        == "consecutive_velocity_difference_over_target_time_interval"
    )


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
