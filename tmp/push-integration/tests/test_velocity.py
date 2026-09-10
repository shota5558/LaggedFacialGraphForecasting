from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting.motion_features import (
    DisplacementSeries,
    MotionFeatureError,
    compute_displacement,
    compute_velocity,
)


def test_velocity_uses_observed_irregular_time_intervals() -> None:
    coordinates = np.array(
        [
            [[0.0, 0.0]],
            [[2.0, 4.0]],
            [[8.0, 1.0]],
        ]
    )
    displacement = compute_displacement(coordinates)
    timestamps = np.array([0.0, 0.5, 2.0])

    velocity = compute_velocity(displacement, timestamps=timestamps)

    assert np.allclose(
        velocity.values,
        np.array([[[4.0, 8.0]], [[4.0, -2.0]]]),
    )
    assert np.allclose(velocity.time_intervals, np.array([0.5, 1.5]))
    assert np.array_equal(velocity.source_frame_indices, np.array([0, 1]))
    assert np.array_equal(velocity.target_frame_indices, np.array([1, 2]))
    assert velocity.method == "displacement_over_observed_time_interval"


def test_uniform_time_intervals_reduce_to_constant_scaling() -> None:
    coordinates = np.array([[[0.0]], [[1.0]], [[3.0]], [[6.0]]])
    displacement = compute_displacement(coordinates)

    velocity = compute_velocity(
        displacement,
        timestamps=np.array([0.0, 0.25, 0.5, 0.75]),
    )

    assert np.array_equal(velocity.values[:, 0, 0], np.array([4.0, 8.0, 12.0]))


def test_nan_displacement_propagates_without_imputation() -> None:
    coordinates = np.array([[[0.0]], [[np.nan]], [[2.0]]])
    displacement = compute_displacement(coordinates)

    velocity = compute_velocity(
        displacement,
        timestamps=np.array([0.0, 1.0, 2.0]),
    )

    assert np.isnan(velocity.values[:, 0, 0]).all()


def test_invalid_timestamps_are_rejected() -> None:
    displacement = compute_displacement(np.zeros((3, 1, 1), dtype=float))

    with pytest.raises(MotionFeatureError, match="strictly increasing"):
        compute_velocity(displacement, timestamps=np.array([0.0, 1.0, 1.0]))
    with pytest.raises(MotionFeatureError, match="exactly one more timestamp"):
        compute_velocity(displacement, timestamps=np.array([0.0, 1.0, 2.0, 3.0]))


def test_nonconsecutive_provenance_is_rejected() -> None:
    displacement = DisplacementSeries(
        values=np.ones((1, 1, 1), dtype=float),
        source_frame_indices=np.array([0]),
        target_frame_indices=np.array([2]),
    )

    with pytest.raises(MotionFeatureError, match="consecutive"):
        compute_velocity(displacement, timestamps=np.array([0.0, 1.0, 2.0]))


def test_velocity_outputs_are_immutable_and_inputs_unchanged() -> None:
    coordinates = np.array([[[0.0]], [[2.0]], [[5.0]]])
    timestamps = np.array([0.0, 1.0, 3.0])
    coordinate_before = coordinates.copy()
    timestamp_before = timestamps.copy()
    displacement = compute_displacement(coordinates)

    velocity = compute_velocity(displacement, timestamps=timestamps)

    assert np.array_equal(coordinates, coordinate_before)
    assert np.array_equal(timestamps, timestamp_before)
    assert not velocity.values.flags.writeable
    assert not velocity.time_intervals.flags.writeable
    assert not velocity.source_frame_indices.flags.writeable
    assert not velocity.target_frame_indices.flags.writeable
