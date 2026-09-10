from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting.motion_features import (
    MotionFeatureError,
    compute_displacement,
)


def test_consecutive_frame_displacement_and_provenance() -> None:
    values = np.array(
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [[1.5, 2.0], [1.0, 5.0]],
            [[4.0, 0.0], [3.0, 6.0]],
        ]
    )

    result = compute_displacement(values)

    expected = np.array(
        [
            [[1.5, 1.0], [-1.0, 2.0]],
            [[2.5, -2.0], [2.0, 1.0]],
        ]
    )
    assert np.array_equal(result.values, expected)
    assert np.array_equal(result.source_frame_indices, np.array([0, 1]))
    assert np.array_equal(result.target_frame_indices, np.array([1, 2]))
    assert result.method == "consecutive_frame_difference"
    assert not result.values.flags.writeable
    assert not result.source_frame_indices.flags.writeable
    assert not result.target_frame_indices.flags.writeable


def test_constant_coordinates_have_zero_displacement() -> None:
    values = np.full((4, 3, 2), 7.5)

    result = compute_displacement(values)

    assert np.array_equal(result.values, np.zeros((3, 3, 2)))


def test_nan_endpoint_propagates_without_imputation() -> None:
    values = np.zeros((3, 2, 2), dtype=float)
    values[1, 0, 0] = np.nan

    result = compute_displacement(values)

    assert np.isnan(result.values[0, 0, 0])
    assert np.isnan(result.values[1, 0, 0])
    assert np.isfinite(result.values[:, 1, :]).all()


def test_displacement_requires_two_frames() -> None:
    with pytest.raises(MotionFeatureError, match="at least two frames"):
        compute_displacement(np.zeros((1, 3, 2), dtype=float))


def test_invalid_coordinate_sequence_is_rejected() -> None:
    with pytest.raises(MotionFeatureError, match="shape"):
        compute_displacement(np.zeros((3, 2), dtype=float))
    with pytest.raises(MotionFeatureError, match="numeric"):
        compute_displacement(np.full((2, 2, 2), "x", dtype=object))


def test_displacement_does_not_modify_input() -> None:
    values = np.arange(4 * 2 * 3, dtype=float).reshape(4, 2, 3)
    before = values.copy()

    compute_displacement(values)

    assert np.array_equal(values, before)
