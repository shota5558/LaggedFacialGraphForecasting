from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting.spatial_normalization import (
    SpatialNormalizationError,
    normalize_rotation,
)


def test_reference_pair_is_aligned_with_positive_first_axis() -> None:
    values = np.array(
        [
            [[1.0, 2.0], [1.0, 4.0], [3.0, 3.0]],
            [[0.0, 0.0], [-3.0, 4.0], [2.0, -1.0]],
        ]
    )

    normalized = normalize_rotation(
        values,
        reference_pair=(0, 1),
        coordinate_axes=(0, 1),
    )

    delta = normalized.values[:, 1, :] - normalized.values[:, 0, :]
    assert np.all(delta[:, 0] > 0.0)
    assert np.allclose(delta[:, 1], 0.0, atol=1e-12)
    assert normalized.reference_pair == (0, 1)
    assert normalized.coordinate_axes == (0, 1)
    assert normalized.method == "frame_reference_pair_axis_alignment"
    assert normalized.angles_radians.shape == (2,)
    assert not normalized.values.flags.writeable
    assert not normalized.angles_radians.flags.writeable


def test_global_planar_rotation_does_not_change_normalized_coordinates() -> None:
    values = np.array([[[1.0, 0.0], [3.0, 2.0], [-2.0, 4.0]]])
    phi = 0.73
    cosine = np.cos(phi)
    sine = np.sin(phi)
    rotated = values.copy()
    rotated[:, :, 0] = values[:, :, 0] * cosine - values[:, :, 1] * sine
    rotated[:, :, 1] = values[:, :, 0] * sine + values[:, :, 1] * cosine

    first = normalize_rotation(
        values,
        reference_pair=(0, 1),
        coordinate_axes=(0, 1),
    )
    second = normalize_rotation(
        rotated,
        reference_pair=(0, 1),
        coordinate_axes=(0, 1),
    )

    assert np.allclose(first.values, second.values, atol=1e-12)


def test_coordinates_outside_rotation_plane_are_preserved() -> None:
    values = np.array([[[0.0, 0.0, 7.0], [0.0, 2.0, 11.0], [1.0, 1.0, 13.0]]])

    normalized = normalize_rotation(
        values,
        reference_pair=(0, 1),
        coordinate_axes=(0, 1),
    )

    assert np.array_equal(normalized.values[:, :, 2], values[:, :, 2])


def test_rotation_reference_and_plane_are_mandatory() -> None:
    values = np.zeros((1, 2, 2), dtype=float)

    with pytest.raises(TypeError):
        normalize_rotation(values, coordinate_axes=(0, 1))  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        normalize_rotation(values, reference_pair=(0, 1))  # type: ignore[call-arg]


def test_zero_projected_reference_distance_is_rejected() -> None:
    values = np.array([[[1.0, 2.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])

    with pytest.raises(SpatialNormalizationError, match="projected reference-pair distance"):
        normalize_rotation(
            values,
            reference_pair=(0, 1),
            coordinate_axes=(0, 1),
        )


def test_nan_in_reference_propagates_instead_of_being_imputed() -> None:
    values = np.array([[[np.nan, 0.0], [1.0, 0.0], [2.0, 1.0]]])

    normalized = normalize_rotation(
        values,
        reference_pair=(0, 1),
        coordinate_axes=(0, 1),
    )

    assert np.isnan(normalized.angles_radians[0])
    assert np.all(np.isnan(normalized.values[0]))


def test_nan_outside_reference_does_not_change_rotation_angle() -> None:
    values = np.array([[[0.0, 0.0], [1.0, 1.0], [np.nan, 2.0]]])

    normalized = normalize_rotation(
        values,
        reference_pair=(0, 1),
        coordinate_axes=(0, 1),
    )

    assert normalized.angles_radians[0] == pytest.approx(np.pi / 4.0)
    assert np.all(np.isnan(normalized.values[0, 2]))


def test_invalid_reference_pair_and_coordinate_axes_are_rejected() -> None:
    values = np.zeros((1, 3, 3), dtype=float)

    with pytest.raises(SpatialNormalizationError, match="exactly two"):
        normalize_rotation(
            values,
            reference_pair=(0,),  # type: ignore[arg-type]
            coordinate_axes=(0, 1),
        )
    with pytest.raises(SpatialNormalizationError, match="duplicate"):
        normalize_rotation(values, reference_pair=(0, 0), coordinate_axes=(0, 1))
    with pytest.raises(SpatialNormalizationError, match="exactly two"):
        normalize_rotation(
            values,
            reference_pair=(0, 1),
            coordinate_axes=(0,),  # type: ignore[arg-type]
        )
    with pytest.raises(SpatialNormalizationError, match="distinct"):
        normalize_rotation(values, reference_pair=(0, 1), coordinate_axes=(0, 0))
    with pytest.raises(SpatialNormalizationError, match="non-negative integers"):
        normalize_rotation(values, reference_pair=(0, 1), coordinate_axes=(-1, 1))
    with pytest.raises(SpatialNormalizationError, match="outside"):
        normalize_rotation(values, reference_pair=(0, 1), coordinate_axes=(0, 3))


def test_rotation_normalization_does_not_modify_input() -> None:
    values = np.array([[[0.0, 0.0], [1.0, 2.0], [3.0, -1.0]]])
    before = values.copy()

    normalize_rotation(
        values,
        reference_pair=(0, 1),
        coordinate_axes=(0, 1),
    )

    assert np.array_equal(values, before)
