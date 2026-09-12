from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting.spatial_normalization import (
    SpatialNormalizationError,
    normalize_scale,
)


def test_reference_pair_distance_becomes_one() -> None:
    values = np.array(
        [
            [[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]],
            [[0.0, 0.0], [0.0, 2.0], [0.0, 4.0]],
        ]
    )

    normalized = normalize_scale(values, reference_pair=(0, 1))

    assert normalized.reference_pair == (0, 1)
    assert normalized.method == "frame_reference_pair_distance"
    assert normalized.scales.shape == (2, 1, 1)
    assert np.allclose(normalized.scales[:, 0, 0], np.array([5.0, 2.0]))
    distances = np.linalg.norm(
        normalized.values[:, 0, :] - normalized.values[:, 1, :], axis=1
    )
    assert np.allclose(distances, 1.0)
    assert not normalized.values.flags.writeable
    assert not normalized.scales.flags.writeable


def test_global_scale_does_not_change_normalized_coordinates() -> None:
    values = np.array([[[0.0, 0.0], [3.0, 4.0], [2.0, -1.0]]])

    first = normalize_scale(values, reference_pair=(0, 1))
    enlarged = normalize_scale(values * 7.5, reference_pair=(0, 1))

    assert np.allclose(first.values, enlarged.values)


def test_reference_pair_is_mandatory_to_avoid_hidden_scientific_default() -> None:
    values = np.zeros((1, 2, 2), dtype=float)

    with pytest.raises(TypeError):
        normalize_scale(values)  # type: ignore[call-arg]


def test_zero_reference_distance_is_rejected() -> None:
    values = np.array([[[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]]])

    with pytest.raises(SpatialNormalizationError, match="distance must be positive"):
        normalize_scale(values, reference_pair=(0, 1))


def test_nan_in_reference_propagates_instead_of_being_imputed() -> None:
    values = np.array([[[np.nan, 0.0], [1.0, 0.0], [2.0, 1.0]]])

    normalized = normalize_scale(values, reference_pair=(0, 1))

    assert np.isnan(normalized.scales[0, 0, 0])
    assert np.all(np.isnan(normalized.values[0]))


def test_nan_outside_reference_does_not_change_scale() -> None:
    values = np.array([[[0.0, 0.0], [3.0, 4.0], [np.nan, 1.0]]])

    normalized = normalize_scale(values, reference_pair=(0, 1))

    assert normalized.scales[0, 0, 0] == pytest.approx(5.0)
    assert np.isnan(normalized.values[0, 2, 0])


def test_invalid_reference_pair_is_rejected() -> None:
    values = np.zeros((1, 3, 2), dtype=float)

    with pytest.raises(SpatialNormalizationError, match="exactly two"):
        normalize_scale(values, reference_pair=(0,))  # type: ignore[arg-type]
    with pytest.raises(SpatialNormalizationError, match="duplicate"):
        normalize_scale(values, reference_pair=(0, 0))
    with pytest.raises(SpatialNormalizationError, match="non-negative integers"):
        normalize_scale(values, reference_pair=(-1, 1))
    with pytest.raises(SpatialNormalizationError, match="outside"):
        normalize_scale(values, reference_pair=(0, 3))


def test_scale_normalization_does_not_modify_input() -> None:
    values = np.array([[[0.0, 0.0], [3.0, 4.0], [2.0, -1.0]]])
    before = values.copy()

    normalize_scale(values, reference_pair=(0, 1))

    assert np.array_equal(values, before)
