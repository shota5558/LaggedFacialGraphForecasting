from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting.spatial_normalization import (
    SpatialNormalizationError,
    normalize_translation,
)


def test_frame_centroid_translation_is_removed() -> None:
    values = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[11.0, -8.0], [13.0, -6.0], [15.0, -4.0]],
        ]
    )

    normalized = normalize_translation(values, reference_indices=(0, 1, 2))

    assert normalized.reference_indices == (0, 1, 2)
    assert np.allclose(np.mean(normalized.values, axis=1), 0.0)
    assert normalized.offsets.shape == (2, 1, 2)
    assert not normalized.values.flags.writeable
    assert not normalized.offsets.flags.writeable


def test_reference_indices_are_mandatory_to_avoid_hidden_scientific_default() -> None:
    values = np.zeros((2, 3, 2), dtype=float)

    with pytest.raises(TypeError):
        normalize_translation(values)  # type: ignore[call-arg]


def test_global_translation_does_not_change_normalized_coordinates() -> None:
    values = np.arange(3 * 4 * 2, dtype=float).reshape(3, 4, 2)
    translation = np.array([[[100.0, -50.0]]])

    first = normalize_translation(values, reference_indices=(0, 2))
    shifted = normalize_translation(values + translation, reference_indices=(0, 2))

    assert np.allclose(first.values, shifted.values)


def test_explicit_reference_centroid_is_zero() -> None:
    values = np.array([[[0.0, 0.0], [2.0, 4.0], [10.0, 20.0]]])

    normalized = normalize_translation(values, reference_indices=(0, 1))

    assert np.allclose(np.mean(normalized.values[:, (0, 1), :], axis=1), 0.0)
    assert np.array_equal(normalized.offsets[0, 0], np.array([1.0, 2.0]))


def test_nan_in_reference_is_propagated_not_imputed() -> None:
    values = np.zeros((2, 3, 2), dtype=float)
    values[1, 0, 0] = np.nan

    normalized = normalize_translation(values, reference_indices=(0, 1))

    assert np.isnan(normalized.offsets[1, 0, 0])
    assert np.all(np.isnan(normalized.values[1, :, 0]))


def test_nan_outside_reference_does_not_change_reference_offset() -> None:
    values = np.zeros((1, 3, 2), dtype=float)
    values[0, 2, 0] = np.nan

    normalized = normalize_translation(values, reference_indices=(0, 1))

    assert np.array_equal(normalized.offsets[0, 0], np.array([0.0, 0.0]))
    assert np.isnan(normalized.values[0, 2, 0])


def test_invalid_reference_indices_are_rejected() -> None:
    values = np.zeros((2, 3, 2), dtype=float)

    with pytest.raises(SpatialNormalizationError, match="must not be empty"):
        normalize_translation(values, reference_indices=())
    with pytest.raises(SpatialNormalizationError, match="non-negative integers"):
        normalize_translation(values, reference_indices=(-1,))
    with pytest.raises(SpatialNormalizationError, match="outside"):
        normalize_translation(values, reference_indices=(3,))
    with pytest.raises(SpatialNormalizationError, match="duplicate"):
        normalize_translation(values, reference_indices=(0, 0))


def test_normalization_does_not_modify_input() -> None:
    values = np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3)
    before = values.copy()

    normalize_translation(values, reference_indices=(0, 1, 2))

    assert np.array_equal(values, before)
