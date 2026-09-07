from __future__ import annotations

import numpy as np

from lagged_facial_graph_forecasting.motion_features import (
    compute_displacement,
    compute_velocity,
)
from lagged_facial_graph_forecasting.spatial_normalization import (
    normalize_rotation,
    normalize_scale,
    normalize_translation,
)


def _apply_similarity_transform(
    canonical: np.ndarray,
    *,
    scale: float,
    angle_radians: float,
    translation: np.ndarray,
) -> np.ndarray:
    cosine = np.cos(angle_radians)
    sine = np.sin(angle_radians)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return scale * (canonical @ rotation.T) + translation


def test_known_local_motion_survives_global_similarity_normalization() -> None:
    """A-17: recover known local motion after removing global head-like motion."""

    # Points 0/1 are frozen reference anchors. Point 2 has a known +0.5 x motion
    # in canonical coordinates between frames. Each observed frame then receives a
    # different global translation, scale, and in-plane rotation.
    canonical_0 = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    canonical_1 = np.array([[-1.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    observed = np.stack(
        [
            _apply_similarity_transform(
                canonical_0,
                scale=2.0,
                angle_radians=np.deg2rad(30.0),
                translation=np.array([5.0, -3.0]),
            ),
            _apply_similarity_transform(
                canonical_1,
                scale=3.0,
                angle_radians=np.deg2rad(-20.0),
                translation=np.array([8.0, 4.0]),
            ),
        ]
    )

    translated = normalize_translation(observed, reference_indices=(0, 1))
    scaled = normalize_scale(translated.values, reference_pair=(0, 1))
    rotated = normalize_rotation(
        scaled.values,
        reference_pair=(0, 1),
        coordinate_axes=(0, 1),
    )

    # Reference-pair distance is 2 in canonical coordinates, so normalized local
    # positions are divided by 2. The only remaining motion is therefore +0.25 x.
    expected_normalized = np.array(
        [
            [[-0.5, 0.0], [0.5, 0.0], [0.0, 0.5]],
            [[-0.5, 0.0], [0.5, 0.0], [0.25, 0.5]],
        ]
    )
    np.testing.assert_allclose(rotated.values, expected_normalized, atol=1e-12)

    displacement = compute_displacement(rotated.values)
    np.testing.assert_allclose(
        displacement.values[0],
        np.array([[0.0, 0.0], [0.0, 0.0], [0.25, 0.0]]),
        atol=1e-12,
    )

    velocity = compute_velocity(
        displacement,
        timestamps=np.array([0.0, 0.5]),
    )
    np.testing.assert_allclose(
        velocity.values[0],
        np.array([[0.0, 0.0], [0.0, 0.0], [0.5, 0.0]]),
        atol=1e-12,
    )

    # Provenance must remain exact; no temporal resampling or inferred frames.
    np.testing.assert_array_equal(displacement.source_frame_indices, np.array([0]))
    np.testing.assert_array_equal(displacement.target_frame_indices, np.array([1]))
    np.testing.assert_array_equal(velocity.source_frame_indices, np.array([0]))
    np.testing.assert_array_equal(velocity.target_frame_indices, np.array([1]))
