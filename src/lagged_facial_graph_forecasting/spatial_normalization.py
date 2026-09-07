"""Deterministic spatial normalization primitives for facial motion preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class SpatialNormalizationError(ValueError):
    """Raised when a spatial normalization contract is invalid."""


def _validate_coordinate_tensor(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 3:
        raise SpatialNormalizationError(
            "coordinate tensor must have shape (frames, points_or_regions, coordinates)"
        )
    if any(size < 1 for size in array.shape):
        raise SpatialNormalizationError("coordinate tensor axes must be non-empty")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise SpatialNormalizationError("coordinate tensor must be numeric")
    return array


def _validate_reference_indices(
    reference_indices: tuple[int, ...],
    *,
    item_count: int,
) -> tuple[int, ...]:
    indices = tuple(reference_indices)
    if not indices:
        raise SpatialNormalizationError("reference_indices must not be empty")

    seen: set[int] = set()
    for index in indices:
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise SpatialNormalizationError(
                "reference_indices must contain non-negative integers"
            )
        if index >= item_count:
            raise SpatialNormalizationError(
                f"reference index {index} is outside item_count={item_count}"
            )
        if index in seen:
            raise SpatialNormalizationError(
                f"duplicate reference index: {index}"
            )
        seen.add(index)
    return indices


def _validate_coordinate_axes(
    coordinate_axes: tuple[int, int],
    *,
    coordinate_dim: int,
) -> tuple[int, int]:
    axes = tuple(coordinate_axes)
    if len(axes) != 2:
        raise SpatialNormalizationError(
            "coordinate_axes must contain exactly two distinct indices"
        )
    first, second = axes
    for axis in axes:
        if not isinstance(axis, int) or isinstance(axis, bool) or axis < 0:
            raise SpatialNormalizationError(
                "coordinate_axes must contain non-negative integers"
            )
        if axis >= coordinate_dim:
            raise SpatialNormalizationError(
                f"coordinate axis {axis} is outside coordinate_dim={coordinate_dim}"
            )
    if first == second:
        raise SpatialNormalizationError("coordinate_axes must be distinct")
    return first, second


@dataclass(frozen=True, slots=True)
class TranslationNormalization:
    """Immutable frame-wise translation-normalized coordinates and provenance."""

    values: np.ndarray
    offsets: np.ndarray
    reference_indices: tuple[int, ...]
    method: str = "frame_reference_centroid"

    def __post_init__(self) -> None:
        values = np.array(self.values, copy=True)
        offsets = np.array(self.offsets, copy=True)
        values.setflags(write=False)
        offsets.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "reference_indices", tuple(self.reference_indices))


def normalize_translation(
    values: np.ndarray,
    *,
    reference_indices: tuple[int, ...],
) -> TranslationNormalization:
    """Remove frame-wise common translation using an explicitly frozen centroid.

    The reference set is intentionally mandatory: the scientific choice of which
    points/regions define head translation must come from frozen experiment
    configuration rather than an API default. The transform contains no learned
    population statistic; each frame is centered by the arithmetic centroid of
    those declared references. Ordinary mean semantics intentionally propagate
    NaN in a reference so missingness remains an A-12/A-13 responsibility.
    """

    array = _validate_coordinate_tensor(values)
    references = _validate_reference_indices(
        reference_indices,
        item_count=int(array.shape[1]),
    )
    offsets = np.mean(array[:, references, :], axis=1, keepdims=True)
    normalized = array - offsets
    return TranslationNormalization(
        values=normalized,
        offsets=offsets,
        reference_indices=references,
    )


@dataclass(frozen=True, slots=True)
class ScaleNormalization:
    """Immutable frame-wise scale-normalized coordinates and provenance."""

    values: np.ndarray
    scales: np.ndarray
    reference_pair: tuple[int, int]
    method: str = "frame_reference_pair_distance"

    def __post_init__(self) -> None:
        values = np.array(self.values, copy=True)
        scales = np.array(self.scales, copy=True)
        values.setflags(write=False)
        scales.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "reference_pair", tuple(self.reference_pair))


def normalize_scale(
    values: np.ndarray,
    *,
    reference_pair: tuple[int, int],
) -> ScaleNormalization:
    """Remove frame-wise global scale using an explicitly frozen reference pair.

    Scale is the Euclidean distance between the two declared reference points or
    regions in each frame. The pair is mandatory so the anatomical/scientific
    anchor cannot be hidden in an API default or selected from outer-test
    performance. The operation is frame-local and contains no population-level fit.

    Missing values in either reference deliberately propagate through the scale and
    normalized frame; missingness repair remains an A-12/A-13 responsibility.
    A finite zero reference distance is rejected because scale normalization would
    otherwise be undefined.
    """

    array = _validate_coordinate_tensor(values)
    references = _validate_reference_indices(
        tuple(reference_pair),
        item_count=int(array.shape[1]),
    )
    if len(references) != 2:
        raise SpatialNormalizationError(
            "reference_pair must contain exactly two distinct indices"
        )

    first, second = references
    distances = np.linalg.norm(array[:, first, :] - array[:, second, :], axis=1)
    if np.any(np.isfinite(distances) & (distances <= 0.0)):
        raise SpatialNormalizationError(
            "reference-pair distance must be positive in every finite frame"
        )

    scales = distances[:, np.newaxis, np.newaxis]
    normalized = array / scales
    return ScaleNormalization(
        values=normalized,
        scales=scales,
        reference_pair=(first, second),
    )


@dataclass(frozen=True, slots=True)
class RotationNormalization:
    """Immutable frame-wise planar rotation-normalized coordinates and provenance."""

    values: np.ndarray
    angles_radians: np.ndarray
    reference_pair: tuple[int, int]
    coordinate_axes: tuple[int, int]
    method: str = "frame_reference_pair_axis_alignment"

    def __post_init__(self) -> None:
        values = np.array(self.values, copy=True)
        angles = np.array(self.angles_radians, copy=True)
        values.setflags(write=False)
        angles.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "angles_radians", angles)
        object.__setattr__(self, "reference_pair", tuple(self.reference_pair))
        object.__setattr__(self, "coordinate_axes", tuple(self.coordinate_axes))


def normalize_rotation(
    values: np.ndarray,
    *,
    reference_pair: tuple[int, int],
    coordinate_axes: tuple[int, int],
) -> RotationNormalization:
    """Align an explicit reference pair with the positive first coordinate axis.

    Rotation is applied independently within the explicitly declared 2-D coordinate
    plane for each frame. The reference pair and coordinate axes are mandatory so
    neither the anatomical anchor nor the rotation plane can be hidden in a default
    or selected from outer-test performance. Coordinate axes outside the chosen
    plane are preserved unchanged. The operation is frame-local and contains no
    population-level fit.

    Missing values in a reference deliberately propagate through the rotated plane;
    missingness repair remains an A-12/A-13 responsibility. A finite zero projected
    reference distance is rejected because the in-plane orientation is undefined.
    """

    array = _validate_coordinate_tensor(values)
    references = _validate_reference_indices(
        tuple(reference_pair),
        item_count=int(array.shape[1]),
    )
    if len(references) != 2:
        raise SpatialNormalizationError(
            "reference_pair must contain exactly two distinct indices"
        )
    axis_x, axis_y = _validate_coordinate_axes(
        tuple(coordinate_axes),
        coordinate_dim=int(array.shape[2]),
    )

    first, second = references
    delta_x = array[:, second, axis_x] - array[:, first, axis_x]
    delta_y = array[:, second, axis_y] - array[:, first, axis_y]
    projected_distance = np.hypot(delta_x, delta_y)
    if np.any(np.isfinite(projected_distance) & (projected_distance <= 0.0)):
        raise SpatialNormalizationError(
            "projected reference-pair distance must be positive in every finite frame"
        )

    angles = np.arctan2(delta_y, delta_x)
    cosine = np.cos(angles)[:, np.newaxis]
    sine = np.sin(angles)[:, np.newaxis]

    normalized = np.array(array, dtype=np.result_type(array.dtype, np.float64), copy=True)
    original_x = array[:, :, axis_x]
    original_y = array[:, :, axis_y]
    normalized[:, :, axis_x] = original_x * cosine + original_y * sine
    normalized[:, :, axis_y] = -original_x * sine + original_y * cosine

    return RotationNormalization(
        values=normalized,
        angles_radians=angles,
        reference_pair=(first, second),
        coordinate_axes=(axis_x, axis_y),
    )
