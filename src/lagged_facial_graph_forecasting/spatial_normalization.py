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
