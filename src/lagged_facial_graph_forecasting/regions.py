"""Canonical facial region definitions and explicit aggregation primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


class RegionDefinitionError(ValueError):
    """Raised when a facial region definition is ambiguous or invalid."""


@dataclass(frozen=True, slots=True)
class RegionDefinition:
    """Immutable ordered mapping from region IDs to raw landmark indices.

    Region order is part of the scientific feature provenance. Definitions are
    extractor-neutral: no landmark count, MediaPipe topology, or facial-region
    names are hard-coded here. Cross-region overlap is rejected by default so a
    raw point has one unambiguous region identity unless the caller explicitly
    freezes ``allow_overlap=True``.
    """

    region_ids: tuple[str, ...]
    landmark_indices: tuple[tuple[int, ...], ...]
    allow_overlap: bool = False

    def __post_init__(self) -> None:
        if not self.region_ids:
            raise RegionDefinitionError("at least one region is required")
        if len(self.region_ids) != len(self.landmark_indices):
            raise RegionDefinitionError(
                "region_ids and landmark_indices must have the same length"
            )
        if not isinstance(self.allow_overlap, bool):
            raise RegionDefinitionError("allow_overlap must be boolean")

        seen_ids: set[str] = set()
        globally_seen: dict[int, str] = {}
        normalized_indices: list[tuple[int, ...]] = []

        for region_id, indices in zip(self.region_ids, self.landmark_indices):
            if (
                not isinstance(region_id, str)
                or not region_id
                or region_id != region_id.strip()
            ):
                raise RegionDefinitionError(
                    "region IDs must be non-empty strings without surrounding whitespace"
                )
            if region_id in seen_ids:
                raise RegionDefinitionError(f"duplicate region ID: {region_id}")
            seen_ids.add(region_id)

            index_tuple = tuple(indices)
            if not index_tuple:
                raise RegionDefinitionError(
                    f"region {region_id!r} must contain at least one landmark index"
                )
            local_seen: set[int] = set()
            for index in index_tuple:
                if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                    raise RegionDefinitionError(
                        "landmark indices must be non-negative integers"
                    )
                if index in local_seen:
                    raise RegionDefinitionError(
                        f"duplicate landmark index {index} in region {region_id!r}"
                    )
                local_seen.add(index)
                if not self.allow_overlap and index in globally_seen:
                    raise RegionDefinitionError(
                        f"landmark index {index} overlaps regions "
                        f"{globally_seen[index]!r} and {region_id!r}"
                    )
                globally_seen.setdefault(index, region_id)
            normalized_indices.append(index_tuple)

        object.__setattr__(self, "region_ids", tuple(self.region_ids))
        object.__setattr__(self, "landmark_indices", tuple(normalized_indices))

    def as_mapping(self) -> dict[str, tuple[int, ...]]:
        """Return an insertion-ordered copy for artifact/config serialization."""

        return dict(zip(self.region_ids, self.landmark_indices))


def build_region_definition(
    regions: Mapping[str, Iterable[int]],
    *,
    allow_overlap: bool = False,
) -> RegionDefinition:
    """Build an immutable RegionDefinition while preserving mapping order."""

    if not isinstance(regions, Mapping):
        raise RegionDefinitionError("regions must be a mapping")
    return RegionDefinition(
        region_ids=tuple(regions.keys()),
        landmark_indices=tuple(tuple(indices) for indices in regions.values()),
        allow_overlap=allow_overlap,
    )


def validate_region_landmark_bounds(
    definition: RegionDefinition,
    *,
    landmark_count: int,
) -> RegionDefinition:
    """Verify all frozen region indices are addressable by a raw landmark tensor."""

    if (
        not isinstance(landmark_count, int)
        or isinstance(landmark_count, bool)
        or landmark_count < 1
    ):
        raise RegionDefinitionError("landmark_count must be a positive integer")

    for region_id, indices in zip(
        definition.region_ids, definition.landmark_indices
    ):
        for index in indices:
            if index >= landmark_count:
                raise RegionDefinitionError(
                    f"landmark index {index} in region {region_id!r} is outside "
                    f"landmark_count={landmark_count}"
                )
    return definition


def aggregate_landmark_regions(
    landmarks: np.ndarray,
    definition: RegionDefinition,
    *,
    method: str,
) -> np.ndarray:
    """Aggregate raw ``(T, P, C)`` landmarks into ordered ``(T, K, C)`` regions.

    The aggregation method must be stated explicitly by the caller so the
    scientific preprocessing choice is never hidden in a default. A-04 supports
    only the arithmetic mean. NumPy's ordinary mean is used deliberately: any NaN
    in a selected region/coordinate propagates instead of being silently dropped,
    imputed, or interpolated before A-12/A-13.
    """

    values = np.asarray(landmarks)
    if values.ndim != 3:
        raise RegionDefinitionError(
            "landmarks must have shape (frames, landmarks, coordinates)"
        )
    if any(size < 1 for size in values.shape):
        raise RegionDefinitionError("landmark tensor axes must be non-empty")
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
        values.dtype, np.bool_
    ):
        raise RegionDefinitionError("landmarks must contain numeric values")
    if not isinstance(definition, RegionDefinition):
        raise RegionDefinitionError("definition must be a RegionDefinition")
    if method != "mean":
        raise RegionDefinitionError("A-04 supports only explicit method='mean'")

    validate_region_landmark_bounds(definition, landmark_count=values.shape[1])
    aggregated = np.stack(
        [
            np.mean(values[:, indices, :], axis=1)
            for indices in definition.landmark_indices
        ],
        axis=1,
    )
    aggregated = np.array(aggregated, copy=True)
    aggregated.setflags(write=False)
    return aggregated
