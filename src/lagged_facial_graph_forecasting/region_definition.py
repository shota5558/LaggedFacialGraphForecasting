"""Explicit facial landmark region-definition contract.

The contract validates a caller-supplied anatomical/topological mapping without
inventing extractor-specific landmark indices. Region values are not aggregated
here; A-04 owns that transformation.
"""

from __future__ import annotations

from dataclasses import dataclass


REGION_DEFINITION_SCHEMA_VERSION = 1


class RegionDefinitionError(ValueError):
    """Raised when a landmark-to-region definition is internally inconsistent."""


def _trimmed_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RegionDefinitionError(f"{field_name} must be non-empty trimmed text")
    return value


@dataclass(frozen=True, slots=True)
class LandmarkRegion:
    """One named region and its explicitly declared landmark indices."""

    name: str
    landmark_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        name = _trimmed_text(self.name, "region name")
        indices = tuple(self.landmark_indices)
        if not indices:
            raise RegionDefinitionError("region landmark_indices must not be empty")
        if any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in indices
        ):
            raise RegionDefinitionError(
                "region landmark_indices must be non-negative integers"
            )
        if len(set(indices)) != len(indices):
            raise RegionDefinitionError(
                "region landmark_indices must be unique within a region"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "landmark_indices", indices)


@dataclass(frozen=True, slots=True)
class RegionDefinition:
    """Versioned, topology-bound mapping from raw landmarks to facial regions.

    ``landmark_topology_id`` identifies the upstream extractor/topology contract;
    it does not select an implementation. ``allow_overlap`` is explicit so the
    scientific policy for landmarks assigned to multiple regions is never implicit.
    Region order is preserved because it becomes canonical ``FaceTimeSeries``
    region order in A-04/A-14.
    """

    landmark_topology_id: str
    landmark_count: int
    regions: tuple[LandmarkRegion, ...]
    allow_overlap: bool
    schema_version: int = REGION_DEFINITION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        topology_id = _trimmed_text(
            self.landmark_topology_id, "landmark_topology_id"
        )
        if (
            not isinstance(self.landmark_count, int)
            or isinstance(self.landmark_count, bool)
            or self.landmark_count < 1
        ):
            raise RegionDefinitionError("landmark_count must be a positive integer")
        if not isinstance(self.allow_overlap, bool):
            raise RegionDefinitionError("allow_overlap must be boolean")
        if self.schema_version != REGION_DEFINITION_SCHEMA_VERSION:
            raise RegionDefinitionError(
                f"schema_version must be {REGION_DEFINITION_SCHEMA_VERSION}"
            )

        regions = tuple(self.regions)
        if not regions:
            raise RegionDefinitionError("regions must not be empty")
        if any(not isinstance(region, LandmarkRegion) for region in regions):
            raise RegionDefinitionError("regions must contain LandmarkRegion values")
        names = tuple(region.name for region in regions)
        if len(set(names)) != len(names):
            raise RegionDefinitionError("region names must be unique")

        seen: dict[int, str] = {}
        for region in regions:
            for index in region.landmark_indices:
                if index >= self.landmark_count:
                    raise RegionDefinitionError(
                        f"landmark index {index} is outside landmark_count={self.landmark_count}"
                    )
                previous = seen.get(index)
                if previous is not None and not self.allow_overlap:
                    raise RegionDefinitionError(
                        f"landmark index {index} is assigned to both {previous!r} and {region.name!r} while allow_overlap=False"
                    )
                seen.setdefault(index, region.name)

        object.__setattr__(self, "landmark_topology_id", topology_id)
        object.__setattr__(self, "regions", regions)

    @property
    def region_ids(self) -> tuple[str, ...]:
        """Return canonical region order without performing aggregation."""

        return tuple(region.name for region in self.regions)
