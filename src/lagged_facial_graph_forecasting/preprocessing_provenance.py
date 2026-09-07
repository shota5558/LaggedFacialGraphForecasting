"""Immutable provenance for Lane A preprocessing semantics.

The scientific contract requires the frozen facial-region definition and the
meaning of each feature dimension to travel with every experiment manifest.
This module keeps that provenance separate from the frozen core-contract ABI.
"""

from __future__ import annotations

from dataclasses import dataclass

from .regions import RegionDefinition


PREPROCESSING_PROVENANCE_SCHEMA_VERSION = 1


class PreprocessingProvenanceError(ValueError):
    """Raised when preprocessing provenance is incomplete or ambiguous."""


def _text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PreprocessingProvenanceError(f"{field_name} must be a non-empty string")
    normalized = value.strip()
    if normalized != value:
        raise PreprocessingProvenanceError(
            f"{field_name} must not contain surrounding whitespace"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class FeatureSemantics:
    """Explicit meaning of one canonical facial-motion feature tensor.

    ``dimensions`` is intentionally unconstrained beyond non-empty unique names.
    The Primary study can therefore preserve the D=2 vector representation
    (for example ``("v_x", "v_y")``) without collapsing a region to one scalar.
    """

    feature_name: str
    dimensions: tuple[str, ...]
    method: str
    units: str
    timestamp_basis: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "feature_name", _text(self.feature_name, "feature_name")
        )
        dimensions = tuple(self.dimensions)
        if not dimensions:
            raise PreprocessingProvenanceError(
                "dimensions must contain at least one named component"
            )
        normalized_dimensions = tuple(
            _text(value, f"dimensions[{index}]")
            for index, value in enumerate(dimensions)
        )
        if len(set(normalized_dimensions)) != len(normalized_dimensions):
            raise PreprocessingProvenanceError("dimensions must be unique")
        object.__setattr__(self, "dimensions", normalized_dimensions)
        object.__setattr__(self, "method", _text(self.method, "method"))
        object.__setattr__(self, "units", _text(self.units, "units"))
        object.__setattr__(
            self,
            "timestamp_basis",
            _text(self.timestamp_basis, "timestamp_basis"),
        )

    def to_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible payload."""

        return {
            "feature_name": self.feature_name,
            "dimensions": list(self.dimensions),
            "method": self.method,
            "units": self.units,
            "timestamp_basis": self.timestamp_basis,
        }


@dataclass(frozen=True, slots=True)
class PreprocessingProvenance:
    """Frozen region definition and feature semantics stored with experiments."""

    region_definition: RegionDefinition
    feature_semantics: FeatureSemantics
    region_aggregation_method: str
    translation_normalization: str
    scale_normalization: str
    rotation_normalization: str
    missing_frame_handling: str
    interpolation_policy: str
    acceleration_primary: bool = False
    schema_version: int = PREPROCESSING_PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.region_definition, RegionDefinition):
            raise PreprocessingProvenanceError(
                "region_definition must be a RegionDefinition"
            )
        if not isinstance(self.feature_semantics, FeatureSemantics):
            raise PreprocessingProvenanceError(
                "feature_semantics must be FeatureSemantics"
            )
        for field_name in (
            "region_aggregation_method",
            "translation_normalization",
            "scale_normalization",
            "rotation_normalization",
            "missing_frame_handling",
            "interpolation_policy",
        ):
            object.__setattr__(
                self,
                field_name,
                _text(getattr(self, field_name), field_name),
            )
        if not isinstance(self.acceleration_primary, bool):
            raise PreprocessingProvenanceError(
                "acceleration_primary must be boolean"
            )
        if self.schema_version != PREPROCESSING_PROVENANCE_SCHEMA_VERSION:
            raise PreprocessingProvenanceError(
                "schema_version must be "
                f"{PREPROCESSING_PROVENANCE_SCHEMA_VERSION}"
            )

    def to_payload(self) -> dict[str, object]:
        """Return deterministic JSON data for an ExperimentManifest."""

        ordered_regions = [
            {
                "region_id": region_id,
                "landmark_indices": list(indices),
            }
            for region_id, indices in zip(
                self.region_definition.region_ids,
                self.region_definition.landmark_indices,
            )
        ]
        return {
            "schema_version": self.schema_version,
            "region_definition": {
                "ordered_regions": ordered_regions,
                "allow_overlap": self.region_definition.allow_overlap,
            },
            "feature_semantics": self.feature_semantics.to_payload(),
            "region_aggregation_method": self.region_aggregation_method,
            "normalization": {
                "translation": self.translation_normalization,
                "scale": self.scale_normalization,
                "rotation": self.rotation_normalization,
            },
            "missingness": {
                "missing_frame_handling": self.missing_frame_handling,
                "interpolation_policy": self.interpolation_policy,
            },
            "acceleration_primary": self.acceleration_primary,
        }
