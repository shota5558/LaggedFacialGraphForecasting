"""Per-frame translation normalization for region-level facial coordinates (A-05)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .region_aggregation import RegionAggregation


class TranslationNormalizationError(ValueError):
    """Raised when translation-normalization provenance is ambiguous or invalid."""


@dataclass(frozen=True, slots=True)
class TranslationNormalization:
    """Immutable translated coordinates plus explicit anchor provenance.

    ``offsets[t]`` is the per-frame centroid of the explicitly frozen reference
    regions. It is a deterministic transform of that frame, not a parameter fit
    across subjects or folds.
    """

    values: np.ndarray
    region_ids: tuple[str, ...]
    reference_region_ids: tuple[str, ...]
    offsets: np.ndarray
    method: str = "reference_region_centroid"

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        offsets = np.asarray(self.offsets)
        region_ids = tuple(self.region_ids)
        reference_region_ids = tuple(self.reference_region_ids)

        if values.ndim != 3 or any(size < 1 for size in values.shape):
            raise TranslationNormalizationError(
                "values must have non-empty shape (T, K, C)"
            )
        if offsets.shape != (values.shape[0], values.shape[2]):
            raise TranslationNormalizationError(
                "offsets must have shape (T, C) matching values"
            )
        if len(region_ids) != values.shape[1]:
            raise TranslationNormalizationError(
                "region_ids must match the region axis of values"
            )
        if not reference_region_ids:
            raise TranslationNormalizationError(
                "reference_region_ids must contain at least one frozen anchor region"
            )
        if len(set(reference_region_ids)) != len(reference_region_ids):
            raise TranslationNormalizationError(
                "reference_region_ids must be unique"
            )
        if any(region_id not in region_ids for region_id in reference_region_ids):
            raise TranslationNormalizationError(
                "reference_region_ids must all exist in region_ids"
            )
        if self.method != "reference_region_centroid":
            raise TranslationNormalizationError(
                "A-05 method must be 'reference_region_centroid'"
            )

        frozen_values = np.array(values, copy=True)
        frozen_offsets = np.array(offsets, copy=True)
        frozen_values.setflags(write=False)
        frozen_offsets.setflags(write=False)
        object.__setattr__(self, "values", frozen_values)
        object.__setattr__(self, "offsets", frozen_offsets)
        object.__setattr__(self, "region_ids", region_ids)
        object.__setattr__(self, "reference_region_ids", reference_region_ids)


def normalize_translation(
    aggregation: RegionAggregation,
    *,
    reference_region_ids: tuple[str, ...],
) -> TranslationNormalization:
    """Remove per-frame global translation using explicit frozen anchor regions.

    The caller must declare the reference regions; A-05 never chooses anchors from
    observed predictive performance. Ordinary ``mean`` is intentional so missing
    anchor coordinates propagate rather than being silently repaired before the
    missingness/interpolation tasks.
    """

    if not isinstance(aggregation, RegionAggregation):
        raise TranslationNormalizationError(
            "aggregation must be a RegionAggregation"
        )
    references = tuple(reference_region_ids)
    if not references:
        raise TranslationNormalizationError(
            "reference_region_ids must contain at least one frozen anchor region"
        )
    if len(set(references)) != len(references):
        raise TranslationNormalizationError("reference_region_ids must be unique")

    region_to_index = {
        region_id: index for index, region_id in enumerate(aggregation.region_ids)
    }
    missing = tuple(region_id for region_id in references if region_id not in region_to_index)
    if missing:
        raise TranslationNormalizationError(
            f"reference regions are not present in aggregation: {missing}"
        )

    reference_indices = [region_to_index[region_id] for region_id in references]
    offsets = np.mean(aggregation.values[:, reference_indices, :], axis=1)
    normalized = aggregation.values - offsets[:, np.newaxis, :]

    return TranslationNormalization(
        values=normalized,
        region_ids=aggregation.region_ids,
        reference_region_ids=references,
        offsets=offsets,
    )
