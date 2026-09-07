"""Adapters between frozen facial-motion contracts and Tigramite OSS objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from tigramite import data_processing as pp

from .contracts import FaceTimeSeries, SplitManifest
from .leakage_guard import assert_discovery_fit_scope


class TigramiteAdapterError(ValueError):
    """Raised when canonical facial time series cannot be converted safely."""


@dataclass(frozen=True, slots=True)
class TigramiteDataFrameBundle:
    """Tigramite DataFrame plus the canonical component-column provenance."""

    dataframe: pp.DataFrame
    subject_ids: tuple[str, ...]
    variable_names: tuple[str, ...]
    variable_components: tuple[tuple[str, str], ...]


def _component_metadata(series: FaceTimeSeries) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    components = tuple(
        (region, dimension)
        for region in series.region_id
        for dimension in series.dimension
    )
    names = tuple(f"{region}.{dimension}" for region, dimension in components)
    return names, components


def face_time_series_to_tigramite_dataframe(
    manifest: SplitManifest,
    series_by_subject: Sequence[FaceTimeSeries],
) -> TigramiteDataFrameBundle:
    """Convert outer-train subject sequences without concatenating boundaries.

    The converter is a discovery-facing API and therefore requires the canonical
    ``SplitManifest``.  Every supplied subject is checked with the existing discovery
    leakage guard before any Tigramite object is constructed; outer-test or otherwise
    out-of-scope subjects fail closed.

    Each allowed subject is supplied to Tigramite as a separate dataset in
    ``analysis_mode='multiple'``.  The canonical ``(T, K, D)`` tensor is flattened only
    across the region/dimension axes, yielding ``(T, K*D)`` continuous variables.
    Elementwise invalid observations are replaced by a constant finite placeholder only
    because Tigramite rejects NaN in ``data``; the same positions are marked ``True`` in
    Tigramite's mask for exclusion by the CI-test masking policy.  No temporal
    interpolation, resampling, fitting, feature selection, or subject concatenation
    occurs here.
    """

    sequences = tuple(series_by_subject)
    if not sequences:
        raise TigramiteAdapterError("at least one FaceTimeSeries is required")
    if any(not isinstance(series, FaceTimeSeries) for series in sequences):
        raise TigramiteAdapterError("all inputs must be FaceTimeSeries instances")

    subject_ids = tuple(series.subject_id for series in sequences)
    if len(set(subject_ids)) != len(subject_ids):
        raise TigramiteAdapterError("subject_id values must be unique")

    # Reuse the canonical Lane-B guard so discovery input can never contain outer-test.
    assert_discovery_fit_scope(manifest, subject_ids)

    reference = sequences[0]
    variable_names, variable_components = _component_metadata(reference)

    data: dict[str, np.ndarray] = {}
    mask: dict[str, np.ndarray] = {}
    for series in sequences:
        if series.region_id != reference.region_id:
            raise TigramiteAdapterError("all subjects must have identical region_id order")
        if series.dimension != reference.dimension:
            raise TigramiteAdapterError("all subjects must have identical dimension order")
        if not np.isclose(series.sampling_rate, reference.sampling_rate, rtol=0.0, atol=0.0):
            raise TigramiteAdapterError("all subjects must have identical sampling_rate")

        values = np.asarray(series.X)
        valid = np.asarray(series.valid_mask, dtype=bool)
        flat_values = np.asarray(values, dtype=float).reshape(values.shape[0], -1)
        flat_valid = valid.reshape(valid.shape[0], -1)

        if np.any(flat_valid & ~np.isfinite(flat_values)):
            raise TigramiteAdapterError("non-finite observations cannot be marked valid")

        # Tigramite DataFrame rejects NaN in data before applying its mask.  A fixed
        # placeholder is safe only together with the corresponding True mask entries;
        # D-02 freezes the ParCorr mask_type used by discovery.
        safe_values = np.array(flat_values, copy=True)
        safe_values[~flat_valid] = 0.0
        data[series.subject_id] = safe_values
        mask[series.subject_id] = ~flat_valid

    dataframe = pp.DataFrame(
        data=data,
        mask=mask,
        var_names=list(variable_names),
        analysis_mode="multiple",
    )
    return TigramiteDataFrameBundle(
        dataframe=dataframe,
        subject_ids=subject_ids,
        variable_names=variable_names,
        variable_components=variable_components,
    )
