"""Primary lag-shift falsification mapping (F-10).

F-10 changes only lag identity. Source region, source component, target component,
feature count, and ParentSet target are preserved exactly. Boundary policy belongs
to F-11; this task only implements the scientific transformation itself.
"""

from __future__ import annotations

import numpy as np

from .contracts import SplitManifest
from .core_contracts import NullMapping, ParentLink, ParentSet
from .leakage_guard import assert_null_construction_scope


class LagShiftMappingError(ValueError):
    """Raised when a lag-shift mapping request violates the F-10 contract."""


def construct_lag_shift_mapping(
    manifest: SplitManifest,
    parent_set: ParentSet,
    *,
    construction_subject_ids: tuple[str, ...],
    lag_delta: int,
) -> NullMapping:
    """Shift every selected cross-region ParentLink by the same non-zero delta.

    Null construction is permitted only from outer-train scope. No data values,
    predictions, metrics, or outer-test outcomes are accepted by this API.
    """

    assert_null_construction_scope(manifest, construction_subject_ids)
    if not isinstance(parent_set, ParentSet):
        raise TypeError("parent_set must be a ParentSet")
    if parent_set.outer_fold != manifest.outer_fold:
        raise LagShiftMappingError("ParentSet outer_fold must match SplitManifest")
    if not isinstance(lag_delta, (int, np.integer)) or isinstance(lag_delta, (bool, np.bool_)):
        raise LagShiftMappingError("lag_delta must be an integer")
    lag_delta = int(lag_delta)
    if lag_delta == 0:
        raise LagShiftMappingError("lag_delta must be non-zero for lag-shift Null")

    mapped = tuple(
        ParentLink(
            source_region=parent.source_region,
            lag=parent.lag + lag_delta,
            source_dimension=parent.source_dimension,
            target_dimension=parent.target_dimension,
        )
        for parent in parent_set.parents
    )

    return NullMapping(
        outer_fold=manifest.outer_fold,
        target_region=parent_set.target_region,
        condition="lag-shift",
        seed=manifest.seed,
        source_parents=parent_set.parents,
        mapped_parents=mapped,
    )
