"""Primary lag-shift falsification mapping (F-10/F-11).

F-10 changes only lag identity. Source region, source component, target component,
feature count, and ParentSet target are preserved exactly. F-11 freezes the valid
Primary lag domain to the already-frozen direct-h=1 discovery domain ``1..10``.
Boundary violations fail closed: clipping, wrapping, dropping features, or choosing
another shift from observed performance would change the Null estimand.
"""

from __future__ import annotations

import numpy as np

from .contracts import SplitManifest
from .core_contracts import NullMapping, ParentLink, ParentSet
from .leakage_guard import assert_null_construction_scope
from .pcmci_lagged_filter import PRIMARY_LAG_MIN
from .pcmci_tau_max import PRIMARY_TAU_MAX


class LagShiftMappingError(ValueError):
    """Raised when a lag-shift mapping request violates the Primary contract."""


def _validate_primary_lag_shift_domain(
    parent_set: ParentSet,
    lag_delta: int,
) -> tuple[int, ...]:
    """Return shifted lags only when source and mapped lags stay in ``1..10``.

    The F-11 policy deliberately does not clip, wrap, drop, or replace links. Such
    behavior would alter feature count or lag identity and could make different
    shifts incomparable. A caller must therefore request only a shift that is valid
    for every selected parent in the frozen ParentSet.
    """

    shifted_lags: list[int] = []
    for parent in parent_set.parents:
        if not PRIMARY_LAG_MIN <= parent.lag <= PRIMARY_TAU_MAX:
            raise LagShiftMappingError(
                "source ParentSet lag is outside the frozen Primary domain "
                f"{PRIMARY_LAG_MIN}..{PRIMARY_TAU_MAX}: {parent.lag}"
            )
        shifted_lag = parent.lag + lag_delta
        if not PRIMARY_LAG_MIN <= shifted_lag <= PRIMARY_TAU_MAX:
            raise LagShiftMappingError(
                "lag-shift would leave the frozen Primary domain "
                f"{PRIMARY_LAG_MIN}..{PRIMARY_TAU_MAX}: "
                f"source lag {parent.lag} with delta {lag_delta} -> {shifted_lag}; "
                "clipping, wrapping, or dropping features is forbidden"
            )
        shifted_lags.append(shifted_lag)
    return tuple(shifted_lags)


def construct_lag_shift_mapping(
    manifest: SplitManifest,
    parent_set: ParentSet,
    *,
    construction_subject_ids: tuple[str, ...],
    lag_delta: int,
) -> NullMapping:
    """Shift every selected cross-region ParentLink by the same non-zero delta.

    Null construction is permitted only from outer-train scope. No data values,
    predictions, metrics, or outer-test outcomes are accepted by this API. Every
    shifted lag must remain in the frozen Primary direct-h=1 domain ``1..10``.
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

    shifted_lags = _validate_primary_lag_shift_domain(parent_set, lag_delta)
    mapped = tuple(
        ParentLink(
            source_region=parent.source_region,
            lag=shifted_lag,
            source_dimension=parent.source_dimension,
            target_dimension=parent.target_dimension,
        )
        for parent, shifted_lag in zip(parent_set.parents, shifted_lags, strict=True)
    )

    return NullMapping(
        outer_fold=manifest.outer_fold,
        target_region=parent_set.target_region,
        condition="lag-shift",
        seed=manifest.seed,
        source_parents=parent_set.parents,
        mapped_parents=mapped,
    )
