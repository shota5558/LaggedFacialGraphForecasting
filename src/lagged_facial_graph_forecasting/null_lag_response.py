"""Deterministic lag-response grid derived from a frozen Primary ParentSet (F-13).

The research plan requires a curve around ``delta=0`` rather than a single
hand-picked shift. It does not freeze an arbitrary finite list of deltas. To avoid
result-dependent tuning, F-13 therefore uses the exhaustive common-shift domain
implied by two already-frozen facts only: every selected parent must remain in the
Primary direct-h=1 lag domain ``1..10``, and every parent must be shifted by the
same integer delta so region/component identity and feature count stay unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import SplitManifest
from .core_contracts import NullMapping, ParentSet
from .leakage_guard import assert_null_construction_scope
from .null_lag_shift import construct_lag_shift_mapping
from .pcmci_lagged_filter import PRIMARY_LAG_MIN
from .pcmci_tau_max import PRIMARY_TAU_MAX


@dataclass(frozen=True, slots=True)
class LagResponseGridPoint:
    """One deterministic point on a lag-response curve.

    ``delta=0`` is the unmodified PCMCI center and therefore has ``mapping=None``.
    Every non-zero point contains a frozen ``lag-shift`` NullMapping.
    """

    delta: int
    mapping: NullMapping | None

    def __post_init__(self) -> None:
        if self.delta == 0 and self.mapping is not None:
            raise ValueError("delta=0 must use the unmodified PCMCI center")
        if self.delta != 0 and self.mapping is None:
            raise ValueError("non-zero lag-response points require a NullMapping")


def feasible_common_lag_deltas(parent_set: ParentSet) -> tuple[int, ...]:
    """Return every common integer shift that preserves all Primary parent lags.

    For an empty ParentSet there is no lag to falsify, so only the PCMCI center is
    defined. For non-empty sets, the returned interval is exhaustive and always
    contains zero. No outer-test values or forecast errors are involved.
    """

    if not isinstance(parent_set, ParentSet):
        raise TypeError("parent_set must be a ParentSet")
    if not parent_set.parents:
        return (0,)

    min_lag = min(parent.lag for parent in parent_set.parents)
    max_lag = max(parent.lag for parent in parent_set.parents)
    if min_lag < PRIMARY_LAG_MIN or max_lag > PRIMARY_TAU_MAX:
        raise ValueError(
            "ParentSet lags must remain in the frozen Primary domain "
            f"{PRIMARY_LAG_MIN}..{PRIMARY_TAU_MAX}"
        )

    min_delta = PRIMARY_LAG_MIN - min_lag
    max_delta = PRIMARY_TAU_MAX - max_lag
    return tuple(range(min_delta, max_delta + 1))


def construct_lag_response_grid(
    manifest: SplitManifest,
    parent_set: ParentSet,
    *,
    construction_subject_ids: tuple[str, ...],
) -> tuple[LagResponseGridPoint, ...]:
    """Construct the exhaustive feasible lag-response curve before outer-test unlock."""

    assert_null_construction_scope(manifest, construction_subject_ids)
    if parent_set.outer_fold != manifest.outer_fold:
        raise ValueError("ParentSet outer_fold must match SplitManifest")

    points: list[LagResponseGridPoint] = []
    for delta in feasible_common_lag_deltas(parent_set):
        if delta == 0:
            points.append(LagResponseGridPoint(delta=0, mapping=None))
            continue
        mapping = construct_lag_shift_mapping(
            manifest,
            parent_set,
            construction_subject_ids=construction_subject_ids,
            lag_delta=delta,
        )
        points.append(LagResponseGridPoint(delta=delta, mapping=mapping))
    return tuple(points)
