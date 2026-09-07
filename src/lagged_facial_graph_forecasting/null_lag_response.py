"""Frozen Primary lag-response grid construction (F-13).

The Primary lag-response estimand uses one predeclared symmetric frame grid:
``[-2, -1, 0, 1, 2]`` around the unmodified PCMCI ParentSet. Every selected
parent is shifted by the same delta. A target-fold is evaluable only when the
entire grid keeps every lag inside the frozen direct-h=1 Primary lag domain
``1..10``. Boundary cases are never clipped, wrapped, made one-sided, or repaired
by dropping features; they are marked unevaluable instead.

This module constructs the target-fold-level grid only. Cross-fold aggregation is
a downstream statistics responsibility, but its eligibility semantics are frozen
in ``configs/scientific_freeze.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import SplitManifest
from .core_contracts import NullMapping, ParentSet
from .leakage_guard import assert_null_construction_scope
from .null_lag_shift import construct_lag_shift_mapping
from .pcmci_lagged_filter import PRIMARY_LAG_MIN
from .pcmci_tau_max import PRIMARY_TAU_MAX


PRIMARY_LAG_RESPONSE_DELTAS: tuple[int, ...] = (-2, -1, 0, 1, 2)
PRIMARY_LAG_RESPONSE_REFERENCE_DELTA = 0

LAG_RESPONSE_EVALUABLE = "evaluable"
LAG_RESPONSE_UNEVALUABLE_INCOMPLETE_GRID = "unevaluable_incomplete_grid"
LAG_RESPONSE_UNEVALUABLE_NO_PARENTS = "unevaluable_no_parents"
_LAG_RESPONSE_STATUSES = {
    LAG_RESPONSE_EVALUABLE,
    LAG_RESPONSE_UNEVALUABLE_INCOMPLETE_GRID,
    LAG_RESPONSE_UNEVALUABLE_NO_PARENTS,
}


@dataclass(frozen=True, slots=True)
class LagResponseGridPoint:
    """One point on the frozen symmetric lag-response grid."""

    delta: int
    mapping: NullMapping | None

    def __post_init__(self) -> None:
        if self.delta not in PRIMARY_LAG_RESPONSE_DELTAS:
            raise ValueError("delta is outside the frozen Primary lag-response grid")
        if self.delta == PRIMARY_LAG_RESPONSE_REFERENCE_DELTA and self.mapping is not None:
            raise ValueError("reference delta=0 must use the unmodified PCMCI center")
        if self.delta != PRIMARY_LAG_RESPONSE_REFERENCE_DELTA and self.mapping is None:
            raise ValueError("non-reference lag-response points require a NullMapping")


@dataclass(frozen=True, slots=True)
class LagResponseGrid:
    """Target-fold evaluability result for the frozen lag-response grid."""

    outer_fold: int
    target_region: str
    status: str
    points: tuple[LagResponseGridPoint, ...]
    infeasible_deltas: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in _LAG_RESPONSE_STATUSES:
            raise ValueError(f"unknown lag-response status: {self.status!r}")

        if self.status == LAG_RESPONSE_EVALUABLE:
            deltas = tuple(point.delta for point in self.points)
            if deltas != PRIMARY_LAG_RESPONSE_DELTAS:
                raise ValueError(
                    "evaluable lag-response grids must contain the complete frozen symmetric grid"
                )
            if self.infeasible_deltas:
                raise ValueError("evaluable lag-response grids cannot have infeasible deltas")
            return

        if self.points:
            raise ValueError("unevaluable lag-response target-folds cannot expose a partial grid")

        if self.status == LAG_RESPONSE_UNEVALUABLE_INCOMPLETE_GRID:
            if not self.infeasible_deltas:
                raise ValueError("incomplete-grid status requires infeasible deltas")
            if any(delta not in PRIMARY_LAG_RESPONSE_DELTAS for delta in self.infeasible_deltas):
                raise ValueError("infeasible deltas must belong to the frozen lag-response grid")
        elif self.infeasible_deltas:
            raise ValueError("empty-parent status cannot have infeasible deltas")

    @property
    def is_evaluable(self) -> bool:
        return self.status == LAG_RESPONSE_EVALUABLE


def _infeasible_fixed_grid_deltas(parent_set: ParentSet) -> tuple[int, ...]:
    """Return fixed-grid deltas that would put any selected parent outside ``1..10``."""

    for parent in parent_set.parents:
        if not PRIMARY_LAG_MIN <= parent.lag <= PRIMARY_TAU_MAX:
            raise ValueError(
                "ParentSet lags must remain in the frozen Primary domain "
                f"{PRIMARY_LAG_MIN}..{PRIMARY_TAU_MAX}: {parent.lag}"
            )

    return tuple(
        delta
        for delta in PRIMARY_LAG_RESPONSE_DELTAS
        if any(
            not PRIMARY_LAG_MIN <= parent.lag + delta <= PRIMARY_TAU_MAX
            for parent in parent_set.parents
        )
    )


def construct_lag_response_grid(
    manifest: SplitManifest,
    parent_set: ParentSet,
    *,
    construction_subject_ids: tuple[str, ...],
) -> LagResponseGrid:
    """Construct the complete frozen grid or mark the target-fold unevaluable.

    Construction remains outer-train-only. No predictions, metrics, or outer-test
    values are accepted. The function never emits a clipped, wrapped, one-sided,
    or feature-dropped partial grid.
    """

    assert_null_construction_scope(manifest, construction_subject_ids)
    if not isinstance(parent_set, ParentSet):
        raise TypeError("parent_set must be a ParentSet")
    if parent_set.outer_fold != manifest.outer_fold:
        raise ValueError("ParentSet outer_fold must match SplitManifest")

    if not parent_set.parents:
        return LagResponseGrid(
            outer_fold=manifest.outer_fold,
            target_region=parent_set.target_region,
            status=LAG_RESPONSE_UNEVALUABLE_NO_PARENTS,
            points=(),
        )

    infeasible_deltas = _infeasible_fixed_grid_deltas(parent_set)
    if infeasible_deltas:
        return LagResponseGrid(
            outer_fold=manifest.outer_fold,
            target_region=parent_set.target_region,
            status=LAG_RESPONSE_UNEVALUABLE_INCOMPLETE_GRID,
            points=(),
            infeasible_deltas=infeasible_deltas,
        )

    points: list[LagResponseGridPoint] = []
    for delta in PRIMARY_LAG_RESPONSE_DELTAS:
        if delta == PRIMARY_LAG_RESPONSE_REFERENCE_DELTA:
            points.append(LagResponseGridPoint(delta=delta, mapping=None))
            continue
        mapping = construct_lag_shift_mapping(
            manifest,
            parent_set,
            construction_subject_ids=construction_subject_ids,
            lag_delta=delta,
        )
        points.append(LagResponseGridPoint(delta=delta, mapping=mapping))

    return LagResponseGrid(
        outer_fold=manifest.outer_fold,
        target_region=parent_set.target_region,
        status=LAG_RESPONSE_EVALUABLE,
        points=tuple(points),
    )
