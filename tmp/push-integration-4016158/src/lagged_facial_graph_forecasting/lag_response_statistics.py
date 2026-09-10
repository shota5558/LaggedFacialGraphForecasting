"""Cross-fold aggregation for the frozen Primary lag-response estimand (G-17).

The target-fold grid itself is constructed by ``null_lag_response``.  This module
implements the downstream statistics contract frozen in Scientific Freeze v6:
only complete-grid target folds may contribute, exactly the same held-out units
must support every delta, and unevaluable target-fold counts remain explicit.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .core_contracts import MetricsResult
from .null_lag_response import (
    LAG_RESPONSE_EVALUABLE,
    LAG_RESPONSE_UNEVALUABLE_INCOMPLETE_GRID,
    LAG_RESPONSE_UNEVALUABLE_NO_PARENTS,
    PRIMARY_LAG_RESPONSE_DELTAS,
    LagResponseGrid,
)


@dataclass(frozen=True, slots=True)
class LagResponseMetricAggregate:
    """Median held-out metric at one lag-response delta for one region/metric."""

    delta: int
    region_id: str
    metric_name: str
    condition: str
    median_value: float
    n_subjects: int
    subject_ids: tuple[str, ...]
    outer_folds: tuple[int, ...]
    n_evaluable_target_folds: int
    n_unevaluable_no_parents: int
    n_unevaluable_incomplete_grid: int

    def __post_init__(self) -> None:
        if self.delta not in PRIMARY_LAG_RESPONSE_DELTAS:
            raise ValueError("delta is outside the frozen Primary lag-response grid")
        for name in ("region_id", "metric_name", "condition"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        median_value = float(self.median_value)
        if not np.isfinite(median_value):
            raise ValueError("median_value must be finite")
        object.__setattr__(self, "median_value", median_value)
        if not isinstance(self.n_subjects, int) or isinstance(self.n_subjects, bool) or self.n_subjects < 1:
            raise ValueError("n_subjects must be a positive integer")
        subject_ids = tuple(self.subject_ids)
        outer_folds = tuple(self.outer_folds)
        if len(subject_ids) != self.n_subjects or len(outer_folds) != self.n_subjects:
            raise ValueError("subject_ids and outer_folds must match n_subjects")
        if len(set(subject_ids)) != len(subject_ids):
            raise ValueError("each held-out subject may contribute only once per aggregate")
        for name in (
            "n_evaluable_target_folds",
            "n_unevaluable_no_parents",
            "n_unevaluable_incomplete_grid",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(self, "subject_ids", subject_ids)
        object.__setattr__(self, "outer_folds", outer_folds)


def aggregate_lag_response_metrics(
    metrics_by_delta: Mapping[int, Sequence[MetricsResult]],
    grids: Sequence[LagResponseGrid],
) -> tuple[LagResponseMetricAggregate, ...]:
    """Aggregate one frozen lag-response metric over identical complete-grid support.

    ``metrics_by_delta`` must contain exactly ``[-2,-1,0,1,2]``.  Every delta must
    contain the same ``outer_fold × subject × region × metric`` keys and the same
    condition label.  Any metric row whose target fold is absent or unevaluable is
    rejected; no clipping, one-sided grid, feature dropping, or support drift can
    enter the aggregate silently.
    """

    if not isinstance(metrics_by_delta, Mapping):
        raise TypeError("metrics_by_delta must be a mapping")
    normalized_keys = tuple(metrics_by_delta.keys())
    if any(not isinstance(delta, int) or isinstance(delta, bool) for delta in normalized_keys):
        raise ValueError("lag-response delta keys must be integers")
    if set(normalized_keys) != set(PRIMARY_LAG_RESPONSE_DELTAS):
        raise ValueError(
            "metrics_by_delta must contain the complete frozen lag-response grid"
        )

    normalized_grids = tuple(grids)
    if not normalized_grids:
        raise ValueError("grids must not be empty")
    if any(not isinstance(grid, LagResponseGrid) for grid in normalized_grids):
        raise TypeError("grids entries must be LagResponseGrid instances")

    grid_by_key: dict[tuple[int, str], LagResponseGrid] = {}
    status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for grid in normalized_grids:
        key = (grid.outer_fold, grid.target_region)
        if key in grid_by_key:
            raise ValueError(f"duplicate lag-response grid for target fold {key!r}")
        grid_by_key[key] = grid
        status_counts[grid.target_region][grid.status] += 1

    indexed_by_delta: dict[
        int, dict[tuple[int, str, str, str], MetricsResult]
    ] = {}
    common_condition: str | None = None
    reference_support: set[tuple[int, str, str, str]] | None = None

    for delta in PRIMARY_LAG_RESPONSE_DELTAS:
        rows = tuple(metrics_by_delta[delta])
        if not rows:
            raise ValueError(f"delta={delta} metric results must not be empty")
        if any(not isinstance(row, MetricsResult) for row in rows):
            raise TypeError("lag-response metric rows must be MetricsResult instances")
        conditions = {row.condition for row in rows}
        if len(conditions) != 1:
            raise ValueError(f"delta={delta} must contain exactly one condition")
        condition = next(iter(conditions))
        if common_condition is None:
            common_condition = condition
        elif condition != common_condition:
            raise ValueError("condition must be identical across every lag-response delta")

        indexed: dict[tuple[int, str, str, str], MetricsResult] = {}
        for row in rows:
            key = (row.outer_fold, row.subject_id, row.region_id, row.metric_name)
            if key in indexed:
                raise ValueError(f"duplicate lag-response metric support key {key!r}")
            grid = grid_by_key.get((row.outer_fold, row.region_id))
            if grid is None:
                raise ValueError(
                    "metric row has no corresponding target-fold lag-response grid: "
                    f"{key!r}"
                )
            if grid.status != LAG_RESPONSE_EVALUABLE:
                raise ValueError(
                    "unevaluable target folds must not contribute lag-response metrics: "
                    f"fold={row.outer_fold}, region={row.region_id}, status={grid.status}"
                )
            indexed[key] = row

        support = set(indexed)
        if reference_support is None:
            reference_support = support
        elif support != reference_support:
            missing = sorted(reference_support - support)
            extra = sorted(support - reference_support)
            raise ValueError(
                "lag-response deltas must use identical held-out support; "
                f"delta={delta}, missing={missing}, extra={extra}"
            )
        indexed_by_delta[delta] = indexed

    assert reference_support is not None
    assert common_condition is not None

    grouped_support: dict[tuple[str, str], list[tuple[int, str, str, str]]] = defaultdict(list)
    for key in sorted(reference_support):
        grouped_support[(key[2], key[3])].append(key)

    aggregates: list[LagResponseMetricAggregate] = []
    for (region_id, metric_name), keys in sorted(grouped_support.items()):
        subject_ids = tuple(key[1] for key in keys)
        if len(set(subject_ids)) != len(subject_ids):
            raise ValueError(
                "each subject must contribute once per region/metric lag-response aggregate"
            )
        outer_folds = tuple(key[0] for key in keys)
        counts = status_counts[region_id]
        for delta in PRIMARY_LAG_RESPONSE_DELTAS:
            values = np.asarray(
                [indexed_by_delta[delta][key].value for key in keys], dtype=float
            )
            aggregates.append(
                LagResponseMetricAggregate(
                    delta=delta,
                    region_id=region_id,
                    metric_name=metric_name,
                    condition=common_condition,
                    median_value=float(np.median(values)),
                    n_subjects=len(keys),
                    subject_ids=subject_ids,
                    outer_folds=outer_folds,
                    n_evaluable_target_folds=counts[LAG_RESPONSE_EVALUABLE],
                    n_unevaluable_no_parents=counts[LAG_RESPONSE_UNEVALUABLE_NO_PARENTS],
                    n_unevaluable_incomplete_grid=counts[
                        LAG_RESPONSE_UNEVALUABLE_INCOMPLETE_GRID
                    ],
                )
            )

    return tuple(aggregates)
