from __future__ import annotations

import pytest

from lagged_facial_graph_forecasting.core_contracts import MetricsResult, NullMapping
from lagged_facial_graph_forecasting.lag_response_statistics import (
    LagResponseMetricAggregate,
    aggregate_lag_response_metrics,
)
from lagged_facial_graph_forecasting.null_lag_response import (
    LAG_RESPONSE_EVALUABLE,
    LAG_RESPONSE_UNEVALUABLE_INCOMPLETE_GRID,
    LAG_RESPONSE_UNEVALUABLE_NO_PARENTS,
    PRIMARY_LAG_RESPONSE_DELTAS,
    LagResponseGrid,
    LagResponseGridPoint,
)


def _grid(*, fold: int, status: str) -> LagResponseGrid:
    if status == LAG_RESPONSE_EVALUABLE:
        points = tuple(
            LagResponseGridPoint(
                delta=delta,
                mapping=None
                if delta == 0
                else NullMapping(
                    outer_fold=fold,
                    target_region="mouth",
                    condition="lag_shift",
                    seed=11,
                    source_parents=(),
                    mapped_parents=(),
                ),
            )
            for delta in PRIMARY_LAG_RESPONSE_DELTAS
        )
        return LagResponseGrid(
            outer_fold=fold,
            target_region="mouth",
            status=status,
            points=points,
        )
    if status == LAG_RESPONSE_UNEVALUABLE_INCOMPLETE_GRID:
        return LagResponseGrid(
            outer_fold=fold,
            target_region="mouth",
            status=status,
            points=(),
            infeasible_deltas=(-2,),
        )
    return LagResponseGrid(
        outer_fold=fold,
        target_region="mouth",
        status=LAG_RESPONSE_UNEVALUABLE_NO_PARENTS,
        points=(),
    )


def _metrics() -> dict[int, tuple[MetricsResult, ...]]:
    rows: dict[int, tuple[MetricsResult, ...]] = {}
    for delta in PRIMARY_LAG_RESPONSE_DELTAS:
        rows[delta] = (
            MetricsResult(
                outer_fold=0,
                subject_id="s0",
                region_id="mouth",
                condition="lag_shift",
                metric_name="velocity_rmse",
                value=2.0 + 0.1 * delta,
                n_valid=20,
            ),
            MetricsResult(
                outer_fold=1,
                subject_id="s1",
                region_id="mouth",
                condition="lag_shift",
                metric_name="velocity_rmse",
                value=4.0 + 0.1 * delta,
                n_valid=20,
            ),
        )
    return rows


def test_lag_response_aggregation_uses_identical_complete_grid_support() -> None:
    results = aggregate_lag_response_metrics(
        _metrics(),
        (
            _grid(fold=0, status=LAG_RESPONSE_EVALUABLE),
            _grid(fold=1, status=LAG_RESPONSE_EVALUABLE),
            _grid(fold=2, status=LAG_RESPONSE_UNEVALUABLE_NO_PARENTS),
            _grid(fold=3, status=LAG_RESPONSE_UNEVALUABLE_INCOMPLETE_GRID),
        ),
    )

    assert len(results) == 5
    assert all(isinstance(row, LagResponseMetricAggregate) for row in results)
    assert [row.delta for row in results] == list(PRIMARY_LAG_RESPONSE_DELTAS)
    assert [row.median_value for row in results] == pytest.approx(
        [2.8, 2.9, 3.0, 3.1, 3.2]
    )
    assert all(row.subject_ids == ("s0", "s1") for row in results)
    assert all(row.outer_folds == (0, 1) for row in results)
    assert all(row.n_evaluable_target_folds == 2 for row in results)
    assert all(row.n_unevaluable_no_parents == 1 for row in results)
    assert all(row.n_unevaluable_incomplete_grid == 1 for row in results)


def test_lag_response_aggregation_fails_closed_on_support_drift() -> None:
    metrics = _metrics()
    metrics[2] = metrics[2][:1]

    with pytest.raises(ValueError, match="identical held-out support"):
        aggregate_lag_response_metrics(
            metrics,
            (
                _grid(fold=0, status=LAG_RESPONSE_EVALUABLE),
                _grid(fold=1, status=LAG_RESPONSE_EVALUABLE),
            ),
        )


def test_lag_response_aggregation_rejects_unevaluable_metric_rows() -> None:
    metrics = _metrics()

    with pytest.raises(ValueError, match="unevaluable target folds"):
        aggregate_lag_response_metrics(
            metrics,
            (
                _grid(fold=0, status=LAG_RESPONSE_EVALUABLE),
                _grid(fold=1, status=LAG_RESPONSE_UNEVALUABLE_NO_PARENTS),
            ),
        )
