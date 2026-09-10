from __future__ import annotations

from dataclasses import replace
from statistics import median

import pytest

from lagged_facial_graph_forecasting.population_response import (
    EdgeLagResponseObservation,
    PopulationResponseContractError,
    aggregate_edge_lag_response,
)


SUPPORT_A = "a" * 64
SUPPORT_B = "b" * 64


@pytest.mark.parametrize("changed_delta", [-1, 0, 1])
@pytest.mark.parametrize("deltas", [(-1, 0, 1), (1, -1, 0)])
def test_edge_response_rejects_reference_error_drift(changed_delta, deltas) -> None:
    rows = [
        _row(edge_id="e1", subject_id="s01", tau_star=2, delta=delta,
             reference_error=1.0, shifted_error=1.0)
        for delta in deltas
    ]
    rows = [
        replace(row, reference_error=2.0, shifted_error=2.0)
        if row.delta == changed_delta else row
        for row in rows
    ]
    with pytest.raises(PopulationResponseContractError, match="reference error"):
        aggregate_edge_lag_response(
            rows, deltas=deltas, aggregate=median, aggregation_id="median_edge_subject"
        )


def _row(
    *,
    edge_id: str,
    subject_id: str,
    tau_star: int,
    delta: int,
    reference_error: float,
    shifted_error: float,
    support: str = SUPPORT_A,
    context_id: str = "self_plus_edge",
) -> EdgeLagResponseObservation:
    return EdgeLagResponseObservation(
        outer_fold=0,
        edge_id=edge_id,
        subject_id=subject_id,
        source_region="eye",
        target_region="mouth",
        source_dimension="vx",
        target_dimension="vy",
        context_id=context_id,
        tau_star=tau_star,
        delta=delta,
        shifted_lag=tau_star + delta,
        sampling_rate_hz=30.0,
        reference_error=reference_error,
        shifted_error=shifted_error,
        reference_support_sha256=support,
        shifted_support_sha256=support,
    )


def test_edge_response_requires_complete_common_grid_and_keeps_edge_counts() -> None:
    rows = []
    for delta, shifted in ((-1, 2.0), (0, 1.0), (1, 0.5)):
        rows.append(
            _row(
                edge_id="e1",
                subject_id="s01",
                tau_star=2,
                delta=delta,
                reference_error=1.0,
                shifted_error=shifted,
            )
        )
        rows.append(
            _row(
                edge_id="e2",
                subject_id="s02",
                tau_star=3,
                delta=delta,
                reference_error=2.0,
                shifted_error={-1: 3.0, 0: 2.0, 1: 1.0}[delta],
            )
        )

    result = aggregate_edge_lag_response(
        rows,
        deltas=(-1, 0, 1),
        aggregate=median,
        aggregation_id="median_edge_subject",
    )

    assert [item.delta for item in result] == [-1, 0, 1]
    assert [item.value for item in result] == pytest.approx([1.0, 0.0, -0.75])
    assert all(item.n_edge_subject == 2 for item in result)
    assert all(item.n_subjects == 2 and item.n_edges == 2 for item in result)
    assert _row(
        edge_id="e1",
        subject_id="s01",
        tau_star=2,
        delta=1,
        reference_error=1.0,
        shifted_error=0.5,
    ).shifted_lag_milliseconds == pytest.approx(100.0)


def test_edge_response_rejects_zero_drift_and_lag_clipping() -> None:
    with pytest.raises(PopulationResponseContractError, match="delta=0"):
        _row(
            edge_id="e1",
            subject_id="s01",
            tau_star=2,
            delta=0,
            reference_error=1.0,
            shifted_error=1.1,
        )

    with pytest.raises(PopulationResponseContractError, match="clipping"):
        EdgeLagResponseObservation(
            outer_fold=0,
            edge_id="e1",
            subject_id="s01",
            source_region="eye",
            target_region="mouth",
            source_dimension="vx",
            target_dimension="vy",
            context_id="self_plus_edge",
            tau_star=1,
            delta=-1,
            shifted_lag=1,
            sampling_rate_hz=30.0,
            reference_error=1.0,
            shifted_error=1.0,
            reference_support_sha256=SUPPORT_A,
            shifted_support_sha256=SUPPORT_A,
        )


def test_edge_response_rejects_incomplete_grid_and_support_drift() -> None:
    with pytest.raises(PopulationResponseContractError, match="complete delta grid"):
        aggregate_edge_lag_response(
            [_row(edge_id="e1", subject_id="s01", tau_star=2, delta=0, reference_error=1.0, shifted_error=1.0)],
            deltas=(-1, 0, 1),
            aggregate=median,
            aggregation_id="median_edge_subject",
        )

    rows = [
        _row(edge_id="e1", subject_id="s01", tau_star=2, delta=-1, reference_error=1.0, shifted_error=2.0),
        _row(edge_id="e1", subject_id="s01", tau_star=2, delta=0, reference_error=1.0, shifted_error=1.0),
        _row(edge_id="e1", subject_id="s01", tau_star=2, delta=1, reference_error=1.0, shifted_error=0.5, support=SUPPORT_B),
    ]
    with pytest.raises(PopulationResponseContractError, match="identity or support"):
        aggregate_edge_lag_response(
            rows,
            deltas=(-1, 0, 1),
            aggregate=median,
            aggregation_id="median_edge_subject",
        )


def test_edge_response_keeps_contexts_separate() -> None:
    rows = [
        _row(edge_id="e1", subject_id="s01", tau_star=2, delta=-1, reference_error=1.0, shifted_error=2.0),
        _row(edge_id="e1", subject_id="s01", tau_star=2, delta=0, reference_error=1.0, shifted_error=1.0),
        _row(edge_id="e1", subject_id="s01", tau_star=2, delta=1, reference_error=1.0, shifted_error=0.5),
        _row(edge_id="e2", subject_id="s02", tau_star=2, delta=-1, reference_error=1.0, shifted_error=2.0, context_id="selected_set"),
    ]
    with pytest.raises(PopulationResponseContractError, match="contexts"):
        aggregate_edge_lag_response(
            rows,
            deltas=(-1, 0, 1),
            aggregate=median,
            aggregation_id="median_edge_subject",
        )
