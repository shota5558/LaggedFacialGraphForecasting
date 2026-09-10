from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting.pcmci_selection_frequency import (
    SelectionFrequencyError,
    aggregate_primary_selection_frequency,
)
from lagged_facial_graph_forecasting.pcmci_tau_max import PRIMARY_TAU_MAX
from lagged_facial_graph_forecasting.tigramite_adapter import TigramiteDataFrameBundle


def _bundle() -> TigramiteDataFrameBundle:
    return TigramiteDataFrameBundle(
        dataframe=None,  # D-13 uses only frozen variable provenance from the bundle.
        subject_ids=("train_1",),
        variable_names=("left_cheek::vx", "mouth::vy"),
        variable_components=(("left_cheek", "vx"), ("mouth", "vy")),
        sampling_rate=25.0,
    )


def _result(graph: np.ndarray, *, summary_results: object = None) -> dict[str, object]:
    return {
        "summary_results": summary_results,
        "boot_results": {"graph": graph},
    }


def test_aggregates_exact_component_lag_frequency_and_zero_frequency_rows() -> None:
    graph = np.full((3, 2, 2, PRIMARY_TAU_MAX + 1), "", dtype="<U3")
    graph[0, 0, 1, 2] = "-->"
    graph[2, 0, 1, 2] = "-->"

    rows = aggregate_primary_selection_frequency(_result(graph), _bundle())

    chosen = next(
        row
        for row in rows
        if row.source_region == "left_cheek"
        and row.source_dimension == "vx"
        and row.target_region == "mouth"
        and row.target_dimension == "vy"
        and row.lag == 2
    )
    assert chosen.selected_count == 2
    assert chosen.bootstrap_count == 3
    assert chosen.frequency == pytest.approx(2 / 3)
    assert chosen.is_interregional is True

    absent = next(
        row
        for row in rows
        if row.source_node_index == 0 and row.target_node_index == 1 and row.lag == 3
    )
    assert absent.selected_count == 0
    assert absent.frequency == 0.0


def test_tau_zero_diagnostics_are_outside_direct_h1_frequency_contract() -> None:
    graph = np.full((2, 2, 2, PRIMARY_TAU_MAX + 1), "", dtype="<U3")
    graph[:, 0, 1, 0] = "o-o"
    graph[:, 0, 1, 1] = "-->"

    rows = aggregate_primary_selection_frequency(_result(graph), _bundle())

    assert all(row.lag >= 1 for row in rows)
    assert not any(row.lag == 0 for row in rows)


def test_rejects_unexpected_positive_lag_tigramite_mark_fail_closed() -> None:
    graph = np.full((2, 2, 2, PRIMARY_TAU_MAX + 1), "", dtype="<U3")
    graph[0, 0, 1, 2] = "o->"

    with pytest.raises(SelectionFrequencyError, match="empty or '-->'"):
        aggregate_primary_selection_frequency(_result(graph), _bundle())


def test_rejects_malformed_bootstrap_graph_shape() -> None:
    malformed = np.full((2, 2, 2, PRIMARY_TAU_MAX), "", dtype="<U3")

    with pytest.raises(SelectionFrequencyError, match="bootstrap graph shape"):
        aggregate_primary_selection_frequency(_result(malformed), _bundle())


def test_summary_majority_cannot_change_raw_bootstrap_frequency() -> None:
    graph = np.full((4, 2, 2, PRIMARY_TAU_MAX + 1), "", dtype="<U3")
    graph[0, 0, 1, 4] = "-->"

    rows_a = aggregate_primary_selection_frequency(
        _result(graph, summary_results={"most_frequent_links": "adversarial"}),
        _bundle(),
    )
    rows_b = aggregate_primary_selection_frequency(
        _result(graph, summary_results={"most_frequent_links": "different"}),
        _bundle(),
    )

    assert rows_a == rows_b
    chosen = next(
        row
        for row in rows_a
        if row.source_node_index == 0 and row.target_node_index == 1 and row.lag == 4
    )
    assert chosen.frequency == 0.25
