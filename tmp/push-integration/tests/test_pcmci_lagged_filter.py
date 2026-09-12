from __future__ import annotations

import pytest

from lagged_facial_graph_forecasting.pcmci_contemporaneous import (
    ContemporaneousLinkDisposition,
    apply_primary_contemporaneous_policy,
)
from lagged_facial_graph_forecasting.pcmci_lagged_filter import (
    PRIMARY_LAGGED_LINK_MARK,
    PRIMARY_LAG_MIN,
    LaggedLinkFilterError,
    filter_primary_lagged_links,
)
from lagged_facial_graph_forecasting.pcmci_link_extraction import SignificantPCMCIPlusLink
from lagged_facial_graph_forecasting.pcmci_tau_max import PRIMARY_TAU_MAX


def _link(*, lag: int, mark: str = "-->", source: int = 0, target: int = 1):
    return SignificantPCMCIPlusLink(
        source_node_index=source,
        target_node_index=target,
        lag=lag,
        link_mark=mark,
        p_value=0.005,
        test_statistic=0.4,
        source_variable_name="left_cheek::vx",
        target_variable_name="mouth::vy",
        source_region="left_cheek",
        source_dimension="vx",
        target_region="mouth",
        target_dimension="vy",
    )


def _disposition(*links: SignificantPCMCIPlusLink) -> ContemporaneousLinkDisposition:
    return ContemporaneousLinkDisposition(
        forecast_candidates=tuple(links),
        excluded_contemporaneous=(),
    )


def test_accepts_directed_lagged_tigramite_links_in_primary_range() -> None:
    links = (_link(lag=1), _link(lag=3, source=1), _link(lag=10))

    accepted = filter_primary_lagged_links(_disposition(*links))

    assert accepted == links
    assert PRIMARY_LAG_MIN == 1
    assert PRIMARY_TAU_MAX == 10
    assert PRIMARY_LAGGED_LINK_MARK == "-->"


def test_empty_lagged_candidate_set_is_valid() -> None:
    assert filter_primary_lagged_links(_disposition()) == ()


def test_tau_zero_is_removed_by_d08_before_d09() -> None:
    contemporaneous = _link(lag=0)
    lagged = _link(lag=1)

    disposition = apply_primary_contemporaneous_policy((contemporaneous, lagged))

    assert disposition.excluded_contemporaneous == (contemporaneous,)
    assert filter_primary_lagged_links(disposition) == (lagged,)


@pytest.mark.parametrize("lag", [11, 99])
def test_rejects_lag_above_primary_tau_max(lag: int) -> None:
    with pytest.raises(LaggedLinkFilterError, match="1 <= tau <= 10"):
        filter_primary_lagged_links(_disposition(_link(lag=lag)))


@pytest.mark.parametrize("mark", ["<--", "o-o", "x-x", "-?>", "o->", ""])
def test_rejects_unexpected_lagged_graph_mark_fail_closed(mark: str) -> None:
    with pytest.raises(LaggedLinkFilterError, match="must use '-->'"):
        filter_primary_lagged_links(_disposition(_link(lag=2, mark=mark)))


def test_preserves_exact_component_provenance_and_order() -> None:
    first = _link(lag=5, source=1, target=0)
    second = _link(lag=2, source=0, target=1)

    accepted = filter_primary_lagged_links(_disposition(first, second))

    assert accepted == (first, second)
    assert accepted[0].source_variable_name == first.source_variable_name
    assert accepted[0].target_dimension == first.target_dimension


def test_rejects_noncanonical_disposition() -> None:
    with pytest.raises(TypeError, match="ContemporaneousLinkDisposition"):
        filter_primary_lagged_links(object())  # type: ignore[arg-type]
