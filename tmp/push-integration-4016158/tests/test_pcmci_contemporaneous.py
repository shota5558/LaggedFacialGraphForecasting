from __future__ import annotations

from inspect import signature

import pytest

from lagged_facial_graph_forecasting.pcmci_contemporaneous import (
    PRIMARY_CONTEMPORANEOUS_POLICY,
    PRIMARY_CONTEMPORANEOUS_POLICY_SCHEMA_VERSION,
    PRIMARY_FORECAST_HORIZON,
    ContemporaneousLinkDisposition,
    ContemporaneousPolicyError,
    apply_primary_contemporaneous_policy,
)
from lagged_facial_graph_forecasting.pcmci_link_extraction import SignificantPCMCIPlusLink


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


def test_tau_zero_is_preserved_for_audit_but_excluded_from_h1_forecasting() -> None:
    contemporaneous = _link(lag=0)
    lagged = _link(lag=1)

    disposition = apply_primary_contemporaneous_policy((contemporaneous, lagged))

    assert disposition.forecast_candidates == (lagged,)
    assert disposition.excluded_contemporaneous == (contemporaneous,)
    assert disposition.policy == PRIMARY_CONTEMPORANEOUS_POLICY
    assert disposition.forecast_horizon == PRIMARY_FORECAST_HORIZON == 1
    assert disposition.schema_version == PRIMARY_CONTEMPORANEOUS_POLICY_SCHEMA_VERSION == 1


def test_reciprocal_contemporaneous_marks_are_both_preserved_audit_only() -> None:
    forward = _link(lag=0, mark="-->", source=0, target=1)
    reverse = _link(lag=0, mark="<--", source=1, target=0)

    disposition = apply_primary_contemporaneous_policy((forward, reverse))

    assert disposition.forecast_candidates == ()
    assert disposition.excluded_contemporaneous == (forward, reverse)


def test_strictly_lagged_candidates_pass_through_without_reordering() -> None:
    links = (_link(lag=3, source=1), _link(lag=1, source=0), _link(lag=10, source=1))

    disposition = apply_primary_contemporaneous_policy(links)

    assert disposition.forecast_candidates == links
    assert disposition.excluded_contemporaneous == ()


def test_policy_api_exposes_no_runtime_horizon_or_policy_override() -> None:
    assert tuple(signature(apply_primary_contemporaneous_policy).parameters) == ("links",)


def test_rejects_negative_lag_contract_corruption() -> None:
    with pytest.raises(ContemporaneousPolicyError, match="non-negative"):
        apply_primary_contemporaneous_policy((_link(lag=-1),))


def test_rejects_non_link_items() -> None:
    with pytest.raises(TypeError, match="SignificantPCMCIPlusLink"):
        apply_primary_contemporaneous_policy((_link(lag=1), object()))  # type: ignore[arg-type]


def test_disposition_rejects_forecast_candidate_at_tau_zero() -> None:
    with pytest.raises(ContemporaneousPolicyError, match="strictly lagged"):
        ContemporaneousLinkDisposition(
            forecast_candidates=(_link(lag=0),),
            excluded_contemporaneous=(),
        )


def test_disposition_rejects_nonzero_link_in_contemporaneous_audit_bucket() -> None:
    with pytest.raises(ContemporaneousPolicyError, match="only tau=0"):
        ContemporaneousLinkDisposition(
            forecast_candidates=(),
            excluded_contemporaneous=(_link(lag=1),),
        )


def test_disposition_rejects_primary_policy_horizon_or_schema_drift() -> None:
    with pytest.raises(ContemporaneousPolicyError, match="policy is frozen"):
        ContemporaneousLinkDisposition((), (), policy="include_tau_zero")
    with pytest.raises(ContemporaneousPolicyError, match="h=1"):
        ContemporaneousLinkDisposition((), (), forecast_horizon=2)
    with pytest.raises(ContemporaneousPolicyError, match="schema_version"):
        ContemporaneousLinkDisposition((), (), schema_version=2)
