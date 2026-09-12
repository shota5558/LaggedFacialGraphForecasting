"""Primary contemporaneous-link disposition for direct h=1 forecasting (D-08).

PCMCI+ may retain contemporaneous (tau=0) relations, but the Primary estimand is a
direct one-step-ahead forecast.  At forecast origin t, a tau=0 parent of Y[t+1]
would be X[t+1], which is not observed yet.  Such links are therefore preserved as
discovery/audit evidence but are never exposed as Primary forecasting candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .pcmci_link_extraction import SignificantPCMCIPlusLink


PRIMARY_CONTEMPORANEOUS_POLICY_SCHEMA_VERSION = 1
PRIMARY_CONTEMPORANEOUS_POLICY = "preserve_audit_exclude_direct_h1_forecasting"
PRIMARY_FORECAST_HORIZON = 1


class ContemporaneousPolicyError(ValueError):
    """Raised when the frozen Primary contemporaneous-link policy is violated."""


@dataclass(frozen=True, slots=True)
class ContemporaneousLinkDisposition:
    """Partition of retained PCMCI+ links under the frozen Primary h=1 policy."""

    forecast_candidates: tuple[SignificantPCMCIPlusLink, ...]
    excluded_contemporaneous: tuple[SignificantPCMCIPlusLink, ...]
    policy: str = PRIMARY_CONTEMPORANEOUS_POLICY
    forecast_horizon: int = PRIMARY_FORECAST_HORIZON
    schema_version: int = PRIMARY_CONTEMPORANEOUS_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.policy != PRIMARY_CONTEMPORANEOUS_POLICY:
            raise ContemporaneousPolicyError(
                f"Primary contemporaneous policy is frozen to {PRIMARY_CONTEMPORANEOUS_POLICY!r}"
            )
        if self.forecast_horizon != PRIMARY_FORECAST_HORIZON:
            raise ContemporaneousPolicyError("Primary forecast horizon is frozen to h=1")
        if self.schema_version != PRIMARY_CONTEMPORANEOUS_POLICY_SCHEMA_VERSION:
            raise ContemporaneousPolicyError(
                f"schema_version must be {PRIMARY_CONTEMPORANEOUS_POLICY_SCHEMA_VERSION}"
            )
        if any(link.lag <= 0 for link in self.forecast_candidates):
            raise ContemporaneousPolicyError(
                "forecast_candidates must contain only strictly lagged links"
            )
        if any(link.lag != 0 for link in self.excluded_contemporaneous):
            raise ContemporaneousPolicyError(
                "excluded_contemporaneous must contain only tau=0 links"
            )


def apply_primary_contemporaneous_policy(
    links: Sequence[SignificantPCMCIPlusLink],
) -> ContemporaneousLinkDisposition:
    """Preserve tau=0 links for audit while excluding them from direct h=1 inputs.

    No policy, horizon, or data-dependent override is accepted from the caller. D-08
    only partitions D-07 records; D-09 remains responsible for validating the exact
    lagged-link marks and the final forecasting link set.
    """

    normalized = tuple(links)
    if any(not isinstance(link, SignificantPCMCIPlusLink) for link in normalized):
        raise TypeError("all links must be SignificantPCMCIPlusLink instances")
    if any(link.lag < 0 for link in normalized):
        raise ContemporaneousPolicyError("PCMCI+ lag must be non-negative")

    forecast_candidates = tuple(link for link in normalized if link.lag > 0)
    excluded_contemporaneous = tuple(link for link in normalized if link.lag == 0)

    return ContemporaneousLinkDisposition(
        forecast_candidates=forecast_candidates,
        excluded_contemporaneous=excluded_contemporaneous,
    )
