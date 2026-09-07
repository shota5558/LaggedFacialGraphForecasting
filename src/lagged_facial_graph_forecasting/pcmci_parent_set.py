"""Canonical ParentSet conversion for Primary PCMCI+ forecasting (D-10).

D-10 is the boundary where validated Tigramite-specific lagged link records become
Tigramite-independent Core Contract objects. The Primary PCMCI forecasting
condition always retains the same target-region Self-history block as the Self
baseline, so same-region discovery links are not duplicated as selected parents.
They remain available in the upstream D-07/D-08 discovery evidence.
"""

from __future__ import annotations

from typing import Sequence

from .core_contracts import ParentLink, ParentSet
from .pcmci_lagged_filter import PRIMARY_LAGGED_LINK_MARK, PRIMARY_LAG_MIN
from .pcmci_link_extraction import SignificantPCMCIPlusLink
from .pcmci_tau_max import PRIMARY_TAU_MAX


class ParentSetConversionError(ValueError):
    """Raised when a raw link bypasses the frozen D-09 lagged-link contract."""


def convert_primary_lagged_links_to_parent_set(
    links: Sequence[SignificantPCMCIPlusLink],
    *,
    outer_fold: int,
    target_region: str,
) -> ParentSet:
    """Convert validated cross-region lagged links for one target region.

    The function accepts the D-09 output domain only: ``1 <= tau <= 10`` and
    Tigramite lagged mark ``'-->'``. Links for other targets are ignored, and
    same-region links are omitted because the fixed Self-history block already
    supplies target-region history in the PCMCI DesignMatrix. Exact source and
    target component identity is preserved in every ``ParentLink``.

    Parent ordering is canonicalized through ``ParentLink`` ordering so the same
    scientific link set yields the same ParentSet independent of caller ordering.
    No p-value re-thresholding, region projection, or Tigramite API object is
    retained past this boundary.
    """

    normalized = tuple(links)
    if any(not isinstance(link, SignificantPCMCIPlusLink) for link in normalized):
        raise TypeError("all links must be SignificantPCMCIPlusLink instances")

    parents: list[ParentLink] = []
    for link in normalized:
        if link.lag < PRIMARY_LAG_MIN or link.lag > PRIMARY_TAU_MAX:
            raise ParentSetConversionError(
                f"Primary lag must satisfy {PRIMARY_LAG_MIN} <= tau <= {PRIMARY_TAU_MAX}; "
                f"got {link.lag}"
            )
        if link.link_mark != PRIMARY_LAGGED_LINK_MARK:
            raise ParentSetConversionError(
                "D-10 accepts only D-09 validated Tigramite lagged links with '-->'; "
                f"got {link.link_mark!r}"
            )
        if link.target_region != target_region:
            continue
        if link.source_region == target_region:
            continue

        parents.append(
            ParentLink(
                source_region=link.source_region,
                lag=link.lag,
                source_dimension=link.source_dimension,
                target_dimension=link.target_dimension,
            )
        )

    return ParentSet(
        outer_fold=outer_fold,
        target_region=target_region,
        parents=tuple(sorted(parents)),
        discovery_method="pcmci_plus",
    )
