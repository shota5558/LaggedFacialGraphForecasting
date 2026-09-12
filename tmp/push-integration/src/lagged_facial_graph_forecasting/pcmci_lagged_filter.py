"""Primary lagged-link validation/filter boundary (D-09).

For PCMCI+ in Tigramite, a retained causal link at tau>0 is encoded as ``-->``.
After D-08 has removed tau=0 from forecasting candidates, D-09 fail-closes on
any lag or mark outside the frozen Primary direct-h=1 domain rather than silently
coercing an unexpected graph state into a forecasting parent.
"""

from __future__ import annotations

from .pcmci_contemporaneous import ContemporaneousLinkDisposition
from .pcmci_link_extraction import SignificantPCMCIPlusLink
from .pcmci_tau_max import PRIMARY_TAU_MAX


PRIMARY_LAGGED_LINK_MARK = "-->"
PRIMARY_LAG_MIN = 1


class LaggedLinkFilterError(ValueError):
    """Raised when a D-08 forecasting candidate violates D-09 semantics."""


def filter_primary_lagged_links(
    disposition: ContemporaneousLinkDisposition,
) -> tuple[SignificantPCMCIPlusLink, ...]:
    """Return canonical Primary lagged links after strict Tigramite validation.

    This function does not re-threshold p-values, deduplicate links, aggregate
    region dimensions, or construct ParentSet. Those concerns belong to earlier or
    later tasks. It only accepts the Tigramite PCMCI+ lagged representation that is
    valid for Primary h=1: ``1 <= tau <= 10`` and ``link_mark == '-->'``.
    """

    if not isinstance(disposition, ContemporaneousLinkDisposition):
        raise TypeError("disposition must be ContemporaneousLinkDisposition")

    accepted: list[SignificantPCMCIPlusLink] = []
    for link in disposition.forecast_candidates:
        if not isinstance(link, SignificantPCMCIPlusLink):
            raise TypeError("forecast candidates must be SignificantPCMCIPlusLink instances")
        if link.lag < PRIMARY_LAG_MIN or link.lag > PRIMARY_TAU_MAX:
            raise LaggedLinkFilterError(
                f"Primary lag must satisfy {PRIMARY_LAG_MIN} <= tau <= {PRIMARY_TAU_MAX}; "
                f"got {link.lag}"
            )
        if link.link_mark != PRIMARY_LAGGED_LINK_MARK:
            raise LaggedLinkFilterError(
                "Tigramite PCMCI+ lagged Primary link must use '-->'; "
                f"got {link.link_mark!r} at tau={link.lag}"
            )
        accepted.append(link)

    return tuple(accepted)
