"""Frozen Tigramite PCMCI+ object construction for Primary discovery (D-03)."""

from __future__ import annotations

from tigramite.pcmci import PCMCI

from .parcorr_factory import make_primary_parcorr
from .tigramite_adapter import TigramiteDataFrameBundle


class PCMCIPlusConfigError(ValueError):
    """Raised when the frozen Primary PCMCI+ construction contract is violated."""


def make_primary_pcmci_plus(bundle: TigramiteDataFrameBundle) -> PCMCI:
    """Construct Tigramite's PCMCI object for the frozen Primary discovery path.

    Tigramite exposes PCMCI+ through ``PCMCI.run_pcmciplus``.  This task only binds
    the D-01 subject-isolated DataFrame to the D-02 frozen ParCorr instance.  It does
    not call discovery and deliberately does not choose ``tau_max`` or ``pc_alpha``;
    those are handled by D-04 and D-05 before D-06 invokes ``run_pcmciplus``.
    """

    if not isinstance(bundle, TigramiteDataFrameBundle):
        raise PCMCIPlusConfigError("bundle must be a TigramiteDataFrameBundle")

    return PCMCI(
        dataframe=bundle.dataframe,
        cond_ind_test=make_primary_parcorr(),
        verbosity=0,
    )
