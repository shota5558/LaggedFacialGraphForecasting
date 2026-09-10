"""Frozen Tigramite ParCorr factory for Primary discovery (D-02)."""

from __future__ import annotations

from tigramite.independence_tests.parcorr import ParCorr


PRIMARY_PARCORR_SIGNIFICANCE = "analytic"
PRIMARY_PARCORR_MASK_TYPE = "xyz"


def make_primary_parcorr() -> ParCorr:
    """Return the frozen Primary conditional-independence test.

    The scientific freeze fixes Primary discovery to Tigramite ParCorr.  We keep
    analytic significance explicit rather than relying on an upstream default.
    ``mask_type='xyz'`` is also mandatory: D-01 translates canonical invalid
    observations into Tigramite's mask, and every invalid sample must be excluded
    regardless of whether the variable participates as X, Y, or conditioning Z.

    This function only configures the OSS implementation; it contains no custom
    conditional-independence algorithm and performs no data-dependent fitting.
    """

    return ParCorr(
        significance=PRIMARY_PARCORR_SIGNIFICANCE,
        mask_type=PRIMARY_PARCORR_MASK_TYPE,
        recycle_residuals=False,
        confidence=None,
        verbosity=0,
    )
