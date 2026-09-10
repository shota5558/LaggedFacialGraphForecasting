from __future__ import annotations


def test_tigramite_pcmciplus_and_parcorr_api_are_importable() -> None:
    """D-00 only verifies the OSS measurement surface; it does not run discovery."""

    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    assert callable(pp.DataFrame)
    assert callable(ParCorr)
    assert callable(PCMCI)
    assert callable(getattr(PCMCI, "run_pcmciplus", None))
