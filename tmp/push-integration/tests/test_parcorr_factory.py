from __future__ import annotations

from tigramite.independence_tests.parcorr import ParCorr

from lagged_facial_graph_forecasting.parcorr_factory import (
    PRIMARY_PARCORR_MASK_TYPE,
    PRIMARY_PARCORR_SIGNIFICANCE,
    make_primary_parcorr,
)


def test_primary_parcorr_factory_returns_frozen_tigramite_configuration() -> None:
    test = make_primary_parcorr()

    assert isinstance(test, ParCorr)
    assert test.measure == "par_corr"
    assert test.significance == PRIMARY_PARCORR_SIGNIFICANCE == "analytic"
    assert test.mask_type == PRIMARY_PARCORR_MASK_TYPE == "xyz"
    assert test.recycle_residuals is False
    assert test.confidence is None
    assert test.verbosity == 0


def test_primary_parcorr_factory_returns_fresh_instances() -> None:
    first = make_primary_parcorr()
    second = make_primary_parcorr()

    assert first is not second
    assert first.dataframe is None
    assert second.dataframe is None
