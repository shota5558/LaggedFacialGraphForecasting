from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting import aligned_indices


def test_lag_one_horizon_one_alignment() -> None:
    aligned = aligned_indices(6, lag=1, horizon=1)

    assert np.array_equal(aligned.source_index, np.array([0, 1, 2, 3, 4]))
    assert np.array_equal(aligned.forecast_origin, np.array([0, 1, 2, 3, 4]))
    assert np.array_equal(aligned.target_index, np.array([1, 2, 3, 4, 5]))
    assert aligned.lag == 1
    assert aligned.horizon == 1


def test_lag_two_horizon_one_alignment() -> None:
    aligned = aligned_indices(6, lag=2, horizon=1)

    assert np.array_equal(aligned.source_index, np.array([0, 1, 2, 3]))
    assert np.array_equal(aligned.forecast_origin, np.array([1, 2, 3, 4]))
    assert np.array_equal(aligned.target_index, np.array([2, 3, 4, 5]))


def test_source_never_occurs_after_forecast_origin() -> None:
    for lag in (1, 2, 3, 4):
        aligned = aligned_indices(10, lag=lag, horizon=1)
        assert np.all(aligned.source_index <= aligned.forecast_origin)
        assert np.all(aligned.target_index == aligned.forecast_origin + 1)
        assert np.all(aligned.source_index == aligned.target_index - lag)


def test_rejects_lag_smaller_than_horizon() -> None:
    with pytest.raises(ValueError, match="lag >= horizon"):
        aligned_indices(8, lag=0, horizon=1)


def test_rejects_non_primary_horizon() -> None:
    with pytest.raises(ValueError, match="frozen to h=1"):
        aligned_indices(8, lag=2, horizon=2)


def test_rejects_lag_with_no_aligned_samples() -> None:
    with pytest.raises(ValueError, match="leaves no aligned samples"):
        aligned_indices(5, lag=5, horizon=1)


@pytest.mark.parametrize("bad_lag", [1.5, True])
def test_rejects_non_integer_lag(bad_lag: object) -> None:
    with pytest.raises(ValueError, match="lag must be an integer"):
        aligned_indices(8, lag=bad_lag, horizon=1)  # type: ignore[arg-type]
