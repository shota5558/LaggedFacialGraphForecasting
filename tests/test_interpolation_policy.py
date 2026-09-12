from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting.interpolation_policy import (
    InterpolationPolicy,
    InterpolationPolicyError,
    apply_interpolation_policy,
)
from lagged_facial_graph_forecasting.missingness import handle_missing_frames


def _missingness_fixture():
    values = np.array(
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [[np.nan, 5.0], [6.0, 7.0]],
            [[8.0, 9.0], [10.0, 11.0]],
        ]
    )
    valid_mask = np.isfinite(values)
    return values, valid_mask, handle_missing_frames(values, valid_mask=valid_mask)


def test_none_policy_preserves_values_validity_and_missingness() -> None:
    values, valid_mask, missingness = _missingness_fixture()

    result = apply_interpolation_policy(
        missingness,
        policy=InterpolationPolicy(method="none"),
    )

    assert np.array_equal(result.values, values, equal_nan=True)
    assert np.array_equal(result.valid_mask, valid_mask)
    assert not result.interpolated_mask.any()
    assert np.isnan(result.values[1, 0, 0])
    assert result.policy.method == "none"


def test_unfrozen_interpolation_method_is_rejected() -> None:
    with pytest.raises(InterpolationPolicyError, match="only method='none'"):
        InterpolationPolicy(method="linear")
