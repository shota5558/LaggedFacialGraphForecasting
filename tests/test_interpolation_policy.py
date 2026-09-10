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


def test_none_policy_preserves_values_and_validity_exactly() -> None:
    values, valid_mask, missingness = _missingness_fixture()

    result = apply_interpolation_policy(
        missingness,
        policy=InterpolationPolicy(method="none"),
    )

    assert np.array_equal(result.values, values, equal_nan=True)
    assert np.array_equal(result.valid_mask, valid_mask)
    assert not result.interpolated_mask.any()
    assert result.policy.method == "none"
    assert result.policy.schema_version == 1


def test_invalid_values_remain_invalid_and_unfilled() -> None:
    _, _, missingness = _missingness_fixture()

    result = apply_interpolation_policy(
        missingness,
        policy=InterpolationPolicy(method="none"),
    )

    assert np.isnan(result.values[1, 0, 0])
    assert not result.valid_mask[1, 0, 0]
    assert not result.interpolated_mask[1, 0, 0]


def test_unfrozen_interpolation_methods_are_rejected() -> None:
    for method in ("linear", "nearest", "spline"):
        with pytest.raises(InterpolationPolicyError, match="only method='none'"):
            InterpolationPolicy(method=method)


def test_policy_schema_version_is_explicit() -> None:
    with pytest.raises(InterpolationPolicyError, match="schema_version must be 1"):
        InterpolationPolicy(method="none", schema_version=2)


def test_policy_must_be_explicit_object() -> None:
    _, _, missingness = _missingness_fixture()

    with pytest.raises(InterpolationPolicyError, match="InterpolationPolicy"):
        apply_interpolation_policy(missingness, policy="none")  # type: ignore[arg-type]


def test_outputs_are_immutable_and_inputs_unchanged() -> None:
    values, valid_mask, missingness = _missingness_fixture()
    values_before = values.copy()
    mask_before = valid_mask.copy()

    result = apply_interpolation_policy(
        missingness,
        policy=InterpolationPolicy(method="none"),
    )

    assert np.array_equal(values, values_before, equal_nan=True)
    assert np.array_equal(valid_mask, mask_before)
    assert not result.values.flags.writeable
    assert not result.valid_mask.flags.writeable
    assert not result.interpolated_mask.flags.writeable
