from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting.missingness import (
    MissingnessError,
    handle_missing_frames,
)


def test_partial_and_fully_missing_frames_are_flagged_without_repair() -> None:
    values = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, np.nan], [7.0, 8.0]],
            [[np.nan, np.nan], [np.nan, np.nan]],
        ]
    )
    valid = np.isfinite(values)

    handled = handle_missing_frames(values, valid_mask=valid)

    assert np.array_equal(handled.frame_has_missing, np.array([False, True, True]))
    assert np.array_equal(handled.frame_fully_missing, np.array([False, False, True]))
    assert np.array_equal(handled.valid_mask, valid)
    assert np.array_equal(handled.values, values, equal_nan=True)
    assert handled.method == "preserve_observed_frames_and_flag_missing"


def test_finite_low_quality_values_may_remain_explicitly_invalid() -> None:
    values = np.ones((2, 1, 2), dtype=float)
    valid = np.array([[[True, False]], [[True, True]]])

    handled = handle_missing_frames(values, valid_mask=valid)

    assert handled.frame_has_missing.tolist() == [True, False]
    assert handled.frame_fully_missing.tolist() == [False, False]
    assert not handled.valid_mask[0, 0, 1]
    assert handled.values[0, 0, 1] == 1.0


def test_nonfinite_value_cannot_be_marked_valid() -> None:
    values = np.array([[[1.0, np.nan]]])
    valid = np.array([[[True, True]]])

    with pytest.raises(MissingnessError, match="non-finite values cannot be marked valid"):
        handle_missing_frames(values, valid_mask=valid)


def test_mask_shape_and_dtype_must_match_contract() -> None:
    values = np.ones((2, 1, 2), dtype=float)

    with pytest.raises(MissingnessError, match="same shape"):
        handle_missing_frames(values, valid_mask=np.ones((2, 1), dtype=bool))
    with pytest.raises(MissingnessError, match="boolean dtype"):
        handle_missing_frames(values, valid_mask=np.ones(values.shape, dtype=int))


def test_missing_frame_handling_rejects_invalid_value_tensor() -> None:
    with pytest.raises(MissingnessError, match="shape"):
        handle_missing_frames(
            np.ones((2, 2), dtype=float),
            valid_mask=np.ones((2, 2), dtype=bool),
        )
    with pytest.raises(MissingnessError, match="numeric"):
        handle_missing_frames(
            np.array([[['x']]]),
            valid_mask=np.ones((1, 1, 1), dtype=bool),
        )


def test_outputs_are_immutable_and_inputs_are_unchanged() -> None:
    values = np.array([[[1.0]], [[np.nan]]])
    valid = np.isfinite(values)
    values_before = values.copy()
    valid_before = valid.copy()

    handled = handle_missing_frames(values, valid_mask=valid)

    assert np.array_equal(values, values_before, equal_nan=True)
    assert np.array_equal(valid, valid_before)
    assert not handled.values.flags.writeable
    assert not handled.valid_mask.flags.writeable
    assert not handled.frame_has_missing.flags.writeable
    assert not handled.frame_fully_missing.flags.writeable
