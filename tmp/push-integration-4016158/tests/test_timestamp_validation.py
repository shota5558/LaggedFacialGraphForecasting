from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting.preprocessing import (
    PreprocessingValidationError,
    validate_timestamps,
)


def test_accepts_strictly_increasing_irregular_observed_timestamps() -> None:
    timestamps = np.array([0.0, 0.031, 0.067, 0.101, 0.142], dtype=float)

    validated = validate_timestamps(timestamps, expected_frame_count=5)

    assert np.array_equal(validated, timestamps)
    assert not validated.flags.writeable


def test_rejects_duplicate_or_decreasing_timestamps() -> None:
    with pytest.raises(PreprocessingValidationError, match="strictly increasing"):
        validate_timestamps(np.array([0.0, 0.03, 0.03, 0.06]))
    with pytest.raises(PreprocessingValidationError, match="strictly increasing"):
        validate_timestamps(np.array([0.0, 0.04, 0.02, 0.06]))


def test_rejects_nonfinite_empty_nonnumeric_and_nonvector_timestamps() -> None:
    with pytest.raises(PreprocessingValidationError, match="finite"):
        validate_timestamps(np.array([0.0, np.nan]))
    with pytest.raises(PreprocessingValidationError, match="finite"):
        validate_timestamps(np.array([0.0, np.inf]))
    with pytest.raises(PreprocessingValidationError, match="must not be empty"):
        validate_timestamps(np.array([], dtype=float))
    with pytest.raises(PreprocessingValidationError, match="numeric"):
        validate_timestamps(np.array(["0", "1"]))
    with pytest.raises(PreprocessingValidationError, match="one-dimensional"):
        validate_timestamps(np.array([[0.0, 1.0]]))


def test_rejects_timestamp_frame_count_mismatch() -> None:
    with pytest.raises(PreprocessingValidationError, match="does not match"):
        validate_timestamps(
            np.array([0.0, 0.1, 0.2]),
            expected_frame_count=4,
        )


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_rejects_invalid_expected_frame_count(invalid: object) -> None:
    with pytest.raises(PreprocessingValidationError, match="positive integer"):
        validate_timestamps(
            np.array([0.0, 0.1]),
            expected_frame_count=invalid,  # type: ignore[arg-type]
        )
