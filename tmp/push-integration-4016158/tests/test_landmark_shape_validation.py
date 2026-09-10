from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting.preprocessing import (
    PreprocessingValidationError,
    validate_landmark_shape,
)


def test_accepts_extractor_neutral_three_axis_landmark_tensor() -> None:
    landmarks = np.arange(5 * 7 * 2, dtype=float).reshape(5, 7, 2)

    validated = validate_landmark_shape(landmarks)

    assert np.array_equal(validated, landmarks)
    assert validated.shape == (5, 7, 2)
    assert not validated.flags.writeable


def test_optional_explicit_shape_expectations_are_enforced() -> None:
    landmarks = np.zeros((4, 8, 3), dtype=float)

    validated = validate_landmark_shape(
        landmarks,
        expected_frame_count=4,
        expected_landmark_count=8,
        expected_coordinate_dim=3,
    )
    assert validated.shape == (4, 8, 3)

    with pytest.raises(PreprocessingValidationError, match="frame count"):
        validate_landmark_shape(landmarks, expected_frame_count=5)
    with pytest.raises(PreprocessingValidationError, match="point count"):
        validate_landmark_shape(landmarks, expected_landmark_count=9)
    with pytest.raises(PreprocessingValidationError, match="coordinate dimension"):
        validate_landmark_shape(landmarks, expected_coordinate_dim=2)


def test_rejects_wrong_rank_empty_axes_and_nonnumeric_values() -> None:
    with pytest.raises(PreprocessingValidationError, match="shape"):
        validate_landmark_shape(np.zeros((4, 8), dtype=float))
    with pytest.raises(PreprocessingValidationError, match="non-empty"):
        validate_landmark_shape(np.empty((0, 8, 2), dtype=float))
    with pytest.raises(PreprocessingValidationError, match="numeric"):
        validate_landmark_shape(np.full((2, 3, 2), "x"))


def test_nan_is_preserved_for_later_missingness_policy() -> None:
    landmarks = np.zeros((3, 4, 2), dtype=float)
    landmarks[1, 2, 0] = np.nan

    validated = validate_landmark_shape(landmarks)

    assert np.isnan(validated[1, 2, 0])


@pytest.mark.parametrize("field", ["frame", "landmark", "coordinate"])
@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_rejects_invalid_explicit_shape_expectations(field: str, invalid: object) -> None:
    kwargs = {
        "expected_frame_count": 3,
        "expected_landmark_count": 4,
        "expected_coordinate_dim": 2,
    }
    kwargs[f"expected_{field}_count" if field != "coordinate" else "expected_coordinate_dim"] = invalid
    if field == "landmark":
        kwargs.pop("expected_landmark_count")
        kwargs["expected_landmark_count"] = invalid
    if field == "frame":
        kwargs.pop("expected_frame_count")
        kwargs["expected_frame_count"] = invalid

    with pytest.raises(PreprocessingValidationError, match="positive integer"):
        validate_landmark_shape(
            np.zeros((3, 4, 2), dtype=float),
            **kwargs,  # type: ignore[arg-type]
        )
