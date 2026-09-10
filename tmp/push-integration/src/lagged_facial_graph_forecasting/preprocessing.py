"""Primary real-data preprocessing primitives built one frozen task at a time."""

from __future__ import annotations

import numpy as np


class PreprocessingValidationError(ValueError):
    """Raised when raw preprocessing input violates an explicit scientific contract."""


def validate_timestamps(
    timestamps: np.ndarray,
    *,
    expected_frame_count: int | None = None,
) -> np.ndarray:
    """Validate observed frame timestamps without repairing or resampling them.

    Timestamps must be one-dimensional, numeric, finite, non-empty, and strictly
    increasing. Irregular positive sampling intervals are allowed: A-01 validates
    temporal ordering only and does not introduce a resampling/interpolation policy.
    When supplied, ``expected_frame_count`` verifies alignment with the landmark
    frame axis without inspecting landmark shape semantics.
    """

    values = np.asarray(timestamps)
    if values.ndim != 1:
        raise PreprocessingValidationError("timestamps must be one-dimensional")
    if values.size == 0:
        raise PreprocessingValidationError("timestamps must not be empty")
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
        values.dtype, np.bool_
    ):
        raise PreprocessingValidationError("timestamps must be numeric")
    if not np.all(np.isfinite(values)):
        raise PreprocessingValidationError("timestamps must be finite")
    if values.size > 1 and not np.all(np.diff(values) > 0):
        raise PreprocessingValidationError("timestamps must be strictly increasing")

    if expected_frame_count is not None:
        if (
            not isinstance(expected_frame_count, int)
            or isinstance(expected_frame_count, bool)
            or expected_frame_count < 1
        ):
            raise PreprocessingValidationError(
                "expected_frame_count must be a positive integer or None"
            )
        if values.size != expected_frame_count:
            raise PreprocessingValidationError(
                "timestamp count does not match expected_frame_count"
            )

    validated = np.array(values, copy=True)
    validated.setflags(write=False)
    return validated


def _optional_positive_integer(value: int | None, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise PreprocessingValidationError(
            f"{field_name} must be a positive integer or None"
        )
    return value


def validate_landmark_shape(
    landmarks: np.ndarray,
    *,
    expected_frame_count: int | None = None,
    expected_landmark_count: int | None = None,
    expected_coordinate_dim: int | None = None,
) -> np.ndarray:
    """Validate the extractor-neutral raw landmark tensor shape ``(T, P, C)``.

    A-02 fixes only the axis semantics and optional explicit expected sizes. It does
    not hard-code a landmark extractor, point count, or 2D/3D coordinate convention.
    Numeric NaN values are intentionally allowed because missing-frame/missing-point
    policy is owned by A-12/A-13 rather than silently handled here.
    """

    values = np.asarray(landmarks)
    if values.ndim != 3:
        raise PreprocessingValidationError(
            "landmarks must have shape (frames, landmarks, coordinates)"
        )
    if any(size < 1 for size in values.shape):
        raise PreprocessingValidationError("landmark tensor axes must be non-empty")
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
        values.dtype, np.bool_
    ):
        raise PreprocessingValidationError("landmarks must contain numeric values")

    expected_frame_count = _optional_positive_integer(
        expected_frame_count, "expected_frame_count"
    )
    expected_landmark_count = _optional_positive_integer(
        expected_landmark_count, "expected_landmark_count"
    )
    expected_coordinate_dim = _optional_positive_integer(
        expected_coordinate_dim, "expected_coordinate_dim"
    )

    frame_count, landmark_count, coordinate_dim = values.shape
    if expected_frame_count is not None and frame_count != expected_frame_count:
        raise PreprocessingValidationError(
            "landmark frame count does not match expected_frame_count"
        )
    if expected_landmark_count is not None and landmark_count != expected_landmark_count:
        raise PreprocessingValidationError(
            "landmark point count does not match expected_landmark_count"
        )
    if expected_coordinate_dim is not None and coordinate_dim != expected_coordinate_dim:
        raise PreprocessingValidationError(
            "landmark coordinate dimension does not match expected_coordinate_dim"
        )

    validated = np.array(values, copy=True)
    validated.setflags(write=False)
    return validated
