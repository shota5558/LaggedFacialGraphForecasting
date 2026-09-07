"""Timestamp validation for extracted facial-landmark sequences (A-01)."""

from __future__ import annotations

import numpy as np

from .landmark_io import RawLandmarkSequence


class TimestampValidationError(ValueError):
    """Raised when landmark timestamps violate the A-01 temporal contract."""


def validate_timestamps(
    timestamps: np.ndarray | None,
    *,
    expected_frame_count: int | None = None,
) -> np.ndarray:
    """Return immutable float timestamps after strict temporal validation.

    Timestamps are required, one-dimensional, finite, and strictly increasing.
    When a frame count is supplied, one timestamp per frame is required.  The
    function performs no interpolation, resampling, or rate estimation.
    """

    if timestamps is None:
        raise TimestampValidationError("timestamps are required")
    array = np.asarray(timestamps)
    if array.ndim != 1:
        raise TimestampValidationError("timestamps must be a one-dimensional array")
    if array.size == 0:
        raise TimestampValidationError("timestamps must not be empty")
    if not np.issubdtype(array.dtype, np.number):
        raise TimestampValidationError("timestamps must be numeric")

    values = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise TimestampValidationError("timestamps must be finite")
    if values.size > 1 and not np.all(np.diff(values) > 0.0):
        raise TimestampValidationError("timestamps must be strictly increasing")

    if expected_frame_count is not None:
        if isinstance(expected_frame_count, bool) or not isinstance(
            expected_frame_count, (int, np.integer)
        ):
            raise TimestampValidationError("expected_frame_count must be an integer")
        if int(expected_frame_count) < 1:
            raise TimestampValidationError("expected_frame_count must be >= 1")
        if values.size != int(expected_frame_count):
            raise TimestampValidationError(
                "timestamp count does not match expected frame count"
            )

    validated = np.array(values, copy=True)
    validated.setflags(write=False)
    return validated


def validate_landmark_timestamps(sequence: RawLandmarkSequence) -> np.ndarray:
    """Validate timestamps against the raw sequence frame axis when available."""

    landmarks = np.asarray(sequence.landmarks)
    if landmarks.ndim == 0:
        # Landmark rank is owned by A-02; avoid inventing frame semantics here.
        raise TimestampValidationError(
            "cannot validate timestamp count before landmark frame axis exists"
        )
    return validate_timestamps(
        sequence.timestamps,
        expected_frame_count=int(landmarks.shape[0]),
    )
