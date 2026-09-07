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
