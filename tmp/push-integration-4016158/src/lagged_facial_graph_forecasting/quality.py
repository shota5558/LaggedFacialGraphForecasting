"""Deterministic quality-flag construction for facial time-series preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class QualityFlagError(ValueError):
    """Raised when quality information violates the explicit preprocessing contract."""


@dataclass(frozen=True, slots=True)
class QualityFlags:
    """Immutable elementwise validity mask and frame-quality provenance."""

    valid_mask: np.ndarray
    finite_mask: np.ndarray
    frame_quality_pass: np.ndarray
    minimum_frame_quality: float | None
    method: str

    def __post_init__(self) -> None:
        valid = np.array(self.valid_mask, dtype=np.bool_, copy=True)
        finite = np.array(self.finite_mask, dtype=np.bool_, copy=True)
        frame_pass = np.array(self.frame_quality_pass, dtype=np.bool_, copy=True)
        valid.setflags(write=False)
        finite.setflags(write=False)
        frame_pass.setflags(write=False)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "finite_mask", finite)
        object.__setattr__(self, "frame_quality_pass", frame_pass)


def build_quality_flags(
    values: np.ndarray,
    *,
    frame_quality: np.ndarray | None = None,
    minimum_frame_quality: float | None = None,
) -> QualityFlags:
    """Construct quality flags without repairing, dropping, or selecting samples.

    Numeric non-finite elements are always marked invalid. An optional externally
    supplied frame-level quality score may additionally invalidate a whole frame,
    but its threshold is mandatory and explicit; this function never estimates a
    threshold from data or predictive performance. Non-finite external quality
    scores fail the threshold and therefore mark the corresponding frame invalid.

    The returned ``valid_mask`` has exactly the same ``(T, K, D)`` shape as the
    input and can be passed into the frozen ``FaceTimeSeries`` contract. This task
    only *flags* observations. Missing-frame handling and interpolation remain the
    responsibilities of A-12 and A-13.
    """

    array = np.asarray(values)
    if array.ndim != 3:
        raise QualityFlagError(
            "values must have shape (frames, points_or_regions, dimensions)"
        )
    if any(size < 1 for size in array.shape):
        raise QualityFlagError("values axes must be non-empty")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise QualityFlagError("values must be numeric")

    finite_mask = np.isfinite(array)
    frame_count = int(array.shape[0])

    if frame_quality is None:
        if minimum_frame_quality is not None:
            raise QualityFlagError(
                "minimum_frame_quality requires explicit frame_quality scores"
            )
        frame_pass = np.ones(frame_count, dtype=np.bool_)
        threshold = None
        method = "finite_values_only"
    else:
        scores = np.asarray(frame_quality)
        if scores.ndim != 1 or scores.shape[0] != frame_count:
            raise QualityFlagError(
                f"frame_quality must have shape ({frame_count},)"
            )
        if not np.issubdtype(scores.dtype, np.number) or np.issubdtype(
            scores.dtype, np.bool_
        ):
            raise QualityFlagError("frame_quality must be numeric")
        if minimum_frame_quality is None:
            raise QualityFlagError(
                "frame_quality requires an explicit minimum_frame_quality"
            )
        threshold = float(minimum_frame_quality)
        if not np.isfinite(threshold):
            raise QualityFlagError("minimum_frame_quality must be finite")
        frame_pass = np.isfinite(scores) & (scores >= threshold)
        method = "finite_values_and_explicit_frame_quality_threshold"

    valid_mask = finite_mask & frame_pass[:, np.newaxis, np.newaxis]
    return QualityFlags(
        valid_mask=valid_mask,
        finite_mask=finite_mask,
        frame_quality_pass=frame_pass,
        minimum_frame_quality=threshold,
        method=method,
    )
