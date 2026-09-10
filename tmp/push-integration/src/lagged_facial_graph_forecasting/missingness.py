"""Missing-observation handling without interpolation for facial time series (A-12)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class MissingnessError(ValueError):
    """Raised when missingness metadata violates the explicit preprocessing contract."""


@dataclass(frozen=True, slots=True)
class MissingFrameHandling:
    """Immutable observed data plus explicit missingness provenance.

    A-12 deliberately preserves the observed time axis and values. Invalid elements
    are represented only through ``valid_mask`` and frame-level summaries; no row is
    dropped and no value is filled. Interpolation, if scientifically frozen, belongs
    to A-13.
    """

    values: np.ndarray
    valid_mask: np.ndarray
    frame_has_missing: np.ndarray
    frame_fully_missing: np.ndarray
    method: str = "preserve_observed_frames_and_flag_missing"

    def __post_init__(self) -> None:
        values = np.array(self.values, copy=True)
        valid = np.array(self.valid_mask, dtype=np.bool_, copy=True)
        has_missing = np.array(self.frame_has_missing, dtype=np.bool_, copy=True)
        fully_missing = np.array(self.frame_fully_missing, dtype=np.bool_, copy=True)
        for array in (values, valid, has_missing, fully_missing):
            array.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "frame_has_missing", has_missing)
        object.__setattr__(self, "frame_fully_missing", fully_missing)


def handle_missing_frames(
    values: np.ndarray,
    *,
    valid_mask: np.ndarray,
) -> MissingFrameHandling:
    """Preserve observed frames and make missingness explicit, without repair.

    ``values`` and ``valid_mask`` must share ``(T, K, D)`` shape. ``False`` mask
    entries may represent non-finite coordinates or externally failed quality flags.
    A finite low-quality value is therefore allowed to be invalid, but a non-finite
    numeric value may never be marked valid.

    This function does not infer unobserved timestamps from gaps because A-01 allows
    irregular positive sampling intervals and no nominal sampling cadence is frozen
    here. It handles missingness *within the observed frame sequence* only. No
    interpolation, extrapolation, smoothing, deletion, resampling, or fitted
    statistic is performed.
    """

    array = np.asarray(values)
    mask = np.asarray(valid_mask)
    if array.ndim != 3:
        raise MissingnessError(
            "values must have shape (frames, points_or_regions, dimensions)"
        )
    if any(size < 1 for size in array.shape):
        raise MissingnessError("values axes must be non-empty")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise MissingnessError("values must be numeric")
    if mask.shape != array.shape:
        raise MissingnessError(
            f"valid_mask must have the same shape as values; got {mask.shape} vs {array.shape}"
        )
    if mask.dtype != np.bool_:
        raise MissingnessError("valid_mask must have boolean dtype")
    if np.any(mask & ~np.isfinite(array)):
        raise MissingnessError("non-finite values cannot be marked valid")

    frame_has_missing = np.any(~mask, axis=(1, 2))
    frame_fully_missing = ~np.any(mask, axis=(1, 2))
    return MissingFrameHandling(
        values=array,
        valid_mask=mask,
        frame_has_missing=frame_has_missing,
        frame_fully_missing=frame_fully_missing,
    )
