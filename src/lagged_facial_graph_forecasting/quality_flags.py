"""Extractor-neutral quality-flag contract for facial preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class QualityFlagError(ValueError):
    """Raised when externally supplied quality flags violate the contract."""


@dataclass(frozen=True, slots=True)
class FrameQualityFlags:
    """Immutable frame-level quality decisions with explicit provenance.

    A-11 does not estimate landmark quality or choose thresholds. ``valid`` must be
    supplied by an upstream extractor/quality procedure whose rule is frozen outside
    this primitive. Missingness is handled separately by A-12/A-13.
    """

    valid: np.ndarray
    source: str
    method: str = "externally_supplied_boolean_quality"

    def __post_init__(self) -> None:
        valid = np.asarray(self.valid)
        if valid.ndim != 1 or valid.size < 1:
            raise QualityFlagError("valid must be a non-empty one-dimensional array")
        if valid.dtype != np.bool_:
            raise QualityFlagError("valid must have boolean dtype")
        if not isinstance(self.source, str) or not self.source.strip():
            raise QualityFlagError("source must be a non-empty provenance string")
        if not isinstance(self.method, str) or not self.method.strip():
            raise QualityFlagError("method must be a non-empty string")

        immutable = np.array(valid, copy=True)
        immutable.setflags(write=False)
        object.__setattr__(self, "valid", immutable)
        object.__setattr__(self, "source", self.source.strip())
        object.__setattr__(self, "method", self.method.strip())


def broadcast_frame_quality(
    flags: FrameQualityFlags,
    *,
    region_count: int,
    dimension_count: int,
) -> np.ndarray:
    """Broadcast frame quality to the canonical ``(T, K, D)`` mask shape.

    This only broadcasts an already-frozen quality decision. It does not inspect
    feature values, infer missingness, impute observations, or select a threshold.
    """

    for value, name in (
        (region_count, "region_count"),
        (dimension_count, "dimension_count"),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise QualityFlagError(f"{name} must be a positive integer")

    mask = np.broadcast_to(
        flags.valid[:, np.newaxis, np.newaxis],
        (flags.valid.size, region_count, dimension_count),
    ).copy()
    mask.setflags(write=False)
    return mask
