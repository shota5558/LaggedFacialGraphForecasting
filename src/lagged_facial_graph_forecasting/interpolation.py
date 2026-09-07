"""Forecast-safe Primary interpolation policy (A-13)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .missingness import MissingFrameHandling


class InterpolationPolicyError(ValueError):
    """Raised when an interpolation policy violates the Primary contract."""


@dataclass(frozen=True, slots=True)
class InterpolationPolicy:
    """Frozen Primary policy.

    Primary deliberately supports only ``none``. Bidirectional interpolation can use
    observations after a forecast origin to construct a predictor at that origin,
    while causal fill methods would introduce additional scientific choices that are
    not part of the frozen Primary specification.
    """

    method: Literal["none"] = "none"

    def __post_init__(self) -> None:
        if self.method != "none":
            raise InterpolationPolicyError(
                "Primary interpolation policy is frozen to 'none'; other methods require an explicit scientific migration"
            )


@dataclass(frozen=True, slots=True)
class InterpolationResult:
    """Immutable pass-through result with explicit interpolation provenance."""

    values: np.ndarray
    valid_mask: np.ndarray
    interpolated_mask: np.ndarray
    method: str = "none"
    forecast_safe: bool = True

    def __post_init__(self) -> None:
        values = np.array(self.values, copy=True)
        valid = np.array(self.valid_mask, dtype=np.bool_, copy=True)
        interpolated = np.array(self.interpolated_mask, dtype=np.bool_, copy=True)
        if values.ndim != 3:
            raise InterpolationPolicyError(
                "values must have shape (frames, points_or_regions, dimensions)"
            )
        if valid.shape != values.shape or interpolated.shape != values.shape:
            raise InterpolationPolicyError(
                "valid_mask and interpolated_mask must match values shape"
            )
        for array in (values, valid, interpolated):
            array.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "interpolated_mask", interpolated)


def apply_interpolation_policy(
    handled: MissingFrameHandling,
    *,
    policy: InterpolationPolicy,
) -> InterpolationResult:
    """Apply the frozen Primary policy without filling any observation.

    A-12 values, observed time support, and validity are preserved exactly. No
    interpolation, extrapolation, smoothing, resampling, imputation, threshold
    fitting, population statistic, or outer-test-dependent operation is performed.
    ``interpolated_mask`` is therefore all false. Downstream design-matrix alignment
    must continue to exclude invalid inputs through the existing validity contract.
    """

    values = np.asarray(handled.values)
    valid = np.asarray(handled.valid_mask)
    if values.ndim != 3 or valid.shape != values.shape or valid.dtype != np.bool_:
        raise InterpolationPolicyError(
            "handled values/valid_mask violate the A-12 contract"
        )
    if policy.method != "none":
        raise InterpolationPolicyError("unsupported Primary interpolation policy")

    return InterpolationResult(
        values=values,
        valid_mask=valid,
        interpolated_mask=np.zeros(values.shape, dtype=np.bool_),
    )
