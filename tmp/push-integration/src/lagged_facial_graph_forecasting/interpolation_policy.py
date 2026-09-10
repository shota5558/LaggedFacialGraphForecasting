"""Explicit interpolation policy for facial time-series preprocessing (A-13)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .missingness import MissingFrameHandling


class InterpolationPolicyError(ValueError):
    """Raised when an interpolation policy violates the frozen Primary contract."""


@dataclass(frozen=True, slots=True)
class InterpolationPolicy:
    """Immutable, explicitly selected interpolation policy.

    Primary currently supports only ``method='none'``. This is deliberate: the
    research plan permits invalid observations to be excluded/flagged but does not
    freeze an imputation rule. Adding an interpolation method later therefore
    requires an explicit scientific/config migration rather than silently creating
    synthetic facial motion.
    """

    method: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise InterpolationPolicyError("schema_version must be 1")
        if self.method != "none":
            raise InterpolationPolicyError(
                "Primary interpolation policy currently supports only method='none'"
            )


@dataclass(frozen=True, slots=True)
class InterpolationResult:
    """Immutable result of applying the explicit interpolation policy."""

    values: np.ndarray
    valid_mask: np.ndarray
    interpolated_mask: np.ndarray
    policy: InterpolationPolicy

    def __post_init__(self) -> None:
        values = np.array(self.values, copy=True)
        valid = np.array(self.valid_mask, dtype=np.bool_, copy=True)
        interpolated = np.array(self.interpolated_mask, dtype=np.bool_, copy=True)
        values.setflags(write=False)
        valid.setflags(write=False)
        interpolated.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "interpolated_mask", interpolated)


def apply_interpolation_policy(
    missingness: MissingFrameHandling,
    *,
    policy: InterpolationPolicy,
) -> InterpolationResult:
    """Apply the frozen interpolation policy without learning from any data split.

    ``method='none'`` preserves values and validity exactly and records an all-false
    ``interpolated_mask``. No fitting, smoothing, extrapolation, resampling, or
    outcome-dependent selection occurs. Missing/quality-invalid observations remain
    invalid for downstream alignment/filtering.
    """

    if not isinstance(policy, InterpolationPolicy):
        raise InterpolationPolicyError("policy must be an InterpolationPolicy")
    if policy.method != "none":  # defensive against malformed/deserialized objects
        raise InterpolationPolicyError("unsupported interpolation method")

    values = np.asarray(missingness.values)
    valid_mask = np.asarray(missingness.valid_mask)
    if values.ndim != 3 or valid_mask.shape != values.shape:
        raise InterpolationPolicyError(
            "missingness values and valid_mask must share (T, K, D) shape"
        )
    if valid_mask.dtype != np.bool_:
        raise InterpolationPolicyError("missingness valid_mask must be boolean")

    return InterpolationResult(
        values=values,
        valid_mask=valid_mask,
        interpolated_mask=np.zeros(values.shape, dtype=np.bool_),
        policy=policy,
    )
