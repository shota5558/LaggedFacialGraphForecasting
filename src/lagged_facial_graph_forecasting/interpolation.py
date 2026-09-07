"""Explicit, non-fitted interpolation policy for A-13 facial preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .missingness import MissingFrameHandling
from .preprocessing import PreprocessingValidationError, validate_timestamps


class InterpolationError(ValueError):
    """Raised when interpolation policy or temporal provenance is invalid."""


@dataclass(frozen=True, slots=True)
class InterpolationPolicy:
    """Frozen interpolation rule with no data-dependent parameter fitting."""

    method: Literal["none", "linear"]
    max_gap_duration: float | None = None

    def __post_init__(self) -> None:
        if self.method not in {"none", "linear"}:
            raise InterpolationError("method must be 'none' or 'linear'")
        if self.method == "none":
            if self.max_gap_duration is not None:
                raise InterpolationError(
                    "method='none' requires max_gap_duration=None"
                )
            return
        if self.max_gap_duration is None:
            raise InterpolationError(
                "method='linear' requires explicit max_gap_duration"
            )
        duration = float(self.max_gap_duration)
        if not np.isfinite(duration) or duration <= 0:
            raise InterpolationError("max_gap_duration must be finite and > 0")
        object.__setattr__(self, "max_gap_duration", duration)


@dataclass(frozen=True, slots=True)
class InterpolationResult:
    """Immutable interpolated values and full interpolation provenance."""

    values: np.ndarray
    valid_mask: np.ndarray
    interpolated_mask: np.ndarray
    method: str
    max_gap_duration: float | None

    def __post_init__(self) -> None:
        values = np.array(self.values, copy=True)
        valid = np.array(self.valid_mask, dtype=np.bool_, copy=True)
        interpolated = np.array(self.interpolated_mask, dtype=np.bool_, copy=True)
        for array in (values, valid, interpolated):
            array.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "interpolated_mask", interpolated)


def apply_interpolation(
    handled: MissingFrameHandling,
    *,
    timestamps: np.ndarray,
    policy: InterpolationPolicy,
) -> InterpolationResult:
    """Apply an explicitly frozen interpolation policy to observed frames.

    ``none`` preserves A-12 exactly. ``linear`` uses the validated observed
    timestamps and only fills non-finite invalid elements that are bracketed by two
    originally valid finite endpoints whose total endpoint-to-endpoint duration is
    no larger than ``max_gap_duration``. It never extrapolates.

    Finite observations invalidated by A-11 quality flags are deliberately *not*
    replaced here: quality rejection and numeric missingness are distinct evidence.
    A different treatment would require a separately frozen scientific policy.

    Policy values are supplied explicitly and never estimated from outer-test (or
    any other) predictive outcomes. No smoothing, resampling, threshold fitting, or
    population-level statistic is performed.
    """

    values = np.asarray(handled.values)
    valid = np.asarray(handled.valid_mask)
    if values.ndim != 3 or valid.shape != values.shape or valid.dtype != np.bool_:
        raise InterpolationError("handled values/valid_mask violate the A-12 contract")

    try:
        time = validate_timestamps(
            timestamps,
            expected_frame_count=int(values.shape[0]),
        )
    except PreprocessingValidationError as exc:
        raise InterpolationError(str(exc)) from exc

    output = np.array(values, copy=True)
    output_valid = np.array(valid, copy=True)
    interpolated = np.zeros(values.shape, dtype=np.bool_)

    if policy.method == "none":
        return InterpolationResult(
            values=output,
            valid_mask=output_valid,
            interpolated_mask=interpolated,
            method="none",
            max_gap_duration=None,
        )

    max_gap_duration = float(policy.max_gap_duration)
    _, region_count, dimension_count = values.shape
    for region in range(region_count):
        for dimension in range(dimension_count):
            original_valid = valid[:, region, dimension]
            valid_indices = np.flatnonzero(original_valid)
            if valid_indices.size < 2:
                continue

            for left, right in zip(valid_indices[:-1], valid_indices[1:]):
                left_i = int(left)
                right_i = int(right)
                if right_i <= left_i + 1:
                    continue
                duration = float(time[right_i] - time[left_i])
                if duration > max_gap_duration:
                    continue

                between = np.arange(left_i + 1, right_i, dtype=np.int64)
                # A-11 finite-but-invalid quality failures remain invalid. Only
                # numeric missingness is interpolation-eligible in this policy.
                eligible = (~original_valid[between]) & (
                    ~np.isfinite(values[between, region, dimension])
                )
                if not np.any(eligible):
                    continue
                fill_indices = between[eligible]
                left_value = float(values[left_i, region, dimension])
                right_value = float(values[right_i, region, dimension])
                fraction = (
                    time[fill_indices] - time[left_i]
                ) / duration
                output[fill_indices, region, dimension] = (
                    left_value + fraction * (right_value - left_value)
                )
                output_valid[fill_indices, region, dimension] = True
                interpolated[fill_indices, region, dimension] = True

    return InterpolationResult(
        values=output,
        valid_mask=output_valid,
        interpolated_mask=interpolated,
        method="linear",
        max_gap_duration=max_gap_duration,
    )
