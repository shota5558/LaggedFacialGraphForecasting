"""Large circular-shift surrogate for Sensitivity pre-implementation (S-IMPL-05).

Each scalar region×dimension component is shifted independently within its own
subject sequence.  The transform uses explicit circular wrapping, preserves the
component's Fourier magnitude/circular autocorrelation, and never concatenates or
moves samples across subject boundaries.  Real-data execution remains controlled
by the common Sensitivity execution barrier.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Sequence

import numpy as np

from .contracts import FaceTimeSeries


CIRCULAR_SHIFT_SCHEMA_VERSION = 1
CIRCULAR_SHIFT_METHOD = "circular_shift"
CIRCULAR_SHIFT_WRAP_SEMANTICS = (
    "positive numpy.roll shift; trailing samples wrap to the beginning of the same "
    "subject/component sequence"
)


class CircularShiftError(ValueError):
    """Raised when a circular surrogate cannot satisfy the frozen contract."""


@dataclass(frozen=True, slots=True)
class CircularShiftPolicy:
    """Policy requiring a non-trivial circular distance from the original alignment."""

    min_circular_distance_fraction: float = 0.25

    def __post_init__(self) -> None:
        value = self.min_circular_distance_fraction
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not (0.0 < float(value) <= 0.5)
        ):
            raise CircularShiftError(
                "min_circular_distance_fraction must satisfy 0 < fraction <= 0.5"
            )

    def minimum_distance(self, sequence_length: int) -> int:
        if not isinstance(sequence_length, int) or isinstance(sequence_length, bool):
            raise CircularShiftError("sequence_length must be an integer")
        if sequence_length < 4:
            raise CircularShiftError("circular shift requires at least four time samples")
        minimum = max(1, int(math.ceil(sequence_length * self.min_circular_distance_fraction)))
        if minimum > sequence_length // 2:
            raise CircularShiftError(
                "shift policy leaves no valid sufficiently-large circular shift"
            )
        return minimum

    def valid_shifts(self, sequence_length: int) -> tuple[int, ...]:
        minimum = self.minimum_distance(sequence_length)
        candidates = tuple(
            shift
            for shift in range(1, sequence_length)
            if min(shift, sequence_length - shift) >= minimum
        )
        if not candidates:
            raise CircularShiftError(
                "shift policy leaves no valid sufficiently-large circular shift"
            )
        return candidates


@dataclass(frozen=True, slots=True, order=True)
class CircularShiftComponent:
    region_id: str
    dimension: str
    shift: int
    circular_distance: int

    def __post_init__(self) -> None:
        if not self.region_id or not self.dimension:
            raise CircularShiftError("component labels must be non-empty")
        if self.shift <= 0 or self.circular_distance <= 0:
            raise CircularShiftError("component shift and circular distance must be positive")


@dataclass(frozen=True, slots=True)
class CircularShiftResult:
    series: FaceTimeSeries
    seed: int
    policy: CircularShiftPolicy
    components: tuple[CircularShiftComponent, ...]
    method: str = CIRCULAR_SHIFT_METHOD
    wrapping_semantics: str = CIRCULAR_SHIFT_WRAP_SEMANTICS
    schema_version: int = CIRCULAR_SHIFT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.series, FaceTimeSeries):
            raise CircularShiftError("series must be a FaceTimeSeries")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise CircularShiftError("seed must be a non-negative integer")
        if not isinstance(self.policy, CircularShiftPolicy):
            raise CircularShiftError("policy must be CircularShiftPolicy")
        if self.method != CIRCULAR_SHIFT_METHOD:
            raise CircularShiftError("circular-shift method provenance is immutable")
        if self.wrapping_semantics != CIRCULAR_SHIFT_WRAP_SEMANTICS:
            raise CircularShiftError("circular-shift wrapping semantics are immutable")
        if self.schema_version != CIRCULAR_SHIFT_SCHEMA_VERSION:
            raise CircularShiftError("unsupported circular-shift schema_version")


def _stable_subject_seed(seed: int, subject_id: str) -> int:
    digest = hashlib.sha256(f"{seed}|{subject_id}|circular_shift".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def circular_shift_face_time_series(
    series: FaceTimeSeries,
    *,
    seed: int,
    policy: CircularShiftPolicy | None = None,
) -> CircularShiftResult:
    """Circularly shift scalar components inside one subject, never across subjects."""

    if not isinstance(series, FaceTimeSeries):
        raise CircularShiftError("series must be a FaceTimeSeries")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise CircularShiftError("seed must be a non-negative integer")
    frozen_policy = policy or CircularShiftPolicy()
    if not isinstance(frozen_policy, CircularShiftPolicy):
        raise CircularShiftError("policy must be CircularShiftPolicy")

    values = np.asarray(series.X, dtype=float)
    valid = np.asarray(series.valid_mask, dtype=bool)
    if not np.all(valid):
        raise CircularShiftError(
            "circular-shift pre-implementation requires fully valid sequences; "
            "missing-data semantics must not be introduced implicitly"
        )
    if not np.all(np.isfinite(values)):
        raise CircularShiftError("circular shift requires finite observations")

    sequence_length = values.shape[0]
    candidates = np.asarray(frozen_policy.valid_shifts(sequence_length), dtype=int)
    component_count = values.shape[1] * values.shape[2]
    rng = np.random.default_rng(_stable_subject_seed(seed, series.subject_id))

    # Avoid accidental identical shifts when the candidate set is large enough.
    # Distinct component shifts better realize the intended destruction of
    # cross-series alignment without changing any marginal series spectrum.
    if candidates.size >= component_count:
        assigned = rng.choice(candidates, size=component_count, replace=False)
    else:
        assigned = rng.choice(candidates, size=component_count, replace=True)

    transformed = np.empty_like(values, dtype=float)
    components: list[CircularShiftComponent] = []
    flat_index = 0
    for region_index, region in enumerate(series.region_id):
        for dimension_index, dimension in enumerate(series.dimension):
            shift = int(assigned[flat_index])
            flat_index += 1
            transformed[:, region_index, dimension_index] = np.roll(
                values[:, region_index, dimension_index], shift
            )
            components.append(
                CircularShiftComponent(
                    region_id=region,
                    dimension=dimension,
                    shift=shift,
                    circular_distance=min(shift, sequence_length - shift),
                )
            )

    surrogate = FaceTimeSeries(
        X=transformed,
        subject_id=series.subject_id,
        time_index=np.array(series.time_index, copy=True),
        region_id=series.region_id,
        dimension=series.dimension,
        valid_mask=np.array(series.valid_mask, copy=True),
        sampling_rate=series.sampling_rate,
    )
    return CircularShiftResult(
        series=surrogate,
        seed=seed,
        policy=frozen_policy,
        components=tuple(components),
    )


def circular_shift_subjects(
    series_by_subject: Sequence[FaceTimeSeries],
    *,
    seed: int,
    policy: CircularShiftPolicy | None = None,
) -> tuple[CircularShiftResult, ...]:
    """Apply the surrogate independently per subject with order-invariant seeds."""

    sequences = tuple(series_by_subject)
    if not sequences:
        raise CircularShiftError("at least one FaceTimeSeries is required")
    if any(not isinstance(series, FaceTimeSeries) for series in sequences):
        raise CircularShiftError("all inputs must be FaceTimeSeries instances")
    subject_ids = tuple(series.subject_id for series in sequences)
    if len(set(subject_ids)) != len(subject_ids):
        raise CircularShiftError("subject_id values must be unique within a surrogate batch")

    return tuple(
        circular_shift_face_time_series(series, seed=seed, policy=policy)
        for series in sequences
    )
