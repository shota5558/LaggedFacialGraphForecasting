"""Phase-shuffled autocorrelation-preserving surrogate (S-IMPL-04).

Each subject and scalar region×dimension series is phase-randomized independently
in the Fourier domain while preserving its Fourier magnitudes.  This keeps the
individual spectral/autocorrelation structure while breaking cross-series temporal
alignment.  Subject boundaries and canonical FaceTimeSeries metadata are preserved.

Missing/invalid observations are rejected in this pre-implementation task rather
than silently interpolated, because interpolation would alter the spectrum being
claimed as preserved.  Every public execution entrypoint requires the common
Sensitivity execution boundary, so real-data use is fail-closed before Primary Freeze.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from .contracts import FaceTimeSeries
from .sensitivity_execution import (
    SensitivityExperimentConfig,
    assert_sensitivity_execution_allowed,
)


PHASE_SHUFFLE_SCHEMA_VERSION = 1
PHASE_SHUFFLE_METHOD = "phase_shuffle"


class PhaseShuffleError(ValueError):
    """Raised when a phase-shuffle surrogate cannot satisfy its frozen contract."""


@dataclass(frozen=True, slots=True)
class PhaseShuffleResult:
    """One transformed subject sequence with deterministic surrogate provenance."""

    series: FaceTimeSeries
    seed: int
    method: str = PHASE_SHUFFLE_METHOD
    schema_version: int = PHASE_SHUFFLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.series, FaceTimeSeries):
            raise PhaseShuffleError("series must be a FaceTimeSeries")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise PhaseShuffleError("seed must be a non-negative integer")
        if self.method != PHASE_SHUFFLE_METHOD:
            raise PhaseShuffleError("phase-shuffle method provenance is immutable")
        if self.schema_version != PHASE_SHUFFLE_SCHEMA_VERSION:
            raise PhaseShuffleError("unsupported phase-shuffle schema_version")


def _stable_component_seed(
    seed: int,
    *,
    subject_id: str,
    region_id: str,
    dimension: str,
) -> int:
    payload = f"{seed}|{subject_id}|{region_id}|{dimension}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _phase_randomize_1d(values: np.ndarray, *, seed: int) -> np.ndarray:
    signal = np.asarray(values, dtype=float)
    if signal.ndim != 1:
        raise PhaseShuffleError("phase randomization requires a one-dimensional series")
    if signal.size < 2:
        raise PhaseShuffleError("phase randomization requires at least two samples")
    if not np.all(np.isfinite(signal)):
        raise PhaseShuffleError("phase randomization requires finite observations")

    spectrum = np.fft.rfft(signal)
    randomized = np.array(spectrum, copy=True)
    rng = np.random.default_rng(seed)
    stop = randomized.size - 1 if signal.size % 2 == 0 else randomized.size
    if stop > 1:
        phases = rng.uniform(0.0, 2.0 * np.pi, size=stop - 1)
        randomized[1:stop] *= np.exp(1j * phases)

    transformed = np.fft.irfft(randomized, n=signal.size)
    return np.asarray(transformed, dtype=float)


def phase_shuffle_face_time_series(
    execution: SensitivityExperimentConfig,
    series: FaceTimeSeries,
    *,
    seed: int,
    repository_root: str | Path | None = None,
) -> PhaseShuffleResult:
    """Phase-shuffle one subject after enforcing the shared execution barrier."""

    assert_sensitivity_execution_allowed(execution, repository_root=repository_root)
    if not isinstance(series, FaceTimeSeries):
        raise PhaseShuffleError("series must be a FaceTimeSeries")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise PhaseShuffleError("seed must be a non-negative integer")

    values = np.asarray(series.X)
    valid = np.asarray(series.valid_mask, dtype=bool)
    if not np.all(valid):
        raise PhaseShuffleError(
            "phase-shuffle spectrum preservation is undefined with invalid samples; "
            "preprocess or use an explicitly frozen missing-data policy first"
        )
    if np.any(~np.isfinite(values)):
        raise PhaseShuffleError("phase-shuffle input must contain only finite observations")

    transformed = np.empty(values.shape, dtype=float)
    for region_index, region in enumerate(series.region_id):
        for dimension_index, dimension in enumerate(series.dimension):
            component_seed = _stable_component_seed(
                seed,
                subject_id=series.subject_id,
                region_id=region,
                dimension=dimension,
            )
            transformed[:, region_index, dimension_index] = _phase_randomize_1d(
                values[:, region_index, dimension_index], seed=component_seed
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
    return PhaseShuffleResult(series=surrogate, seed=seed)


def phase_shuffle_subjects(
    execution: SensitivityExperimentConfig,
    series_by_subject: Sequence[FaceTimeSeries],
    *,
    seed: int,
    repository_root: str | Path | None = None,
) -> tuple[PhaseShuffleResult, ...]:
    """Transform subjects independently after enforcing the shared execution barrier."""

    assert_sensitivity_execution_allowed(execution, repository_root=repository_root)
    sequences = tuple(series_by_subject)
    if not sequences:
        raise PhaseShuffleError("at least one FaceTimeSeries is required")
    if any(not isinstance(series, FaceTimeSeries) for series in sequences):
        raise PhaseShuffleError("all inputs must be FaceTimeSeries instances")
    subject_ids = tuple(series.subject_id for series in sequences)
    if len(set(subject_ids)) != len(subject_ids):
        raise PhaseShuffleError("subject_id values must be unique within a surrogate batch")

    return tuple(
        phase_shuffle_face_time_series(
            execution,
            series,
            seed=seed,
            repository_root=repository_root,
        )
        for series in sequences
    )
