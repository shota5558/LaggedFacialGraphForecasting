"""Subject-level FaceTimeSeries assembly for canonical preprocessing output (A-14)."""

from __future__ import annotations

import numpy as np

from .contracts import ContractError, FaceTimeSeries


class SubjectSequenceAssemblyError(ValueError):
    """Raised when subject-level preprocessing output cannot be assembled safely."""


def _immutable_copy(values: np.ndarray, *, dtype: np.dtype | None = None) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def assemble_subject_sequence(
    values: np.ndarray,
    *,
    valid_mask: np.ndarray,
    subject_id: str,
    time_index: np.ndarray,
    region_id: tuple[str, ...],
    dimension: tuple[str, ...],
    sampling_rate: float,
) -> FaceTimeSeries:
    """Assemble one subject's processed sequence into the frozen FaceTimeSeries contract.

    A-14 is intentionally a boundary/assembly step rather than another preprocessing
    algorithm. ``values`` and ``valid_mask`` are preserved exactly; no interpolation,
    smoothing, resampling, normalization, imputation, sample deletion, or statistic
    fitting occurs here. ``sampling_rate`` is explicit rather than estimated from the
    observed timestamps, so this function cannot learn a preprocessing parameter from
    outer-test data.

    Non-finite values are permitted only where ``valid_mask`` is false. This retains
    the A-11/A-12 missingness semantics while preventing invalid numeric observations
    from silently entering downstream discovery or forecasting as usable evidence.
    """

    array = np.asarray(values)
    mask = np.asarray(valid_mask)
    times = np.asarray(time_index)

    if array.ndim != 3:
        raise SubjectSequenceAssemblyError(
            "values must have canonical shape (T, K, D)"
        )
    if mask.shape != array.shape:
        raise SubjectSequenceAssemblyError(
            "valid_mask must have the same shape as values"
        )
    if mask.dtype != np.bool_:
        raise SubjectSequenceAssemblyError("valid_mask must have boolean dtype")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
        raise SubjectSequenceAssemblyError("values must be numeric")
    if np.any(mask & ~np.isfinite(array)):
        raise SubjectSequenceAssemblyError(
            "non-finite values cannot be marked valid"
        )

    try:
        return FaceTimeSeries(
            X=_immutable_copy(array),
            subject_id=subject_id,
            time_index=_immutable_copy(times),
            region_id=tuple(region_id),
            dimension=tuple(dimension),
            valid_mask=_immutable_copy(mask, dtype=np.bool_),
            sampling_rate=sampling_rate,
        )
    except ContractError as exc:
        raise SubjectSequenceAssemblyError(str(exc)) from exc
