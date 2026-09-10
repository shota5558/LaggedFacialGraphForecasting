"""Subject-isolated assembly into the canonical FaceTimeSeries contract (A-14)."""

from __future__ import annotations

import numpy as np

from .contracts import ContractError, FaceTimeSeries
from .interpolation_policy import InterpolationResult


class SequenceAssemblyError(ValueError):
    """Raised when processed subject data cannot form one FaceTimeSeries."""


def assemble_subject_sequence(
    processed: InterpolationResult,
    *,
    subject_id: str,
    time_index: np.ndarray,
    region_id: tuple[str, ...],
    dimension: tuple[str, ...],
    sampling_rate: float,
) -> FaceTimeSeries:
    """Assemble exactly one subject into the canonical ``FaceTimeSeries``.

    This function deliberately has no collection-of-subjects input: subject mixing
    cannot occur inside the assembly API. ``time_index`` is the authoritative
    observed temporal provenance and is never inferred, resampled, or repaired here.
    ``sampling_rate`` is explicit metadata supplied by the acquisition/preprocessing
    configuration; it is not estimated from outer-test data.

    A-13 Primary policy must remain ``method='none'``. Invalid observations stay in
    place under the elementwise validity mask; no row deletion, interpolation,
    imputation, smoothing, fitting, feature selection, or split-dependent operation
    is performed.
    """

    if not isinstance(processed, InterpolationResult):
        raise SequenceAssemblyError("processed must be an InterpolationResult")
    if processed.policy.method != "none":
        raise SequenceAssemblyError(
            "Primary subject assembly requires the frozen no-interpolation policy"
        )

    values = np.asarray(processed.values)
    valid_mask = np.asarray(processed.valid_mask)
    if values.ndim != 3 or valid_mask.shape != values.shape:
        raise SequenceAssemblyError(
            "processed values and valid_mask must share (T, K, D) shape"
        )
    if valid_mask.dtype != np.bool_:
        raise SequenceAssemblyError("processed valid_mask must be boolean")
    if np.any(valid_mask & ~np.isfinite(values)):
        raise SequenceAssemblyError("non-finite values cannot be marked valid")

    try:
        return FaceTimeSeries(
            X=np.array(values, copy=True),
            subject_id=subject_id,
            time_index=np.array(time_index, copy=True),
            region_id=tuple(region_id),
            dimension=tuple(dimension),
            valid_mask=np.array(valid_mask, dtype=np.bool_, copy=True),
            sampling_rate=sampling_rate,
        )
    except ContractError as exc:
        raise SequenceAssemblyError(str(exc)) from exc
