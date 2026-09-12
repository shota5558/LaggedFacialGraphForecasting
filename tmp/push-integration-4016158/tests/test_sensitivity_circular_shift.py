from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lagged_facial_graph_forecasting.contracts import FaceTimeSeries
from lagged_facial_graph_forecasting.sensitivity_circular_shift import (
    CIRCULAR_SHIFT_WRAP_SEMANTICS,
    CircularShiftError,
    CircularShiftPolicy,
    circular_shift_face_time_series,
    circular_shift_subjects,
)
from lagged_facial_graph_forecasting.sensitivity_execution import (
    load_sensitivity_experiment_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _execution():
    return load_sensitivity_experiment_config(
        ROOT / "configs/sensitivity_preimplementation.yaml", repository_root=ROOT
    )


def _series(subject_id: str, values: np.ndarray, sampling_rate: float = 25.0) -> FaceTimeSeries:
    tensor = np.asarray(values, dtype=float)
    if tensor.ndim == 2:
        tensor = tensor[:, :, None]
    return FaceTimeSeries(
        X=tensor,
        subject_id=subject_id,
        time_index=np.arange(tensor.shape[0], dtype=float) / sampling_rate,
        region_id=tuple(f"region_{index}" for index in range(tensor.shape[1])),
        dimension=tuple(f"d{index}" for index in range(tensor.shape[2])),
        valid_mask=np.ones_like(tensor, dtype=bool),
        sampling_rate=sampling_rate,
    )


def _circular_autocorrelation(values: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fft(values)
    return np.fft.ifft(spectrum * np.conjugate(spectrum)).real


def test_policy_requires_sufficiently_large_circular_distance() -> None:
    policy = CircularShiftPolicy(min_circular_distance_fraction=0.25)
    candidates = policy.valid_shifts(100)
    assert candidates
    assert min(min(shift, 100 - shift) for shift in candidates) >= 25
    assert 0 not in candidates
    with pytest.raises(CircularShiftError, match="fraction"):
        CircularShiftPolicy(min_circular_distance_fraction=0.0)
    with pytest.raises(CircularShiftError, match="fraction"):
        CircularShiftPolicy(min_circular_distance_fraction=0.51)


def test_circular_shift_preserves_metadata_wrap_semantics_and_large_shift() -> None:
    execution = _execution()
    values = np.arange(64 * 2, dtype=float).reshape(64, 2, 1)
    source = _series("subject_a", values)
    policy = CircularShiftPolicy(min_circular_distance_fraction=0.25)
    result = circular_shift_face_time_series(
        execution, source, seed=505, policy=policy, repository_root=ROOT
    )

    assert result.wrapping_semantics == CIRCULAR_SHIFT_WRAP_SEMANTICS
    assert result.series.subject_id == source.subject_id
    assert result.series.X.shape == source.X.shape
    np.testing.assert_array_equal(result.series.time_index, source.time_index)
    assert result.series.region_id == source.region_id
    assert result.series.dimension == source.dimension
    assert result.series.sampling_rate == source.sampling_rate
    np.testing.assert_array_equal(result.series.valid_mask, source.valid_mask)
    for component in result.components:
        assert component.circular_distance >= 16
    first_shift = result.components[0].shift
    np.testing.assert_array_equal(
        result.series.X[:, 0, 0], np.roll(source.X[:, 0, 0], first_shift)
    )


def test_circular_shift_preserves_component_circular_autocorrelation() -> None:
    execution = _execution()
    rng = np.random.default_rng(21)
    source = _series("subject_a", rng.normal(size=(257, 3, 2)))
    shifted = circular_shift_face_time_series(
        execution, source, seed=606, repository_root=ROOT
    ).series

    for region_index in range(source.X.shape[1]):
        for dimension_index in range(source.X.shape[2]):
            np.testing.assert_allclose(
                _circular_autocorrelation(shifted.X[:, region_index, dimension_index]),
                _circular_autocorrelation(source.X[:, region_index, dimension_index]),
                rtol=1e-11,
                atol=1e-11,
            )


def test_independent_large_shifts_destroy_cross_series_alignment() -> None:
    execution = _execution()
    rng = np.random.default_rng(44)
    shared = rng.normal(size=2048)
    source = _series("subject_a", np.stack([shared, shared], axis=1))
    result = circular_shift_face_time_series(
        execution, source, seed=707, repository_root=ROOT
    )
    before = float(np.corrcoef(source.X[:, 0, 0], source.X[:, 1, 0])[0, 1])
    after = float(np.corrcoef(result.series.X[:, 0, 0], result.series.X[:, 1, 0])[0, 1])

    assert result.components[0].shift != result.components[1].shift
    assert before > 0.999
    assert abs(after) < 0.10


def test_circular_shift_is_deterministic_order_invariant_and_subject_local() -> None:
    execution = _execution()
    rng = np.random.default_rng(99)
    a = _series("subject_a", rng.normal(size=(128, 2, 1)))
    b = _series("subject_b", rng.normal(size=(96, 2, 1)))

    forward = circular_shift_subjects(
        execution, (a, b), seed=808, repository_root=ROOT
    )
    reverse = circular_shift_subjects(
        execution, (b, a), seed=808, repository_root=ROOT
    )
    forward_by_id = {item.series.subject_id: item for item in forward}
    reverse_by_id = {item.series.subject_id: item for item in reverse}

    assert forward_by_id["subject_a"].series.X.shape[0] == 128
    assert forward_by_id["subject_b"].series.X.shape[0] == 96
    np.testing.assert_array_equal(
        forward_by_id["subject_a"].series.X,
        reverse_by_id["subject_a"].series.X,
    )
    np.testing.assert_array_equal(
        forward_by_id["subject_b"].series.X,
        reverse_by_id["subject_b"].series.X,
    )
    assert forward_by_id["subject_a"].components == reverse_by_id["subject_a"].components
    assert forward_by_id["subject_b"].components == reverse_by_id["subject_b"].components


def test_circular_shift_rejects_missing_values_without_implicit_policy() -> None:
    execution = _execution()
    rng = np.random.default_rng(3)
    source = _series("subject_a", rng.normal(size=(64, 2, 1)))
    mask = np.array(source.valid_mask, copy=True)
    mask[7, 1, 0] = False
    invalid = FaceTimeSeries(
        X=np.array(source.X, copy=True),
        subject_id=source.subject_id,
        time_index=np.array(source.time_index, copy=True),
        region_id=source.region_id,
        dimension=source.dimension,
        valid_mask=mask,
        sampling_rate=source.sampling_rate,
    )
    with pytest.raises(CircularShiftError, match="fully valid"):
        circular_shift_face_time_series(
            execution, invalid, seed=1, repository_root=ROOT
        )
