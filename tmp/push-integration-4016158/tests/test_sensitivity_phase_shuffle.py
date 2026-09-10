from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lagged_facial_graph_forecasting.contracts import FaceTimeSeries
from lagged_facial_graph_forecasting.sensitivity_execution import (
    load_sensitivity_experiment_config,
)
from lagged_facial_graph_forecasting.sensitivity_phase_shuffle import (
    PhaseShuffleError,
    phase_shuffle_face_time_series,
    phase_shuffle_subjects,
)


ROOT = Path(__file__).resolve().parents[1]


def _execution():
    return load_sensitivity_experiment_config(
        ROOT / "configs/sensitivity_preimplementation.yaml", repository_root=ROOT
    )


def _coherent_series(subject_id: str, seed: int, length: int = 256) -> FaceTimeSeries:
    rng = np.random.default_rng(seed)
    shared = np.convolve(rng.normal(size=length + 8), np.ones(9) / 9.0, mode="valid")
    x = shared + rng.normal(scale=0.02, size=length)
    y = 0.9 * shared + rng.normal(scale=0.02, size=length)
    values = np.stack([x, y], axis=1)[:, :, None]
    return FaceTimeSeries(
        X=values,
        subject_id=subject_id,
        time_index=np.arange(length, dtype=float) / 25.0,
        region_id=("left_cheek", "mouth"),
        dimension=("value",),
        valid_mask=np.ones_like(values, dtype=bool),
        sampling_rate=25.0,
    )


def test_phase_shuffle_preserves_subject_shape_sampling_metadata_and_spectrum() -> None:
    execution = _execution()
    source = _coherent_series("subject_a", 81)
    result = phase_shuffle_face_time_series(execution, source, seed=4404, repository_root=ROOT)
    shuffled = result.series

    assert shuffled.subject_id == source.subject_id
    assert shuffled.X.shape == source.X.shape
    assert shuffled.region_id == source.region_id
    assert shuffled.dimension == source.dimension
    assert shuffled.sampling_rate == source.sampling_rate
    np.testing.assert_array_equal(shuffled.time_index, source.time_index)
    np.testing.assert_array_equal(shuffled.valid_mask, source.valid_mask)

    for region_index in range(source.X.shape[1]):
        original_magnitude = np.abs(np.fft.rfft(source.X[:, region_index, 0]))
        shuffled_magnitude = np.abs(np.fft.rfft(shuffled.X[:, region_index, 0]))
        np.testing.assert_allclose(shuffled_magnitude, original_magnitude, rtol=1e-11, atol=1e-11)


def test_phase_shuffle_destroys_cross_series_temporal_alignment_on_broadband_fixture() -> None:
    execution = _execution()
    source = _coherent_series("subject_a", 88)
    before = float(np.corrcoef(source.X[:, 0, 0], source.X[:, 1, 0])[0, 1])
    after_abs_correlations = []
    for seed in range(20):
        shuffled = phase_shuffle_face_time_series(
            execution, source, seed=seed, repository_root=ROOT
        ).series
        after = float(np.corrcoef(shuffled.X[:, 0, 0], shuffled.X[:, 1, 0])[0, 1])
        after_abs_correlations.append(abs(after))

    assert before > 0.95
    assert float(np.median(after_abs_correlations)) < 0.30
    assert sum(value < 0.35 for value in after_abs_correlations) >= 14


def test_phase_shuffle_is_deterministic_and_subject_order_invariant() -> None:
    execution = _execution()
    first_subject = _coherent_series("subject_a", 101)
    second_subject = _coherent_series("subject_b", 202)

    forward = phase_shuffle_subjects(
        execution, (first_subject, second_subject), seed=501, repository_root=ROOT
    )
    reverse = phase_shuffle_subjects(
        execution, (second_subject, first_subject), seed=501, repository_root=ROOT
    )
    forward_by_id = {result.series.subject_id: result.series.X for result in forward}
    reverse_by_id = {result.series.subject_id: result.series.X for result in reverse}

    np.testing.assert_allclose(forward_by_id["subject_a"], reverse_by_id["subject_a"])
    np.testing.assert_allclose(forward_by_id["subject_b"], reverse_by_id["subject_b"])
    repeat = phase_shuffle_face_time_series(
        execution, first_subject, seed=501, repository_root=ROOT
    ).series
    np.testing.assert_allclose(forward_by_id["subject_a"], repeat.X)


def test_phase_shuffle_never_mutates_input_arrays() -> None:
    execution = _execution()
    source = _coherent_series("subject_a", 77)
    x_before = np.array(source.X, copy=True)
    time_before = np.array(source.time_index, copy=True)
    mask_before = np.array(source.valid_mask, copy=True)

    result = phase_shuffle_face_time_series(
        execution, source, seed=19, repository_root=ROOT
    )

    np.testing.assert_array_equal(source.X, x_before)
    np.testing.assert_array_equal(source.time_index, time_before)
    np.testing.assert_array_equal(source.valid_mask, mask_before)
    assert not np.shares_memory(result.series.X, source.X)


def test_phase_shuffle_rejects_invalid_samples_instead_of_implicit_imputation() -> None:
    execution = _execution()
    source = _coherent_series("subject_a", 66)
    invalid_mask = np.array(source.valid_mask, copy=True)
    invalid_mask[10, 0, 0] = False
    invalid = FaceTimeSeries(
        X=np.array(source.X, copy=True),
        subject_id=source.subject_id,
        time_index=np.array(source.time_index, copy=True),
        region_id=source.region_id,
        dimension=source.dimension,
        valid_mask=invalid_mask,
        sampling_rate=source.sampling_rate,
    )

    with pytest.raises(PhaseShuffleError, match="invalid samples"):
        phase_shuffle_face_time_series(
            execution, invalid, seed=7, repository_root=ROOT
        )
