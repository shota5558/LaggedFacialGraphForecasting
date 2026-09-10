from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting.interpolation_policy import (
    InterpolationPolicy,
    InterpolationResult,
    apply_interpolation_policy,
)
from lagged_facial_graph_forecasting.missingness import handle_missing_frames
from lagged_facial_graph_forecasting.sequence_assembly import (
    SequenceAssemblyError,
    assemble_subject_sequence,
)


def _processed_fixture():
    values = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, np.nan], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ]
    )
    valid = np.isfinite(values)
    missingness = handle_missing_frames(values, valid_mask=valid)
    processed = apply_interpolation_policy(
        missingness,
        policy=InterpolationPolicy(method="none"),
    )
    return values, valid, processed


def test_assembles_one_subject_with_exact_provenance() -> None:
    values, valid, processed = _processed_fixture()
    time_index = np.array([0.0, 0.04, 0.09])

    series = assemble_subject_sequence(
        processed,
        subject_id="subject-01",
        time_index=time_index,
        region_id=("left_eye", "mouth"),
        dimension=("vx", "vy"),
        sampling_rate=25.0,
    )

    assert series.subject_id == "subject-01"
    assert np.array_equal(series.X, values, equal_nan=True)
    assert np.array_equal(series.valid_mask, valid)
    assert np.array_equal(series.time_index, time_index)
    assert series.region_id == ("left_eye", "mouth")
    assert series.dimension == ("vx", "vy")
    assert series.sampling_rate == 25.0


def test_invalid_observation_remains_in_place_under_mask() -> None:
    _, _, processed = _processed_fixture()

    series = assemble_subject_sequence(
        processed,
        subject_id="subject-01",
        time_index=np.array([0.0, 1.0, 2.0]),
        region_id=("left_eye", "mouth"),
        dimension=("vx", "vy"),
        sampling_rate=1.0,
    )

    assert np.isnan(series.X[1, 0, 1])
    assert not series.valid_mask[1, 0, 1]
    assert series.X.shape[0] == 3


def test_nonfinite_observation_cannot_be_marked_valid_at_assembly_boundary() -> None:
    processed = InterpolationResult(
        values=np.array([[[1.0, np.nan]], [[2.0, 3.0]]], dtype=np.float64),
        valid_mask=np.ones((2, 1, 2), dtype=np.bool_),
        interpolated_mask=np.zeros((2, 1, 2), dtype=np.bool_),
        policy=InterpolationPolicy(method="none"),
    )

    with pytest.raises(SequenceAssemblyError, match="non-finite values cannot be marked valid"):
        assemble_subject_sequence(
            processed,
            subject_id="subject-invalid-evidence",
            time_index=np.array([0.0, 1.0]),
            region_id=("mouth",),
            dimension=("vx", "vy"),
            sampling_rate=1.0,
        )


def test_time_region_dimension_and_subject_contracts_are_enforced() -> None:
    _, _, processed = _processed_fixture()

    with pytest.raises(SequenceAssemblyError):
        assemble_subject_sequence(
            processed,
            subject_id="",
            time_index=np.array([0.0, 1.0, 2.0]),
            region_id=("left_eye", "mouth"),
            dimension=("vx", "vy"),
            sampling_rate=1.0,
        )
    with pytest.raises(SequenceAssemblyError):
        assemble_subject_sequence(
            processed,
            subject_id="subject-01",
            time_index=np.array([0.0, 1.0]),
            region_id=("left_eye", "mouth"),
            dimension=("vx", "vy"),
            sampling_rate=1.0,
        )
    with pytest.raises(SequenceAssemblyError):
        assemble_subject_sequence(
            processed,
            subject_id="subject-01",
            time_index=np.array([0.0, 1.0, 2.0]),
            region_id=("left_eye",),
            dimension=("vx", "vy"),
            sampling_rate=1.0,
        )


def test_input_arrays_are_copied_into_series() -> None:
    _, _, processed = _processed_fixture()
    time_index = np.array([0.0, 1.0, 2.0])

    series = assemble_subject_sequence(
        processed,
        subject_id="subject-01",
        time_index=time_index,
        region_id=("left_eye", "mouth"),
        dimension=("vx", "vy"),
        sampling_rate=1.0,
    )

    time_index[0] = 99.0
    assert series.time_index[0] == 0.0
