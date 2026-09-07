from __future__ import annotations

import numpy as np

from lagged_facial_graph_forecasting.core_contract_io import (
    dumps_core_contract,
    loads_core_contract,
    serialize_core_contract,
)
from lagged_facial_graph_forecasting.interpolation_policy import (
    InterpolationPolicy,
    apply_interpolation_policy,
)
from lagged_facial_graph_forecasting.missingness import handle_missing_frames
from lagged_facial_graph_forecasting.sequence_assembly import assemble_subject_sequence


def _assembled_sequence():
    values = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, np.nan], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ],
        dtype=np.float64,
    )
    valid = np.isfinite(values)
    missingness = handle_missing_frames(values, valid_mask=valid)
    processed = apply_interpolation_policy(
        missingness,
        policy=InterpolationPolicy(method="none"),
    )
    return assemble_subject_sequence(
        processed,
        subject_id="subject-serialization",
        time_index=np.array([0.0, 0.04, 0.09], dtype=np.float64),
        region_id=("left_eye", "mouth"),
        dimension=("vx", "vy"),
        sampling_rate=25.0,
    )


def test_assembled_face_time_series_round_trips_through_frozen_serializer() -> None:
    original = _assembled_sequence()

    text = dumps_core_contract(original)
    restored = loads_core_contract(text)

    assert type(restored) is type(original)
    assert restored.subject_id == original.subject_id
    assert np.array_equal(restored.X, original.X, equal_nan=True)
    assert np.array_equal(restored.time_index, original.time_index)
    assert restored.region_id == original.region_id
    assert restored.dimension == original.dimension
    assert np.array_equal(restored.valid_mask, original.valid_mask)
    assert restored.sampling_rate == original.sampling_rate


def test_assembled_sequence_serialization_is_deterministic_and_versioned() -> None:
    sequence = _assembled_sequence()

    encoded_once = dumps_core_contract(sequence)
    encoded_twice = dumps_core_contract(sequence)
    envelope = serialize_core_contract(sequence)

    assert encoded_once == encoded_twice
    assert envelope["schema_version"] == 1
    assert envelope["contract_type"] == "FaceTimeSeries"
    assert dumps_core_contract(loads_core_contract(encoded_once)) == encoded_once
