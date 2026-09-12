from __future__ import annotations

import numpy as np

from lagged_facial_graph_forecasting.core_contract_io import (
    dumps_core_contract,
    loads_core_contract,
)
from lagged_facial_graph_forecasting.interpolation_policy import (
    InterpolationPolicy,
    apply_interpolation_policy,
)
from lagged_facial_graph_forecasting.missingness import handle_missing_frames
from lagged_facial_graph_forecasting.sequence_assembly import assemble_subject_sequence


def test_a14_face_time_series_round_trip_preserves_missingness_and_provenance() -> None:
    values = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, np.nan], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ],
        dtype=np.float64,
    )
    valid_mask = np.isfinite(values)
    handled = handle_missing_frames(values, valid_mask=valid_mask)
    processed = apply_interpolation_policy(
        handled,
        policy=InterpolationPolicy(method="none"),
    )
    series = assemble_subject_sequence(
        processed,
        subject_id="subject-01",
        time_index=np.array([0.0, 0.04, 0.11], dtype=np.float64),
        region_id=("left_eye", "mouth"),
        dimension=("vx", "vy"),
        sampling_rate=25.0,
    )

    encoded = dumps_core_contract(series)
    restored = loads_core_contract(encoded)

    assert type(restored) is type(series)
    assert dumps_core_contract(restored) == encoded
    assert restored.subject_id == series.subject_id
    assert restored.region_id == series.region_id
    assert restored.dimension == series.dimension
    assert restored.sampling_rate == series.sampling_rate
    assert np.array_equal(restored.time_index, series.time_index)
    assert np.array_equal(restored.valid_mask, series.valid_mask)
    assert np.array_equal(restored.X, series.X, equal_nan=True)
    assert np.isnan(restored.X[1, 0, 1])
    assert not restored.valid_mask[1, 0, 1]


def test_face_time_series_serialization_is_deterministic_for_same_contract() -> None:
    values = np.arange(12, dtype=np.float64).reshape(3, 2, 2)
    valid_mask = np.ones(values.shape, dtype=bool)
    handled = handle_missing_frames(values, valid_mask=valid_mask)
    processed = apply_interpolation_policy(
        handled,
        policy=InterpolationPolicy(method="none"),
    )
    series = assemble_subject_sequence(
        processed,
        subject_id="subject-02",
        time_index=np.array([0.0, 0.05, 0.13], dtype=np.float64),
        region_id=("left_eye", "mouth"),
        dimension=("vx", "vy"),
        sampling_rate=20.0,
    )

    assert dumps_core_contract(series) == dumps_core_contract(series)
