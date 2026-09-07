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


def _assembled_series():
    values = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, np.nan], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ],
        dtype=np.float64,
    )
    missingness = handle_missing_frames(values, valid_mask=np.isfinite(values))
    processed = apply_interpolation_policy(
        missingness,
        policy=InterpolationPolicy(method="none"),
    )
    return assemble_subject_sequence(
        processed,
        subject_id="subject-01",
        time_index=np.array([0.0, 0.04, 0.09], dtype=np.float64),
        region_id=("left_eye", "mouth"),
        dimension=("vx", "vy"),
        sampling_rate=25.0,
    )


def test_assembled_subject_sequence_round_trips_through_existing_core_serializer() -> None:
    series = _assembled_series()

    encoded = dumps_core_contract(series)
    restored = loads_core_contract(encoded)

    assert type(restored) is type(series)
    assert restored.subject_id == series.subject_id
    assert np.array_equal(restored.X, series.X, equal_nan=True)
    assert np.array_equal(restored.time_index, series.time_index)
    assert restored.region_id == series.region_id
    assert restored.dimension == series.dimension
    assert np.array_equal(restored.valid_mask, series.valid_mask)
    assert restored.sampling_rate == series.sampling_rate


def test_subject_sequence_serialization_is_deterministic_and_preserves_invalid_evidence() -> None:
    series = _assembled_series()

    encoded_once = dumps_core_contract(series)
    restored = loads_core_contract(encoded_once)
    encoded_twice = dumps_core_contract(restored)

    assert encoded_twice == encoded_once
    assert np.isnan(restored.X[1, 0, 1])
    assert not restored.valid_mask[1, 0, 1]
    assert restored.X.shape == series.X.shape
