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
from lagged_facial_graph_forecasting.quality import build_quality_flags
from lagged_facial_graph_forecasting.sequence_assembly import assemble_subject_sequence


def test_quality_invalid_finite_frame_is_never_revalidated_by_a11_to_a15_chain() -> None:
    values = np.arange(24, dtype=np.float64).reshape(3, 4, 2)
    original = values.copy()

    quality = build_quality_flags(
        values,
        frame_quality=np.array([0.9, 0.2, 0.95], dtype=np.float64),
        minimum_frame_quality=0.5,
    )
    assert np.isfinite(values[1]).all()
    assert not quality.valid_mask[1].any()

    missingness = handle_missing_frames(values, valid_mask=quality.valid_mask)
    processed = apply_interpolation_policy(
        missingness,
        policy=InterpolationPolicy(method="none"),
    )
    series = assemble_subject_sequence(
        processed,
        subject_id="subject-01",
        time_index=np.array([0.0, 0.04, 0.08], dtype=np.float64),
        region_id=("left_eye", "right_eye", "mouth", "jaw"),
        dimension=("vx", "vy"),
        sampling_rate=25.0,
    )
    restored = loads_core_contract(dumps_core_contract(series))

    assert np.array_equal(restored.X, original)
    assert not restored.valid_mask[1].any()
    assert restored.valid_mask[0].all()
    assert restored.valid_mask[2].all()
    assert not processed.interpolated_mask.any()
