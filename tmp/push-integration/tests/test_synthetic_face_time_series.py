from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting import (
    FaceTimeSeries,
    generate_synthetic_face_time_series,
)


def test_generator_returns_canonical_face_time_series() -> None:
    series = generate_synthetic_face_time_series(
        subject_id="synthetic-001",
        n_steps=32,
        region_ids=("eye", "mouth"),
        dimensions=("vx", "vy"),
        sampling_rate=25.0,
        seed=17,
    )

    assert isinstance(series, FaceTimeSeries)
    assert series.X.shape == (32, 2, 2)
    assert series.subject_id == "synthetic-001"
    assert series.region_id == ("eye", "mouth")
    assert series.dimension == ("vx", "vy")
    assert np.array_equal(series.time_index, np.arange(32, dtype=float))
    assert np.all(series.valid_mask)
    assert series.sampling_rate == 25.0


def test_generator_is_deterministic_for_same_seed() -> None:
    first = generate_synthetic_face_time_series(subject_id="s01", seed=123)
    second = generate_synthetic_face_time_series(subject_id="s01", seed=123)

    assert np.array_equal(first.X, second.X)
    assert np.array_equal(first.time_index, second.time_index)


def test_generator_changes_signal_for_different_seed() -> None:
    first = generate_synthetic_face_time_series(subject_id="s01", seed=123)
    second = generate_synthetic_face_time_series(subject_id="s01", seed=124)

    assert not np.array_equal(first.X, second.X)


@pytest.mark.parametrize("n_steps", [0, 1, -1])
def test_rejects_too_short_sequence(n_steps: int) -> None:
    with pytest.raises(ValueError, match="n_steps"):
        generate_synthetic_face_time_series(subject_id="s01", n_steps=n_steps)


@pytest.mark.parametrize("coefficient", [-1.0, 1.0, 1.2, np.inf, np.nan])
def test_rejects_unstable_autoregressive_coefficient(coefficient: float) -> None:
    with pytest.raises(ValueError, match="autoregressive_coefficient"):
        generate_synthetic_face_time_series(
            subject_id="s01", autoregressive_coefficient=coefficient
        )


def test_rejects_invalid_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        generate_synthetic_face_time_series(subject_id="s01", seed=-1)
