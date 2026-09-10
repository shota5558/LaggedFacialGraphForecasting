from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting import ContractError, FaceTimeSeries


def _valid_kwargs() -> dict:
    X = np.arange(24, dtype=float).reshape(4, 3, 2)
    return {
        "X": X,
        "subject_id": "subject-001",
        "time_index": np.array([0.0, 1.0, 2.0, 3.0]),
        "region_id": ("left_eye", "mouth", "jaw"),
        "dimension": ("vx", "vy"),
        "valid_mask": np.ones_like(X, dtype=bool),
        "sampling_rate": 30.0,
    }


def test_face_time_series_accepts_canonical_shape_and_metadata() -> None:
    series = FaceTimeSeries(**_valid_kwargs())

    assert series.X.shape == (4, 3, 2)
    assert series.subject_id == "subject-001"
    assert series.time_index.shape == (4,)
    assert series.region_id == ("left_eye", "mouth", "jaw")
    assert series.dimension == ("vx", "vy")
    assert series.valid_mask.shape == series.X.shape
    assert series.valid_mask.dtype == np.bool_
    assert series.sampling_rate == 30.0


def test_rejects_non_tkd_X() -> None:
    kwargs = _valid_kwargs()
    kwargs["X"] = np.zeros((4, 3), dtype=float)

    with pytest.raises(ContractError, match="shape \\(T, K, D\\)"):
        FaceTimeSeries(**kwargs)


def test_rejects_time_index_length_mismatch() -> None:
    kwargs = _valid_kwargs()
    kwargs["time_index"] = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ContractError, match="time_index must have shape"):
        FaceTimeSeries(**kwargs)


def test_rejects_non_increasing_time_index() -> None:
    kwargs = _valid_kwargs()
    kwargs["time_index"] = np.array([0.0, 1.0, 1.0, 2.0])

    with pytest.raises(ContractError, match="strictly increasing"):
        FaceTimeSeries(**kwargs)


def test_rejects_region_count_mismatch() -> None:
    kwargs = _valid_kwargs()
    kwargs["region_id"] = ("left_eye", "mouth")

    with pytest.raises(ContractError, match="region_id must contain 3 entries"):
        FaceTimeSeries(**kwargs)


def test_rejects_duplicate_region_ids() -> None:
    kwargs = _valid_kwargs()
    kwargs["region_id"] = ("mouth", "mouth", "jaw")

    with pytest.raises(ContractError, match="region_id entries must be unique"):
        FaceTimeSeries(**kwargs)


def test_rejects_dimension_count_mismatch() -> None:
    kwargs = _valid_kwargs()
    kwargs["dimension"] = ("vx",)

    with pytest.raises(ContractError, match="dimension must contain 2 entries"):
        FaceTimeSeries(**kwargs)


def test_rejects_valid_mask_shape_mismatch() -> None:
    kwargs = _valid_kwargs()
    kwargs["valid_mask"] = np.ones((4, 3), dtype=bool)

    with pytest.raises(ContractError, match="valid_mask must match"):
        FaceTimeSeries(**kwargs)


def test_rejects_non_boolean_valid_mask() -> None:
    kwargs = _valid_kwargs()
    kwargs["valid_mask"] = np.ones((4, 3, 2), dtype=np.int8)

    with pytest.raises(ContractError, match="boolean dtype"):
        FaceTimeSeries(**kwargs)


@pytest.mark.parametrize("sampling_rate", [0.0, -1.0, np.inf, np.nan])
def test_rejects_invalid_sampling_rate(sampling_rate: float) -> None:
    kwargs = _valid_kwargs()
    kwargs["sampling_rate"] = sampling_rate

    with pytest.raises(ContractError, match="sampling_rate"):
        FaceTimeSeries(**kwargs)
