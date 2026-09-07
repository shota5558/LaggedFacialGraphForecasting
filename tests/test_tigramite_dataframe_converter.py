from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting.contracts import (
    FaceTimeSeries,
    InnerFold,
    SplitManifest,
)
from lagged_facial_graph_forecasting.leakage_guard import LeakageGuardError
from lagged_facial_graph_forecasting.tigramite_adapter import (
    TigramiteAdapterError,
    face_time_series_to_tigramite_dataframe,
)


def _manifest() -> SplitManifest:
    return SplitManifest(
        outer_fold=0,
        train_subject_ids=("s1", "s2", "s3"),
        test_subject_ids=("outer_test",),
        inner_folds=(
            InnerFold(
                inner_train_subject_ids=("s1", "s2"),
                inner_val_subject_ids=("s3",),
            ),
        ),
        seed=17,
    )


def _series(subject_id: str, *, length: int, sampling_rate: float = 25.0) -> FaceTimeSeries:
    values = np.arange(length * 4, dtype=float).reshape(length, 2, 2)
    valid = np.ones_like(values, dtype=bool)
    if subject_id == "s2":
        values[1, 0, 1] = np.nan
        valid[1, 0, 1] = False
    return FaceTimeSeries(
        X=values,
        subject_id=subject_id,
        time_index=np.arange(length, dtype=float) / sampling_rate,
        region_id=("left_eye", "mouth"),
        dimension=("vx", "vy"),
        valid_mask=valid,
        sampling_rate=sampling_rate,
    )


def test_converter_keeps_subjects_as_separate_tigramite_datasets() -> None:
    bundle = face_time_series_to_tigramite_dataframe(
        _manifest(),
        (_series("s1", length=4), _series("s2", length=5)),
    )

    frame = bundle.dataframe
    assert frame.analysis_mode == "multiple"
    assert frame.datasets == ["s1", "s2"]
    assert frame.T == {"s1": 4, "s2": 5}
    assert frame.N == 4
    assert bundle.subject_ids == ("s1", "s2")
    assert bundle.variable_names == (
        "left_eye.vx",
        "left_eye.vy",
        "mouth.vx",
        "mouth.vy",
    )
    assert bundle.variable_components == (
        ("left_eye", "vx"),
        ("left_eye", "vy"),
        ("mouth", "vx"),
        ("mouth", "vy"),
    )


def test_invalid_values_are_finite_placeholders_under_tigramite_mask() -> None:
    bundle = face_time_series_to_tigramite_dataframe(
        _manifest(),
        (_series("s1", length=4), _series("s2", length=5)),
    )

    frame = bundle.dataframe
    # [region=left_eye, dimension=vy] maps to flattened column 1.
    assert frame.values["s2"][1, 1] == 0.0
    assert bool(frame.mask["s2"][1, 1]) is True
    assert np.isfinite(frame.values["s2"]).all()
    assert not frame.mask["s2"][0, 1]


def test_converter_rejects_outer_test_before_dataframe_construction() -> None:
    with pytest.raises(LeakageGuardError, match="outer-test subjects"):
        face_time_series_to_tigramite_dataframe(
            _manifest(),
            (_series("outer_test", length=4),),
        )


def test_converter_rejects_subject_or_schema_mixing() -> None:
    manifest = _manifest()
    s1 = _series("s1", length=4)
    with pytest.raises(TigramiteAdapterError, match="subject_id values must be unique"):
        face_time_series_to_tigramite_dataframe(manifest, (s1, s1))

    incompatible_rate = _series("s2", length=4, sampling_rate=30.0)
    with pytest.raises(TigramiteAdapterError, match="sampling_rate"):
        face_time_series_to_tigramite_dataframe(manifest, (s1, incompatible_rate))

    incompatible_region = FaceTimeSeries(
        X=np.zeros((4, 2, 2)),
        subject_id="s3",
        time_index=np.arange(4, dtype=float) / 25.0,
        region_id=("mouth", "left_eye"),
        dimension=("vx", "vy"),
        valid_mask=np.ones((4, 2, 2), dtype=bool),
        sampling_rate=25.0,
    )
    with pytest.raises(TigramiteAdapterError, match="region_id order"):
        face_time_series_to_tigramite_dataframe(manifest, (s1, incompatible_region))
