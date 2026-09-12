from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lagged_facial_graph_forecasting.landmark_io import (
    LandmarkReadError,
    RawLandmarkSequence,
    read_landmark_file,
)


def test_reads_npy_without_imposing_shape_semantics(tmp_path: Path) -> None:
    landmarks = np.arange(24, dtype=float).reshape(2, 4, 3)
    path = tmp_path / "subject01.npy"
    np.save(path, landmarks)

    loaded = read_landmark_file(path)

    assert isinstance(loaded, RawLandmarkSequence)
    assert np.array_equal(loaded.landmarks, landmarks)
    assert loaded.timestamps is None
    assert loaded.source_path == path.as_posix()
    assert not loaded.landmarks.flags.writeable


def test_reads_npz_landmarks_and_optional_timestamps(tmp_path: Path) -> None:
    landmarks = np.arange(40, dtype=np.float64).reshape(5, 4, 2)
    timestamps = np.array([0.0, 0.033, 0.066, 0.099, 0.132])
    path = tmp_path / "subject02.npz"
    np.savez(path, landmarks=landmarks, timestamps=timestamps)

    loaded = read_landmark_file(path)

    assert np.array_equal(loaded.landmarks, landmarks)
    assert np.array_equal(loaded.timestamps, timestamps)
    assert loaded.timestamps is not None
    assert not loaded.timestamps.flags.writeable


def test_npz_allows_custom_keys_without_validating_semantics(tmp_path: Path) -> None:
    landmarks = np.array([[1.0, 2.0, 3.0]])
    timestamps = np.array([10.0])
    path = tmp_path / "custom.npz"
    np.savez(path, points=landmarks, time=timestamps)

    loaded = read_landmark_file(
        path,
        landmark_key="points",
        timestamp_key="time",
    )

    assert np.array_equal(loaded.landmarks, landmarks)
    assert np.array_equal(loaded.timestamps, timestamps)


def test_missing_npz_landmark_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "missing.npz"
    np.savez(path, timestamps=np.arange(3, dtype=float))

    with pytest.raises(LandmarkReadError, match="missing required key"):
        read_landmark_file(path)


def test_unsupported_extension_and_missing_file_are_rejected(tmp_path: Path) -> None:
    text_path = tmp_path / "landmarks.csv"
    text_path.write_text("0,1,2\n", encoding="utf-8")

    with pytest.raises(LandmarkReadError, match="unsupported"):
        read_landmark_file(text_path)
    with pytest.raises(LandmarkReadError, match="does not exist"):
        read_landmark_file(tmp_path / "absent.npy")


def test_pickle_backed_object_arrays_are_not_loaded(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.npy"
    np.save(path, np.array([{"x": 1}], dtype=object), allow_pickle=True)

    with pytest.raises(LandmarkReadError, match="failed to read"):
        read_landmark_file(path)
