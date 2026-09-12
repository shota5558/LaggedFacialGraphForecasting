"""Thin readers for already-extracted facial landmark arrays.

A-00 deliberately does not run a landmark detector and does not impose timestamp
or landmark-shape semantics. Those checks belong to A-01 and A-02 respectively.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class LandmarkReadError(ValueError):
    """Raised when an extracted-landmark container cannot be read safely."""


@dataclass(frozen=True, slots=True)
class RawLandmarkSequence:
    """Extractor-neutral bytes-to-array result prior to semantic validation."""

    landmarks: np.ndarray
    timestamps: np.ndarray | None
    source_path: str

    def __post_init__(self) -> None:
        landmarks = np.array(self.landmarks, copy=True)
        timestamps = (
            None if self.timestamps is None else np.array(self.timestamps, copy=True)
        )
        landmarks.setflags(write=False)
        if timestamps is not None:
            timestamps.setflags(write=False)
        object.__setattr__(self, "landmarks", landmarks)
        object.__setattr__(self, "timestamps", timestamps)


def read_landmark_file(
    path: str | Path,
    *,
    landmark_key: str = "landmarks",
    timestamp_key: str = "timestamps",
) -> RawLandmarkSequence:
    """Read an extracted landmark array from ``.npy`` or ``.npz``.

    ``.npy`` contains only the landmark array. ``.npz`` must contain
    ``landmark_key`` and may contain ``timestamp_key``. Loading always uses
    ``allow_pickle=False`` so the reader never executes serialized Python objects.

    No timestamp monotonicity, landmark rank/shape, coordinate dimension, missing
    value, or face-region assumptions are made here; later Lane-A tasks own those
    contracts explicitly.
    """

    input_path = Path(path)
    if not input_path.is_file():
        raise LandmarkReadError(f"landmark file does not exist: {input_path}")
    if not isinstance(landmark_key, str) or not landmark_key.strip():
        raise LandmarkReadError("landmark_key must be a non-empty string")
    if not isinstance(timestamp_key, str) or not timestamp_key.strip():
        raise LandmarkReadError("timestamp_key must be a non-empty string")

    suffix = input_path.suffix.lower()
    try:
        if suffix == ".npy":
            landmarks = np.load(input_path, allow_pickle=False)
            timestamps = None
        elif suffix == ".npz":
            with np.load(input_path, allow_pickle=False) as archive:
                if landmark_key not in archive.files:
                    raise LandmarkReadError(
                        f"NPZ landmark file is missing required key {landmark_key!r}"
                    )
                landmarks = np.array(archive[landmark_key], copy=True)
                timestamps = (
                    np.array(archive[timestamp_key], copy=True)
                    if timestamp_key in archive.files
                    else None
                )
        else:
            raise LandmarkReadError(
                "unsupported landmark file extension; expected .npy or .npz"
            )
    except (OSError, ValueError) as exc:
        if isinstance(exc, LandmarkReadError):
            raise
        raise LandmarkReadError(f"failed to read landmark file: {input_path}") from exc

    return RawLandmarkSequence(
        landmarks=landmarks,
        timestamps=timestamps,
        source_path=input_path.as_posix(),
    )
