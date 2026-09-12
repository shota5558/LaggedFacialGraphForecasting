"""Fold-level data access lock for scientific dry-run and Primary execution.

The lock makes outer-test access an explicit state transition.  Discovery,
preprocessing fitting, Ridge tuning, and Null construction consume only the
outer-train accessor; outer-test series are unavailable until the caller has
completed and frozen all fold settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .contracts import FaceTimeSeries, SplitManifest
from .runner import OuterTestLockedError


@dataclass(slots=True)
class ScientificFoldDataLock:
    """Expose outer-train immediately and outer-test only after one-way freeze."""

    manifest: SplitManifest
    series_by_subject: Mapping[str, FaceTimeSeries]
    _state: str = field(init=False, default="locked")

    def __post_init__(self) -> None:
        provided = dict(self.series_by_subject)
        required = set(self.manifest.train_subject_ids) | set(self.manifest.test_subject_ids)
        missing = sorted(required - set(provided))
        if missing:
            raise ValueError(f"missing FaceTimeSeries for manifest subjects: {missing}")
        for subject_id, series in provided.items():
            if subject_id != series.subject_id:
                raise ValueError(
                    f"series_by_subject key {subject_id!r} does not match "
                    f"FaceTimeSeries.subject_id {series.subject_id!r}"
                )
        self.series_by_subject = provided

    @property
    def outer_test_locked(self) -> bool:
        return self._state != "frozen"

    def outer_train_series(self) -> tuple[FaceTimeSeries, ...]:
        """Return exactly the frozen manifest's outer-train subjects."""

        if self._state != "locked":
            raise RuntimeError("outer-train selection stage is closed after fold freeze")
        return tuple(self.series_by_subject[sid] for sid in self.manifest.train_subject_ids)

    def freeze(self) -> None:
        """Irreversibly close selection and unlock one outer-test evaluation stage."""

        if self._state != "locked":
            raise RuntimeError("fold may be frozen exactly once")
        self._state = "frozen"

    def outer_test_series(self) -> tuple[FaceTimeSeries, ...]:
        """Return outer-test series only after all fold settings have been frozen."""

        if self._state != "frozen":
            raise OuterTestLockedError(
                "outer-test is locked until preprocessing, discovery, tuning, "
                "and Null construction are frozen"
            )
        return tuple(self.series_by_subject[sid] for sid in self.manifest.test_subject_ids)

    def close_after_evaluation(self) -> None:
        """Prevent a second outer-test evaluation pass from becoming adaptive."""

        if self._state != "frozen":
            raise RuntimeError("outer-test can be closed only after fold freeze")
        self._state = "evaluated"
