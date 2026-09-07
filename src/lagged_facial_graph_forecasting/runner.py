"""Minimal one-fold orchestration for the V0 vertical slice."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .artifacts import write_v0_result_json
from .contracts import DesignMatrix, FaceTimeSeries, PredictionArtifact, SplitManifest
from .design_matrix import build_self_history_design_matrix
from .forecaster import (
    FittedRidgeForecaster,
    fit_ridge_forecaster,
    predict_ridge_forecaster,
)
from .metrics import velocity_rmse


class OuterTestLockedError(RuntimeError):
    """Raised when outer-test evaluation is requested before freeze or after use."""


@dataclass(frozen=True, slots=True)
class FoldRunResult:
    """Outputs of one frozen V0 outer-test evaluation."""

    prediction: PredictionArtifact
    velocity_rmse: float
    artifact_path: Path


def _concatenate_design_matrices(matrices: Iterable[DesignMatrix]) -> DesignMatrix:
    items = tuple(matrices)
    if not items:
        raise ValueError("at least one DesignMatrix is required")

    first = items[0]
    for index, matrix in enumerate(items[1:], start=1):
        if matrix.feature_names != first.feature_names:
            raise ValueError(f"matrix {index} feature_names do not match")
        if matrix.feature_lags != first.feature_lags:
            raise ValueError(f"matrix {index} feature_lags do not match")
        if matrix.y.ndim != first.y.ndim or matrix.y.shape[1:] != first.y.shape[1:]:
            raise ValueError(f"matrix {index} target shape does not match")

    return DesignMatrix(
        X=np.concatenate([matrix.X for matrix in items], axis=0),
        y=np.concatenate([matrix.y for matrix in items], axis=0),
        subject_id=tuple(
            subject_id for matrix in items for subject_id in matrix.subject_id
        ),
        region_id=tuple(region_id for matrix in items for region_id in matrix.region_id),
        forecast_origin=np.concatenate(
            [matrix.forecast_origin for matrix in items], axis=0
        ),
        target_time=np.concatenate([matrix.target_time for matrix in items], axis=0),
        feature_names=first.feature_names,
        feature_lags=first.feature_lags,
        valid_mask=np.concatenate([matrix.valid_mask for matrix in items], axis=0),
    )


@dataclass(slots=True)
class MinimalFoldRunner:
    """V0 runner with explicit outer-test lock and one-shot frozen evaluation."""

    manifest: SplitManifest
    series_by_subject: Mapping[str, FaceTimeSeries]
    _state: str = field(init=False, default="locked")
    _fitted: FittedRidgeForecaster | None = field(init=False, default=None)
    _target_region: str | None = field(init=False, default=None)
    _lags: tuple[int, ...] | None = field(init=False, default=None)
    _training_matrix: DesignMatrix | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        provided = dict(self.series_by_subject)
        required = set(self.manifest.train_subject_ids) | set(self.manifest.test_subject_ids)
        missing = sorted(required - set(provided))
        if missing:
            raise ValueError(f"missing FaceTimeSeries for manifest subjects: {missing}")
        for key, series in provided.items():
            if key != series.subject_id:
                raise ValueError(
                    f"series_by_subject key {key!r} does not match "
                    f"FaceTimeSeries.subject_id {series.subject_id!r}"
                )
        self.series_by_subject = provided

    @property
    def outer_test_locked(self) -> bool:
        return self._state != "frozen"

    @property
    def training_subject_ids(self) -> tuple[str, ...]:
        if self._training_matrix is None:
            return ()
        return tuple(dict.fromkeys(self._training_matrix.subject_id))

    def fit_self_history(
        self,
        *,
        target_region: str,
        lags: Iterable[int] = (1,),
        alpha: float = 1.0,
    ) -> FittedRidgeForecaster:
        """Fit on outer-train subjects only; outer-test remains locked."""

        if self._state != "locked" or self._fitted is not None:
            raise RuntimeError("fold training may only be performed once before freeze")

        normalized_lags = tuple(lags)
        train_matrices = tuple(
            build_self_history_design_matrix(
                self.series_by_subject[subject_id],
                target_region=target_region,
                lags=normalized_lags,
                horizon=1,
            )
            for subject_id in self.manifest.train_subject_ids
        )
        training_matrix = _concatenate_design_matrices(train_matrices)

        observed_training_subjects = set(training_matrix.subject_id)
        expected_training_subjects = set(self.manifest.train_subject_ids)
        if observed_training_subjects != expected_training_subjects:
            raise RuntimeError("training matrix subject provenance does not match outer-train")
        if observed_training_subjects & set(self.manifest.test_subject_ids):
            raise RuntimeError("outer-test subject entered the training matrix")

        self._training_matrix = training_matrix
        self._fitted = fit_ridge_forecaster(training_matrix, alpha=alpha)
        self._target_region = target_region
        self._lags = tuple(sorted(set(self._fitted.feature_lags)))
        return self._fitted

    def freeze(self) -> None:
        """Freeze fitted V0 settings and unlock exactly one outer-test evaluation."""

        if self._state != "locked" or self._fitted is None:
            raise RuntimeError("fold can be frozen only after successful training")
        self._state = "frozen"

    def evaluate_outer_test(self, output_path: str | Path) -> FoldRunResult:
        """Evaluate frozen Self-history Ridge on outer-test exactly once."""

        if self._state != "frozen":
            raise OuterTestLockedError(
                "outer-test is locked until fold freeze and after final evaluation"
            )
        if self._fitted is None or self._target_region is None or self._lags is None:
            raise RuntimeError("frozen runner is missing fitted state")

        test_matrices = tuple(
            build_self_history_design_matrix(
                self.series_by_subject[subject_id],
                target_region=self._target_region,
                lags=self._lags,
                horizon=1,
            )
            for subject_id in self.manifest.test_subject_ids
        )
        test_matrix = _concatenate_design_matrices(test_matrices)

        prediction = predict_ridge_forecaster(
            self._fitted,
            test_matrix,
            outer_fold=self.manifest.outer_fold,
            condition="self",
        )
        metric = velocity_rmse(prediction)
        artifact_path = write_v0_result_json(
            output_path,
            matrix=test_matrix,
            prediction=prediction,
            fitted=self._fitted,
            velocity_rmse_value=metric,
        )

        self._state = "evaluated"
        return FoldRunResult(
            prediction=prediction,
            velocity_rmse=metric,
            artifact_path=artifact_path,
        )
