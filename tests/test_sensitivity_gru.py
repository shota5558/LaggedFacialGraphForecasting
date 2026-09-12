from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from lagged_facial_graph_forecasting.contracts import DesignMatrix, InnerFold, PredictionArtifact, SplitManifest
from lagged_facial_graph_forecasting.core_contracts import ExperimentConfig
from lagged_facial_graph_forecasting.sensitivity_execution import (
    PRIMARY_FREEZE_MANIFEST_PATH,
    SensitivityExecutionError,
    SensitivityExecutionMode,
    SensitivityExperimentConfig,
    load_sensitivity_experiment_config,
)
from lagged_facial_graph_forecasting.sensitivity_gru import (
    GRUConfig,
    SensitivityGRUError,
    fit_sensitivity_gru,
    predict_sensitivity_gru,
)


ROOT = Path(__file__).resolve().parents[1]


def _execution():
    return load_sensitivity_experiment_config(
        ROOT / "configs/sensitivity_preimplementation.yaml", repository_root=ROOT
    )


def _real_execution() -> SensitivityExperimentConfig:
    return SensitivityExperimentConfig(
        experiment=ExperimentConfig(
            experiment_id="gru-real-barrier-test",
            scientific_config_path="configs/scientific_freeze.yaml",
            run_config_path="configs/sensitivity_preimplementation.yaml",
            artifact_root="artifacts/sensitivity",
            seed=1,
        ),
        namespace="sensitivity",
        execution_mode=SensitivityExecutionMode.REAL,
        primary_freeze_manifest=PRIMARY_FREEZE_MANIFEST_PATH,
    )


def _manifest() -> SplitManifest:
    return SplitManifest(
        outer_fold=0,
        train_subject_ids=("train_a", "train_b", "validation"),
        test_subject_ids=("outer_test",),
        inner_folds=(InnerFold(inner_train_subject_ids=("train_a", "train_b"), inner_val_subject_ids=("validation",)),),
        seed=701,
    )


def _matrix(subject_ids: tuple[str, ...], *, rows_per_subject: int = 24) -> DesignMatrix:
    X_parts = []
    y_parts = []
    subject_rows: list[str] = []
    region_rows: list[str] = []
    forecast_parts = []
    target_parts = []
    for subject_index, subject_id in enumerate(subject_ids):
        t = np.arange(rows_per_subject, dtype=float)
        x0 = np.sin(0.21 * t) + 0.03 * subject_index
        x1 = np.cos(0.17 * t) - 0.02 * subject_index
        X = np.column_stack((x0, x1))
        y = 0.65 * x0 - 0.35 * x1 + 0.05
        X_parts.append(X)
        y_parts.append(y)
        subject_rows.extend([subject_id] * rows_per_subject)
        region_rows.extend(["mouth"] * rows_per_subject)
        base = subject_index * 10.0
        forecast_parts.append(base + t / 25.0)
        target_parts.append(base + (t + 1.0) / 25.0)

    X_all = np.vstack(X_parts)
    y_all = np.concatenate(y_parts)
    return DesignMatrix(
        X=X_all,
        y=y_all,
        subject_id=tuple(subject_rows),
        region_id=tuple(region_rows),
        target_dimensions=("value",),
        forecast_origin=np.concatenate(forecast_parts),
        target_time=np.concatenate(target_parts),
        feature_names=("mouth.value", "cheek.value"),
        feature_lags=(1, 1),
        valid_mask=np.ones(X_all.shape[0], dtype=bool),
    )


def _config(seed: int = 91) -> GRUConfig:
    # Unit tests exercise contracts and data flow, not optimizer convergence.
    return GRUConfig(
        sequence_length=3,
        hidden_size=8,
        num_layers=1,
        learning_rate=0.03,
        weight_decay=0.0,
        epochs=3,
        seed=seed,
    )


def test_gru_tiny_synthetic_fit_and_prediction_artifact_compatibility() -> None:
    execution = _execution()
    manifest = _manifest()
    train = _matrix(("train_a", "train_b"))
    validation = _matrix(("validation",))
    test = _matrix(("outer_test",))
    fitted = fit_sensitivity_gru(
        execution,
        manifest,
        train,
        validation,
        inner_fold_index=0,
        config=_config(),
        repository_root=ROOT,
    )
    artifact = predict_sensitivity_gru(
        execution, fitted, manifest, test, repository_root=ROOT
    )

    assert fitted.training_sequence_count == 2 * (24 - 2)
    assert fitted.validation_sequence_count == 24 - 2
    assert np.isfinite(fitted.final_training_loss)
    assert np.isfinite(fitted.final_validation_loss)
    assert isinstance(artifact, PredictionArtifact)
    assert artifact.condition == "gru"
    assert artifact.outer_fold == manifest.outer_fold
    assert artifact.target_dimensions == test.target_dimensions
    assert artifact.subject_id == test.subject_id
    assert np.count_nonzero(artifact.valid_mask) == 24 - 2
    assert np.all(np.isfinite(artifact.y_pred[artifact.valid_mask]))
    assert np.all(np.isnan(artifact.y_pred[~artifact.valid_mask]))


def test_gru_fit_rejects_outer_test_or_wrong_inner_scope() -> None:
    execution = _execution()
    manifest = _manifest()
    validation = _matrix(("validation",))
    with pytest.raises(SensitivityGRUError, match="inner-train"):
        fit_sensitivity_gru(
            execution,
            manifest,
            _matrix(("train_a", "outer_test")),
            validation,
            inner_fold_index=0,
            config=_config(),
            repository_root=ROOT,
        )


def test_gru_prediction_rejects_non_outer_test_subjects() -> None:
    execution = _execution()
    manifest = _manifest()
    fitted = fit_sensitivity_gru(
        execution,
        manifest,
        _matrix(("train_a", "train_b")),
        _matrix(("validation",)),
        inner_fold_index=0,
        config=_config(),
        repository_root=ROOT,
    )
    with pytest.raises(SensitivityGRUError, match="outer-test"):
        predict_sensitivity_gru(
            execution,
            fitted,
            manifest,
            _matrix(("validation",)),
            repository_root=ROOT,
        )


def test_gru_real_fit_fails_closed_before_primary_freeze(tmp_path) -> None:
    with pytest.raises(SensitivityExecutionError, match="PRIMARY FREEZE"):
        fit_sensitivity_gru(
            _real_execution(),
            _manifest(),
            _matrix(("train_a", "train_b")),
            _matrix(("validation",)),
            inner_fold_index=0,
            config=_config(),
            repository_root=tmp_path,
        )
