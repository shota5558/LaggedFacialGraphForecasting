from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from lagged_facial_graph_forecasting.contracts import (
    DesignMatrix,
    InnerFold,
    PredictionArtifact,
    SplitManifest,
)
from lagged_facial_graph_forecasting.sensitivity_gru import (
    GRUConfig,
    SensitivityGRUError,
    fit_sensitivity_gru,
    predict_sensitivity_gru,
)


def _manifest() -> SplitManifest:
    return SplitManifest(
        outer_fold=0,
        train_subject_ids=("train_a", "train_b", "validation"),
        test_subject_ids=("outer_test",),
        inner_folds=(
            InnerFold(
                inner_train_subject_ids=("train_a", "train_b"),
                inner_val_subject_ids=("validation",),
            ),
        ),
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
    return GRUConfig(
        sequence_length=3,
        hidden_size=8,
        num_layers=1,
        learning_rate=0.03,
        weight_decay=0.0,
        epochs=160,
        seed=seed,
    )


def test_gru_tiny_synthetic_fit_and_prediction_artifact_compatibility() -> None:
    manifest = _manifest()
    train = _matrix(("train_a", "train_b"))
    validation = _matrix(("validation",))
    test = _matrix(("outer_test",))

    fitted = fit_sensitivity_gru(
        manifest, train, validation, inner_fold_index=0, config=_config()
    )
    artifact = predict_sensitivity_gru(fitted, manifest, test)

    assert fitted.training_sequence_count == 2 * (24 - 2)
    assert fitted.validation_sequence_count == 24 - 2
    assert np.isfinite(fitted.final_training_loss)
    assert fitted.final_training_loss < 0.01
    assert np.isfinite(fitted.final_validation_loss)
    assert isinstance(artifact, PredictionArtifact)
    assert artifact.condition == "gru"
    assert artifact.outer_fold == manifest.outer_fold
    assert artifact.target_dimensions == test.target_dimensions
    assert artifact.subject_id == test.subject_id
    assert np.count_nonzero(artifact.valid_mask) == 24 - 2
    assert np.all(np.isfinite(artifact.y_pred[artifact.valid_mask]))
    assert np.all(np.isnan(artifact.y_pred[~artifact.valid_mask]))


def test_gru_same_seed_gives_deterministic_predictions() -> None:
    manifest = _manifest()
    train = _matrix(("train_a", "train_b"))
    validation = _matrix(("validation",))
    test = _matrix(("outer_test",))

    first = fit_sensitivity_gru(
        manifest, train, validation, inner_fold_index=0, config=_config(seed=222)
    )
    second = fit_sensitivity_gru(
        manifest, train, validation, inner_fold_index=0, config=_config(seed=222)
    )
    first_prediction = predict_sensitivity_gru(first, manifest, test)
    second_prediction = predict_sensitivity_gru(second, manifest, test)

    np.testing.assert_array_equal(first_prediction.valid_mask, second_prediction.valid_mask)
    np.testing.assert_allclose(
        first_prediction.y_pred[first_prediction.valid_mask],
        second_prediction.y_pred[second_prediction.valid_mask],
        rtol=0.0,
        atol=0.0,
    )
    assert first.final_training_loss == second.final_training_loss
    assert first.final_validation_loss == second.final_validation_loss


def test_gru_fit_rejects_outer_test_or_wrong_inner_scope() -> None:
    manifest = _manifest()
    validation = _matrix(("validation",))

    with pytest.raises(SensitivityGRUError, match="inner-train"):
        fit_sensitivity_gru(
            manifest,
            _matrix(("train_a", "outer_test")),
            validation,
            inner_fold_index=0,
            config=_config(),
        )


def test_gru_prediction_rejects_non_outer_test_subjects() -> None:
    manifest = _manifest()
    fitted = fit_sensitivity_gru(
        manifest,
        _matrix(("train_a", "train_b")),
        _matrix(("validation",)),
        inner_fold_index=0,
        config=_config(),
    )
    with pytest.raises(SensitivityGRUError, match="outer-test"):
        predict_sensitivity_gru(fitted, manifest, _matrix(("validation",)))
