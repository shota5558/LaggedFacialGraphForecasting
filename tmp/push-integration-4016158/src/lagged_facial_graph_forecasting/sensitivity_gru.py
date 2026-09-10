"""PyTorch GRU forecaster adapter for Sensitivity only (S-IMPL-07).

The recurrent implementation is delegated to ``torch.nn.GRU``. This module only
provides contract adaptation, subject-scope guards, deterministic fitting, sequence
window construction, and PredictionArtifact conversion. Every public fit/predict
entrypoint requires the common Sensitivity execution boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .contracts import DesignMatrix, PredictionArtifact, SplitManifest
from .sensitivity_execution import (
    SensitivityExperimentConfig,
    assert_sensitivity_execution_allowed,
)


SENSITIVITY_GRU_SCHEMA_VERSION = 1
SENSITIVITY_GRU_METHOD = "gru"


class SensitivityGRUError(ValueError):
    """Raised when the GRU Sensitivity adapter violates scope or data contracts."""


@dataclass(frozen=True, slots=True)
class GRUConfig:
    sequence_length: int = 4
    hidden_size: int = 8
    num_layers: int = 1
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    epochs: int = 100
    seed: int = 0
    schema_version: int = SENSITIVITY_GRU_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for value, name in (
            (self.sequence_length, "sequence_length"),
            (self.hidden_size, "hidden_size"),
            (self.num_layers, "num_layers"),
            (self.epochs, "epochs"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise SensitivityGRUError(f"{name} must be a positive integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise SensitivityGRUError("seed must be a non-negative integer")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise SensitivityGRUError("learning_rate must be finite and > 0")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise SensitivityGRUError("weight_decay must be finite and >= 0")
        if self.schema_version != SENSITIVITY_GRU_SCHEMA_VERSION:
            raise SensitivityGRUError("unsupported GRU config schema_version")


class _TorchGRURegressor(nn.Module):
    def __init__(self, input_size: int, output_size: int, config: GRUConfig) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(config.hidden_size, output_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(values)
        return self.head(output[:, -1, :])


@dataclass(slots=True)
class FittedSensitivityGRU:
    model: _TorchGRURegressor
    config: GRUConfig
    feature_names: tuple[str, ...]
    feature_lags: tuple[int, ...]
    target_dimensions: tuple[str, ...]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    train_subject_ids: tuple[str, ...]
    validation_subject_ids: tuple[str, ...]
    training_sequence_count: int
    validation_sequence_count: int
    final_training_loss: float
    final_validation_loss: float
    method: str = SENSITIVITY_GRU_METHOD
    schema_version: int = SENSITIVITY_GRU_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class _SequenceBatch:
    X: np.ndarray
    y: np.ndarray
    row_indices: np.ndarray


def _unique_subjects(matrix: DesignMatrix) -> tuple[str, ...]:
    return tuple(dict.fromkeys(matrix.subject_id))


def _assert_exact_inner_fold_scope(
    manifest: SplitManifest,
    train_matrix: DesignMatrix,
    validation_matrix: DesignMatrix,
    *,
    inner_fold_index: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(manifest, SplitManifest):
        raise SensitivityGRUError("manifest must be a SplitManifest")
    if (
        not isinstance(inner_fold_index, int)
        or isinstance(inner_fold_index, bool)
        or not 0 <= inner_fold_index < len(manifest.inner_folds)
    ):
        raise SensitivityGRUError("inner_fold_index is out of range")

    train_subjects = _unique_subjects(train_matrix)
    validation_subjects = _unique_subjects(validation_matrix)
    expected = manifest.inner_folds[inner_fold_index]
    if set(train_subjects) != set(expected.inner_train_subject_ids):
        raise SensitivityGRUError(
            "GRU training subjects must exactly match the selected inner-train fold"
        )
    if set(validation_subjects) != set(expected.inner_val_subject_ids):
        raise SensitivityGRUError(
            "GRU validation subjects must exactly match the selected inner-validation fold"
        )
    if set(train_subjects) & set(validation_subjects):
        raise SensitivityGRUError("GRU train/validation subjects must be disjoint")
    outer_test = set(manifest.test_subject_ids)
    if set(train_subjects) & outer_test or set(validation_subjects) & outer_test:
        raise SensitivityGRUError("outer-test subjects are forbidden during GRU fitting")
    return train_subjects, validation_subjects


def _assert_prediction_scope(manifest: SplitManifest, matrix: DesignMatrix) -> None:
    subjects = set(_unique_subjects(matrix))
    expected = set(manifest.test_subject_ids)
    if not subjects:
        raise SensitivityGRUError("prediction matrix must contain at least one subject")
    if not subjects <= expected:
        raise SensitivityGRUError("GRU final prediction may contain only outer-test subjects")


def _check_matrix_provenance(
    matrix: DesignMatrix,
    *,
    feature_names: tuple[str, ...],
    feature_lags: tuple[int, ...],
    target_dimensions: tuple[str, ...],
) -> None:
    if matrix.feature_names != feature_names:
        raise SensitivityGRUError("feature_names do not match fitted GRU provenance")
    if matrix.feature_lags != feature_lags:
        raise SensitivityGRUError("feature_lags do not match fitted GRU provenance")
    if matrix.target_dimensions != target_dimensions:
        raise SensitivityGRUError("target_dimensions do not match fitted GRU provenance")


def _fit_scaler(matrix: DesignMatrix) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(matrix.valid_mask, dtype=bool)
    if not np.any(valid):
        raise SensitivityGRUError("GRU training matrix has no valid rows")
    values = np.asarray(matrix.X[valid], dtype=float)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return mean, scale


def _sequence_batch(
    matrix: DesignMatrix,
    *,
    sequence_length: int,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
) -> _SequenceBatch:
    y = np.asarray(matrix.y, dtype=float)
    y_2d = y.reshape(-1, 1) if y.ndim == 1 else y
    windows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    row_indices: list[int] = []

    for subject in _unique_subjects(matrix):
        indices = np.flatnonzero(np.asarray(matrix.subject_id) == subject)
        if indices.size < sequence_length:
            continue
        origins = np.asarray(matrix.forecast_origin)[indices]
        if np.any(np.diff(origins) <= 0):
            raise SensitivityGRUError(
                "rows for each subject must be strictly ordered by forecast_origin"
            )
        for end_position in range(sequence_length - 1, indices.size):
            window_indices = indices[end_position - sequence_length + 1 : end_position + 1]
            target_index = int(indices[end_position])
            if not np.all(matrix.valid_mask[window_indices]) or not matrix.valid_mask[target_index]:
                continue
            window = (
                np.asarray(matrix.X[window_indices], dtype=float) - scaler_mean
            ) / scaler_scale
            windows.append(window)
            targets.append(np.asarray(y_2d[target_index], dtype=float))
            row_indices.append(target_index)

    if not windows:
        return _SequenceBatch(
            X=np.empty((0, sequence_length, matrix.X.shape[1]), dtype=float),
            y=np.empty((0, y_2d.shape[1]), dtype=float),
            row_indices=np.empty((0,), dtype=np.int64),
        )
    return _SequenceBatch(
        X=np.stack(windows),
        y=np.stack(targets),
        row_indices=np.asarray(row_indices, dtype=np.int64),
    )


def _configure_torch_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def fit_sensitivity_gru(
    execution: SensitivityExperimentConfig,
    manifest: SplitManifest,
    train_matrix: DesignMatrix,
    validation_matrix: DesignMatrix,
    *,
    inner_fold_index: int,
    config: GRUConfig,
    repository_root: str | Path | None = None,
) -> FittedSensitivityGRU:
    """Fit a deterministic GRU only after the shared execution barrier passes."""

    assert_sensitivity_execution_allowed(execution, repository_root=repository_root)
    if not isinstance(config, GRUConfig):
        raise SensitivityGRUError("config must be GRUConfig")
    train_subjects, validation_subjects = _assert_exact_inner_fold_scope(
        manifest,
        train_matrix,
        validation_matrix,
        inner_fold_index=inner_fold_index,
    )
    _check_matrix_provenance(
        validation_matrix,
        feature_names=train_matrix.feature_names,
        feature_lags=train_matrix.feature_lags,
        target_dimensions=train_matrix.target_dimensions,
    )

    scaler_mean, scaler_scale = _fit_scaler(train_matrix)
    train_batch = _sequence_batch(
        train_matrix,
        sequence_length=config.sequence_length,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
    )
    validation_batch = _sequence_batch(
        validation_matrix,
        sequence_length=config.sequence_length,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
    )
    if train_batch.X.shape[0] < 1:
        raise SensitivityGRUError("GRU training split yields no valid sequence windows")
    if validation_batch.X.shape[0] < 1:
        raise SensitivityGRUError("GRU validation split yields no valid sequence windows")

    _configure_torch_determinism(config.seed)
    model = _TorchGRURegressor(
        input_size=train_matrix.X.shape[1],
        output_size=train_batch.y.shape[1],
        config=config,
    ).to(dtype=torch.float64, device="cpu")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    loss_fn = nn.MSELoss()
    X_train = torch.as_tensor(train_batch.X, dtype=torch.float64)
    y_train = torch.as_tensor(train_batch.y, dtype=torch.float64)
    X_validation = torch.as_tensor(validation_batch.X, dtype=torch.float64)
    y_validation = torch.as_tensor(validation_batch.y, dtype=torch.float64)

    model.train()
    training_loss = float("nan")
    for _ in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(X_train)
        loss = loss_fn(prediction, y_train)
        loss.backward()
        optimizer.step()
        training_loss = float(loss.detach().cpu().item())

    model.eval()
    with torch.no_grad():
        validation_loss = float(
            loss_fn(model(X_validation), y_validation).detach().cpu().item()
        )

    return FittedSensitivityGRU(
        model=model,
        config=config,
        feature_names=train_matrix.feature_names,
        feature_lags=train_matrix.feature_lags,
        target_dimensions=train_matrix.target_dimensions,
        scaler_mean=np.array(scaler_mean, copy=True),
        scaler_scale=np.array(scaler_scale, copy=True),
        train_subject_ids=train_subjects,
        validation_subject_ids=validation_subjects,
        training_sequence_count=int(train_batch.X.shape[0]),
        validation_sequence_count=int(validation_batch.X.shape[0]),
        final_training_loss=training_loss,
        final_validation_loss=validation_loss,
    )


def predict_sensitivity_gru(
    execution: SensitivityExperimentConfig,
    fitted: FittedSensitivityGRU,
    manifest: SplitManifest,
    matrix: DesignMatrix,
    *,
    condition: str = "gru",
    repository_root: str | Path | None = None,
) -> PredictionArtifact:
    """Predict outer-test rows only after the shared execution barrier passes."""

    assert_sensitivity_execution_allowed(execution, repository_root=repository_root)
    if not isinstance(fitted, FittedSensitivityGRU):
        raise SensitivityGRUError("fitted must be FittedSensitivityGRU")
    _assert_prediction_scope(manifest, matrix)
    _check_matrix_provenance(
        matrix,
        feature_names=fitted.feature_names,
        feature_lags=fitted.feature_lags,
        target_dimensions=fitted.target_dimensions,
    )
    if not isinstance(condition, str) or not condition.strip():
        raise SensitivityGRUError("condition must be a non-empty string")

    batch = _sequence_batch(
        matrix,
        sequence_length=fitted.config.sequence_length,
        scaler_mean=fitted.scaler_mean,
        scaler_scale=fitted.scaler_scale,
    )
    y_pred = np.full(np.asarray(matrix.y).shape, np.nan, dtype=float)
    prediction_valid = np.zeros(np.asarray(matrix.valid_mask).shape, dtype=bool)
    if batch.X.shape[0] > 0:
        fitted.model.eval()
        with torch.no_grad():
            predicted = fitted.model(
                torch.as_tensor(batch.X, dtype=torch.float64)
            ).detach().cpu().numpy()
        if matrix.y.ndim == 1:
            y_pred[batch.row_indices] = predicted[:, 0]
        else:
            y_pred[batch.row_indices] = predicted
        prediction_valid[batch.row_indices] = True

    return PredictionArtifact(
        outer_fold=manifest.outer_fold,
        subject_id=matrix.subject_id,
        region_id=matrix.region_id,
        target_dimensions=matrix.target_dimensions,
        condition=condition,
        forecast_origin=matrix.forecast_origin,
        target_time=matrix.target_time,
        y_true=matrix.y,
        y_pred=y_pred,
        valid_mask=prediction_valid,
    )
