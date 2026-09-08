from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from lagged_facial_graph_forecasting.contracts import DesignMatrix, InnerFold, SplitManifest
from lagged_facial_graph_forecasting.metrics import velocity_rmse_by_subject_region
from lagged_facial_graph_forecasting.sensitivity_execution import (
    assert_sensitivity_execution_allowed,
    load_sensitivity_experiment_config,
)
from lagged_facial_graph_forecasting.sensitivity_gru import (
    GRUConfig,
    fit_sensitivity_gru,
    predict_sensitivity_gru,
)
from lagged_facial_graph_forecasting.sensitivity_results import (
    SensitivityMetricArtifact,
    SoftwareVersion,
    dumps_sensitivity_metric,
    loads_sensitivity_metric,
)
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _manifest() -> SplitManifest:
    return SplitManifest(
        outer_fold=0,
        train_subject_ids=("train_a", "validation"),
        test_subject_ids=("outer_test",),
        inner_folds=(
            InnerFold(
                inner_train_subject_ids=("train_a",),
                inner_val_subject_ids=("validation",),
            ),
        ),
        seed=9901,
    )


def _matrix(subject_id: str, rows: int = 18) -> DesignMatrix:
    t = np.arange(rows, dtype=float)
    x0 = np.sin(0.2 * t)
    x1 = np.cos(0.13 * t)
    X = np.column_stack((x0, x1))
    y = 0.7 * x0 - 0.25 * x1
    return DesignMatrix(
        X=X,
        y=y,
        subject_id=(subject_id,) * rows,
        region_id=("mouth",) * rows,
        target_dimensions=("value",),
        forecast_origin=t / 25.0,
        target_time=(t + 1.0) / 25.0,
        feature_names=("mouth.value", "source.value"),
        feature_lags=(1, 1),
        valid_mask=np.ones(rows, dtype=bool),
    )


def test_gru_synthetic_integration_reaches_sensitivity_result_namespace() -> None:
    execution = load_sensitivity_experiment_config(
        ROOT / "configs/sensitivity_preimplementation.yaml", repository_root=ROOT
    )
    assert_sensitivity_execution_allowed(execution, repository_root=ROOT)
    manifest = _manifest()
    config = GRUConfig(
        sequence_length=3,
        hidden_size=6,
        epochs=30,
        learning_rate=0.03,
        seed=9911,
    )
    fitted = fit_sensitivity_gru(
        manifest,
        _matrix("train_a"),
        _matrix("validation"),
        inner_fold_index=0,
        config=config,
    )
    prediction = predict_sensitivity_gru(fitted, manifest, _matrix("outer_test"))
    metric = velocity_rmse_by_subject_region(prediction)[0]
    result = SensitivityMetricArtifact(
        method_id="gru",
        source_primary_freeze_manifest="artifacts/primary/freeze_manifest.json",
        outer_fold=metric.outer_fold,
        subject_id=metric.subject_id,
        region_id=metric.region_id,
        horizon=1,
        metric=metric,
        config_path="configs/sensitivity_preimplementation.yaml",
        seed=config.seed,
        software_versions=(
            SoftwareVersion("numpy", np.__version__),
            SoftwareVersion("torch", torch.__version__),
        ),
    )
    encoded = dumps_sensitivity_metric(result)
    restored = loads_sensitivity_metric(encoded)

    assert restored == result
    assert restored.namespace == "sensitivity"
    assert restored.method_id == "gru"
    assert np.count_nonzero(prediction.valid_mask) > 0
