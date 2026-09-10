from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting.contracts import FaceTimeSeries
from lagged_facial_graph_forecasting.core_contracts import ExperimentConfig
from lagged_facial_graph_forecasting.sensitivity_circular_shift import (
    circular_shift_face_time_series,
)
from lagged_facial_graph_forecasting.sensitivity_execution import (
    PRIMARY_FREEZE_MANIFEST_PATH,
    SensitivityExecutionError,
    SensitivityExecutionMode,
    SensitivityExperimentConfig,
)
from lagged_facial_graph_forecasting.sensitivity_horizon import (
    build_sensitivity_self_history_design_matrix,
)
from lagged_facial_graph_forecasting.sensitivity_phase_shuffle import (
    phase_shuffle_face_time_series,
)


def _execution(mode: SensitivityExecutionMode) -> SensitivityExperimentConfig:
    return SensitivityExperimentConfig(
        experiment=ExperimentConfig(
            experiment_id="component-barrier-test",
            scientific_config_path="configs/scientific_freeze.yaml",
            run_config_path="configs/sensitivity_preimplementation.yaml",
            artifact_root="artifacts/sensitivity",
            seed=77,
        ),
        namespace="sensitivity",
        execution_mode=mode,
        primary_freeze_manifest=PRIMARY_FREEZE_MANIFEST_PATH,
    )


def _series() -> FaceTimeSeries:
    values = np.arange(32 * 2, dtype=float).reshape(32, 2, 1)
    return FaceTimeSeries(
        X=values,
        subject_id="fixture_subject",
        time_index=np.arange(32, dtype=float) / 25.0,
        region_id=("source", "target"),
        dimension=("value",),
        valid_mask=np.ones_like(values, dtype=bool),
        sampling_rate=25.0,
    )


@pytest.mark.parametrize("component", ["phase", "circular", "horizon"])
def test_real_component_entrypoints_fail_closed_without_primary_freeze(
    tmp_path, component: str
) -> None:
    execution = _execution(SensitivityExecutionMode.REAL)
    series = _series()

    with pytest.raises(SensitivityExecutionError, match="PRIMARY FREEZE"):
        if component == "phase":
            phase_shuffle_face_time_series(
                execution, series, seed=1, repository_root=tmp_path
            )
        elif component == "circular":
            circular_shift_face_time_series(
                execution, series, seed=1, repository_root=tmp_path
            )
        else:
            build_sensitivity_self_history_design_matrix(
                execution,
                series,
                target_region="target",
                horizon=2,
                lags=(2, 3),
                repository_root=tmp_path,
            )


def test_synthetic_component_entrypoints_remain_available_without_primary_freeze(
    tmp_path,
) -> None:
    execution = _execution(SensitivityExecutionMode.SYNTHETIC)
    series = _series()

    phase_shuffle_face_time_series(execution, series, seed=1, repository_root=tmp_path)
    circular_shift_face_time_series(execution, series, seed=1, repository_root=tmp_path)
    matrix = build_sensitivity_self_history_design_matrix(
        execution,
        series,
        target_region="target",
        horizon=2,
        lags=(2, 3),
        repository_root=tmp_path,
    )
    assert matrix.horizon == 2
