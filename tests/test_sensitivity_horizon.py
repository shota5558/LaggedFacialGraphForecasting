from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting.contracts import FaceTimeSeries
from lagged_facial_graph_forecasting.core_contracts import ParentLink, ParentSet
from lagged_facial_graph_forecasting.sensitivity_horizon import (
    SensitivityHorizonError,
    build_sensitivity_full_history_design_matrix,
    build_sensitivity_pcmci_parent_design_matrix,
    build_sensitivity_persistence_design_matrix,
    build_sensitivity_self_history_design_matrix,
)


def _series(length: int = 12) -> FaceTimeSeries:
    time = np.arange(length, dtype=float)
    values = np.empty((length, 2, 1), dtype=float)
    values[:, 0, 0] = time
    values[:, 1, 0] = 100.0 + time
    return FaceTimeSeries(
        X=values,
        subject_id="synthetic_subject",
        time_index=time / 25.0,
        region_id=("source", "target"),
        dimension=("value",),
        valid_mask=np.ones_like(values, dtype=bool),
        sampling_rate=25.0,
    )


def test_h2_self_matrix_preserves_forecast_origin_target_alignment_and_horizon_provenance() -> None:
    result = build_sensitivity_self_history_design_matrix(
        _series(), target_region="target", horizon=2, lags=(2, 3)
    )
    matrix = result.design_matrix

    assert result.horizon == 2
    assert result.condition == "self"
    assert matrix.feature_lags == (2, 3)
    np.testing.assert_allclose(matrix.target_time - matrix.forecast_origin, 2.0 / 25.0)
    # First target is index 3 because the common support lag is 3.
    assert matrix.y[0] == 103.0
    assert matrix.X[0, 0] == 101.0  # target-relative tau=2 -> forecast origin index 1
    assert matrix.X[0, 1] == 100.0  # tau=3 -> index 0


def test_h3_persistence_uses_forecast_origin_observation_as_tau_equal_h() -> None:
    result = build_sensitivity_persistence_design_matrix(
        _series(), target_region="target", horizon=3
    )
    matrix = result.design_matrix

    assert result.horizon == 3
    assert result.condition == "persistence"
    assert matrix.feature_lags == (3,)
    np.testing.assert_allclose(matrix.target_time - matrix.forecast_origin, 3.0 / 25.0)
    assert matrix.y[0] == 103.0
    assert matrix.X[0, 0] == 100.0


def test_h2_full_matrix_rejects_any_feature_lag_before_forecast_origin() -> None:
    with pytest.raises(SensitivityHorizonError, match="tau >= h"):
        build_sensitivity_full_history_design_matrix(
            _series(), target_region="target", horizon=2, lags=(1, 2)
        )


def test_h2_pcmci_rejects_unavailable_parent_instead_of_silent_drop() -> None:
    parent_set = ParentSet(
        outer_fold=0,
        target_region="target",
        parents=(
            ParentLink(
                source_region="source",
                lag=1,
                source_dimension="value",
                target_dimension="value",
            ),
        ),
        discovery_method="pcmci_plus",
    )

    with pytest.raises(SensitivityHorizonError, match="tau < h=2"):
        build_sensitivity_pcmci_parent_design_matrix(
            _series(),
            parent_set=parent_set,
            horizon=2,
            self_lags=(2,),
            target_dimension="value",
        )


def test_h2_pcmci_accepts_only_directly_observable_parent_and_preserves_lag_provenance() -> None:
    parent_set = ParentSet(
        outer_fold=0,
        target_region="target",
        parents=(
            ParentLink(
                source_region="source",
                lag=3,
                source_dimension="value",
                target_dimension="value",
            ),
        ),
        discovery_method="pcmci_plus",
    )
    result = build_sensitivity_pcmci_parent_design_matrix(
        _series(),
        parent_set=parent_set,
        horizon=2,
        self_lags=(2,),
        target_dimension="value",
    )

    assert result.condition == "pcmci"
    assert result.horizon == 2
    assert all(lag >= 2 for lag in result.design_matrix.feature_lags)
    assert 3 in result.design_matrix.feature_lags
    np.testing.assert_allclose(
        result.design_matrix.target_time - result.design_matrix.forecast_origin,
        2.0 / 25.0,
    )


def test_multi_horizon_adapter_cannot_be_used_to_relabel_primary_h1() -> None:
    with pytest.raises(SensitivityHorizonError, match="Primary remains h=1"):
        build_sensitivity_self_history_design_matrix(
            _series(), target_region="target", horizon=1, lags=(1,)
        )
