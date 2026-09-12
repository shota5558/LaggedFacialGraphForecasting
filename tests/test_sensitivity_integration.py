from __future__ import annotations

from pathlib import Path

import numpy as np

from lagged_facial_graph_forecasting.contracts import FaceTimeSeries, InnerFold, SplitManifest
from lagged_facial_graph_forecasting.design_matrix import build_pcmci_parent_design_matrix, build_self_history_design_matrix
from lagged_facial_graph_forecasting.forecaster import fit_ridge_forecaster, predict_ridge_forecaster
from lagged_facial_graph_forecasting.metrics import velocity_rmse_by_subject_region
from lagged_facial_graph_forecasting.sensitivity_circular_shift import circular_shift_face_time_series
from lagged_facial_graph_forecasting.sensitivity_execution import assert_sensitivity_execution_allowed, load_sensitivity_experiment_config
from lagged_facial_graph_forecasting.sensitivity_gpdc import gpdc_result_to_parent_set, run_sensitivity_pcmciplus_gpdc
from lagged_facial_graph_forecasting.sensitivity_horizon import build_sensitivity_self_history_design_matrix
from lagged_facial_graph_forecasting.sensitivity_lpcmci import load_lpcmci_config, lpcmci_graph_to_parent_set, lpcmci_result_to_graph_artifact, run_sensitivity_lpcmci
from lagged_facial_graph_forecasting.sensitivity_phase_shuffle import phase_shuffle_face_time_series
from lagged_facial_graph_forecasting.sensitivity_results import SensitivityMetricArtifact, SoftwareVersion, dumps_sensitivity_metric, loads_sensitivity_metric
from lagged_facial_graph_forecasting.tigramite_adapter import face_time_series_to_tigramite_dataframe


ROOT = Path(__file__).resolve().parents[1]
TARGET = "mouth"


def _manifest() -> SplitManifest:
    return SplitManifest(
        outer_fold=0,
        train_subject_ids=("train_a", "train_b"),
        test_subject_ids=("outer_test",),
        inner_folds=(InnerFold(inner_train_subject_ids=("train_a",), inner_val_subject_ids=("train_b",)),),
        seed=909,
    )


def _series(subject_id: str, seed: int, length: int = 64) -> FaceTimeSeries:
    rng = np.random.default_rng(seed)
    source = rng.normal(size=length)
    target = rng.normal(scale=0.08, size=length)
    for time in range(2, length):
        target[time] += 1.15 * source[time - 2] + 0.25 * target[time - 1]
    values = np.stack((source, target), axis=1)[:, :, None]
    return FaceTimeSeries(
        X=values, subject_id=subject_id,
        time_index=np.arange(length, dtype=float) / 25.0,
        region_id=("source", TARGET), dimension=("value",),
        valid_mask=np.ones_like(values, dtype=bool), sampling_rate=25.0,
    )


def _execution():
    config = load_sensitivity_experiment_config(
        ROOT / "configs/sensitivity_preimplementation.yaml", repository_root=ROOT
    )
    assert_sensitivity_execution_allowed(config, repository_root=ROOT)
    return config


def _serialize_result(execution, method_id: str, prediction, *, horizon: int, seed: int) -> str:
    metric = velocity_rmse_by_subject_region(prediction)[0]
    result = SensitivityMetricArtifact(
        method_id=method_id,
        source_primary_freeze_manifest="artifacts/primary/freeze_manifest.json",
        outer_fold=metric.outer_fold, subject_id=metric.subject_id, region_id=metric.region_id,
        horizon=horizon, metric=metric,
        config_path="configs/sensitivity_preimplementation.yaml", seed=seed,
        software_versions=(SoftwareVersion("numpy", np.__version__),),
    )
    encoded = dumps_sensitivity_metric(execution, result, repository_root=ROOT)
    restored = loads_sensitivity_metric(encoded)
    assert restored == result
    assert restored.namespace == "sensitivity"
    return encoded


def _ridge_from_parent_set(parent_set, train: FaceTimeSeries, test: FaceTimeSeries, condition: str):
    train_matrix = build_pcmci_parent_design_matrix(train, parent_set=parent_set, self_lags=(1,), target_dimension="value")
    test_matrix = build_pcmci_parent_design_matrix(test, parent_set=parent_set, self_lags=(1,), target_dimension="value")
    fitted = fit_ridge_forecaster(train_matrix, alpha=1.0)
    return predict_ridge_forecaster(fitted, test_matrix, outer_fold=_manifest().outer_fold, condition=condition)


def _ridge_from_self(train: FaceTimeSeries, test: FaceTimeSeries, condition: str):
    train_matrix = build_self_history_design_matrix(train, target_region=TARGET, lags=(1, 2), target_dimension="value")
    test_matrix = build_self_history_design_matrix(test, target_region=TARGET, lags=(1, 2), target_dimension="value")
    fitted = fit_ridge_forecaster(train_matrix, alpha=1.0)
    return predict_ridge_forecaster(fitted, test_matrix, outer_fold=_manifest().outer_fold, condition=condition)


def test_gpdc_synthetic_end_to_end_reaches_sensitivity_result_namespace() -> None:
    manifest = _manifest(); execution = _execution(); train = _series("train_a", 1001); test = _series("outer_test", 1003)
    bundle = face_time_series_to_tigramite_dataframe(manifest, (train,))
    discovery, provenance = run_sensitivity_pcmciplus_gpdc(execution, manifest, bundle, seed=1201)
    parent_set = gpdc_result_to_parent_set(discovery, bundle, outer_fold=manifest.outer_fold, target_region=TARGET)
    encoded = _serialize_result(execution, "pcmci_plus_gpdc", _ridge_from_parent_set(parent_set, train, test, "gpdc"), horizon=1, seed=provenance.seed)
    assert parent_set.discovery_method == "pcmci_plus_gpdc"; assert '"namespace":"sensitivity"' in encoded


def test_lpcmci_synthetic_end_to_end_reaches_sensitivity_result_namespace() -> None:
    manifest = _manifest(); execution = _execution(); train = _series("train_a", 2001); test = _series("outer_test", 2003)
    bundle = face_time_series_to_tigramite_dataframe(manifest, (train,)); method = load_lpcmci_config(ROOT / "configs/sensitivity_lpcmci.yaml")
    discovery = run_sensitivity_lpcmci(execution, manifest, bundle, method, seed=2201)
    graph = lpcmci_result_to_graph_artifact(discovery, bundle, outer_fold=manifest.outer_fold, seed=2201)
    parent_set = lpcmci_graph_to_parent_set(graph, target_region=TARGET)
    encoded = _serialize_result(execution, "lpcmci", _ridge_from_parent_set(parent_set, train, test, "lpcmci"), horizon=1, seed=graph.seed)
    assert parent_set.discovery_method == "lpcmci"; assert '"namespace":"sensitivity"' in encoded


def test_phase_shuffle_synthetic_end_to_end_reaches_sensitivity_result_namespace() -> None:
    execution = _execution(); train = _series("train_a", 3001); test = _series("outer_test", 3003); seed = 3201
    shuffled_train = phase_shuffle_face_time_series(execution, train, seed=seed, repository_root=ROOT).series
    shuffled_test = phase_shuffle_face_time_series(execution, test, seed=seed, repository_root=ROOT).series
    encoded = _serialize_result(
        execution,
        "phase_shuffle",
        _ridge_from_self(shuffled_train, shuffled_test, "phase_shuffle"),
        horizon=1,
        seed=seed,
    )
    assert '"namespace":"sensitivity"' in encoded


def test_circular_shift_synthetic_end_to_end_reaches_sensitivity_result_namespace() -> None:
    execution = _execution(); train = _series("train_a", 4001); test = _series("outer_test", 4003); seed = 4201
    shifted_train = circular_shift_face_time_series(execution, train, seed=seed, repository_root=ROOT).series
    shifted_test = circular_shift_face_time_series(execution, test, seed=seed, repository_root=ROOT).series
    encoded = _serialize_result(
        execution,
        "circular_shift",
        _ridge_from_self(shifted_train, shifted_test, "circular_shift"),
        horizon=1,
        seed=seed,
    )
    assert '"namespace":"sensitivity"' in encoded


def test_multi_horizon_synthetic_end_to_end_preserves_horizon_provenance() -> None:
    execution = _execution(); train = _series("train_a", 5001); test = _series("outer_test", 5003)
    train_wrapped = build_sensitivity_self_history_design_matrix(execution, train, target_region=TARGET, horizon=3, lags=(3, 4), target_dimension="value", repository_root=ROOT)
    test_wrapped = build_sensitivity_self_history_design_matrix(execution, test, target_region=TARGET, horizon=3, lags=(3, 4), target_dimension="value", repository_root=ROOT)
    fitted = fit_ridge_forecaster(train_wrapped.design_matrix, alpha=1.0)
    prediction = predict_ridge_forecaster(fitted, test_wrapped.design_matrix, outer_fold=_manifest().outer_fold, condition="h3")
    restored = loads_sensitivity_metric(_serialize_result(execution, "multi_horizon", prediction, horizon=3, seed=5201))
    assert train_wrapped.horizon == 3; assert test_wrapped.horizon == 3; assert restored.horizon == 3
    assert all(lag >= 3 for lag in test_wrapped.design_matrix.feature_lags)
