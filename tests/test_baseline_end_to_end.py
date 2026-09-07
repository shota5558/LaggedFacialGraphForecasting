from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lagged_facial_graph_forecasting.artifact_registry import ArtifactRegistry
from lagged_facial_graph_forecasting.artifacts import write_v0_result_json
from lagged_facial_graph_forecasting.core_contracts import ExperimentConfig
from lagged_facial_graph_forecasting.design_matrix import (
    build_full_history_design_matrix,
    build_persistence_design_matrix,
    build_self_history_design_matrix,
)
from lagged_facial_graph_forecasting.failure_handling import atomic_artifact_output
from lagged_facial_graph_forecasting.forecaster import predict_ridge_forecaster
from lagged_facial_graph_forecasting.metrics import velocity_rmse
from lagged_facial_graph_forecasting.ridge_tuning import (
    aggregate_ridge_inner_cv_scores,
    evaluate_ridge_inner_cv,
    refit_selected_ridge,
    select_ridge_alpha,
)
from lagged_facial_graph_forecasting.runner import _concatenate_design_matrices
from lagged_facial_graph_forecasting.splits import build_subject_split_manifest
from lagged_facial_graph_forecasting.synthetic import generate_synthetic_face_time_series


_TARGET_REGION = "left_eye"
_LAGS = (1, 2, 3)
_ALIGNMENT_LAG = 3
_ALPHA_GRID = (0.1, 1.0, 10.0)


def _experiment_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="i1-baseline-gate",
        scientific_config_path="configs/scientific_freeze.yaml",
        run_config_path="configs/primary_run.yaml",
        artifact_root="artifacts/primary",
        seed=20260908,
    )


def _series_by_subject() -> dict[str, object]:
    return {
        subject_id: generate_synthetic_face_time_series(
            subject_id=subject_id,
            n_steps=90,
            seed=100 + index,
        )
        for index, subject_id in enumerate(("s01", "s02", "s03", "s04", "s05"))
    }


def _builder(condition: str):
    if condition == "persistence":
        return lambda series: build_persistence_design_matrix(
            series,
            target_region=_TARGET_REGION,
            horizon=1,
            alignment_lag=_ALIGNMENT_LAG,
        )
    if condition == "self":
        return lambda series: build_self_history_design_matrix(
            series,
            target_region=_TARGET_REGION,
            lags=_LAGS,
            horizon=1,
            alignment_lag=_ALIGNMENT_LAG,
        )
    if condition == "full":
        return lambda series: build_full_history_design_matrix(
            series,
            target_region=_TARGET_REGION,
            lags=_LAGS,
            horizon=1,
            alignment_lag=_ALIGNMENT_LAG,
        )
    raise AssertionError(f"unexpected baseline condition: {condition}")


def _run_gate(repository_root: Path):
    series_by_subject = _series_by_subject()
    manifest = build_subject_split_manifest(
        tuple(series_by_subject),
        outer_fold=0,
        test_subject_count=1,
        n_inner_folds=2,
        seed=20260908,
    )
    config = _experiment_config()
    registry = ArtifactRegistry(config, repository_root=repository_root)

    results = {}
    common_test_provenance = None
    for condition in ("persistence", "self", "full"):
        build_matrix = _builder(condition)

        # Synthetic FaceTimeSeries is already the contract-level input for Gate 1;
        # no learned real-data preprocessing is introduced before Lane A.
        train_matrix = _concatenate_design_matrices(
            build_matrix(series_by_subject[subject_id])
            for subject_id in manifest.train_subject_ids
        )
        assert set(train_matrix.subject_id) == set(manifest.train_subject_ids)
        assert not (set(train_matrix.subject_id) & set(manifest.test_subject_ids))

        scores = evaluate_ridge_inner_cv(
            train_matrix,
            split_manifest=manifest,
            alpha_grid=_ALPHA_GRID,
        )
        aggregates = aggregate_ridge_inner_cv_scores(scores)
        selection = select_ridge_alpha(aggregates)
        fitted = refit_selected_ridge(
            train_matrix,
            split_manifest=manifest,
            selection=selection,
        )

        # Outer-test materialization is deliberately after tuning + frozen refit.
        test_matrix = _concatenate_design_matrices(
            build_matrix(series_by_subject[subject_id])
            for subject_id in manifest.test_subject_ids
        )
        prediction = predict_ridge_forecaster(
            fitted,
            test_matrix,
            outer_fold=manifest.outer_fold,
            condition=condition,
        )
        rmse = velocity_rmse(prediction)

        assert set(prediction.subject_id) == set(manifest.test_subject_ids)
        assert prediction.condition == condition
        assert np.isfinite(rmse)
        assert rmse >= 0.0
        assert np.array_equal(prediction.valid_mask, test_matrix.valid_mask)

        provenance = (
            prediction.subject_id,
            prediction.region_id,
            prediction.forecast_origin.tobytes(),
            prediction.target_time.tobytes(),
            prediction.y_true.tobytes(),
        )
        if common_test_provenance is None:
            common_test_provenance = provenance
        else:
            assert provenance == common_test_provenance

        final_path = Path("artifacts/primary/i1") / f"{condition}.json"
        with atomic_artifact_output(
            final_path,
            config=config,
            repository_root=repository_root,
        ) as staging:
            write_v0_result_json(
                staging,
                matrix=test_matrix,
                prediction=prediction,
                fitted=fitted,
                velocity_rmse_value=rmse,
            )
        entry = registry.register_file(
            final_path,
            artifact_type="baseline_result",
            outer_fold=manifest.outer_fold,
            condition=condition,
        )
        results[condition] = {
            "selection": selection,
            "prediction": prediction,
            "rmse": rmse,
            "artifact": entry,
            "bytes": (repository_root / final_path).read_bytes(),
        }

    assert tuple(entry.condition for entry in registry.entries) == (
        "full",
        "persistence",
        "self",
    )
    return manifest, results


def test_baseline_end_to_end_gate_is_leak_free_deterministic_and_complete(
    tmp_path: Path,
) -> None:
    manifest_a, results_a = _run_gate(tmp_path / "run_a")
    manifest_b, results_b = _run_gate(tmp_path / "run_b")

    assert manifest_a == manifest_b
    for condition in ("persistence", "self", "full"):
        left = results_a[condition]
        right = results_b[condition]
        assert left["selection"] == right["selection"]
        assert left["rmse"] == right["rmse"]
        assert np.array_equal(left["prediction"].y_pred, right["prediction"].y_pred)
        assert left["artifact"].sha256 == right["artifact"].sha256
        assert left["bytes"] == right["bytes"]


def test_baseline_inner_cv_rejects_outer_test_subject_rows() -> None:
    series_by_subject = _series_by_subject()
    manifest = build_subject_split_manifest(
        tuple(series_by_subject),
        outer_fold=0,
        test_subject_count=1,
        n_inner_folds=2,
        seed=20260908,
    )
    leaked_matrix = _concatenate_design_matrices(
        build_self_history_design_matrix(
            series_by_subject[subject_id],
            target_region=_TARGET_REGION,
            lags=_LAGS,
            horizon=1,
            alignment_lag=_ALIGNMENT_LAG,
        )
        for subject_id in (*manifest.train_subject_ids, *manifest.test_subject_ids)
    )

    with pytest.raises(ValueError, match="outer-test"):
        evaluate_ridge_inner_cv(
            leaked_matrix,
            split_manifest=manifest,
            alpha_grid=_ALPHA_GRID,
        )
