from __future__ import annotations

import json

import numpy as np

from lagged_facial_graph_forecasting import (
    MinimalFoldRunner,
    build_subject_split_manifest,
    generate_synthetic_face_time_series,
)


def _v0_fixture():
    subject_ids = tuple(f"s{index:02d}" for index in range(1, 7))
    series_by_subject = {
        subject_id: generate_synthetic_face_time_series(
            subject_id=subject_id,
            n_steps=64,
            seed=900 + index,
        )
        for index, subject_id in enumerate(subject_ids)
    }
    manifest = build_subject_split_manifest(
        subject_ids,
        outer_fold=4,
        test_subject_count=1,
        n_inner_folds=2,
        seed=23,
    )
    return manifest, series_by_subject


def _run_once(manifest, series_by_subject, output_path):
    runner = MinimalFoldRunner(manifest, series_by_subject)
    runner.fit_self_history(target_region="mouth", lags=(1, 2), alpha=1.0)
    assert runner.outer_test_locked
    assert set(runner.training_subject_ids) == set(manifest.train_subject_ids)
    assert set(runner.training_subject_ids).isdisjoint(manifest.test_subject_ids)

    runner.freeze()
    return runner.evaluate_outer_test(output_path)


def test_v0_self_only_vertical_slice_end_to_end_is_traceable_and_deterministic(
    tmp_path,
) -> None:
    manifest, series_by_subject = _v0_fixture()

    first = _run_once(manifest, series_by_subject, tmp_path / "first.json")
    second = _run_once(manifest, series_by_subject, tmp_path / "second.json")

    assert first.artifact_path.exists()
    assert second.artifact_path.exists()
    assert first.artifact_path.read_bytes() == second.artifact_path.read_bytes()
    assert first.velocity_rmse == second.velocity_rmse
    np.testing.assert_array_equal(first.prediction.valid_mask, second.prediction.valid_mask)
    np.testing.assert_allclose(
        first.prediction.y_pred,
        second.prediction.y_pred,
        equal_nan=True,
    )

    payload = json.loads(first.artifact_path.read_text(encoding="utf-8"))

    # Gate V0 provenance: fold, subject, region, target dimensions, time alignment,
    # features/lags, validity mask, prediction, and ground truth survive the full path.
    assert payload["schema_version"] == 2
    assert payload["outer_fold"] == manifest.outer_fold
    assert payload["condition"] == "self"
    assert payload["feature_provenance"] == {
        "feature_names": ["mouth.vx", "mouth.vy", "mouth.vx", "mouth.vy"],
        "feature_lags": [1, 1, 2, 2],
    }
    assert payload["target_provenance"] == {
        "target_dimensions": ["vx", "vy"],
    }
    assert payload["model"]["forecaster"] == "ridge"
    assert payload["model"]["alpha"] == 1.0
    assert np.isfinite(payload["metric"]["velocity_rmse"])

    rows = payload["rows"]
    assert set(rows["subject_id"]) == set(manifest.test_subject_ids)
    assert set(rows["subject_id"]).isdisjoint(manifest.train_subject_ids)
    assert set(rows["region_id"]) == {"mouth"}

    row_count = len(rows["subject_id"])
    assert row_count > 0
    assert len(rows["forecast_origin"]) == row_count
    assert len(rows["target_time"]) == row_count
    assert len(rows["valid_mask"]) == row_count
    assert len(rows["y_pred"]) == row_count
    assert len(rows["y_true"]) == row_count

    assert all(
        target_time > forecast_origin
        for forecast_origin, target_time in zip(
            rows["forecast_origin"], rows["target_time"], strict=True
        )
    )
    assert all(isinstance(value, bool) for value in rows["valid_mask"])
