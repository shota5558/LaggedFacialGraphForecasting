from __future__ import annotations

import json

import numpy as np
import pytest

from lagged_facial_graph_forecasting import (
    MinimalFoldRunner,
    OuterTestLockedError,
    build_subject_split_manifest,
    generate_synthetic_face_time_series,
)


def _fixture():
    subject_ids = tuple(f"s{index:02d}" for index in range(1, 7))
    series_by_subject = {
        subject_id: generate_synthetic_face_time_series(
            subject_id=subject_id,
            n_steps=48,
            seed=100 + index,
        )
        for index, subject_id in enumerate(subject_ids)
    }
    manifest = build_subject_split_manifest(
        subject_ids,
        outer_fold=2,
        test_subject_count=1,
        n_inner_folds=2,
        seed=17,
    )
    return manifest, series_by_subject


def test_outer_test_remains_locked_until_explicit_freeze(tmp_path) -> None:
    manifest, series_by_subject = _fixture()
    runner = MinimalFoldRunner(manifest, series_by_subject)

    assert runner.outer_test_locked
    with pytest.raises(OuterTestLockedError, match="outer-test is locked"):
        runner.evaluate_outer_test(tmp_path / "too_early.json")

    runner.fit_self_history(target_region="mouth", lags=(1, 2), alpha=1.0)

    assert runner.outer_test_locked
    assert set(runner.training_subject_ids) == set(manifest.train_subject_ids)
    assert set(runner.training_subject_ids).isdisjoint(manifest.test_subject_ids)
    with pytest.raises(OuterTestLockedError, match="outer-test is locked"):
        runner.evaluate_outer_test(tmp_path / "still_locked.json")

    runner.freeze()
    assert not runner.outer_test_locked

    result = runner.evaluate_outer_test(tmp_path / "result.json")

    assert result.artifact_path.exists()
    assert np.isfinite(result.velocity_rmse)
    assert set(result.prediction.subject_id) == set(manifest.test_subject_ids)
    assert set(result.prediction.subject_id).isdisjoint(manifest.train_subject_ids)
    assert runner.outer_test_locked

    payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    assert payload["outer_fold"] == manifest.outer_fold
    assert payload["condition"] == "self"

    with pytest.raises(OuterTestLockedError, match="outer-test is locked"):
        runner.evaluate_outer_test(tmp_path / "second_evaluation.json")


def test_freeze_before_training_is_rejected() -> None:
    manifest, series_by_subject = _fixture()
    runner = MinimalFoldRunner(manifest, series_by_subject)

    with pytest.raises(RuntimeError, match="only after successful training"):
        runner.freeze()


def test_runner_rejects_missing_manifest_subject() -> None:
    manifest, series_by_subject = _fixture()
    series_by_subject.pop(manifest.test_subject_ids[0])

    with pytest.raises(ValueError, match="missing FaceTimeSeries"):
        MinimalFoldRunner(manifest, series_by_subject)
