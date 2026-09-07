from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lagged_facial_graph_forecasting.contracts import FaceTimeSeries, InnerFold, SplitManifest
from lagged_facial_graph_forecasting.core_contract_io import dumps_core_contract
from lagged_facial_graph_forecasting.core_contracts import ParentLink, ParentSet
from lagged_facial_graph_forecasting.design_matrix import (
    build_full_history_design_matrix,
    build_pcmci_parent_design_matrix,
    build_persistence_design_matrix,
    build_self_history_design_matrix,
)
from lagged_facial_graph_forecasting.forecaster import predict_ridge_forecaster
from lagged_facial_graph_forecasting.leakage_guard import (
    assert_discovery_fit_scope,
    assert_preprocessing_fit_scope,
    assert_ridge_tuning_scope,
)
from lagged_facial_graph_forecasting.metrics import velocity_rmse_by_subject_region
from lagged_facial_graph_forecasting.null_lag_response import (
    LAG_RESPONSE_EVALUABLE,
    PRIMARY_LAG_RESPONSE_DELTAS,
    construct_lag_response_grid,
)
from lagged_facial_graph_forecasting.null_matched_sparsity import (
    build_matched_sparsity_candidate_space,
    construct_matched_sparsity_mapping,
    sample_matched_sparsity_parents,
)
from lagged_facial_graph_forecasting.null_random_region import construct_random_region_mapping
from lagged_facial_graph_forecasting.null_time_shuffle import (
    apply_time_shuffle_to_design_matrix,
    construct_time_shuffle_mapping,
)
from lagged_facial_graph_forecasting.pcmci_contemporaneous import apply_primary_contemporaneous_policy
from lagged_facial_graph_forecasting.pcmci_lagged_filter import filter_primary_lagged_links
from lagged_facial_graph_forecasting.pcmci_link_extraction import extract_significant_pcmciplus_links
from lagged_facial_graph_forecasting.pcmci_parent_set import convert_primary_lagged_links_to_parent_set
from lagged_facial_graph_forecasting.pcmci_runner import run_primary_pcmciplus
from lagged_facial_graph_forecasting.ridge_tuning import (
    aggregate_ridge_inner_cv_scores,
    evaluate_ridge_inner_cv,
    refit_selected_ridge,
    select_ridge_alpha,
)
from lagged_facial_graph_forecasting.runner import OuterTestLockedError, _concatenate_design_matrices
from lagged_facial_graph_forecasting.scientific_fold_lock import ScientificFoldDataLock
from lagged_facial_graph_forecasting.tigramite_adapter import face_time_series_to_tigramite_dataframe


_SOURCE_REGION = "left_cheek"
_TARGET_REGION = "mouth"
_TARGET_DIMENSION = "vy"
_TRUE_LAG = 5
_REGIONS = (_SOURCE_REGION, _TARGET_REGION, "right_cheek", "jaw", "eye")
_DIMENSIONS = ("vx", "vy")
_PRIMARY_LAGS = tuple(range(1, 11))
_ALPHA_GRID = (0.1, 1.0, 10.0)
_SEED = 20260908


def _manifest() -> SplitManifest:
    return SplitManifest(
        outer_fold=0,
        train_subject_ids=("train_1", "train_2", "train_3"),
        test_subject_ids=("outer_test",),
        inner_folds=(
            InnerFold(("train_1", "train_2"), ("train_3",)),
            InnerFold(("train_1", "train_3"), ("train_2",)),
            InnerFold(("train_2", "train_3"), ("train_1",)),
        ),
        seed=_SEED,
    )


def _series(subject_id: str, seed: int, length: int = 560) -> FaceTimeSeries:
    rng = np.random.default_rng(seed)
    values = rng.normal(scale=0.30, size=(length, len(_REGIONS), len(_DIMENSIONS)))
    source = rng.normal(size=length)
    target_noise = rng.normal(scale=0.07, size=length)
    source_index = _REGIONS.index(_SOURCE_REGION)
    target_index = _REGIONS.index(_TARGET_REGION)

    values[:, source_index, 0] = source
    values[:_TRUE_LAG, target_index, 1] = target_noise[:_TRUE_LAG]
    values[_TRUE_LAG:, target_index, 1] = (
        1.9 * source[:-_TRUE_LAG] + target_noise[_TRUE_LAG:]
    )

    return FaceTimeSeries(
        X=values,
        subject_id=subject_id,
        time_index=np.arange(length, dtype=float) / 30.0,
        region_id=_REGIONS,
        dimension=_DIMENSIONS,
        valid_mask=np.ones_like(values, dtype=bool),
        sampling_rate=30.0,
    )


def _series_by_subject() -> dict[str, FaceTimeSeries]:
    return {
        "train_1": _series("train_1", 7101),
        "train_2": _series("train_2", 7102),
        "train_3": _series("train_3", 7103),
        "outer_test": _series("outer_test", 7999),
    }


def _target_component_parent_set(parent_set: ParentSet) -> ParentSet:
    return ParentSet(
        outer_fold=parent_set.outer_fold,
        target_region=parent_set.target_region,
        parents=tuple(
            parent
            for parent in parent_set.parents
            if parent.target_dimension == _TARGET_DIMENSION
        ),
        discovery_method=parent_set.discovery_method,
    )


def _mapped_parent_set(mapping) -> ParentSet:
    return ParentSet(
        outer_fold=mapping.outer_fold,
        target_region=mapping.target_region,
        parents=mapping.mapped_parents,
        discovery_method=f"null:{mapping.condition}",
    )


def _pcmci_matrix(series: FaceTimeSeries, parent_set: ParentSet):
    return build_pcmci_parent_design_matrix(
        series,
        parent_set=parent_set,
        self_lags=_PRIMARY_LAGS,
        horizon=1,
        alignment_lag=10,
        target_dimension=_TARGET_DIMENSION,
    )


def _baseline_matrix(series: FaceTimeSeries, condition: str):
    common = dict(
        target_region=_TARGET_REGION,
        target_dimension=_TARGET_DIMENSION,
        horizon=1,
        alignment_lag=10,
    )
    if condition == "persistence":
        return build_persistence_design_matrix(series, **common)
    if condition == "self":
        return build_self_history_design_matrix(series, lags=_PRIMARY_LAGS, **common)
    if condition == "full":
        return build_full_history_design_matrix(series, lags=_PRIMARY_LAGS, **common)
    raise AssertionError(condition)


def _time_shuffle_matrix(series: FaceTimeSeries, parent_set: ParentSet, mapping):
    """Apply the frozen shuffle within one subject before any cross-subject concat."""

    return apply_time_shuffle_to_design_matrix(
        _pcmci_matrix(series, parent_set),
        mapping,
    )


def _write(path: Path, text: str) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path.read_bytes()


def _run_one_fold(root: Path):
    manifest = _manifest()
    access = ScientificFoldDataLock(manifest, _series_by_subject())

    with pytest.raises(OuterTestLockedError, match="outer-test is locked"):
        access.outer_test_series()

    train_series = access.outer_train_series()
    train_ids = tuple(series.subject_id for series in train_series)
    assert_preprocessing_fit_scope(manifest, train_ids)
    assert_discovery_fit_scope(manifest, train_ids)

    # Contract-level synthetic velocity data need no learned normalization, but the
    # fit scope is still audited before discovery. Actual PCMCI+ + ParCorr receives
    # only outer-train subjects.
    bundle = face_time_series_to_tigramite_dataframe(manifest, train_series)
    pcmci_result = run_primary_pcmciplus(manifest, bundle)
    raw_links = extract_significant_pcmciplus_links(pcmci_result, bundle)
    disposition = apply_primary_contemporaneous_policy(raw_links)
    lagged_links = filter_primary_lagged_links(disposition)
    discovered = convert_primary_lagged_links_to_parent_set(
        lagged_links,
        outer_fold=manifest.outer_fold,
        target_region=_TARGET_REGION,
    )
    parent_set = _target_component_parent_set(discovered)
    expected = ParentLink(_SOURCE_REGION, _TRUE_LAG, "vx", _TARGET_DIMENSION)
    assert expected in parent_set.parents
    assert parent_set.parents

    # All Primary Null construction is complete while outer-test remains locked.
    lag_response = construct_lag_response_grid(
        manifest,
        parent_set,
        construction_subject_ids=train_ids,
    )
    assert lag_response.status == LAG_RESPONSE_EVALUABLE
    assert tuple(point.delta for point in lag_response.points) == PRIMARY_LAG_RESPONSE_DELTAS

    random_region = construct_random_region_mapping(
        manifest,
        parent_set,
        construction_subject_ids=train_ids,
        candidate_regions=_REGIONS,
    )
    candidates = build_matched_sparsity_candidate_space(
        manifest,
        parent_set,
        construction_subject_ids=train_ids,
        region_ids=_REGIONS,
        dimension_ids=_DIMENSIONS,
    )
    sampled = sample_matched_sparsity_parents(
        manifest,
        parent_set,
        construction_subject_ids=train_ids,
        candidates=candidates,
    )
    matched = construct_matched_sparsity_mapping(
        manifest,
        parent_set,
        construction_subject_ids=train_ids,
        sampled_parents=sampled,
    )
    time_shuffle = construct_time_shuffle_mapping(
        manifest,
        parent_set,
        construction_subject_ids=train_ids,
    )

    structural_mappings = {
        "random-region": random_region,
        "matched-sparsity": matched,
    }
    for point in lag_response.points:
        if point.delta != 0:
            structural_mappings[f"lag-shift:{point.delta:+d}"] = point.mapping

    train_matrices = {
        condition: _concatenate_design_matrices(
            _baseline_matrix(series, condition) for series in train_series
        )
        for condition in ("persistence", "self", "full")
    }
    train_matrices["pcmci"] = _concatenate_design_matrices(
        _pcmci_matrix(series, parent_set) for series in train_series
    )
    for condition, mapping in structural_mappings.items():
        assert mapping is not None
        mapped_parent_set = _mapped_parent_set(mapping)
        train_matrices[condition] = _concatenate_design_matrices(
            _pcmci_matrix(series, mapped_parent_set) for series in train_series
        )
    train_matrices["time-shuffle"] = _concatenate_design_matrices(
        _time_shuffle_matrix(series, parent_set, time_shuffle)
        for series in train_series
    )

    selections = {}
    fitted = {}
    assert_ridge_tuning_scope(manifest, train_ids)
    for condition, matrix in train_matrices.items():
        assert set(matrix.subject_id) == set(manifest.train_subject_ids)
        assert not (set(matrix.subject_id) & set(manifest.test_subject_ids))
        scores = evaluate_ridge_inner_cv(
            matrix,
            split_manifest=manifest,
            alpha_grid=_ALPHA_GRID,
        )
        selection = select_ridge_alpha(aggregate_ridge_inner_cv_scores(scores))
        selections[condition] = selection
        fitted[condition] = refit_selected_ridge(
            matrix,
            split_manifest=manifest,
            selection=selection,
        )

    # FOLD FREEZE: ParentSet, NullMappings, seed, and all Ridge selections are now
    # fixed. Only after this state transition can outer-test data be obtained.
    access.freeze()
    assert not access.outer_test_locked
    test_series = access.outer_test_series()

    test_matrices = {
        condition: _concatenate_design_matrices(
            _baseline_matrix(series, condition) for series in test_series
        )
        for condition in ("persistence", "self", "full")
    }
    test_matrices["pcmci"] = _concatenate_design_matrices(
        _pcmci_matrix(series, parent_set) for series in test_series
    )
    for condition, mapping in structural_mappings.items():
        assert mapping is not None
        mapped_parent_set = _mapped_parent_set(mapping)
        test_matrices[condition] = _concatenate_design_matrices(
            _pcmci_matrix(series, mapped_parent_set) for series in test_series
        )
    test_matrices["time-shuffle"] = _concatenate_design_matrices(
        _time_shuffle_matrix(series, parent_set, time_shuffle)
        for series in test_series
    )

    predictions = {}
    metrics = {}
    reference = test_matrices["pcmci"]
    for condition, matrix in test_matrices.items():
        assert set(matrix.subject_id) == set(manifest.test_subject_ids)
        assert np.array_equal(matrix.forecast_origin, reference.forecast_origin)
        assert np.array_equal(matrix.target_time, reference.target_time)
        assert np.array_equal(matrix.y, reference.y)
        assert np.array_equal(matrix.valid_mask, reference.valid_mask)
        prediction = predict_ridge_forecaster(
            fitted[condition],
            matrix,
            outer_fold=manifest.outer_fold,
            condition=condition,
        )
        metric_results = velocity_rmse_by_subject_region(prediction)
        assert len(metric_results) == 1
        predictions[condition] = prediction
        metrics[condition] = metric_results[0]

    access.close_after_evaluation()
    with pytest.raises(OuterTestLockedError):
        access.outer_test_series()

    artifact_bytes: dict[str, bytes] = {}
    artifact_bytes["split_manifest.json"] = _write(
        root / "split_manifest.json", dumps_core_contract(manifest)
    )
    artifact_bytes["parent_set.json"] = _write(
        root / "parent_set.json", dumps_core_contract(parent_set)
    )
    for condition, mapping in sorted(structural_mappings.items()):
        assert mapping is not None
        name = f"null_{condition.replace(':', '_')}.json"
        artifact_bytes[name] = _write(root / name, dumps_core_contract(mapping))
    artifact_bytes["null_time-shuffle.json"] = _write(
        root / "null_time-shuffle.json", dumps_core_contract(time_shuffle)
    )

    selection_payload = {
        condition: {
            "alpha": selection.alpha,
            "mean_rmse": selection.mean_rmse,
            "fold_count": selection.fold_count,
            "total_val_valid": selection.total_val_valid,
        }
        for condition, selection in sorted(selections.items())
    }
    artifact_bytes["ridge_selections.json"] = _write(
        root / "ridge_selections.json",
        json.dumps(selection_payload, sort_keys=True, separators=(",", ":")),
    )
    lag_payload = {
        "status": lag_response.status,
        "deltas": [point.delta for point in lag_response.points],
        "reference_delta": 0,
    }
    artifact_bytes["lag_response.json"] = _write(
        root / "lag_response.json",
        json.dumps(lag_payload, sort_keys=True, separators=(",", ":")),
    )

    for condition in sorted(predictions):
        safe = condition.replace(":", "_")
        artifact_bytes[f"prediction_{safe}.json"] = _write(
            root / f"prediction_{safe}.json", dumps_core_contract(predictions[condition])
        )
        artifact_bytes[f"metric_{safe}.json"] = _write(
            root / f"metric_{safe}.json", dumps_core_contract(metrics[condition])
        )

    freeze_payload = {
        "outer_fold": manifest.outer_fold,
        "seed": manifest.seed,
        "parent_set_sha256": hashlib.sha256(artifact_bytes["parent_set.json"]).hexdigest(),
        "ridge_conditions": sorted(selections),
        "null_conditions": sorted(set([*structural_mappings, "time-shuffle"])),
        "outer_test_usage": "frozen_final_evaluation_only",
        "evaluated_once": True,
    }
    artifact_bytes["fold_freeze.json"] = _write(
        root / "fold_freeze.json",
        json.dumps(freeze_payload, sort_keys=True, separators=(",", ":")),
    )

    hashes = {
        name: hashlib.sha256(content).hexdigest()
        for name, content in sorted(artifact_bytes.items())
    }
    artifact_bytes["artifact_manifest.json"] = _write(
        root / "artifact_manifest.json",
        json.dumps(hashes, sort_keys=True, separators=(",", ":")),
    )
    return manifest, parent_set, selections, predictions, metrics, artifact_bytes


def test_one_outer_fold_scientific_dry_run_is_leak_free_complete_and_reproducible(
    tmp_path: Path,
) -> None:
    first = _run_one_fold(tmp_path / "run_a")
    second = _run_one_fold(tmp_path / "run_b")

    manifest_a, parent_a, selections_a, predictions_a, metrics_a, bytes_a = first
    manifest_b, parent_b, selections_b, predictions_b, metrics_b, bytes_b = second

    assert manifest_a == manifest_b
    assert parent_a == parent_b
    assert selections_a == selections_b
    assert metrics_a == metrics_b
    assert set(predictions_a) == set(predictions_b)
    assert {
        "persistence",
        "self",
        "full",
        "pcmci",
        "random-region",
        "matched-sparsity",
        "time-shuffle",
        "lag-shift:-2",
        "lag-shift:-1",
        "lag-shift:+1",
        "lag-shift:+2",
    } <= set(predictions_a)
    for condition in predictions_a:
        left = predictions_a[condition]
        right = predictions_b[condition]
        assert np.array_equal(left.y_pred, right.y_pred)
        assert np.array_equal(left.y_true, right.y_true)
        assert np.array_equal(left.valid_mask, right.valid_mask)

    assert bytes_a == bytes_b
    required = {
        "split_manifest.json",
        "parent_set.json",
        "ridge_selections.json",
        "lag_response.json",
        "null_random-region.json",
        "null_matched-sparsity.json",
        "null_time-shuffle.json",
        "fold_freeze.json",
        "artifact_manifest.json",
    }
    assert required <= set(bytes_a)
