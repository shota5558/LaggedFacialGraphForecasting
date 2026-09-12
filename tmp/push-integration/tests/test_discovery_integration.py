from __future__ import annotations

import numpy as np

from lagged_facial_graph_forecasting.contracts import FaceTimeSeries, InnerFold, SplitManifest
from lagged_facial_graph_forecasting.core_contract_io import dumps_core_contract, loads_core_contract
from lagged_facial_graph_forecasting.core_contracts import ParentLink, ParentSet
from lagged_facial_graph_forecasting.design_matrix import build_pcmci_parent_design_matrix
from lagged_facial_graph_forecasting.forecaster import predict_ridge_forecaster
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
from lagged_facial_graph_forecasting.runner import _concatenate_design_matrices
from lagged_facial_graph_forecasting.tigramite_adapter import face_time_series_to_tigramite_dataframe


_SOURCE_REGION = "left_cheek"
_TARGET_REGION = "mouth"
_TARGET_DIMENSION = "vy"
_TRUE_LAG = 2
_ALIGNMENT_LAG = 10
_ALPHA_GRID = (0.1, 1.0, 10.0)


def _manifest() -> SplitManifest:
    return SplitManifest(
        outer_fold=0,
        train_subject_ids=("train_1", "train_2", "train_3"),
        test_subject_ids=("outer_test",),
        inner_folds=(
            InnerFold(("train_1", "train_2"), ("train_3",)),
            InnerFold(("train_2", "train_3"), ("train_1",)),
            InnerFold(("train_1", "train_3"), ("train_2",)),
        ),
        seed=20260908,
    )


def _series(subject_id: str, seed: int, length: int = 420) -> FaceTimeSeries:
    rng = np.random.default_rng(seed)
    values = rng.normal(scale=0.30, size=(length, 2, 2))
    source = rng.normal(size=length)
    target_noise = rng.normal(scale=0.08, size=length)

    values[:, 0, 0] = source
    values[:_TRUE_LAG, 1, 1] = target_noise[:_TRUE_LAG]
    values[_TRUE_LAG:, 1, 1] = 1.7 * source[:-_TRUE_LAG] + target_noise[_TRUE_LAG:]

    return FaceTimeSeries(
        X=values,
        subject_id=subject_id,
        time_index=np.arange(length, dtype=float) / 25.0,
        region_id=(_SOURCE_REGION, _TARGET_REGION),
        dimension=("vx", "vy"),
        valid_mask=np.ones_like(values, dtype=bool),
        sampling_rate=25.0,
    )


def _discover_parent_set(
    manifest: SplitManifest,
    series_by_subject: dict[str, FaceTimeSeries],
) -> ParentSet:
    train_series = tuple(series_by_subject[sid] for sid in manifest.train_subject_ids)
    bundle = face_time_series_to_tigramite_dataframe(manifest, train_series)
    result = run_primary_pcmciplus(manifest, bundle)
    links = extract_significant_pcmciplus_links(result, bundle)
    disposition = apply_primary_contemporaneous_policy(links)
    lagged = filter_primary_lagged_links(disposition)
    return convert_primary_lagged_links_to_parent_set(
        lagged,
        outer_fold=manifest.outer_fold,
        target_region=_TARGET_REGION,
    )


def _build_matrix(series: FaceTimeSeries, parent_set: ParentSet):
    return build_pcmci_parent_design_matrix(
        series,
        parent_set=parent_set,
        self_lags=(1, 2, 3),
        horizon=1,
        alignment_lag=_ALIGNMENT_LAG,
        target_dimension=_TARGET_DIMENSION,
    )


def test_gate2_discovery_parentset_ridge_prediction_is_reproducible_and_leak_free() -> None:
    manifest = _manifest()
    series_by_subject = {
        "train_1": _series("train_1", 9101),
        "train_2": _series("train_2", 9102),
        "train_3": _series("train_3", 9103),
        "outer_test": _series("outer_test", 9999),
    }

    # Discovery receives outer-train only. Repeat the same frozen discovery to
    # require a reproducible ParentSet before any outer-test matrix is materialized.
    parent_set_a = _discover_parent_set(manifest, series_by_subject)
    parent_set_b = _discover_parent_set(manifest, series_by_subject)
    assert parent_set_a == parent_set_b
    assert ParentLink(_SOURCE_REGION, _TRUE_LAG, "vx", _TARGET_DIMENSION) in parent_set_a.parents

    serialized = dumps_core_contract(parent_set_a)
    restored = loads_core_contract(serialized)
    assert isinstance(restored, ParentSet)
    assert restored == parent_set_a
    assert dumps_core_contract(restored) == serialized

    train_matrix = _concatenate_design_matrices(
        _build_matrix(series_by_subject[sid], restored)
        for sid in manifest.train_subject_ids
    )
    assert set(train_matrix.subject_id) == set(manifest.train_subject_ids)
    assert not (set(train_matrix.subject_id) & set(manifest.test_subject_ids))
    assert f"{_SOURCE_REGION}.vx" in train_matrix.feature_names
    known_feature = train_matrix.feature_names.index(f"{_SOURCE_REGION}.vx")
    assert train_matrix.feature_lags[known_feature] == _TRUE_LAG

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

    # Outer-test is unlocked only after discovery, ParentSet serialization,
    # inner-CV selection, and complete outer-train refit are frozen.
    test_matrix = _concatenate_design_matrices(
        _build_matrix(series_by_subject[sid], restored)
        for sid in manifest.test_subject_ids
    )
    prediction = predict_ridge_forecaster(
        fitted,
        test_matrix,
        outer_fold=manifest.outer_fold,
        condition="pcmci",
    )

    assert set(prediction.subject_id) == set(manifest.test_subject_ids)
    assert prediction.condition == "pcmci"
    assert prediction.target_dimensions == (_TARGET_DIMENSION,)
    assert prediction.y_true.ndim == 1
    assert prediction.y_pred.ndim == 1
    assert np.all(np.isfinite(prediction.y_pred[prediction.valid_mask]))
    assert fitted.feature_names == train_matrix.feature_names == test_matrix.feature_names
    assert fitted.feature_lags == train_matrix.feature_lags == test_matrix.feature_lags
