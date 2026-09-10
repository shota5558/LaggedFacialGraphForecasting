from __future__ import annotations

import numpy as np

from lagged_facial_graph_forecasting.contracts import FaceTimeSeries, InnerFold, SplitManifest
from lagged_facial_graph_forecasting.core_contracts import NullMapping, ParentLink, ParentSet
from lagged_facial_graph_forecasting.design_matrix import build_pcmci_parent_design_matrix
from lagged_facial_graph_forecasting.forecaster import (
    fit_ridge_forecaster,
    predict_ridge_forecaster,
)
from lagged_facial_graph_forecasting.null_lag_shift import construct_lag_shift_mapping
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


REGIONS = ("mouth", "jaw", "eye", "cheek", "brow")
DIMENSIONS = ("vx", "vy")
ALIGNMENT_LAG = 10
SELF_LAGS = (1,)


def _manifest() -> SplitManifest:
    return SplitManifest(
        outer_fold=2,
        train_subject_ids=("train_a", "train_b"),
        test_subject_ids=("test_a",),
        inner_folds=(InnerFold(("train_a",), ("train_b",)),),
        seed=2026,
    )


def _series(subject_id: str, *, phase: float) -> FaceTimeSeries:
    t = np.arange(64, dtype=float)
    X = np.zeros((len(t), len(REGIONS), len(DIMENSIONS)), dtype=float)

    for region_index in range(len(REGIONS)):
        offset = 0.17 * region_index
        X[:, region_index, 0] = (
            np.sin(0.11 * t + phase + offset)
            + 0.015 * t
            + 0.03 * region_index
        )
        X[:, region_index, 1] = (
            np.cos(0.07 * t + 0.5 * phase + offset)
            - 0.01 * t
            + 0.02 * region_index
        )

    # Give the target a stable dependence on its own history and two cross-region
    # channels. The integration test does not assert performance superiority; it only
    # requires every frozen condition to traverse the same forecasting backbone.
    mouth = REGIONS.index("mouth")
    jaw = REGIONS.index("jaw")
    eye = REGIONS.index("eye")
    for index in range(4, len(t)):
        X[index, mouth, 0] = (
            0.55 * X[index - 1, mouth, 0]
            + 0.30 * X[index - 2, jaw, 0]
            + 0.15 * X[index - 4, eye, 1]
        )

    return FaceTimeSeries(
        X=X,
        subject_id=subject_id,
        time_index=t,
        region_id=REGIONS,
        dimension=DIMENSIONS,
        valid_mask=np.ones_like(X, dtype=bool),
        sampling_rate=30.0,
    )


def _parent_set() -> ParentSet:
    return ParentSet(
        outer_fold=2,
        target_region="mouth",
        parents=(
            ParentLink("jaw", 2, "vx", "vx"),
            ParentLink("eye", 4, "vy", "vx"),
        ),
        discovery_method="pcmci_plus",
    )


def _mapped_parent_set(mapping: NullMapping) -> ParentSet:
    return ParentSet(
        outer_fold=mapping.outer_fold,
        target_region=mapping.target_region,
        parents=mapping.mapped_parents,
        discovery_method=f"null:{mapping.condition}",
    )


def _matrix(series: FaceTimeSeries, parent_set: ParentSet):
    return build_pcmci_parent_design_matrix(
        series,
        parent_set=parent_set,
        self_lags=SELF_LAGS,
        horizon=1,
        alignment_lag=ALIGNMENT_LAG,
        target_dimension="vx",
    )


def _null_mappings() -> dict[str, NullMapping]:
    manifest = _manifest()
    parent_set = _parent_set()
    scope = manifest.train_subject_ids

    lag_shift = construct_lag_shift_mapping(
        manifest,
        parent_set,
        construction_subject_ids=scope,
        lag_delta=1,
    )
    random_region = construct_random_region_mapping(
        manifest,
        parent_set,
        construction_subject_ids=scope,
        candidate_regions=REGIONS,
    )
    candidates = build_matched_sparsity_candidate_space(
        manifest,
        parent_set,
        construction_subject_ids=scope,
        region_ids=REGIONS,
        dimension_ids=DIMENSIONS,
    )
    sampled = sample_matched_sparsity_parents(
        manifest,
        parent_set,
        construction_subject_ids=scope,
        candidates=candidates,
    )
    matched_sparsity = construct_matched_sparsity_mapping(
        manifest,
        parent_set,
        construction_subject_ids=scope,
        sampled_parents=sampled,
    )
    time_shuffle = construct_time_shuffle_mapping(
        manifest,
        parent_set,
        construction_subject_ids=scope,
    )

    return {
        "lag-shift": lag_shift,
        "random-region": random_region,
        "matched-sparsity": matched_sparsity,
        "time-shuffle": time_shuffle,
    }


def test_primary_nulls_integrate_through_same_design_ridge_prediction_backbone() -> None:
    manifest = _manifest()
    parent_set = _parent_set()
    mappings = _null_mappings()
    train_series = _series("train_a", phase=0.0)
    test_series = _series("test_a", phase=0.35)

    train_matrices = {"pcmci": _matrix(train_series, parent_set)}
    test_matrices = {"pcmci": _matrix(test_series, parent_set)}

    for condition in ("lag-shift", "random-region", "matched-sparsity"):
        mapped_parent_set = _mapped_parent_set(mappings[condition])
        train_matrices[condition] = _matrix(train_series, mapped_parent_set)
        test_matrices[condition] = _matrix(test_series, mapped_parent_set)

    # Time-shuffle uses the same PCMCI-selected ParentSet and then destroys only the
    # cross-region temporal row correspondence inside each subject/support.
    train_matrices["time-shuffle"] = apply_time_shuffle_to_design_matrix(
        train_matrices["pcmci"], mappings["time-shuffle"]
    )
    test_matrices["time-shuffle"] = apply_time_shuffle_to_design_matrix(
        test_matrices["pcmci"], mappings["time-shuffle"]
    )

    reference_test = test_matrices["pcmci"]
    fitted_models = {}
    predictions = {}
    for condition, train_matrix in train_matrices.items():
        fitted = fit_ridge_forecaster(train_matrix, alpha=1.0)
        artifact = predict_ridge_forecaster(
            fitted,
            test_matrices[condition],
            outer_fold=manifest.outer_fold,
            condition=condition,
        )
        fitted_models[condition] = fitted
        predictions[condition] = artifact

    assert set(predictions) == {
        "pcmci",
        "lag-shift",
        "random-region",
        "matched-sparsity",
        "time-shuffle",
    }

    # Null identity changes must not change which outer-test samples are evaluated.
    for condition, matrix in test_matrices.items():
        assert matrix.subject_id == reference_test.subject_id
        assert matrix.region_id == reference_test.region_id
        assert matrix.target_dimensions == reference_test.target_dimensions
        assert np.array_equal(matrix.forecast_origin, reference_test.forecast_origin)
        assert np.array_equal(matrix.target_time, reference_test.target_time)
        assert np.array_equal(matrix.y, reference_test.y)
        assert np.array_equal(matrix.valid_mask, reference_test.valid_mask)
        assert set(matrix.subject_id) == set(manifest.test_subject_ids)

        artifact = predictions[condition]
        assert artifact.outer_fold == manifest.outer_fold
        assert artifact.condition == condition
        assert set(artifact.subject_id) == set(manifest.test_subject_ids)
        assert np.array_equal(artifact.y_true, reference_test.y)
        assert np.array_equal(artifact.valid_mask, reference_test.valid_mask)
        assert np.all(np.isfinite(artifact.y_pred[artifact.valid_mask]))

    # Every condition uses exactly the same sklearn StandardScaler -> Ridge pipeline
    # and the same fixed alpha. Only the frozen input information set changes.
    for fitted in fitted_models.values():
        assert tuple(fitted.pipeline.named_steps) == ("scaler", "ridge")
        assert fitted.alpha == 1.0
        assert fitted.training_row_count == int(
            np.count_nonzero(train_matrices["pcmci"].valid_mask)
        )

    # Same support plus matched cardinality means every condition enters Ridge with
    # the same model capacity; structural controls change only their intended input.
    feature_counts = {matrix.X.shape[1] for matrix in train_matrices.values()}
    assert feature_counts == {train_matrices["pcmci"].X.shape[1]}

    # Time-shuffle keeps exact feature provenance while changing valid-row values;
    # structural Nulls encode their mapped region/lag provenance explicitly.
    assert test_matrices["time-shuffle"].feature_names == reference_test.feature_names
    assert test_matrices["time-shuffle"].feature_lags == reference_test.feature_lags
    assert not np.array_equal(
        test_matrices["time-shuffle"].X[reference_test.valid_mask],
        reference_test.X[reference_test.valid_mask],
    )
    for condition in ("lag-shift", "random-region", "matched-sparsity"):
        assert (
            test_matrices[condition].feature_names,
            test_matrices[condition].feature_lags,
        ) != (reference_test.feature_names, reference_test.feature_lags)
