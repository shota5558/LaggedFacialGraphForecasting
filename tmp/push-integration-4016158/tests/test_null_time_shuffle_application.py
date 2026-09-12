from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting.contracts import DesignMatrix, InnerFold, SplitManifest
from lagged_facial_graph_forecasting.core_contracts import ParentLink, ParentSet
from lagged_facial_graph_forecasting.null_time_shuffle import (
    TimeShuffleMappingError,
    apply_time_shuffle_to_design_matrix,
    construct_time_shuffle_mapping,
    generate_time_shuffle_permutation,
)


def _manifest() -> SplitManifest:
    return SplitManifest(
        outer_fold=3,
        train_subject_ids=("train_a", "train_b"),
        test_subject_ids=("test_a",),
        inner_folds=(InnerFold(("train_a",), ("train_b",)),),
        seed=404,
    )


def _mapping(*, source_region: str = "jaw"):
    return construct_time_shuffle_mapping(
        _manifest(),
        ParentSet(
            outer_fold=3,
            target_region="mouth",
            parents=(ParentLink(source_region, 2, "vx", "vx"),),
            discovery_method="pcmci_plus",
        ),
        construction_subject_ids=("train_a", "train_b"),
    )


def _empty_mapping():
    return construct_time_shuffle_mapping(
        _manifest(),
        ParentSet(
            outer_fold=3,
            target_region="mouth",
            parents=(),
            discovery_method="pcmci_plus",
        ),
        construction_subject_ids=("train_a", "train_b"),
    )


def _matrix(
    *,
    subject_ids: tuple[str, ...] = ("s01", "s01", "s01", "s01"),
    region_ids: tuple[str, ...] = ("mouth", "mouth", "mouth", "mouth"),
) -> DesignMatrix:
    return DesignMatrix(
        X=np.array(
            [
                [10.0, 20.0, 100.0],
                [11.0, 21.0, 101.0],
                [12.0, 22.0, 102.0],
                [13.0, 23.0, 103.0],
            ]
        ),
        y=np.array([30.0, 31.0, 32.0, 33.0]),
        subject_id=subject_ids,
        region_id=region_ids,
        target_dimensions=("vx",),
        forecast_origin=np.array([1.0, 2.0, 3.0, 4.0]),
        target_time=np.array([2.0, 3.0, 4.0, 5.0]),
        feature_names=("mouth.vx", "mouth.vy", "jaw.vx"),
        feature_lags=(1, 1, 2),
        valid_mask=np.array([True, False, True, True]),
    )


def _seed_permutation(matrix: DesignMatrix | None = None) -> tuple[int, ...]:
    matrix = _matrix() if matrix is None else matrix
    return generate_time_shuffle_permutation(
        _mapping(), row_count=int(np.count_nonzero(matrix.valid_mask))
    )


def test_application_shuffles_only_cross_region_block_within_valid_support() -> None:
    matrix = _matrix()
    permutation = _seed_permutation(matrix)
    shuffled = apply_time_shuffle_to_design_matrix(matrix, _mapping())

    # Self-history is fixed for every row.
    assert np.array_equal(shuffled.X[:, :2], matrix.X[:, :2])

    valid_rows = np.flatnonzero(matrix.valid_mask)
    expected_cross = np.array(matrix.X[:, 2], copy=True)
    expected_cross[valid_rows] = matrix.X[valid_rows[np.asarray(permutation)], 2]
    assert np.array_equal(shuffled.X[:, 2], expected_cross)

    # Invalid row 1 is neither donor nor recipient and sample support is unchanged.
    assert np.array_equal(shuffled.valid_mask, matrix.valid_mask)
    assert shuffled.X[1, 2] == matrix.X[1, 2]

    assert np.array_equal(shuffled.y, matrix.y)
    assert np.array_equal(shuffled.forecast_origin, matrix.forecast_origin)
    assert np.array_equal(shuffled.target_time, matrix.target_time)
    assert shuffled.subject_id == matrix.subject_id
    assert shuffled.region_id == matrix.region_id
    assert shuffled.target_dimensions == matrix.target_dimensions
    assert shuffled.feature_names == matrix.feature_names
    assert shuffled.feature_lags == matrix.feature_lags


def test_application_is_deterministic_from_frozen_seed_without_explicit_permutation() -> None:
    first = apply_time_shuffle_to_design_matrix(_matrix(), _mapping())
    second = apply_time_shuffle_to_design_matrix(_matrix(), _mapping())
    assert np.array_equal(first.X, second.X)
    assert np.array_equal(first.y, second.y)
    assert np.array_equal(first.valid_mask, second.valid_mask)


def test_application_accepts_explicit_replay_only_when_it_matches_f41_generator() -> None:
    matrix = _matrix()
    expected = _seed_permutation(matrix)
    automatic = apply_time_shuffle_to_design_matrix(matrix, _mapping())
    replay = apply_time_shuffle_to_design_matrix(
        matrix, _mapping(), permutation=expected
    )
    assert np.array_equal(automatic.X, replay.X)


def test_application_rejects_arbitrary_valid_nonidentity_permutation() -> None:
    matrix = _matrix()
    expected = _seed_permutation(matrix)
    alternatives = (
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    )
    arbitrary = next(candidate for candidate in alternatives if candidate != expected)
    with pytest.raises(TimeShuffleMappingError, match="frozen-seed F-41"):
        apply_time_shuffle_to_design_matrix(
            matrix, _mapping(), permutation=arbitrary
        )


def test_application_rejects_cross_region_provenance_not_matching_parentset() -> None:
    with pytest.raises(TimeShuffleMappingError, match="provenance"):
        apply_time_shuffle_to_design_matrix(_matrix(), _mapping(source_region="eye"))


def test_application_rejects_cross_subject_permutation() -> None:
    matrix = _matrix(subject_ids=("s01", "s01", "s02", "s02"))
    with pytest.raises(TimeShuffleMappingError, match="within one subject"):
        apply_time_shuffle_to_design_matrix(matrix, _mapping())


def test_application_rejects_target_region_mismatch() -> None:
    matrix = _matrix(region_ids=("jaw", "jaw", "jaw", "jaw"))
    with pytest.raises(TimeShuffleMappingError, match="target_region"):
        apply_time_shuffle_to_design_matrix(matrix, _mapping())


def test_application_rejects_permutation_length_not_matching_valid_support() -> None:
    with pytest.raises(TimeShuffleMappingError, match="length"):
        apply_time_shuffle_to_design_matrix(
            _matrix(), _mapping(), permutation=(1, 0)
        )


def test_application_rejects_identity_permutation() -> None:
    with pytest.raises(TimeShuffleMappingError, match="identity"):
        apply_time_shuffle_to_design_matrix(
            _matrix(), _mapping(), permutation=(0, 1, 2)
        )


def test_application_rejects_invalid_permutation_indices() -> None:
    with pytest.raises(TimeShuffleMappingError, match="every valid-row index"):
        apply_time_shuffle_to_design_matrix(
            _matrix(), _mapping(), permutation=(0, 0, 2)
        )


def test_application_requires_at_least_two_valid_rows() -> None:
    matrix = DesignMatrix(
        X=np.array([[10.0, 20.0, 100.0], [11.0, 21.0, 101.0]]),
        y=np.array([30.0, 31.0]),
        subject_id=("s01", "s01"),
        region_id=("mouth", "mouth"),
        target_dimensions=("vx",),
        forecast_origin=np.array([1.0, 2.0]),
        target_time=np.array([2.0, 3.0]),
        feature_names=("mouth.vx", "mouth.vy", "jaw.vx"),
        feature_lags=(1, 1, 2),
        valid_mask=np.array([True, False]),
    )
    with pytest.raises(TimeShuffleMappingError, match="row_count"):
        apply_time_shuffle_to_design_matrix(matrix, _mapping())


def test_application_rejects_multioutput_matrix() -> None:
    matrix = DesignMatrix(
        X=np.array([[10.0, 20.0, 100.0], [11.0, 21.0, 101.0]]),
        y=np.array([[30.0, 40.0], [31.0, 41.0]]),
        subject_id=("s01", "s01"),
        region_id=("mouth", "mouth"),
        target_dimensions=("vx", "vy"),
        forecast_origin=np.array([1.0, 2.0]),
        target_time=np.array([2.0, 3.0]),
        feature_names=("mouth.vx", "mouth.vy", "jaw.vx"),
        feature_lags=(1, 1, 2),
        valid_mask=np.array([True, True]),
    )
    with pytest.raises(TimeShuffleMappingError, match="scalar target-component"):
        apply_time_shuffle_to_design_matrix(matrix, _mapping())


def test_application_rejects_noop_when_target_component_has_no_cross_region_parent() -> None:
    base = _matrix()
    self_only = DesignMatrix(
        X=np.array(base.X[:, :2], copy=True),
        y=np.array(base.y, copy=True),
        subject_id=base.subject_id,
        region_id=base.region_id,
        target_dimensions=base.target_dimensions,
        forecast_origin=np.array(base.forecast_origin, copy=True),
        target_time=np.array(base.target_time, copy=True),
        feature_names=base.feature_names[:2],
        feature_lags=base.feature_lags[:2],
        valid_mask=np.array(base.valid_mask, copy=True),
    )
    with pytest.raises(TimeShuffleMappingError, match="unevaluable without cross-region"):
        apply_time_shuffle_to_design_matrix(self_only, _empty_mapping())
