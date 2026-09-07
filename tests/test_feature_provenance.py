from __future__ import annotations

import numpy as np

from lagged_facial_graph_forecasting import (
    FaceTimeSeries,
    ParentLink,
    ParentSet,
    build_full_history_design_matrix,
    build_pcmci_parent_design_matrix,
    build_self_history_design_matrix,
)


def _series() -> FaceTimeSeries:
    X = np.arange(7 * 3 * 2, dtype=float).reshape(7, 3, 2)
    return FaceTimeSeries(
        X=X,
        subject_id="s01",
        time_index=np.arange(7, dtype=float),
        region_id=("mouth", "jaw", "eye"),
        dimension=("vx", "vy"),
        valid_mask=np.ones_like(X, dtype=bool),
        sampling_rate=30.0,
    )


def _assert_column_provenance(matrix, expected: tuple[tuple[str, int], ...]) -> None:
    observed = tuple(zip(matrix.feature_names, matrix.feature_lags, strict=True))
    assert observed == expected
    assert matrix.X.shape[1] == len(matrix.feature_names) == len(matrix.feature_lags)
    assert all(lag >= 1 for lag in matrix.feature_lags)


def test_self_history_feature_provenance_is_exact_and_column_aligned() -> None:
    matrix = build_self_history_design_matrix(
        _series(), target_region="mouth", lags=(1, 3), horizon=1
    )

    _assert_column_provenance(
        matrix,
        (
            ("mouth.vx", 1),
            ("mouth.vy", 1),
            ("mouth.vx", 3),
            ("mouth.vy", 3),
        ),
    )


def test_full_history_feature_provenance_identifies_region_dimension_and_lag() -> None:
    matrix = build_full_history_design_matrix(
        _series(), target_region="mouth", lags=(2,), horizon=1
    )

    _assert_column_provenance(
        matrix,
        (
            ("mouth.vx", 2),
            ("mouth.vy", 2),
            ("jaw.vx", 2),
            ("jaw.vy", 2),
            ("eye.vx", 2),
            ("eye.vy", 2),
        ),
    )


def test_pcmci_feature_provenance_matches_frozen_parent_set_exactly() -> None:
    parent_set = ParentSet(
        outer_fold=0,
        target_region="mouth",
        parents=(
            ParentLink(source_region="eye", lag=3),
            ParentLink(source_region="jaw", lag=1),
        ),
    )
    matrix = build_pcmci_parent_design_matrix(
        _series(), parent_set=parent_set, horizon=1
    )

    _assert_column_provenance(
        matrix,
        (
            ("eye.vx", 3),
            ("eye.vy", 3),
            ("jaw.vx", 1),
            ("jaw.vy", 1),
        ),
    )
    assert tuple(name.split(".", 1)[0] for name in matrix.feature_names) == (
        "eye",
        "eye",
        "jaw",
        "jaw",
    )
