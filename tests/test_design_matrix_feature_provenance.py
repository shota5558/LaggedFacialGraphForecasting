from __future__ import annotations

import numpy as np

from lagged_facial_graph_forecasting import FaceTimeSeries, ParentLink, ParentSet
from lagged_facial_graph_forecasting.design_matrix import (
    build_full_history_design_matrix,
    build_pcmci_parent_design_matrix,
    build_persistence_design_matrix,
    build_self_history_design_matrix,
    decode_feature_name,
    encode_feature_name,
)


def _series() -> FaceTimeSeries:
    X = np.arange(6 * 2 * 2, dtype=float).reshape(6, 2, 2)
    return FaceTimeSeries(
        X=X,
        subject_id="s01",
        time_index=np.arange(6, dtype=float),
        region_id=("mouth", "jaw"),
        dimension=("vx", "vy"),
        valid_mask=np.ones_like(X, dtype=bool),
        sampling_rate=30.0,
    )


def _provenance(matrix) -> tuple[tuple[str, int], ...]:
    return tuple(zip(matrix.feature_names, matrix.feature_lags, strict=True))


def test_persistence_and_self_feature_provenance_is_lossless() -> None:
    series = _series()

    persistence = build_persistence_design_matrix(series, target_region="mouth")
    self_history = build_self_history_design_matrix(
        series, target_region="mouth", lags=(1, 3)
    )

    assert _provenance(persistence) == (("mouth.vx", 1), ("mouth.vy", 1))
    assert _provenance(self_history) == (
        ("mouth.vx", 1),
        ("mouth.vy", 1),
        ("mouth.vx", 3),
        ("mouth.vy", 3),
    )
    assert set(persistence.region_id) == {"mouth"}
    assert set(self_history.region_id) == {"mouth"}


def test_full_feature_provenance_tracks_region_dimension_and_lag() -> None:
    matrix = build_full_history_design_matrix(
        _series(), target_region="mouth", lags=(2, 1)
    )

    assert _provenance(matrix) == (
        ("mouth.vx", 1),
        ("mouth.vy", 1),
        ("jaw.vx", 1),
        ("jaw.vy", 1),
        ("mouth.vx", 2),
        ("mouth.vy", 2),
        ("jaw.vx", 2),
        ("jaw.vy", 2),
    )
    assert set(matrix.region_id) == {"mouth"}


def test_pcmci_feature_provenance_matches_parent_set_without_reselection() -> None:
    parent_set = ParentSet(
        outer_fold=2,
        target_region="mouth",
        parents=(
            ParentLink(source_region="jaw", lag=3),
            ParentLink(source_region="mouth", lag=1),
        ),
    )

    matrix = build_pcmci_parent_design_matrix(_series(), parent_set=parent_set)

    assert _provenance(matrix) == (
        ("jaw.vx", 3),
        ("jaw.vy", 3),
        ("mouth.vx", 1),
        ("mouth.vy", 1),
    )
    assert set(matrix.region_id) == {parent_set.target_region}
    assert tuple(dict.fromkeys(matrix.feature_lags)) == tuple(
        parent.lag for parent in parent_set.parents
    )


def test_feature_name_codec_is_lossless_for_reserved_characters() -> None:
    provenance = (
        ("mouth.part", "v.x"),
        ("jaw\\side", "v\\y"),
        ("eye.part\\inner", "velocity.x\\raw"),
    )

    encoded = tuple(encode_feature_name(region, dimension) for region, dimension in provenance)

    assert len(set(encoded)) == len(encoded)
    assert tuple(decode_feature_name(name) for name in encoded) == provenance
    assert encode_feature_name("mouth", "vx") == "mouth.vx"
    assert decode_feature_name("mouth.vx") == ("mouth", "vx")


def test_matrix_feature_names_round_trip_to_exact_source_region_and_dimension() -> None:
    X = np.arange(5 * 2 * 2, dtype=float).reshape(5, 2, 2)
    series = FaceTimeSeries(
        X=X,
        subject_id="s-special",
        time_index=np.arange(5, dtype=float),
        region_id=("mouth.part", "jaw\\side"),
        dimension=("v.x", "v\\y"),
        valid_mask=np.ones_like(X, dtype=bool),
        sampling_rate=30.0,
    )

    matrix = build_full_history_design_matrix(
        series, target_region="mouth.part", lags=(1,)
    )

    decoded = tuple(decode_feature_name(name) for name in matrix.feature_names)
    assert decoded == (
        ("mouth.part", "v.x"),
        ("mouth.part", "v\\y"),
        ("jaw\\side", "v.x"),
        ("jaw\\side", "v\\y"),
    )
    assert matrix.feature_lags == (1, 1, 1, 1)
    assert set(matrix.region_id) == {"mouth.part"}
