from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting.quality import (
    QualityFlagError,
    build_quality_flags,
)


def test_finite_values_define_elementwise_validity_without_quality_scores() -> None:
    values = np.array(
        [
            [[1.0, 2.0], [3.0, np.nan]],
            [[4.0, 5.0], [np.inf, 6.0]],
        ]
    )

    flags = build_quality_flags(values)

    expected = np.isfinite(values)
    assert np.array_equal(flags.finite_mask, expected)
    assert np.array_equal(flags.valid_mask, expected)
    assert np.array_equal(flags.frame_quality_pass, np.array([True, True]))
    assert flags.minimum_frame_quality is None
    assert flags.method == "finite_values_only"


def test_explicit_frame_quality_threshold_invalidates_entire_frame() -> None:
    values = np.ones((3, 2, 2), dtype=float)
    scores = np.array([0.95, 0.70, 0.80])

    flags = build_quality_flags(
        values,
        frame_quality=scores,
        minimum_frame_quality=0.80,
    )

    assert np.array_equal(flags.frame_quality_pass, np.array([True, False, True]))
    assert np.all(flags.valid_mask[0])
    assert not np.any(flags.valid_mask[1])
    assert np.all(flags.valid_mask[2])
    assert flags.minimum_frame_quality == pytest.approx(0.80)
    assert flags.method == "finite_values_and_explicit_frame_quality_threshold"


def test_nonfinite_external_quality_fails_explicit_threshold() -> None:
    values = np.ones((3, 1, 1), dtype=float)

    flags = build_quality_flags(
        values,
        frame_quality=np.array([0.9, np.nan, np.inf]),
        minimum_frame_quality=0.5,
    )

    assert np.array_equal(flags.frame_quality_pass, np.array([True, False, False]))


def test_frame_quality_never_repairs_nonfinite_elements() -> None:
    values = np.array([[[1.0, np.nan]], [[2.0, 3.0]]])

    flags = build_quality_flags(
        values,
        frame_quality=np.array([1.0, 1.0]),
        minimum_frame_quality=0.0,
    )

    assert not flags.valid_mask[0, 0, 1]
    assert flags.valid_mask[0, 0, 0]
    assert np.all(flags.valid_mask[1])


def test_quality_threshold_cannot_be_inferred_or_used_without_scores() -> None:
    values = np.ones((2, 1, 1), dtype=float)

    with pytest.raises(QualityFlagError, match="requires an explicit minimum_frame_quality"):
        build_quality_flags(values, frame_quality=np.array([0.9, 0.8]))
    with pytest.raises(QualityFlagError, match="requires explicit frame_quality"):
        build_quality_flags(values, minimum_frame_quality=0.5)


def test_invalid_quality_inputs_are_rejected() -> None:
    values = np.ones((2, 1, 1), dtype=float)

    with pytest.raises(QualityFlagError, match="shape"):
        build_quality_flags(
            values,
            frame_quality=np.array([0.8]),
            minimum_frame_quality=0.5,
        )
    with pytest.raises(QualityFlagError, match="numeric"):
        build_quality_flags(
            values,
            frame_quality=np.array([True, False]),
            minimum_frame_quality=0.5,
        )
    with pytest.raises(QualityFlagError, match="finite"):
        build_quality_flags(
            values,
            frame_quality=np.array([0.8, 0.9]),
            minimum_frame_quality=np.nan,
        )


def test_quality_flags_are_immutable_and_inputs_are_unchanged() -> None:
    values = np.array([[[1.0]], [[2.0]]], dtype=float)
    scores = np.array([0.8, 0.9], dtype=float)
    values_before = values.copy()
    scores_before = scores.copy()

    flags = build_quality_flags(
        values,
        frame_quality=scores,
        minimum_frame_quality=0.7,
    )

    assert np.array_equal(values, values_before)
    assert np.array_equal(scores, scores_before)
    assert not flags.valid_mask.flags.writeable
    assert not flags.finite_mask.flags.writeable
    assert not flags.frame_quality_pass.flags.writeable
