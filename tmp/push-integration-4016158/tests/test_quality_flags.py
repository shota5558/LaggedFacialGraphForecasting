from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting.quality import (
    QualityFlagError,
    build_quality_flags,
)


def test_finite_values_only_produces_elementwise_mask() -> None:
    values = np.array(
        [
            [[1.0, 2.0], [3.0, np.nan]],
            [[4.0, 5.0], [np.inf, 6.0]],
        ]
    )

    result = build_quality_flags(values)

    assert np.array_equal(result.valid_mask, np.isfinite(values))
    assert np.array_equal(result.finite_mask, np.isfinite(values))
    assert np.array_equal(result.frame_quality_pass, np.array([True, True]))
    assert result.minimum_frame_quality is None
    assert result.method == "finite_values_only"


def test_explicit_frame_quality_threshold_invalidates_whole_frame() -> None:
    values = np.ones((3, 2, 2), dtype=float)
    scores = np.array([0.9, 0.49, 0.5])

    result = build_quality_flags(
        values,
        frame_quality=scores,
        minimum_frame_quality=0.5,
    )

    assert np.array_equal(result.frame_quality_pass, np.array([True, False, True]))
    assert result.valid_mask[0].all()
    assert not result.valid_mask[1].any()
    assert result.valid_mask[2].all()
    assert result.minimum_frame_quality == pytest.approx(0.5)
    assert result.method == "finite_values_and_explicit_frame_quality_threshold"


def test_nonfinite_quality_score_fails_threshold_without_imputation() -> None:
    values = np.ones((2, 1, 2), dtype=float)

    result = build_quality_flags(
        values,
        frame_quality=np.array([0.9, np.nan]),
        minimum_frame_quality=0.5,
    )

    assert np.array_equal(result.frame_quality_pass, np.array([True, False]))
    assert not result.valid_mask[1].any()


def test_external_quality_does_not_hide_elementwise_nonfinite_values() -> None:
    values = np.ones((2, 2, 1), dtype=float)
    values[0, 1, 0] = np.nan

    result = build_quality_flags(
        values,
        frame_quality=np.array([1.0, 1.0]),
        minimum_frame_quality=0.5,
    )

    assert not result.valid_mask[0, 1, 0]
    assert result.valid_mask[0, 0, 0]


def test_threshold_and_score_must_be_supplied_together() -> None:
    values = np.ones((2, 1, 1), dtype=float)

    with pytest.raises(QualityFlagError, match="requires explicit frame_quality"):
        build_quality_flags(values, minimum_frame_quality=0.5)
    with pytest.raises(QualityFlagError, match="requires an explicit minimum"):
        build_quality_flags(values, frame_quality=np.array([0.9, 0.8]))


def test_invalid_quality_inputs_are_rejected() -> None:
    values = np.ones((2, 1, 1), dtype=float)

    with pytest.raises(QualityFlagError, match="shape"):
        build_quality_flags(np.ones((2, 2), dtype=float))
    with pytest.raises(QualityFlagError, match="numeric"):
        build_quality_flags(np.full((2, 1, 1), "x", dtype=object))
    with pytest.raises(QualityFlagError, match="shape"):
        build_quality_flags(
            values,
            frame_quality=np.array([0.9]),
            minimum_frame_quality=0.5,
        )
    with pytest.raises(QualityFlagError, match="numeric"):
        build_quality_flags(
            values,
            frame_quality=np.array(["good", "bad"], dtype=object),
            minimum_frame_quality=0.5,
        )
    with pytest.raises(QualityFlagError, match="finite"):
        build_quality_flags(
            values,
            frame_quality=np.array([0.9, 0.8]),
            minimum_frame_quality=np.nan,
        )


def test_quality_outputs_are_immutable_and_input_is_unchanged() -> None:
    values = np.array([[[1.0]], [[np.nan]], [[3.0]]])
    scores = np.array([0.9, 0.8, 0.7])
    values_before = values.copy()
    scores_before = scores.copy()

    result = build_quality_flags(
        values,
        frame_quality=scores,
        minimum_frame_quality=0.75,
    )

    assert np.array_equal(values, values_before, equal_nan=True)
    assert np.array_equal(scores, scores_before)
    assert not result.valid_mask.flags.writeable
    assert not result.finite_mask.flags.writeable
    assert not result.frame_quality_pass.flags.writeable
