from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting.region_aggregation import (
    RegionAggregationError,
    aggregate_regions,
)
from lagged_facial_graph_forecasting.regions import build_region_definition


def test_mean_aggregation_preserves_region_order_and_coordinate_axes() -> None:
    landmarks = np.array(
        [
            [[0.0, 0.0], [2.0, 2.0], [10.0, 20.0], [14.0, 24.0]],
            [[1.0, 3.0], [3.0, 5.0], [12.0, 22.0], [16.0, 26.0]],
        ]
    )
    definition = build_region_definition(
        {"eye": (0, 1), "mouth": (2, 3)}
    )

    aggregated = aggregate_regions(landmarks, definition)

    assert aggregated.region_ids == ("eye", "mouth")
    assert aggregated.method == "mean"
    assert aggregated.values.shape == (2, 2, 2)
    assert np.array_equal(
        aggregated.values,
        np.array(
            [
                [[1.0, 1.0], [12.0, 22.0]],
                [[2.0, 4.0], [14.0, 24.0]],
            ]
        ),
    )
    assert not aggregated.values.flags.writeable


def test_single_landmark_region_is_identity_for_that_point() -> None:
    landmarks = np.arange(3 * 4 * 3, dtype=float).reshape(3, 4, 3)
    definition = build_region_definition({"point": (2,)})

    aggregated = aggregate_regions(landmarks, definition)

    assert np.array_equal(aggregated.values[:, 0, :], landmarks[:, 2, :])


def test_nan_is_propagated_not_silently_ignored() -> None:
    landmarks = np.zeros((2, 3, 2), dtype=float)
    landmarks[1, 1, 0] = np.nan
    definition = build_region_definition({"region": (0, 1)})

    aggregated = aggregate_regions(landmarks, definition)

    assert np.isnan(aggregated.values[1, 0, 0])
    assert aggregated.values[1, 0, 1] == 0.0


def test_out_of_bounds_region_index_is_rejected() -> None:
    landmarks = np.zeros((2, 3, 2), dtype=float)
    definition = build_region_definition({"region": (3,)})

    with pytest.raises(ValueError, match="outside"):
        aggregate_regions(landmarks, definition)


def test_non_frozen_aggregation_method_is_rejected() -> None:
    landmarks = np.zeros((2, 3, 2), dtype=float)
    definition = build_region_definition({"region": (0, 1)})

    with pytest.raises(RegionAggregationError, match="only the frozen 'mean'"):
        aggregate_regions(landmarks, definition, method="median")


def test_aggregation_does_not_modify_raw_landmarks() -> None:
    landmarks = np.arange(2 * 4 * 2, dtype=float).reshape(2, 4, 2)
    before = landmarks.copy()
    definition = build_region_definition({"a": (0, 1), "b": (2, 3)})

    aggregate_regions(landmarks, definition)

    assert np.array_equal(landmarks, before)
