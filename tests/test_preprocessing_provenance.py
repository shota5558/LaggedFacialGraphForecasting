from __future__ import annotations

import json

import pytest

from lagged_facial_graph_forecasting.preprocessing_provenance import (
    FeatureSemantics,
    PreprocessingProvenance,
    PreprocessingProvenanceError,
)
from lagged_facial_graph_forecasting.regions import build_region_definition


def _provenance() -> PreprocessingProvenance:
    return PreprocessingProvenance(
        region_definition=build_region_definition(
            {
                "left_eye": (0, 1, 2),
                "mouth": (3, 4, 5),
            }
        ),
        feature_semantics=FeatureSemantics(
            feature_name="velocity",
            dimensions=("v_x", "v_y"),
            method="displacement_over_observed_time_interval",
            units="normalized_coordinate_per_timestamp_unit",
            timestamp_basis="observed_frame_timestamps",
        ),
        region_aggregation_method="mean",
        translation_normalization="reference_centroid",
        scale_normalization="reference_pair_distance",
        rotation_normalization="reference_pair_axis_alignment",
        missing_frame_handling="preserve_invalidity",
        interpolation_policy="none",
        acceleration_primary=False,
    )


def test_preprocessing_provenance_json_round_trip_preserves_region_order_and_d2() -> None:
    original = _provenance()
    payload = json.loads(json.dumps(original.to_payload(), sort_keys=True))
    restored = PreprocessingProvenance.from_payload(payload)

    assert restored == original
    assert restored.region_definition.region_ids == ("left_eye", "mouth")
    assert restored.feature_semantics.dimensions == ("v_x", "v_y")


def test_feature_semantics_rejects_duplicate_dimension_names() -> None:
    with pytest.raises(PreprocessingProvenanceError, match="dimensions must be unique"):
        FeatureSemantics(
            feature_name="velocity",
            dimensions=("v_x", "v_x"),
            method="displacement_over_observed_time_interval",
            units="normalized_coordinate_per_timestamp_unit",
            timestamp_basis="observed_frame_timestamps",
        )


def test_preprocessing_provenance_requires_explicit_nonempty_policy_names() -> None:
    with pytest.raises(
        PreprocessingProvenanceError,
        match="interpolation_policy must be a non-empty string",
    ):
        PreprocessingProvenance(
            region_definition=build_region_definition({"mouth": (0, 1)}),
            feature_semantics=FeatureSemantics(
                feature_name="velocity",
                dimensions=("v_x", "v_y"),
                method="displacement_over_observed_time_interval",
                units="normalized_coordinate_per_timestamp_unit",
                timestamp_basis="observed_frame_timestamps",
            ),
            region_aggregation_method="mean",
            translation_normalization="reference_centroid",
            scale_normalization="reference_pair_distance",
            rotation_normalization="reference_pair_axis_alignment",
            missing_frame_handling="preserve_invalidity",
            interpolation_policy="",
            acceleration_primary=False,
        )


def test_preprocessing_provenance_rejects_manifest_field_drift() -> None:
    payload = _provenance().to_payload()
    payload["unexpected"] = "drift"

    with pytest.raises(PreprocessingProvenanceError, match="fields are incompatible"):
        PreprocessingProvenance.from_payload(payload)
