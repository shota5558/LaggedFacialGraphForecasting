from __future__ import annotations

import pytest

from lagged_facial_graph_forecasting.regions import (
    RegionDefinition,
    RegionDefinitionError,
    build_region_definition,
    validate_region_landmark_bounds,
)


def test_builds_ordered_extractor_neutral_region_definition() -> None:
    definition = build_region_definition(
        {
            "left_eye": (4, 5, 6),
            "mouth": (10, 11, 12, 13),
        }
    )

    assert definition.region_ids == ("left_eye", "mouth")
    assert definition.landmark_indices == ((4, 5, 6), (10, 11, 12, 13))
    assert definition.as_mapping() == {
        "left_eye": (4, 5, 6),
        "mouth": (10, 11, 12, 13),
    }
    assert definition.allow_overlap is False


def test_does_not_hardcode_extractor_landmark_count_or_region_names() -> None:
    definition = build_region_definition({"custom_region": (0, 999)})

    assert definition.region_ids == ("custom_region",)
    assert definition.landmark_indices == ((0, 999),)


def test_bounds_are_checked_only_when_landmark_count_is_known() -> None:
    definition = build_region_definition({"region_a": (0, 3)})

    assert validate_region_landmark_bounds(definition, landmark_count=4) is definition
    with pytest.raises(RegionDefinitionError, match="outside"):
        validate_region_landmark_bounds(definition, landmark_count=3)


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_rejects_invalid_landmark_count(invalid: object) -> None:
    definition = build_region_definition({"region_a": (0,)})

    with pytest.raises(RegionDefinitionError, match="positive integer"):
        validate_region_landmark_bounds(
            definition,
            landmark_count=invalid,  # type: ignore[arg-type]
        )


def test_rejects_duplicate_region_ids_and_invalid_region_ids() -> None:
    with pytest.raises(RegionDefinitionError, match="duplicate region ID"):
        RegionDefinition(
            region_ids=("eye", "eye"),
            landmark_indices=((0,), (1,)),
        )
    with pytest.raises(RegionDefinitionError, match="region IDs"):
        RegionDefinition(region_ids=(" eye",), landmark_indices=((0,),))


def test_rejects_empty_duplicate_or_invalid_landmark_indices() -> None:
    with pytest.raises(RegionDefinitionError, match="at least one landmark"):
        build_region_definition({"eye": ()})
    with pytest.raises(RegionDefinitionError, match="duplicate landmark"):
        build_region_definition({"eye": (1, 1)})
    with pytest.raises(RegionDefinitionError, match="non-negative integers"):
        build_region_definition({"eye": (-1,)})
    with pytest.raises(RegionDefinitionError, match="non-negative integers"):
        build_region_definition({"eye": (True,)})


def test_cross_region_overlap_requires_explicit_freeze() -> None:
    regions = {"eye": (1, 2), "cheek": (2, 3)}

    with pytest.raises(RegionDefinitionError, match="overlaps"):
        build_region_definition(regions)

    definition = build_region_definition(regions, allow_overlap=True)
    assert definition.allow_overlap is True
    assert definition.as_mapping() == {"eye": (1, 2), "cheek": (2, 3)}
