from __future__ import annotations

from lagged_facial_graph_forecasting.contracts import InnerFold, SplitManifest
from lagged_facial_graph_forecasting.core_contracts import ParentLink, ParentSet
from lagged_facial_graph_forecasting.null_random_region import construct_random_region_mapping


def test_random_region_preserves_feature_count_under_collision_pressure() -> None:
    manifest = SplitManifest(
        outer_fold=2,
        train_subject_ids=("train_a", "train_b"),
        test_subject_ids=("test_a",),
        inner_folds=(InnerFold(("train_a",), ("train_b",)),),
        seed=722,
    )
    parent_set = ParentSet(
        outer_fold=2,
        target_region="mouth",
        parents=(
            ParentLink("left_cheek", 4, "vx", "vy"),
            ParentLink("right_cheek", 4, "vx", "vy"),
            ParentLink("jaw", 4, "vx", "vy"),
        ),
        discovery_method="pcmci_plus",
    )

    mapping = construct_random_region_mapping(
        manifest,
        parent_set,
        construction_subject_ids=manifest.train_subject_ids,
        candidate_regions=(
            "left_cheek",
            "right_cheek",
            "jaw",
            "left_eye",
            "right_eye",
            "left_brow",
            "right_brow",
            "mouth",
        ),
    )

    assert mapping.source_parents == parent_set.parents
    assert len(mapping.mapped_parents) == len(parent_set.parents) == 3
    assert len(set(mapping.mapped_parents)) == len(mapping.mapped_parents)
    for source, mapped in zip(mapping.source_parents, mapping.mapped_parents, strict=True):
        assert mapped.lag == source.lag
        assert mapped.source_dimension == source.source_dimension
        assert mapped.target_dimension == source.target_dimension
