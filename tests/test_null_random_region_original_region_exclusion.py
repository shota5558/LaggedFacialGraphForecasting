from __future__ import annotations

from lagged_facial_graph_forecasting.contracts import InnerFold, SplitManifest
from lagged_facial_graph_forecasting.core_contracts import ParentLink, ParentSet
from lagged_facial_graph_forecasting.null_random_region import construct_random_region_mapping


def test_random_region_excludes_each_parents_original_source_region() -> None:
    manifest = SplitManifest(
        outer_fold=3,
        train_subject_ids=("train_a", "train_b"),
        test_subject_ids=("test_a",),
        inner_folds=(InnerFold(("train_a",), ("train_b",)),),
        seed=823,
    )
    parent_set = ParentSet(
        outer_fold=3,
        target_region="mouth",
        parents=(
            ParentLink("left_cheek", 2, "vx", "vy"),
            ParentLink("right_cheek", 5, "vy", "vx"),
            ParentLink("jaw", 8, "vx", "vx"),
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

    assert len(mapping.mapped_parents) == len(mapping.source_parents)
    for source, mapped in zip(mapping.source_parents, mapping.mapped_parents, strict=True):
        assert mapped.source_region != source.source_region
        assert mapped.source_region != parent_set.target_region
        assert mapped.lag == source.lag
        assert mapped.source_dimension == source.source_dimension
        assert mapped.target_dimension == source.target_dimension
