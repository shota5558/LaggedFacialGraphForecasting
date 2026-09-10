from __future__ import annotations

from lagged_facial_graph_forecasting.splits import (
    build_grouped_kfold_split_manifests,
    build_loso_split_manifests,
    build_subject_split_manifest,
)


SUBJECT_IDS = tuple(f"s{index:02d}" for index in range(1, 11))
SEED = 20260908


def test_all_subject_split_builders_are_deterministic_for_same_inputs_and_seed() -> None:
    single_a = build_subject_split_manifest(
        SUBJECT_IDS,
        outer_fold=2,
        test_subject_count=2,
        n_inner_folds=3,
        seed=SEED,
    )
    single_b = build_subject_split_manifest(
        SUBJECT_IDS,
        outer_fold=2,
        test_subject_count=2,
        n_inner_folds=3,
        seed=SEED,
    )
    assert single_a == single_b

    loso_a = build_loso_split_manifests(
        SUBJECT_IDS,
        n_inner_folds=3,
        seed=SEED,
    )
    loso_b = build_loso_split_manifests(
        SUBJECT_IDS,
        n_inner_folds=3,
        seed=SEED,
    )
    assert loso_a == loso_b

    grouped_a = build_grouped_kfold_split_manifests(
        SUBJECT_IDS,
        n_outer_folds=5,
        n_inner_folds=3,
        seed=SEED,
    )
    grouped_b = build_grouped_kfold_split_manifests(
        SUBJECT_IDS,
        n_outer_folds=5,
        n_inner_folds=3,
        seed=SEED,
    )
    assert grouped_a == grouped_b
