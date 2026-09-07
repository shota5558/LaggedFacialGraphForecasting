from __future__ import annotations

import pytest

from lagged_facial_graph_forecasting import (
    build_loso_split_manifests,
    build_subject_split_manifest,
)


SUBJECTS = tuple(f"s{index:02d}" for index in range(1, 9))


def test_subject_split_is_deterministic_for_same_seed() -> None:
    first = build_subject_split_manifest(
        SUBJECTS, test_subject_count=2, n_inner_folds=3, seed=42
    )
    second = build_subject_split_manifest(
        SUBJECTS, test_subject_count=2, n_inner_folds=3, seed=42
    )

    assert first == second


def test_outer_train_and_test_are_subject_disjoint() -> None:
    manifest = build_subject_split_manifest(
        SUBJECTS, test_subject_count=2, n_inner_folds=3, seed=7
    )

    assert set(manifest.train_subject_ids).isdisjoint(manifest.test_subject_ids)
    assert set(manifest.train_subject_ids) | set(manifest.test_subject_ids) == set(SUBJECTS)


def test_inner_folds_use_outer_train_only() -> None:
    manifest = build_subject_split_manifest(
        SUBJECTS, test_subject_count=2, n_inner_folds=3, seed=7
    )
    outer_train = set(manifest.train_subject_ids)
    outer_test = set(manifest.test_subject_ids)

    for fold in manifest.inner_folds:
        inner_subjects = set(fold.inner_train_subject_ids) | set(fold.inner_val_subject_ids)
        assert inner_subjects <= outer_train
        assert inner_subjects.isdisjoint(outer_test)


def test_each_outer_train_subject_is_validation_once() -> None:
    manifest = build_subject_split_manifest(
        SUBJECTS, test_subject_count=2, n_inner_folds=3, seed=19
    )
    validation_subjects = [
        subject_id
        for fold in manifest.inner_folds
        for subject_id in fold.inner_val_subject_ids
    ]

    assert sorted(validation_subjects) == sorted(manifest.train_subject_ids)


def test_different_seed_changes_outer_split_for_fixture() -> None:
    first = build_subject_split_manifest(
        SUBJECTS, test_subject_count=2, n_inner_folds=3, seed=1
    )
    second = build_subject_split_manifest(
        SUBJECTS, test_subject_count=2, n_inner_folds=3, seed=2
    )

    assert first.test_subject_ids != second.test_subject_ids


def test_rejects_duplicate_subject_ids() -> None:
    with pytest.raises(ValueError, match="subject_ids must be unique"):
        build_subject_split_manifest(("s01", "s01", "s02", "s03"))


def test_rejects_insufficient_subject_count() -> None:
    with pytest.raises(ValueError, match="need at least"):
        build_subject_split_manifest(
            ("s01", "s02", "s03"), test_subject_count=1, n_inner_folds=3
        )


def test_loso_holds_each_subject_out_exactly_once() -> None:
    manifests = build_loso_split_manifests(SUBJECTS, n_inner_folds=3, seed=41)

    assert len(manifests) == len(SUBJECTS)
    assert [manifest.outer_fold for manifest in manifests] == list(range(len(SUBJECTS)))
    assert [manifest.test_subject_ids[0] for manifest in manifests] == list(SUBJECTS)

    for manifest in manifests:
        assert len(manifest.test_subject_ids) == 1
        assert set(manifest.train_subject_ids).isdisjoint(manifest.test_subject_ids)
        assert set(manifest.train_subject_ids) | set(manifest.test_subject_ids) == set(SUBJECTS)


def test_loso_inner_folds_never_contain_held_out_subject() -> None:
    manifests = build_loso_split_manifests(SUBJECTS, n_inner_folds=3, seed=7)

    for manifest in manifests:
        held_out = set(manifest.test_subject_ids)
        for inner_fold in manifest.inner_folds:
            inner_subjects = set(inner_fold.inner_train_subject_ids) | set(
                inner_fold.inner_val_subject_ids
            )
            assert inner_subjects <= set(manifest.train_subject_ids)
            assert inner_subjects.isdisjoint(held_out)


def test_loso_is_deterministic_for_same_subject_order_and_seed() -> None:
    first = build_loso_split_manifests(SUBJECTS, n_inner_folds=3, seed=99)
    second = build_loso_split_manifests(SUBJECTS, n_inner_folds=3, seed=99)

    assert first == second
