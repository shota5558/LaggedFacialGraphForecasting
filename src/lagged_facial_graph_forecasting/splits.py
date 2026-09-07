"""Subject-level splitting utilities shared by all downstream modules."""

from __future__ import annotations

import numpy as np

from .contracts import InnerFold, SplitManifest


def _validate_subject_ids(subject_ids: tuple[str, ...]) -> tuple[str, ...]:
    subject_ids = tuple(subject_ids)
    if len(subject_ids) != len(set(subject_ids)):
        raise ValueError("subject_ids must be unique")
    if any(
        not isinstance(subject_id, str) or not subject_id.strip()
        for subject_id in subject_ids
    ):
        raise ValueError("subject_ids entries must be non-empty strings")
    return subject_ids


def _validate_inner_split_parameters(n_inner_folds: int, seed: int) -> None:
    if not isinstance(n_inner_folds, int) or isinstance(n_inner_folds, bool):
        raise ValueError("n_inner_folds must be an integer")
    if n_inner_folds < 2:
        raise ValueError("n_inner_folds must be >= 2")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")


def _build_inner_folds(
    train_subject_ids: tuple[str, ...],
    *,
    n_inner_folds: int,
    rng: np.random.Generator,
) -> tuple[InnerFold, ...]:
    if len(train_subject_ids) < n_inner_folds:
        raise ValueError(
            f"need at least {n_inner_folds} outer-train subjects for "
            f"n_inner_folds={n_inner_folds}"
        )

    inner_order = list(train_subject_ids)
    rng.shuffle(inner_order)
    validation_chunks = [
        tuple(chunk.tolist())
        for chunk in np.array_split(
            np.asarray(inner_order, dtype=object), n_inner_folds
        )
    ]

    inner_folds: list[InnerFold] = []
    for validation_ids in validation_chunks:
        validation_set = set(validation_ids)
        inner_train_ids = tuple(
            subject_id
            for subject_id in train_subject_ids
            if subject_id not in validation_set
        )
        inner_folds.append(
            InnerFold(
                inner_train_subject_ids=inner_train_ids,
                inner_val_subject_ids=validation_ids,
            )
        )
    return tuple(inner_folds)


def build_subject_split_manifest(
    subject_ids: tuple[str, ...],
    *,
    outer_fold: int = 0,
    test_subject_count: int = 1,
    n_inner_folds: int = 2,
    seed: int = 0,
) -> SplitManifest:
    """Build one deterministic subject-isolated outer split with inner folds.

    All downstream modules receive the returned ``SplitManifest``; they do not
    perform their own splitting.
    """

    subject_ids = _validate_subject_ids(subject_ids)
    if not isinstance(test_subject_count, int) or isinstance(test_subject_count, bool):
        raise ValueError("test_subject_count must be an integer")
    if test_subject_count < 1:
        raise ValueError("test_subject_count must be >= 1")
    _validate_inner_split_parameters(n_inner_folds, seed)

    minimum_subjects = test_subject_count + n_inner_folds
    if len(subject_ids) < minimum_subjects:
        raise ValueError(
            f"need at least {minimum_subjects} subjects for "
            f"test_subject_count={test_subject_count} and n_inner_folds={n_inner_folds}"
        )

    rng = np.random.default_rng(seed)
    permuted = list(subject_ids)
    rng.shuffle(permuted)

    test_subject_ids = tuple(permuted[:test_subject_count])
    train_subject_ids = tuple(permuted[test_subject_count:])
    inner_folds = _build_inner_folds(
        train_subject_ids,
        n_inner_folds=n_inner_folds,
        rng=rng,
    )

    return SplitManifest(
        outer_fold=outer_fold,
        train_subject_ids=train_subject_ids,
        test_subject_ids=test_subject_ids,
        inner_folds=inner_folds,
        seed=seed,
    )


def build_loso_split_manifests(
    subject_ids: tuple[str, ...],
    *,
    n_inner_folds: int = 2,
    seed: int = 0,
) -> tuple[SplitManifest, ...]:
    """Build deterministic leave-one-subject-out outer folds.

    Each validated subject is held out exactly once. Input subject order defines
    ``outer_fold`` numbering; ``seed`` affects only inner-fold assignment inside
    each outer-train partition. No outcome or measurement data are consulted.
    """

    subject_ids = _validate_subject_ids(subject_ids)
    _validate_inner_split_parameters(n_inner_folds, seed)
    minimum_subjects = n_inner_folds + 1
    if len(subject_ids) < minimum_subjects:
        raise ValueError(
            f"need at least {minimum_subjects} subjects for LOSO with "
            f"n_inner_folds={n_inner_folds}"
        )

    manifests: list[SplitManifest] = []
    for outer_fold, test_subject_id in enumerate(subject_ids):
        train_subject_ids = tuple(
            subject_id
            for subject_id in subject_ids
            if subject_id != test_subject_id
        )
        rng = np.random.default_rng(np.random.SeedSequence([seed, outer_fold]))
        inner_folds = _build_inner_folds(
            train_subject_ids,
            n_inner_folds=n_inner_folds,
            rng=rng,
        )
        manifests.append(
            SplitManifest(
                outer_fold=outer_fold,
                train_subject_ids=train_subject_ids,
                test_subject_ids=(test_subject_id,),
                inner_folds=inner_folds,
                seed=seed,
            )
        )

    return tuple(manifests)


def build_grouped_kfold_split_manifests(
    subject_ids: tuple[str, ...],
    *,
    n_outer_folds: int,
    n_inner_folds: int = 2,
    seed: int = 0,
) -> tuple[SplitManifest, ...]:
    """Build deterministic grouped K-fold outer splits at subject granularity.

    Subject IDs are the groups: each subject appears in outer-test exactly once
    and is never split across train/test. The outer partition depends only on the
    validated subject IDs and seed, never on outcomes or facial measurements.
    """

    subject_ids = _validate_subject_ids(subject_ids)
    _validate_inner_split_parameters(n_inner_folds, seed)
    if not isinstance(n_outer_folds, int) or isinstance(n_outer_folds, bool):
        raise ValueError("n_outer_folds must be an integer")
    if n_outer_folds < 2:
        raise ValueError("n_outer_folds must be >= 2")
    if n_outer_folds > len(subject_ids):
        raise ValueError("n_outer_folds must not exceed the number of subjects")

    rng = np.random.default_rng(seed)
    outer_order = list(subject_ids)
    rng.shuffle(outer_order)
    test_chunks = [
        tuple(chunk.tolist())
        for chunk in np.array_split(
            np.asarray(outer_order, dtype=object), n_outer_folds
        )
    ]

    manifests: list[SplitManifest] = []
    all_subjects = set(subject_ids)
    for outer_fold, test_subject_ids in enumerate(test_chunks):
        test_set = set(test_subject_ids)
        train_subject_ids = tuple(
            subject_id for subject_id in subject_ids if subject_id not in test_set
        )
        if len(train_subject_ids) < n_inner_folds:
            raise ValueError(
                "each grouped outer-train partition must contain at least "
                f"n_inner_folds={n_inner_folds} subjects"
            )
        inner_rng = np.random.default_rng(
            np.random.SeedSequence([seed, outer_fold, n_outer_folds])
        )
        inner_folds = _build_inner_folds(
            train_subject_ids,
            n_inner_folds=n_inner_folds,
            rng=inner_rng,
        )
        manifest = SplitManifest(
            outer_fold=outer_fold,
            train_subject_ids=train_subject_ids,
            test_subject_ids=test_subject_ids,
            inner_folds=inner_folds,
            seed=seed,
        )
        if set(manifest.train_subject_ids) | set(manifest.test_subject_ids) != all_subjects:
            raise RuntimeError("grouped outer split lost a subject")
        manifests.append(manifest)

    return tuple(manifests)
