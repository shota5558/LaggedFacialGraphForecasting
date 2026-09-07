"""Minimal subject-level splitting for the V0 vertical slice."""

from __future__ import annotations

import numpy as np

from .contracts import InnerFold, SplitManifest


def build_subject_split_manifest(
    subject_ids: tuple[str, ...],
    *,
    outer_fold: int = 0,
    test_subject_count: int = 1,
    n_inner_folds: int = 2,
    seed: int = 0,
) -> SplitManifest:
    """Build one deterministic subject-isolated outer split with inner folds.

    This is intentionally minimal V0 infrastructure. All downstream modules receive
    the returned ``SplitManifest``; they do not perform their own splitting.
    """

    subject_ids = tuple(subject_ids)
    if len(subject_ids) != len(set(subject_ids)):
        raise ValueError("subject_ids must be unique")
    if any(not isinstance(subject_id, str) or not subject_id.strip() for subject_id in subject_ids):
        raise ValueError("subject_ids entries must be non-empty strings")
    if not isinstance(test_subject_count, int) or isinstance(test_subject_count, bool):
        raise ValueError("test_subject_count must be an integer")
    if test_subject_count < 1:
        raise ValueError("test_subject_count must be >= 1")
    if not isinstance(n_inner_folds, int) or isinstance(n_inner_folds, bool):
        raise ValueError("n_inner_folds must be an integer")
    if n_inner_folds < 2:
        raise ValueError("n_inner_folds must be >= 2")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

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

    inner_order = list(train_subject_ids)
    rng.shuffle(inner_order)
    validation_chunks = [
        tuple(chunk.tolist())
        for chunk in np.array_split(np.asarray(inner_order, dtype=object), n_inner_folds)
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

    return SplitManifest(
        outer_fold=outer_fold,
        train_subject_ids=train_subject_ids,
        test_subject_ids=test_subject_ids,
        inner_folds=tuple(inner_folds),
        seed=seed,
    )
