"""Subject-level split contract used as the single source of split truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .contracts import ContractError


def _subject_ids(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    subject_ids = tuple(values)
    if not subject_ids:
        raise ContractError(f"{field_name} must not be empty")
    if any(not isinstance(value, str) or not value.strip() for value in subject_ids):
        raise ContractError(f"{field_name} entries must be non-empty strings")
    if len(set(subject_ids)) != len(subject_ids):
        raise ContractError(f"{field_name} entries must be unique")
    return subject_ids


@dataclass(frozen=True, slots=True)
class InnerFold:
    """One subject-grouped inner train/validation partition."""

    inner_train_subject_ids: tuple[str, ...]
    inner_val_subject_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        train_ids = _subject_ids(self.inner_train_subject_ids, "inner_train_subject_ids")
        val_ids = _subject_ids(self.inner_val_subject_ids, "inner_val_subject_ids")

        overlap = set(train_ids) & set(val_ids)
        if overlap:
            raise ContractError(
                "inner train/validation subject overlap is forbidden: "
                f"{sorted(overlap)}"
            )

        object.__setattr__(self, "inner_train_subject_ids", train_ids)
        object.__setattr__(self, "inner_val_subject_ids", val_ids)


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Frozen outer/inner subject split manifest.

    Every downstream module receives this manifest rather than constructing its
    own split. Each inner fold must partition the complete outer-train subject
    set, which structurally prevents outer-test subjects from entering tuning.
    """

    outer_fold: int
    train_subject_ids: tuple[str, ...]
    test_subject_ids: tuple[str, ...]
    inner_folds: tuple[InnerFold, ...]
    seed: int

    def __post_init__(self) -> None:
        if isinstance(self.outer_fold, bool) or not isinstance(self.outer_fold, int):
            raise ContractError("outer_fold must be an integer")
        if self.outer_fold < 0:
            raise ContractError("outer_fold must be >= 0")

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ContractError("seed must be an integer")
        if self.seed < 0:
            raise ContractError("seed must be >= 0")

        train_ids = _subject_ids(self.train_subject_ids, "train_subject_ids")
        test_ids = _subject_ids(self.test_subject_ids, "test_subject_ids")
        outer_train = set(train_ids)
        outer_test = set(test_ids)

        overlap = outer_train & outer_test
        if overlap:
            raise ContractError(
                "outer train/test subject overlap is forbidden: " f"{sorted(overlap)}"
            )

        inner_folds = tuple(self.inner_folds)
        if not inner_folds:
            raise ContractError("inner_folds must contain at least one fold")

        for index, fold in enumerate(inner_folds):
            if not isinstance(fold, InnerFold):
                raise ContractError(f"inner_folds[{index}] must be an InnerFold")

            inner_train = set(fold.inner_train_subject_ids)
            inner_val = set(fold.inner_val_subject_ids)
            inner_subjects = inner_train | inner_val

            if inner_subjects != outer_train:
                missing = sorted(outer_train - inner_subjects)
                unexpected = sorted(inner_subjects - outer_train)
                raise ContractError(
                    f"inner_folds[{index}] must partition outer-train subjects; "
                    f"missing={missing}, unexpected={unexpected}"
                )

            if inner_subjects & outer_test:
                raise ContractError(
                    f"inner_folds[{index}] contains outer-test subjects"
                )

        object.__setattr__(self, "train_subject_ids", train_ids)
        object.__setattr__(self, "test_subject_ids", test_ids)
        object.__setattr__(self, "inner_folds", inner_folds)
