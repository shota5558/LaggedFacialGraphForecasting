from __future__ import annotations

import pytest

from lagged_facial_graph_forecasting import ContractError, InnerFold, SplitManifest


def _valid_manifest() -> SplitManifest:
    return SplitManifest(
        outer_fold=0,
        train_subject_ids=("s01", "s02", "s03", "s04"),
        test_subject_ids=("s05", "s06"),
        inner_folds=(
            InnerFold(
                inner_train_subject_ids=("s01", "s02"),
                inner_val_subject_ids=("s03", "s04"),
            ),
            InnerFold(
                inner_train_subject_ids=("s03", "s04"),
                inner_val_subject_ids=("s01", "s02"),
            ),
        ),
        seed=123,
    )


def test_split_manifest_accepts_subject_isolated_nested_split() -> None:
    manifest = _valid_manifest()

    assert manifest.outer_fold == 0
    assert manifest.seed == 123
    assert set(manifest.train_subject_ids).isdisjoint(manifest.test_subject_ids)
    for fold in manifest.inner_folds:
        assert set(fold.inner_train_subject_ids).isdisjoint(fold.inner_val_subject_ids)
        assert set(fold.inner_train_subject_ids) <= set(manifest.train_subject_ids)
        assert set(fold.inner_val_subject_ids) <= set(manifest.train_subject_ids)


def test_rejects_outer_train_test_subject_overlap() -> None:
    with pytest.raises(ContractError, match="outer train/test subject overlap"):
        SplitManifest(
            outer_fold=0,
            train_subject_ids=("s01", "s02"),
            test_subject_ids=("s02", "s03"),
            inner_folds=(
                InnerFold(
                    inner_train_subject_ids=("s01",),
                    inner_val_subject_ids=("s02",),
                ),
            ),
            seed=123,
        )


def test_rejects_inner_train_validation_subject_overlap() -> None:
    with pytest.raises(ContractError, match="inner train/validation subject overlap"):
        InnerFold(
            inner_train_subject_ids=("s01", "s02"),
            inner_val_subject_ids=("s02", "s03"),
        )


def test_rejects_outer_test_subject_in_inner_fold() -> None:
    with pytest.raises(ContractError, match="contains outer-test subjects"):
        SplitManifest(
            outer_fold=0,
            train_subject_ids=("s01", "s02", "s03"),
            test_subject_ids=("s04",),
            inner_folds=(
                InnerFold(
                    inner_train_subject_ids=("s01", "s02"),
                    inner_val_subject_ids=("s04",),
                ),
            ),
            seed=7,
        )


def test_rejects_subject_outside_outer_train_in_inner_fold() -> None:
    with pytest.raises(ContractError, match="outside outer-train"):
        SplitManifest(
            outer_fold=1,
            train_subject_ids=("s01", "s02", "s03"),
            test_subject_ids=("s04",),
            inner_folds=(
                InnerFold(
                    inner_train_subject_ids=("s01", "s02"),
                    inner_val_subject_ids=("unknown",),
                ),
            ),
            seed=7,
        )


def test_rejects_duplicate_subject_ids() -> None:
    with pytest.raises(ContractError, match="train_subject_ids entries must be unique"):
        SplitManifest(
            outer_fold=0,
            train_subject_ids=("s01", "s01"),
            test_subject_ids=("s02",),
            inner_folds=(
                InnerFold(
                    inner_train_subject_ids=("s01",),
                    inner_val_subject_ids=("s02",),
                ),
            ),
            seed=1,
        )


@pytest.mark.parametrize("field,value", [("outer_fold", -1), ("seed", -1)])
def test_rejects_negative_fold_or_seed(field: str, value: int) -> None:
    kwargs = {
        "outer_fold": 0,
        "train_subject_ids": ("s01", "s02"),
        "test_subject_ids": ("s03",),
        "inner_folds": (
            InnerFold(
                inner_train_subject_ids=("s01",),
                inner_val_subject_ids=("s02",),
            ),
        ),
        "seed": 1,
    }
    kwargs[field] = value

    with pytest.raises(ContractError, match=field):
        SplitManifest(**kwargs)
