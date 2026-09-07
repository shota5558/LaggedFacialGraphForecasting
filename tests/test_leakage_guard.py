from __future__ import annotations

import pytest

from lagged_facial_graph_forecasting.contracts import InnerFold, SplitManifest
from lagged_facial_graph_forecasting.leakage_guard import (
    LeakageGuardError,
    assert_discovery_fit_scope,
    assert_null_construction_scope,
    assert_preprocessing_fit_scope,
    assert_ridge_tuning_scope,
)


def _manifest() -> SplitManifest:
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


def test_preprocessing_scope_accepts_outer_train_subjects_only() -> None:
    manifest = _manifest()
    assert assert_preprocessing_fit_scope(manifest, ("s01", "s03")) == ("s01", "s03")


def test_preprocessing_scope_rejects_outer_test_subject() -> None:
    with pytest.raises(LeakageGuardError, match="cannot access outer-test subjects"):
        assert_preprocessing_fit_scope(_manifest(), ("s01", "s05"))


def test_preprocessing_scope_rejects_subject_outside_manifest_outer_train() -> None:
    with pytest.raises(LeakageGuardError, match="outside outer-train"):
        assert_preprocessing_fit_scope(_manifest(), ("s01", "unknown"))


def test_preprocessing_scope_rejects_duplicate_subject_provenance() -> None:
    with pytest.raises(LeakageGuardError, match="must be unique"):
        assert_preprocessing_fit_scope(_manifest(), ("s01", "s01"))


def test_preprocessing_scope_rejects_empty_subject_scope() -> None:
    with pytest.raises(LeakageGuardError, match="at least one subject"):
        assert_preprocessing_fit_scope(_manifest(), ())


def test_discovery_scope_accepts_outer_train_subjects_only() -> None:
    assert assert_discovery_fit_scope(_manifest(), ("s02", "s04")) == ("s02", "s04")


def test_discovery_scope_rejects_outer_test_subject_before_discovery() -> None:
    with pytest.raises(LeakageGuardError, match="discovery fit cannot access outer-test subjects"):
        assert_discovery_fit_scope(_manifest(), ("s02", "s06"))


def test_ridge_tuning_scope_accepts_outer_train_subjects_only() -> None:
    assert assert_ridge_tuning_scope(_manifest(), ("s01", "s02")) == ("s01", "s02")


def test_ridge_tuning_scope_rejects_outer_test_subject_before_tuning() -> None:
    with pytest.raises(LeakageGuardError, match="ridge tuning cannot access outer-test subjects"):
        assert_ridge_tuning_scope(_manifest(), ("s01", "s05"))


def test_null_construction_scope_accepts_outer_train_subjects_only() -> None:
    assert assert_null_construction_scope(_manifest(), ("s03", "s04")) == ("s03", "s04")


def test_null_construction_scope_rejects_outer_test_subject_before_mapping() -> None:
    with pytest.raises(LeakageGuardError, match="null construction cannot access outer-test subjects"):
        assert_null_construction_scope(_manifest(), ("s03", "s06"))
