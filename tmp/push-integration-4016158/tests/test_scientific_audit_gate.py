from __future__ import annotations

from dataclasses import replace

import pytest

from lagged_facial_graph_forecasting.scientific_audit import (
    ScientificAuditError,
    ScientificAuditSnapshot,
    audit_scientific_snapshots,
)


def _digest(char: str) -> str:
    return char * 64


def _snapshot(**overrides) -> ScientificAuditSnapshot:
    values = {
        "scientific_config_sha256": _digest("a"),
        "seed": 20260908,
        "train_subject_ids": ("train_1", "train_2", "train_3"),
        "test_subject_ids": ("outer_test",),
        "preprocessing_fit_subject_ids": ("train_1", "train_2", "train_3"),
        "discovery_subject_ids": ("train_1", "train_2", "train_3"),
        "ridge_tuning_subject_ids": ("train_1", "train_2", "train_3"),
        "null_construction_subject_ids": ("train_1", "train_2", "train_3"),
        "split_manifest_sha256": _digest("b"),
        "parent_set_sha256": _digest("c"),
        "ridge_selection_sha256": _digest("d"),
        "null_mapping_sha256": (
            ("lag-shift:-2", _digest("1")),
            ("lag-shift:-1", _digest("2")),
            ("lag-shift:+1", _digest("3")),
            ("lag-shift:+2", _digest("4")),
            ("random-region", _digest("5")),
            ("matched-sparsity", _digest("6")),
            ("time-shuffle", _digest("7")),
        ),
        "prediction_sha256": (
            ("persistence", _digest("8")),
            ("self", _digest("9")),
            ("full", _digest("0")),
            ("pcmci", _digest("a")),
            ("random-region", _digest("b")),
        ),
        "metrics_sha256": (
            ("persistence", _digest("c")),
            ("self", _digest("d")),
            ("full", _digest("e")),
            ("pcmci", _digest("f")),
            ("random-region", _digest("1")),
        ),
        "fold_frozen_before_outer_test": True,
        "outer_test_evaluation_count": 1,
    }
    values.update(overrides)
    return ScientificAuditSnapshot(**values)


def test_gate4_passes_only_for_identical_leak_free_scientific_evidence() -> None:
    first = _snapshot()
    second = _snapshot()

    result = audit_scientific_snapshots(first, second)

    assert result.status == "PASS"
    assert result.checks == (
        "outer_test_not_in_preprocessing_fit",
        "outer_test_not_in_discovery",
        "outer_test_not_in_ridge_tuning",
        "outer_test_not_in_null_construction",
        "same_scientific_config",
        "same_seed",
        "same_split_manifest",
        "same_parent_set",
        "same_ridge_selection",
        "same_null_mappings",
        "same_predictions",
        "same_metrics",
        "fold_frozen_before_outer_test",
        "single_outer_test_evaluation",
    )


@pytest.mark.parametrize(
    "scope_field",
    (
        "preprocessing_fit_subject_ids",
        "discovery_subject_ids",
        "ridge_tuning_subject_ids",
        "null_construction_subject_ids",
    ),
)
def test_gate4_rejects_outer_test_subject_in_every_selection_scope(scope_field: str) -> None:
    with pytest.raises(ScientificAuditError, match="outer-test"):
        _snapshot(**{scope_field: ("train_1", "outer_test")})


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    (
        ("scientific_config_sha256", _digest("f")),
        ("seed", 17),
        ("split_manifest_sha256", _digest("e")),
        ("parent_set_sha256", _digest("d")),
        ("ridge_selection_sha256", _digest("c")),
        ("null_mapping_sha256", (("time-shuffle", _digest("b")),)),
        ("prediction_sha256", (("pcmci", _digest("c")),)),
        ("metrics_sha256", (("pcmci", _digest("d")),)),
    ),
)
def test_gate4_rejects_any_same_config_seed_reproducibility_mismatch(
    field_name: str,
    changed_value,
) -> None:
    first = _snapshot()
    second = replace(first, **{field_name: changed_value})

    with pytest.raises(ScientificAuditError, match=field_name):
        audit_scientific_snapshots(first, second)


def test_gate4_rejects_outer_test_before_freeze() -> None:
    with pytest.raises(ScientificAuditError, match="before fold freeze"):
        _snapshot(fold_frozen_before_outer_test=False)


def test_gate4_rejects_repeated_outer_test_evaluation() -> None:
    with pytest.raises(ScientificAuditError, match="exactly one"):
        _snapshot(outer_test_evaluation_count=2)


def test_gate4_rejects_unknown_subject_in_selection_scope() -> None:
    with pytest.raises(ScientificAuditError, match="outside outer-train"):
        _snapshot(discovery_subject_ids=("train_1", "unknown_subject"))
