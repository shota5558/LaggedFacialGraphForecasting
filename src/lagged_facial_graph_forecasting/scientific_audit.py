"""Gate 4 scientific leakage and reproducibility audit.

This module does not rerun discovery or forecasting.  It consumes immutable
evidence produced by a completed one-fold dry run and fails closed unless every
selection scope is outer-train-only and two same-config/same-seed executions are
bitwise-identical at the scientific artifact boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ScientificAuditError(RuntimeError):
    """Raised when Gate 4 leakage or reproducibility requirements are violated."""


def _subject_tuple(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    normalized = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in normalized):
        raise ScientificAuditError(f"{field_name} must contain non-empty subject IDs")
    if len(set(normalized)) != len(normalized):
        raise ScientificAuditError(f"{field_name} must not contain duplicates")
    return normalized


def _digest(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ScientificAuditError(f"{field_name} must be a lowercase SHA256 digest")
    return value


def _digest_map(
    values: tuple[tuple[str, str], ...],
    field_name: str,
) -> tuple[tuple[str, str], ...]:
    normalized = tuple(values)
    names: set[str] = set()
    checked: list[tuple[str, str]] = []
    for name, digest in normalized:
        if not isinstance(name, str) or not name.strip():
            raise ScientificAuditError(f"{field_name} condition names must be non-empty")
        if name in names:
            raise ScientificAuditError(f"{field_name} contains duplicate condition {name!r}")
        names.add(name)
        checked.append((name, _digest(digest, f"{field_name}[{name!r}]")))
    return tuple(sorted(checked))


@dataclass(frozen=True, slots=True)
class ScientificAuditSnapshot:
    """Immutable evidence required to decide the Gate 4 scientific audit."""

    scientific_config_sha256: str
    seed: int
    train_subject_ids: tuple[str, ...]
    test_subject_ids: tuple[str, ...]
    preprocessing_fit_subject_ids: tuple[str, ...]
    discovery_subject_ids: tuple[str, ...]
    ridge_tuning_subject_ids: tuple[str, ...]
    null_construction_subject_ids: tuple[str, ...]
    split_manifest_sha256: str
    parent_set_sha256: str
    ridge_selection_sha256: str
    null_mapping_sha256: tuple[tuple[str, str], ...]
    prediction_sha256: tuple[tuple[str, str], ...]
    metrics_sha256: tuple[tuple[str, str], ...]
    fold_frozen_before_outer_test: bool
    outer_test_evaluation_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scientific_config_sha256",
            _digest(self.scientific_config_sha256, "scientific_config_sha256"),
        )
        for field_name in (
            "split_manifest_sha256",
            "parent_set_sha256",
            "ridge_selection_sha256",
        ):
            object.__setattr__(self, field_name, _digest(getattr(self, field_name), field_name))

        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ScientificAuditError("seed must be a non-negative integer")

        train = _subject_tuple(self.train_subject_ids, "train_subject_ids")
        test = _subject_tuple(self.test_subject_ids, "test_subject_ids")
        if not train or not test:
            raise ScientificAuditError("train_subject_ids and test_subject_ids must be non-empty")
        if set(train) & set(test):
            raise ScientificAuditError("outer-train and outer-test subjects must be disjoint")
        object.__setattr__(self, "train_subject_ids", train)
        object.__setattr__(self, "test_subject_ids", test)

        outer_train = set(train)
        outer_test = set(test)
        for field_name in (
            "preprocessing_fit_subject_ids",
            "discovery_subject_ids",
            "ridge_tuning_subject_ids",
            "null_construction_subject_ids",
        ):
            scope = _subject_tuple(getattr(self, field_name), field_name)
            leaked = set(scope) & outer_test
            if leaked:
                raise ScientificAuditError(
                    f"{field_name} contains outer-test subjects: {sorted(leaked)}"
                )
            outside = set(scope) - outer_train
            if outside:
                raise ScientificAuditError(
                    f"{field_name} contains subjects outside outer-train: {sorted(outside)}"
                )
            object.__setattr__(self, field_name, scope)

        for field_name in (
            "null_mapping_sha256",
            "prediction_sha256",
            "metrics_sha256",
        ):
            normalized = _digest_map(getattr(self, field_name), field_name)
            if not normalized:
                raise ScientificAuditError(f"{field_name} must not be empty")
            object.__setattr__(self, field_name, normalized)

        if not isinstance(self.fold_frozen_before_outer_test, bool):
            raise ScientificAuditError("fold_frozen_before_outer_test must be boolean")
        if not self.fold_frozen_before_outer_test:
            raise ScientificAuditError("outer-test was accessed before fold freeze")
        if (
            not isinstance(self.outer_test_evaluation_count, int)
            or isinstance(self.outer_test_evaluation_count, bool)
            or self.outer_test_evaluation_count != 1
        ):
            raise ScientificAuditError(
                "outer_test_evaluation_count must be exactly one for the dry-run audit"
            )


@dataclass(frozen=True, slots=True)
class ScientificAuditResult:
    """Successful Gate 4 audit result."""

    status: str
    checks: tuple[str, ...]


_REPRODUCIBILITY_FIELDS = (
    "scientific_config_sha256",
    "seed",
    "train_subject_ids",
    "test_subject_ids",
    "preprocessing_fit_subject_ids",
    "discovery_subject_ids",
    "ridge_tuning_subject_ids",
    "null_construction_subject_ids",
    "split_manifest_sha256",
    "parent_set_sha256",
    "ridge_selection_sha256",
    "null_mapping_sha256",
    "prediction_sha256",
    "metrics_sha256",
    "fold_frozen_before_outer_test",
    "outer_test_evaluation_count",
)


def audit_scientific_snapshots(
    first: ScientificAuditSnapshot,
    second: ScientificAuditSnapshot,
) -> ScientificAuditResult:
    """Require two same-config/same-seed dry runs to match at every frozen boundary."""

    if not isinstance(first, ScientificAuditSnapshot) or not isinstance(
        second, ScientificAuditSnapshot
    ):
        raise TypeError("first and second must be ScientificAuditSnapshot instances")

    for field_name in _REPRODUCIBILITY_FIELDS:
        if getattr(first, field_name) != getattr(second, field_name):
            raise ScientificAuditError(
                f"reproducibility mismatch for {field_name}"
            )

    return ScientificAuditResult(
        status="PASS",
        checks=(
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
        ),
    )
