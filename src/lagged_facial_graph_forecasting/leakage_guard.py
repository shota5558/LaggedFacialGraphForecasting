"""Subject-scope guards that prevent outer-test leakage into fitting/selection."""

from __future__ import annotations

from collections.abc import Iterable

from .contracts import SplitManifest


class LeakageGuardError(ValueError):
    """Raised when an operation attempts to access subjects outside outer-train."""


def _assert_outer_train_only_subjects(
    manifest: SplitManifest,
    subject_ids: Iterable[str],
    *,
    operation: str,
) -> tuple[str, ...]:
    if not isinstance(manifest, SplitManifest):
        raise TypeError("manifest must be a SplitManifest")

    normalized = tuple(subject_ids)
    if not normalized:
        raise LeakageGuardError(f"{operation} requires at least one subject")
    if any(not isinstance(subject_id, str) or not subject_id.strip() for subject_id in normalized):
        raise LeakageGuardError(f"{operation} subject IDs must be non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise LeakageGuardError(f"{operation} subject IDs must be unique")

    supplied = set(normalized)
    outer_train = set(manifest.train_subject_ids)
    outer_test = set(manifest.test_subject_ids)

    leaked_test = supplied & outer_test
    if leaked_test:
        raise LeakageGuardError(
            f"{operation} cannot access outer-test subjects: {sorted(leaked_test)}"
        )

    outside = supplied - outer_train
    if outside:
        raise LeakageGuardError(
            f"{operation} contains subjects outside outer-train: {sorted(outside)}"
        )

    return normalized


def assert_preprocessing_fit_scope(
    manifest: SplitManifest,
    subject_ids: Iterable[str],
) -> tuple[str, ...]:
    """Assert that preprocessing parameter fitting uses outer-train subjects only."""

    return _assert_outer_train_only_subjects(
        manifest,
        subject_ids,
        operation="preprocessing fit",
    )


def assert_discovery_fit_scope(
    manifest: SplitManifest,
    subject_ids: Iterable[str],
) -> tuple[str, ...]:
    """Assert that PCMCI+ discovery uses outer-train subjects only."""

    return _assert_outer_train_only_subjects(
        manifest,
        subject_ids,
        operation="discovery fit",
    )
