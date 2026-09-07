"""Deterministic SplitManifest serialization and strict loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import ContractError, InnerFold, SplitManifest


SPLIT_MANIFEST_SCHEMA_VERSION = 1
_ROOT_KEYS = {
    "schema_version",
    "outer_fold",
    "seed",
    "train_subject_ids",
    "test_subject_ids",
    "inner_folds",
}
_INNER_FOLD_KEYS = {
    "inner_train_subject_ids",
    "inner_val_subject_ids",
}


class SplitManifestReadError(ValueError):
    """Raised when a serialized split manifest is malformed or incompatible."""


def build_split_manifest_payload(manifest: SplitManifest) -> dict[str, Any]:
    """Convert a validated SplitManifest into its versioned JSON payload."""

    if not isinstance(manifest, SplitManifest):
        raise TypeError("manifest must be a SplitManifest")

    return {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "outer_fold": manifest.outer_fold,
        "seed": manifest.seed,
        "train_subject_ids": list(manifest.train_subject_ids),
        "test_subject_ids": list(manifest.test_subject_ids),
        "inner_folds": [
            {
                "inner_train_subject_ids": list(fold.inner_train_subject_ids),
                "inner_val_subject_ids": list(fold.inner_val_subject_ids),
            }
            for fold in manifest.inner_folds
        ],
    }


def write_split_manifest_json(path: str | Path, manifest: SplitManifest) -> Path:
    """Write a deterministic UTF-8 JSON representation of ``manifest``."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_split_manifest_payload(manifest)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def _require_exact_keys(value: Any, expected: set[str], *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SplitManifestReadError(f"{context} must be a JSON object")
    keys = set(value)
    missing = sorted(expected - keys)
    extra = sorted(keys - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing keys={missing}")
        if extra:
            details.append(f"unexpected keys={extra}")
        raise SplitManifestReadError(f"{context} key mismatch: {'; '.join(details)}")
    return value


def _require_list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise SplitManifestReadError(f"{field} must be a JSON array")
    return value


def read_split_manifest_json(path: str | Path) -> SplitManifest:
    """Read a versioned split artifact and revalidate the SplitManifest contract."""

    input_path = Path(path)
    if not input_path.is_file():
        raise SplitManifestReadError(f"split manifest file does not exist: {input_path}")

    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise SplitManifestReadError("split manifest must be UTF-8 encoded") from exc
    except json.JSONDecodeError as exc:
        raise SplitManifestReadError(f"invalid split manifest JSON: {exc.msg}") from exc

    root = _require_exact_keys(payload, _ROOT_KEYS, context="split manifest")
    version = root["schema_version"]
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != SPLIT_MANIFEST_SCHEMA_VERSION
    ):
        raise SplitManifestReadError(
            f"unsupported split manifest schema_version={version!r}; "
            f"expected {SPLIT_MANIFEST_SCHEMA_VERSION}"
        )

    train_subject_ids = _require_list(
        root["train_subject_ids"], field="train_subject_ids"
    )
    test_subject_ids = _require_list(
        root["test_subject_ids"], field="test_subject_ids"
    )
    inner_payloads = _require_list(root["inner_folds"], field="inner_folds")

    inner_folds: list[InnerFold] = []
    for index, raw_fold in enumerate(inner_payloads):
        fold = _require_exact_keys(
            raw_fold,
            _INNER_FOLD_KEYS,
            context=f"inner_folds[{index}]",
        )
        inner_train = _require_list(
            fold["inner_train_subject_ids"],
            field=f"inner_folds[{index}].inner_train_subject_ids",
        )
        inner_val = _require_list(
            fold["inner_val_subject_ids"],
            field=f"inner_folds[{index}].inner_val_subject_ids",
        )
        try:
            inner_folds.append(
                InnerFold(
                    inner_train_subject_ids=tuple(inner_train),
                    inner_val_subject_ids=tuple(inner_val),
                )
            )
        except ContractError as exc:
            raise SplitManifestReadError(
                f"inner_folds[{index}] violates InnerFold contract: {exc}"
            ) from exc

    try:
        return SplitManifest(
            outer_fold=root["outer_fold"],
            train_subject_ids=tuple(train_subject_ids),
            test_subject_ids=tuple(test_subject_ids),
            inner_folds=tuple(inner_folds),
            seed=root["seed"],
        )
    except ContractError as exc:
        raise SplitManifestReadError(
            f"serialized split violates SplitManifest contract: {exc}"
        ) from exc
