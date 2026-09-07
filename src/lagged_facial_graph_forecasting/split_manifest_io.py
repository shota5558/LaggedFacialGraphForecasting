"""Deterministic SplitManifest serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import SplitManifest


SPLIT_MANIFEST_SCHEMA_VERSION = 1


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
