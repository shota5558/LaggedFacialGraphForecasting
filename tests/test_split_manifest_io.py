from __future__ import annotations

import json

import pytest

from lagged_facial_graph_forecasting.split_manifest_io import (
    SPLIT_MANIFEST_SCHEMA_VERSION,
    SplitManifestReadError,
    build_split_manifest_payload,
    read_split_manifest_json,
    write_split_manifest_json,
)
from lagged_facial_graph_forecasting.splits import build_subject_split_manifest


def _manifest():
    return build_subject_split_manifest(
        tuple(f"s{index:02d}" for index in range(1, 7)),
        outer_fold=2,
        test_subject_count=2,
        n_inner_folds=2,
        seed=17,
    )


def test_split_manifest_reader_round_trips_exact_contract(tmp_path) -> None:
    manifest = _manifest()
    output_path = tmp_path / "split.json"
    write_split_manifest_json(output_path, manifest)

    assert read_split_manifest_json(output_path) == manifest


def test_split_manifest_reader_rejects_unknown_schema_version(tmp_path) -> None:
    payload = build_split_manifest_payload(_manifest())
    payload["schema_version"] = SPLIT_MANIFEST_SCHEMA_VERSION + 1
    path = tmp_path / "future.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SplitManifestReadError, match="unsupported.*schema_version"):
        read_split_manifest_json(path)


def test_split_manifest_reader_rejects_unknown_keys(tmp_path) -> None:
    payload = build_split_manifest_payload(_manifest())
    payload["unexpected"] = True
    path = tmp_path / "extra.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SplitManifestReadError, match="unexpected keys"):
        read_split_manifest_json(path)


def test_split_manifest_reader_rejects_outer_test_leakage_into_inner_fold(tmp_path) -> None:
    payload = build_split_manifest_payload(_manifest())
    leaked_subject = payload["test_subject_ids"][0]
    payload["inner_folds"][0]["inner_val_subject_ids"].append(leaked_subject)
    path = tmp_path / "leaked.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SplitManifestReadError, match="outer-test"):
        read_split_manifest_json(path)


def test_split_manifest_reader_rejects_invalid_json(tmp_path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(SplitManifestReadError, match="invalid split manifest JSON"):
        read_split_manifest_json(path)
