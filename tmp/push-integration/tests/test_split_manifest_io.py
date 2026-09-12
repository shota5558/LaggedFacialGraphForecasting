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


def test_split_manifest_payload_preserves_all_split_provenance() -> None:
    manifest = _manifest()
    payload = build_split_manifest_payload(manifest)

    assert payload["schema_version"] == SPLIT_MANIFEST_SCHEMA_VERSION
    assert payload["outer_fold"] == manifest.outer_fold
    assert payload["seed"] == manifest.seed
    assert payload["train_subject_ids"] == list(manifest.train_subject_ids)
    assert payload["test_subject_ids"] == list(manifest.test_subject_ids)
    assert payload["inner_folds"] == [
        {
            "inner_train_subject_ids": list(fold.inner_train_subject_ids),
            "inner_val_subject_ids": list(fold.inner_val_subject_ids),
        }
        for fold in manifest.inner_folds
    ]


def test_split_manifest_writer_is_deterministic(tmp_path) -> None:
    manifest = _manifest()
    first_path = tmp_path / "first" / "split.json"
    second_path = tmp_path / "second" / "split.json"

    write_split_manifest_json(first_path, manifest)
    write_split_manifest_json(second_path, manifest)

    first = first_path.read_bytes()
    second = second_path.read_bytes()
    assert first == second
    assert first.endswith(b"\n")
    assert json.loads(first.decode("utf-8")) == build_split_manifest_payload(manifest)


def test_split_manifest_writer_does_not_serialize_runtime_object_repr(tmp_path) -> None:
    output_path = tmp_path / "split.json"
    write_split_manifest_json(output_path, _manifest())

    serialized = output_path.read_text(encoding="utf-8")
    assert "SplitManifest(" not in serialized
    assert "InnerFold(" not in serialized


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
