from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest
import yaml

from lagged_facial_graph_forecasting import (
    PrimaryExecutionMode,
    RunnerConfigError,
    ScientificConfigError,
    load_primary_experiment_config,
    load_scientific_config,
)


ROOT = Path(__file__).resolve().parents[1]
SCIENTIFIC = ROOT / "configs" / "scientific_freeze.yaml"


def _raw_scientific() -> dict:
    return yaml.safe_load(SCIENTIFIC.read_text(encoding="utf-8"))


def _write_run(tmp_path: Path, *, mode: str = "preflight") -> Path:
    configs = tmp_path / "configs"
    configs.mkdir()
    scientific = _raw_scientific()
    scientific_path = configs / "scientific_freeze.yaml"
    scientific_path.write_text(yaml.safe_dump(scientific, sort_keys=False), encoding="utf-8")
    run = {
        "schema_version": 4,
        "experiment_id": "primary-test",
        "scientific_config_path": "configs/scientific_freeze.yaml",
        "run_config_path": "configs/primary_run.yaml",
        "artifact_root": "artifacts/primary",
        "seed": 20260911,
        "protocol_id": "primary-2026-09-12-adopted",
        "execution_mode": mode,
        "preflight_manifest_path": None,
        "executable_freeze_path": None,
    }
    path = configs / "primary_run.yaml"
    path.write_text(yaml.safe_dump(run, sort_keys=False), encoding="utf-8")
    return path


def test_repository_uses_adopted_protocol_and_explicit_preflight_state() -> None:
    scientific = load_scientific_config(SCIENTIFIC)
    run = load_primary_experiment_config("configs/primary_run.yaml", repository_root=ROOT)

    assert scientific["schema_version"] == 7
    assert scientific["protocol_id"] == "primary-2026-09-12-adopted"
    assert scientific["protocol_status"] == "preflight_required"
    assert scientific["primary"]["discovery"]["tau_max"] is None
    assert scientific["primary"]["matched_sparsity"]["repeat_count"] == 1000
    assert run.run_schema_version == 4
    assert run.execution_mode is PrimaryExecutionMode.PREFLIGHT


def test_scientific_loader_rejects_legacy_v6(tmp_path: Path) -> None:
    config = _raw_scientific()
    config["schema_version"] = 6
    with pytest.raises(ScientificConfigError, match="schema_version"):
        load_scientific_config(
            _write_scientific(tmp_path, config), repository_root=tmp_path
        )


def test_real_runner_rejects_missing_preflight_facts(tmp_path: Path) -> None:
    with pytest.raises(RunnerConfigError, match="preflight facts"):
        load_primary_experiment_config(
            _write_run(tmp_path, mode="real"), repository_root=tmp_path
        )


def test_real_materialization_requires_dynamic_lag_and_freeze_hash(tmp_path: Path) -> None:
    config = _raw_scientific()
    config["protocol_status"] = "executable"
    preflight = config["preflight"]
    data = tmp_path / "data"
    artifacts = tmp_path / "artifacts" / "primary"
    data.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    paths = {
        "dataset_manifest_path": data / "dataset.json",
        "actual_subject_group_inventory_path": data / "inventory.json",
        "outer_split_manifest_path": artifacts / "splits.json",
        "common_support_manifest_path": artifacts / "support.json",
    }
    for field, path in paths.items():
        path.write_text("{}\n", encoding="utf-8")
        preflight[field] = path.relative_to(tmp_path).as_posix()
    preflight["dataset_manifest_sha256"] = _sha256(paths["dataset_manifest_path"])
    preflight["extractor_package_version"] = "mediapipe-test"
    preflight["extractor_package_sha256"] = "a" * 64
    preflight["model_asset_sha256"] = "b" * 64
    preflight["actual_fps"] = 30
    preflight["actual_cadence"] = "30fps"
    preflight["actual_self_history_frames"] = 30
    preflight["outer_split_manifest_sha256"] = _sha256(paths["outer_split_manifest_path"])
    preflight["common_support_manifest_sha256"] = _sha256(paths["common_support_manifest_path"])
    config["primary"]["discovery"]["tau_max"] = 15
    config["primary"]["lag_response"]["delta_frames"] = list(range(-2, 3))
    freeze_path = artifacts / "executable_freeze.json"
    preflight["executable_freeze_path"] = freeze_path.relative_to(tmp_path).as_posix()
    scientific_path = tmp_path / "configs" / "scientific_freeze.yaml"
    scientific_path.parent.mkdir()
    authority_dir = tmp_path / "docs"
    authority_dir.mkdir()
    for name in (
        "authoritative_primary_experiment_spec_2026-09-12.md",
        "adopted_decisions_2026-09-12.md",
        "primary_experiment_recommendation_2026-09-11.md",
    ):
        source = ROOT / "docs" / name
        target = authority_dir / name
        shutil.copyfile(source, target)
    config["authority"]["specification_sha256"] = _sha256(
        authority_dir / "authoritative_primary_experiment_spec_2026-09-12.md"
    )
    config["authority"]["adoption_record_sha256"] = _sha256(
        authority_dir / "adopted_decisions_2026-09-12.md"
    )
    config["authority"]["recommendation_sha256"] = _sha256(
        authority_dir / "primary_experiment_recommendation_2026-09-11.md"
    )
    scientific_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    freeze_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_id": "primary-2026-09-12-adopted",
                "freeze_status": "PASS",
                "config_sha256": _sha256(scientific_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    # The config is intentionally rewritten after adding the freeze path; this
    # first check proves stale dynamic values fail before any execution.
    with pytest.raises(ScientificConfigError, match="lag-response deltas"):
        load_scientific_config(
            scientific_path, require_materialized=True, repository_root=tmp_path
        )


def _write_scientific(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "scientific_freeze.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
