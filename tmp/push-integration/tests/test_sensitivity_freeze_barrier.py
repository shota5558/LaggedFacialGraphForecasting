from __future__ import annotations

from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

from lagged_facial_graph_forecasting.core_contracts import ExperimentConfig
from lagged_facial_graph_forecasting.sensitivity_execution import (
    PRIMARY_ARTIFACT_ROOT,
    PRIMARY_FREEZE_MANIFEST_PATH,
    PRIMARY_FREEZE_REQUIRED_ARTIFACT_TYPES,
    SensitivityExecutionError,
    SensitivityExecutionMode,
    SensitivityExperimentConfig,
    assert_real_sensitivity_artifact_provenance,
    assert_sensitivity_execution_allowed,
    resolve_sensitivity_output_path,
    validate_primary_freeze_manifest,
)


def _config(mode: SensitivityExecutionMode) -> SensitivityExperimentConfig:
    return SensitivityExperimentConfig(
        experiment=ExperimentConfig(
            experiment_id="sensitivity-barrier-test",
            scientific_config_path="configs/scientific_freeze.yaml",
            run_config_path="configs/sensitivity_preimplementation.yaml",
            artifact_root="artifacts/sensitivity",
            seed=1010,
        ),
        namespace="sensitivity",
        execution_mode=mode,
        primary_freeze_manifest=PRIMARY_FREEZE_MANIFEST_PATH,
    )


def _write_complete_freeze(root) -> dict:
    primary = root / PRIMARY_ARTIFACT_ROOT
    primary.mkdir(parents=True, exist_ok=True)
    entries = []
    for artifact_type in sorted(PRIMARY_FREEZE_REQUIRED_ARTIFACT_TYPES):
        relative_path = f"{PRIMARY_ARTIFACT_ROOT}/{artifact_type}.json"
        content = json.dumps(
            {"artifact_type": artifact_type, "fixture": "synthetic-freeze-contract"},
            sort_keys=True,
        ).encode("utf-8")
        path = root / relative_path
        path.write_bytes(content)
        entries.append(
            {
                "artifact_type": artifact_type,
                "relative_path": relative_path,
                "sha256": sha256(content).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "freeze_status": "PASS",
        "artifact_root": PRIMARY_ARTIFACT_ROOT,
        "artifacts": entries,
    }
    (root / PRIMARY_FREEZE_MANIFEST_PATH).write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return manifest


def test_synthetic_execution_remains_allowed_without_primary_freeze_manifest(tmp_path) -> None:
    assert_sensitivity_execution_allowed(
        _config(SensitivityExecutionMode.SYNTHETIC), repository_root=tmp_path
    )


def test_real_execution_fails_closed_when_primary_freeze_manifest_is_missing(tmp_path) -> None:
    with pytest.raises(SensitivityExecutionError, match="manifest exists"):
        assert_sensitivity_execution_allowed(
            _config(SensitivityExecutionMode.REAL), repository_root=tmp_path
        )


def test_real_execution_accepts_only_complete_hash_verified_primary_freeze(tmp_path) -> None:
    expected = _write_complete_freeze(tmp_path)
    config = _config(SensitivityExecutionMode.REAL)

    validated = validate_primary_freeze_manifest(config, repository_root=tmp_path)
    assert validated == expected
    assert_sensitivity_execution_allowed(config, repository_root=tmp_path)


def test_incomplete_primary_freeze_manifest_is_rejected(tmp_path) -> None:
    manifest = _write_complete_freeze(tmp_path)
    manifest["artifacts"] = [
        entry
        for entry in manifest["artifacts"]
        if entry["artifact_type"] != "software_versions"
    ]
    (tmp_path / PRIMARY_FREEZE_MANIFEST_PATH).write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(SensitivityExecutionError, match="incomplete"):
        assert_sensitivity_execution_allowed(
            _config(SensitivityExecutionMode.REAL), repository_root=tmp_path
        )


def test_primary_freeze_hash_mismatch_is_rejected(tmp_path) -> None:
    manifest = _write_complete_freeze(tmp_path)
    metrics = next(
        entry for entry in manifest["artifacts"] if entry["artifact_type"] == "metrics"
    )
    (tmp_path / metrics["relative_path"]).write_text("tampered", encoding="utf-8")

    with pytest.raises(SensitivityExecutionError, match="hash mismatch"):
        assert_sensitivity_execution_allowed(
            _config(SensitivityExecutionMode.REAL), repository_root=tmp_path
        )


def test_real_sensitivity_artifact_must_reference_exact_validated_freeze(tmp_path) -> None:
    _write_complete_freeze(tmp_path)
    config = _config(SensitivityExecutionMode.REAL)
    matching = SimpleNamespace(source_primary_freeze_manifest=PRIMARY_FREEZE_MANIFEST_PATH)
    assert_real_sensitivity_artifact_provenance(
        config, matching, repository_root=tmp_path
    )

    wrong = SimpleNamespace(
        source_primary_freeze_manifest="artifacts/primary/other_manifest.json"
    )
    with pytest.raises(SensitivityExecutionError, match="exact validated"):
        assert_real_sensitivity_artifact_provenance(
            config, wrong, repository_root=tmp_path
        )


def test_sensitivity_output_path_cannot_escape_into_primary_namespace(tmp_path) -> None:
    config = _config(SensitivityExecutionMode.SYNTHETIC)
    primary_target = tmp_path / "artifacts/primary/forbidden.json"
    with pytest.raises(SensitivityExecutionError, match="Sensitivity output path"):
        resolve_sensitivity_output_path(
            config, primary_target, repository_root=tmp_path
        )
