"""Loader and preflight gate for the adopted 2026-09-12 Primary protocol."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

import yaml


class ScientificConfigError(ValueError):
    """Raised when the adopted protocol is missing, malformed, or stale."""


SCIENTIFIC_CONFIG_SCHEMA_VERSION = 7
PRIMARY_PROTOCOL_ID = "primary-2026-09-12-adopted"
LEGACY_SCIENTIFIC_SCHEMA_VERSION = 6
LEGACY_PRIMARY_ARTIFACT_MANIFEST = "artifacts/primary/freeze_manifest.json"
EXECUTABLE_FREEZE_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ANY = object()


def _template() -> dict[str, Any]:
    return {
        "schema_version": SCIENTIFIC_CONFIG_SCHEMA_VERSION,
        "protocol_id": PRIMARY_PROTOCOL_ID,
        "protocol_status": _ANY,
        "authority": {
            "specification": "docs/authoritative_primary_experiment_spec_2026-09-12.md",
            "specification_sha256": _ANY,
            "adoption_record": "docs/adopted_decisions_2026-09-12.md",
            "adoption_record_sha256": _ANY,
            "recommendation": "docs/primary_experiment_recommendation_2026-09-11.md",
            "recommendation_sha256": _ANY,
        },
        "compatibility": {
            "supersedes_schema_version": LEGACY_SCIENTIFIC_SCHEMA_VERSION,
            "legacy_artifacts": "reject",
            "explicit_migration_required": True,
        },
        "primary": {
            "horizon": 1,
            "conditions": ["self_speaking", "self_non_speaking"],
            "analysis": [
                "predictive_gain_landscape",
                "four_condition_comparison",
                "selection_enrichment",
                "selected_lag_centered_response",
                "discovery_stability",
                "data_execution_audit",
            ],
            "representation": {
                "input": "normalized_2d_displacement",
                "node_unit": "landmark_coordinate_scalar",
                "forecast_projection": "region_block_or",
                "forecast_feature_rule": "all_usable_source_scalars",
                "raw_scalar_links_preserved": True,
            },
            "discovery": {
                "method": "pcmci_plus",
                "ci_test": "parcorr",
                "pc_alpha": 0.01,
                "significance": "analytic",
                "mask_type": "xyz",
                "recycle_residuals": False,
                "tau_min": 0,
                "tau_max": _ANY,
                "tau_max_rule": "floor_half_fps",
                "contemporaneous_forecast_input": "forbidden",
                "contemp_collider_rule": "majority",
                "conflict_resolution": True,
                "reset_lagged_links": False,
                "max_combinations": 1,
                "max_conds_dim": None,
                "max_conds_py": None,
                "max_conds_px": None,
                "max_conds_px_lagged": None,
                "fdr_method": "none",
            },
            "lag": {
                "candidate_rule": "integer_frames_1_through_floor_half_fps",
                "bands_ms": [
                    {"label": "B1", "lower_exclusive": 0, "upper_inclusive": 100},
                    {"label": "B2", "lower_exclusive": 100, "upper_inclusive": 250},
                    {"label": "B3", "lower_exclusive": 250, "upper_inclusive": 500},
                ],
            },
            "self_history": {
                "selection": "pilot_60_20_20_shortest_within_1_percent",
                "candidate_seconds": [0.5, 1.0, 2.0],
                "optional_extension_seconds": 4.0,
                "history_frames": _ANY,
            },
            "forecaster": {
                "method": "ridge",
                "intercept": True,
                "target_standardization": False,
                "input_standardization": "train_only",
                "subject_equal_weight": True,
                "alpha_grid": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
                "alpha_selection": "inner_subject_equal_main_error",
                "alpha_tie_tolerance": 1e-12,
                "alpha_tie_break": "larger",
                "outer_test_grid_extension": "forbidden",
            },
            "matched_sparsity": {
                "repeat_count": 1000,
                "sampling_unit": "source_region_lag_block_scalar_count_strata",
                "within_set_replacement": "forbidden",
                "selected_overlap": "allowed",
                "repeat_overlap": "allowed",
                "lag_distribution_matching": "forbidden",
            },
            "lag_response": {
                "estimand": "self_plus_single_selected_source_region_lag_block",
                "delta_frames": _ANY,
                "delta_rule": "symmetric_floor_tenth_fps",
                "reference_delta": 0,
                "complete_symmetric_grid": True,
                "clipping": "forbidden",
                "wrapping": "forbidden",
                "one_sided_grid": "forbidden",
                "boundary_policy": "unevaluable_without_common_grid",
            },
            "statistics": {
                "paired_unit": "subject",
                "point_aggregation": "median",
                "bootstrap_unit": "dependency_group",
                "bootstrap": {
                    "confidence_level": 0.95,
                    "n_resamples": 10000,
                    "method": "percentile_linear",
                    "shared_indices_across_outputs": True,
                    "minimum_evaluable_groups": 10,
                },
                "held_out_significance": "not_primary",
                "fdr": "not_primary",
                "full_noninferiority_margin": None,
            },
            "discovery_stability": {
                "resampling_unit": "dependency_group",
                "repeat_count": 100,
                "time_order": "preserved",
                "duplicate_group": "independent_series_copy",
                "concatenation": "forbidden",
                "parent_set_redefinition": "forbidden",
            },
        },
        "measurement": {
            "regions": [
                "left_brow",
                "right_brow",
                "left_eyelid",
                "right_eyelid",
                "left_cheek",
                "right_cheek",
                "mouth",
                "jaw",
            ],
            "landmark_count": 29,
            "scalar_component_count": 58,
            "extractor": "mediapipe_face_landmarker",
            "mode": "IMAGE",
            "num_faces": 1,
            "detection_threshold": 0.5,
            "presence_threshold": 0.5,
            "blendshapes": False,
            "pose_matrix": True,
            "point_mapping": {
                "left_brow": [336, 296, 300],
                "right_brow": [107, 66, 70],
                "left_eyelid": [385, 386, 380, 374],
                "right_eyelid": [158, 159, 153, 145],
                "left_cheek": [425, 280],
                "right_cheek": [205, 50],
                "mouth": [61, 291, 13, 14, 37, 267, 84, 314],
                "jaw": [176, 152, 400],
            },
        },
        "data": {
            "dataset_family": "NoXi",
            "distribution_policy": "single_distribution_only",
            "fallback_to_caviares": "forbidden",
            "conditions": ["self_speaking", "self_non_speaking"],
            "condition_boundary_exclusion_ms": 200,
            "calibration_seconds": 10,
            "normalization": "eye_corner_similarity_transform",
            "normalization_reference_distance_min_px": 20,
            "pose_abs_yaw_pitch_max_degrees": 20,
            "missingness": "no_interpolation_or_resampling",
            "point_valid_subject_fraction": 0.8,
            "point_variance_min": 1e-12,
        },
        "split": {
            "seed": 20260911,
            "unit": "dependency_group",
            "pilot_min_subjects": 5,
            "outer_folds": 5,
            "outer_fallback_folds": 3,
            "inner_folds": 3,
            "pilot_excluded_from_formal": True,
            "loso_fallback": "forbidden",
            "frame_split": "forbidden",
        },
        "preflight": {
            "dataset_manifest_path": _ANY,
            "dataset_manifest_sha256": _ANY,
            "extractor_package_version": _ANY,
            "extractor_package_sha256": _ANY,
            "model_asset_sha256": _ANY,
            "actual_fps": _ANY,
            "actual_cadence": _ANY,
            "actual_subject_group_inventory_path": _ANY,
            "actual_self_history_frames": _ANY,
            "outer_split_manifest_path": _ANY,
            "outer_split_manifest_sha256": _ANY,
            "common_support_manifest_path": _ANY,
            "common_support_manifest_sha256": _ANY,
            "executable_freeze_path": _ANY,
        },
        "artifacts": {
            "primary_root": "artifacts/primary",
            "sensitivity_root": "artifacts/sensitivity",
        },
    }


_EXPECTED = _template()


def _validate_template(actual: Any, expected: Any, path: str = "root") -> None:
    if expected is _ANY:
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ScientificConfigError(f"{path} must be a mapping")
        expected_keys = set(expected)
        actual_keys = set(actual)
        if actual_keys != expected_keys:
            raise ScientificConfigError(
                f"{path} keys differ from adopted protocol; "
                f"missing={sorted(expected_keys - actual_keys)}, "
                f"extra={sorted(actual_keys - expected_keys)}"
            )
        for key, expected_value in expected.items():
            _validate_template(actual[key], expected_value, f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or actual != expected:
            raise ScientificConfigError(
                f"{path} differs from adopted protocol: expected={expected!r}, actual={actual!r}"
            )
        return
    if actual != expected:
        raise ScientificConfigError(
            f"{path} differs from adopted protocol: expected={expected!r}, actual={actual!r}"
        )


def _relative_path(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScientificConfigError(f"{field_name} must be a non-empty repository-relative path")
    path = PurePosixPath(value.strip())
    if path.is_absolute() or ".." in path.parts:
        raise ScientificConfigError(f"{field_name} must be a repository-relative POSIX path")
    return path.as_posix()


def _sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ScientificConfigError(f"{field_name} must be 64 lowercase hexadecimal characters")
    return value


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_materialization(
    config: Mapping[str, Any],
    *,
    repository_root: str | Path | None,
    config_path: Path | None,
) -> None:
    if config["protocol_status"] != "executable":
        raise ScientificConfigError(
            "real execution requires protocol_status='executable'; preflight facts are incomplete"
        )

    root = Path(repository_root or Path.cwd()).resolve()
    authority = config["authority"]
    for path_field, hash_field in (
        ("specification", "specification_sha256"),
        ("adoption_record", "adoption_record_sha256"),
        ("recommendation", "recommendation_sha256"),
    ):
        authority_path = (root / _relative_path(authority[path_field], f"authority.{path_field}")).resolve()
        if not authority_path.is_file():
            raise ScientificConfigError(f"authority.{path_field} does not exist")
        expected_hash = _sha256(authority[hash_field], f"authority.{hash_field}")
        if _hash_file(authority_path) != expected_hash:
            raise ScientificConfigError(f"authority.{path_field} hash does not match")

    preflight = config["preflight"]
    path_fields = (
        "dataset_manifest_path",
        "actual_subject_group_inventory_path",
        "outer_split_manifest_path",
        "common_support_manifest_path",
        "executable_freeze_path",
    )
    hash_fields = (
        "dataset_manifest_sha256",
        "extractor_package_sha256",
        "model_asset_sha256",
        "outer_split_manifest_sha256",
        "common_support_manifest_sha256",
    )
    for field_name in path_fields:
        relative = _relative_path(preflight[field_name], f"preflight.{field_name}")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ScientificConfigError(f"preflight.{field_name} escapes repository_root") from exc
        if not path.is_file():
            raise ScientificConfigError(f"preflight.{field_name} does not exist: {relative}")
    for field_name in hash_fields:
        _sha256(preflight[field_name], f"preflight.{field_name}")

    fps = preflight["actual_fps"]
    if (
        not isinstance(fps, (int, float))
        or isinstance(fps, bool)
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise ScientificConfigError("preflight.actual_fps must be a positive number")
    history = preflight["actual_self_history_frames"]
    if not isinstance(history, int) or isinstance(history, bool) or history < 1:
        raise ScientificConfigError("preflight.actual_self_history_frames must be a positive integer")
    cadence = preflight["actual_cadence"]
    if not isinstance(cadence, str) or not cadence.strip():
        raise ScientificConfigError("preflight.actual_cadence must be a non-empty string")
    version = preflight["extractor_package_version"]
    if not isinstance(version, str) or not version.strip():
        raise ScientificConfigError("preflight.extractor_package_version must be non-empty")
    if config["primary"]["discovery"]["tau_max"] != int(fps * 0.5):
        raise ScientificConfigError("preflight tau_max must equal floor(0.5 * actual_fps)")
    expected_deltas = list(range(-int(fps * 0.1), int(fps * 0.1) + 1))
    if config["primary"]["lag_response"]["delta_frames"] != expected_deltas:
        raise ScientificConfigError("preflight lag-response deltas must equal floor(0.1 * actual_fps)")

    freeze_path = (root / _relative_path(preflight["executable_freeze_path"], "preflight.executable_freeze_path")).resolve()
    try:
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ScientificConfigError("executable freeze record is not valid JSON") from exc
    required = {"schema_version", "protocol_id", "freeze_status", "config_sha256"}
    if not isinstance(freeze, dict) or set(freeze) < required:
        raise ScientificConfigError("executable freeze record is incomplete")
    if freeze["schema_version"] != EXECUTABLE_FREEZE_SCHEMA_VERSION:
        raise ScientificConfigError("unsupported executable freeze schema_version")
    if freeze["protocol_id"] != PRIMARY_PROTOCOL_ID or freeze["freeze_status"] != "PASS":
        raise ScientificConfigError("executable freeze record does not belong to the adopted protocol")
    if config_path is not None:
        expected_config_sha = _hash_file(config_path)
        if freeze["config_sha256"] != expected_config_sha:
            raise ScientificConfigError("executable freeze config_sha256 does not match scientific config")


def validate_scientific_config(
    config: Mapping[str, Any],
    *,
    require_materialized: bool = False,
    repository_root: str | Path | None = None,
    config_path: str | Path | None = None,
) -> None:
    """Validate the adopted protocol, optionally requiring real preflight facts."""

    _validate_template(config, _EXPECTED)
    if config["protocol_status"] not in {"preflight_required", "executable"}:
        raise ScientificConfigError("protocol_status must be 'preflight_required' or 'executable'")
    for field_name in (
        "specification_sha256",
        "adoption_record_sha256",
        "recommendation_sha256",
    ):
        value = config["authority"][field_name]
        if value is not None:
            _sha256(value, f"authority.{field_name}")
    for field_name in (
        "dataset_manifest_sha256",
        "extractor_package_sha256",
        "model_asset_sha256",
        "outer_split_manifest_sha256",
        "common_support_manifest_sha256",
    ):
        value = config["preflight"][field_name]
        if value is not None:
            _sha256(value, f"preflight.{field_name}")
    if require_materialized or config["protocol_status"] == "executable":
        validate_path = Path(config_path).resolve() if config_path is not None else None
        _validate_materialization(
            config,
            repository_root=repository_root,
            config_path=validate_path,
        )


def load_scientific_config(
    path: str | Path,
    *,
    require_materialized: bool = False,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load and validate the adopted protocol YAML.

    The default accepts the repository's explicit ``preflight_required`` state;
    real runners must pass ``require_materialized=True``.
    """

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path(repository_root or Path.cwd()) / config_path
    config_path = config_path.resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ScientificConfigError(f"scientific config not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ScientificConfigError(f"invalid YAML in scientific config: {config_path}") from exc
    if not isinstance(raw, dict):
        raise ScientificConfigError("scientific config root must be a mapping")
    validate_scientific_config(
        raw,
        require_materialized=require_materialized,
        repository_root=repository_root or config_path.parent.parent,
        config_path=config_path,
    )
    if raw["artifacts"]["primary_root"] == raw["artifacts"]["sensitivity_root"]:
        raise ScientificConfigError("Primary and Sensitivity artifact roots must be separate")
    return raw
