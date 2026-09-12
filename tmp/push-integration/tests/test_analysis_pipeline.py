from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from lagged_facial_graph_forecasting.analysis_pipeline import (
    AnalysisInputs,
    AnalysisOutputError,
    SYNTHETIC_NOTICE,
    edge_lag_source,
    generate_analysis_outputs,
    lag_response_source,
    null_distribution_source,
    table_t07,
    table_t04,
    table_t08,
)

FIXTURE_ROOT = Path("data/mock_analysis")


def _inputs() -> AnalysisInputs:
    return AnalysisInputs.from_directory(FIXTURE_ROOT, include_sensitivity=True)


def _config() -> dict[str, object]:
    return json.loads((FIXTURE_ROOT / "mock_primary_config.json").read_text(encoding="utf-8"))


def test_issue23_generates_t01_t09_f01_f14_from_mock_with_provenance(tmp_path: Path) -> None:
    result = generate_analysis_outputs(
        _inputs(), repository_root=tmp_path, include_sensitivity=True, allow_mock_sensitivity=True
    )
    assert result.is_synthetic is True
    assert result.publication_ready is False
    assert result.output_root == (tmp_path / "artifacts/mock_analysis").resolve()

    table_names = {
        "T01_dataset_outer_fold_summary.csv", "T02_primary_frozen_configuration.md",
        "T03_primary_condition_performance.csv", "T04_pcmci_vs_self_paired_effect.csv",
        "T05_pcmci_vs_full_sparsity.csv", "T06_region_wise_effect.csv",
        "T07_falsification_summary.csv", "T08_edge_lag_stability.csv",
        "T09_sensitivity_summary.csv",
    }
    for name in table_names:
        assert (result.output_root / "tables" / name).is_file(), name

    figure_names = {
        "F01_primary_condition_comparison.png", "F02_subject_paired_pcmci_vs_self.png",
        "F03_incremental_gain_by_region.png", "F04_lag_response_curve.png",
        "F05_falsification_effect_comparison.png", "F06_performance_sparsity.png",
        "F07_region_edge_stability_heatmap.png", "F08_region_lag_stability_heatmap.png",
        "F09_outer_fold_effect_distribution.png", "F10_sensitivity_forest_plot.png",
        "F11_forecast_trajectory_example.png", "F13_prediction_error_distribution.png",
        "F14_metric_concordance.png",
    }
    for name in figure_names:
        assert (result.output_root / "figures" / name).is_file(), name
    assert (result.output_root / "figures/F12_data_quality/F12a_subject_frame_counts.png").is_file()

    t04 = pd.read_csv(result.output_root / "tables/T04_pcmci_vs_self_paired_effect.csv")
    assert (t04["is_synthetic"] == True).all()  # noqa: E712
    assert (t04["synthetic_notice"] == SYNTHETIC_NOTICE).all()
    assert (t04["median_paired_difference"] > 0).all()
    f04 = pd.read_csv(result.output_root / "tables/F04_lag_response_source.csv")
    assert f04["delta_frames"].tolist() == [-2, -1, 0, 1, 2]
    assert f04.loc[f04.delta_frames == 0, "median_delta_error"].iloc[0] == pytest.approx(0.0)
    assert f04["same_support_across_deltas"].all()

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["publication_ready"] is False
    assert manifest["sensitivity_validation_scope"] == "software_only_mock"
    registry = pd.read_csv(result.registry_path)
    assert not registry.empty
    assert registry["is_synthetic"].all()
    assert registry["output_id"].str.match(r"^(T\d\d|F\d\d|CAPTIONS)$").all()
    assert registry["source_artifacts"].notna().all()
    assert manifest["artifact_registry"]["record_count"] == len(registry)
    assert manifest["artifact_registry"]["sha256"] == hashlib.sha256(result.registry_path.read_bytes()).hexdigest()


def test_issue23_outputs_are_deterministic_for_same_config_seed(tmp_path: Path) -> None:
    first = generate_analysis_outputs(
        _inputs(), repository_root=tmp_path, include_sensitivity=True, allow_mock_sensitivity=True
    )

    def hashes() -> dict[str, str]:
        return {
            p.relative_to(first.output_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in first.artifact_paths
        }

    h1 = hashes()
    generate_analysis_outputs(
        _inputs(), repository_root=tmp_path, include_sensitivity=True, allow_mock_sensitivity=True
    )
    assert hashes() == h1


def test_synthetic_publication_ready_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(AnalysisOutputError, match="never be publication-ready"):
        generate_analysis_outputs(_inputs(), repository_root=tmp_path, publication_ready=True)


def test_t04_rejects_missing_exact_pair_and_uses_self_minus_pcmci_sign() -> None:
    metrics = pd.read_csv(FIXTURE_ROOT / "mock_metrics.csv")
    table, _ = table_t04(metrics, _config())
    assert (table.median_paired_difference > 0).all()
    missing = metrics.drop(metrics[(metrics.condition == "pcmci")].index[0])
    with pytest.raises(AnalysisOutputError, match="no silent dropping"):
        table_t04(missing, _config())


def test_lag_response_marks_incomplete_target_fold_unevaluable_without_clipping() -> None:
    lag = pd.read_csv(FIXTURE_ROOT / "mock_lag_response.csv")
    bad_idx = lag[(lag.subject_id == "MOCK_S01") & (lag.delta_frames == 2)].index[0]
    source = lag_response_source(lag.drop(index=bad_idx), _config())
    assert source.unevaluable_target_folds.nunique() == 1
    assert int(source.unevaluable_target_folds.iloc[0]) == 1
    assert source.delta_frames.tolist() == [-2, -1, 0, 1, 2]
    assert source.same_region_identity.all()
    assert source.same_feature_count.all()
    cfg = _config()
    cfg["lag_response"] = dict(cfg["lag_response"])
    cfg["lag_response"]["clipping"] = "allowed"
    with pytest.raises(AnalysisOutputError, match="clipping must be forbidden"):
        lag_response_source(lag, cfg)


def test_null_provenance_is_outer_train_only_and_matched_sparsity_is_exact() -> None:
    metrics = pd.read_csv(FIXTURE_ROOT / "mock_metrics.csv")
    nulls = pd.read_csv(FIXTURE_ROOT / "mock_null_metrics.csv")
    source = null_distribution_source(metrics, nulls, _config())
    assert set(source.mapping_scope) == {"outer_train_only"}
    assert source.input_count_preserved.all()
    matched = source[source.condition == "matched_sparsity"]
    assert (matched.pcmci_feature_count == matched.null_feature_count).all()
    random_region = source[source.condition.str.startswith("random_region")]
    assert random_region.lag_identity_preserved.all()
    keys = ["outer_fold", "subject_id", "region_id", "condition", "metric_name", "replicate_id"]
    assert not source.duplicated(keys).any()


def _matched_repeat_fixture(values: tuple[float, ...]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    metrics = pd.DataFrame([
        {
            "outer_fold": index,
            "subject_id": subject,
            "region_id": "mouth",
            "condition": "pcmci",
            "metric_name": "velocity_rmse",
            "metric_direction": "lower_is_better",
            "value": 0.0,
            "n_valid": 4,
        }
        for index, subject in enumerate(("S01", "S02"))
    ])
    nulls = pd.DataFrame([
        {
            "outer_fold": outer_fold,
            "subject_id": subject,
            "region_id": "mouth",
            "condition": "matched_sparsity",
            "metric_name": "velocity_rmse",
            "metric_direction": "lower_is_better",
            "value": value,
            "n_valid": 4,
            "replicate_id": f"R{index:03d}",
            "seed": 7,
            "mapping_scope": "outer_train_only",
            "pcmci_feature_count": 2,
            "null_feature_count": 2,
            "input_count_preserved": True,
            "lag_identity_preserved": True,
        }
        for outer_fold, subject in enumerate(("S01", "S02"))
        for index, value in enumerate(values)
    ])
    config = _config()
    config["primary"] = dict(config["primary"])
    config["primary"]["matched_sparsity"] = {
        "repeat_count": len(values),
        "seed_source": "split_manifest_seed",
        "repeat_aggregation": "median_error_across_repeats_per_subject_region",
    }
    return metrics, nulls, config


def test_matched_sparsity_aggregates_repeat_median_before_region_summary() -> None:
    metrics, nulls, config = _matched_repeat_fixture((0.0, 0.0, 9.0))

    summary, _ = table_t07(metrics, nulls, config)

    row = summary.loc[summary.null_condition == "matched_sparsity"].iloc[0]
    assert row.median_null_minus_pcmci == pytest.approx(0.0)
    assert row.mean_null_minus_pcmci == pytest.approx(0.0)
    assert row.repeat_aggregation == "median_error_across_repeats_per_subject_region"


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda frame: frame.iloc[:-1], "exactly 3 unique repeats"),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True), "replicate identity"),
        (lambda frame: frame.assign(seed=float("nan")), "replicate seed"),
    ],
)
def test_matched_sparsity_rejects_incomplete_duplicate_or_unseeded_repeats(mutator, match) -> None:
    metrics, nulls, config = _matched_repeat_fixture((0.0, 0.0, 9.0))

    with pytest.raises(AnalysisOutputError, match=match):
        null_distribution_source(metrics, mutator(nulls), config)


def test_primary_analysis_rejects_metric_and_seed_fallbacks() -> None:
    metrics = pd.read_csv(FIXTURE_ROOT / "mock_metrics.csv")
    config = _config()
    config["evaluation"] = {}
    with pytest.raises(AnalysisOutputError, match="explicitly frozen"):
        table_t04(metrics, config)

    config = _config()
    config.pop("seed")
    with pytest.raises(AnalysisOutputError, match="explicit non-negative integer"):
        table_t04(metrics, config)


def test_stability_validates_tau_max_frequency_and_missing_vs_zero() -> None:
    edges = pd.read_csv(FIXTURE_ROOT / "mock_edge_stability.csv")
    table = table_t08(edges)
    assert table.lag.between(1, 10).all()
    bad = edges.copy(); bad.loc[0, "lag"] = 11
    with pytest.raises(AnalysisOutputError, match="1..10"):
        table_t08(bad)
    bad = edges.copy(); bad.loc[0, "outer_fold_selection_frequency"] = 0.5
    with pytest.raises(AnalysisOutputError, match="does not reconcile"):
        table_t08(bad)

    zero = edges.iloc[[0]].copy()
    zero["source_region"] = "zero_source"
    zero["target_region"] = "zero_target"
    zero["lag"] = 5
    zero["fold_selected_count"] = 0
    zero["outer_fold_selection_frequency"] = 0.0
    zero["bootstrap_selected_count"] = 0
    zero["bootstrap_selection_frequency"] = 0.0
    expanded = table_t08(pd.concat([edges, zero], ignore_index=True))
    source = edge_lag_source(expanded)
    observed_zero = source[(source.relation == "zero_source→zero_target") & (source.lag == 5)].iloc[0]
    missing = source[(source.relation == "zero_source→zero_target") & (source.lag == 6)].iloc[0]
    assert bool(observed_zero.observed) is True
    assert observed_zero.fold_selection_frequency == pytest.approx(0.0)
    assert bool(missing.observed) is False
    assert pd.isna(missing.fold_selection_frequency)


def test_primary_outputs_can_be_generated_without_sensitivity_execution(tmp_path: Path, monkeypatch) -> None:
    import lagged_facial_graph_forecasting.analysis_pipeline as pipeline
    original_read, original_hash = pipeline._read_csv, pipeline._file_hash
    def read(path):
        assert "sensitivity" not in path.name
        return original_read(path)
    def digest(path):
        assert "sensitivity" not in path.name
        return original_hash(path)
    monkeypatch.setattr(pipeline, "_read_csv", read)
    monkeypatch.setattr(pipeline, "_file_hash", digest)
    result = generate_analysis_outputs(_inputs(), repository_root=tmp_path, include_sensitivity=False)
    assert (result.output_root / "tables/T03_primary_condition_performance.csv").is_file()
    assert not (result.output_root / "tables/T09_sensitivity_summary.csv").exists()
    assert not (result.output_root / "figures/F10_sensitivity_forest_plot.png").exists()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["sensitivity_included"] is False
    assert manifest["sensitivity_validation_scope"] == "not_generated"


def test_sensitivity_requires_primary_freeze_outside_explicit_mock_verification(tmp_path: Path) -> None:
    with pytest.raises(AnalysisOutputError, match="before Primary freeze"):
        generate_analysis_outputs(_inputs(), repository_root=tmp_path, include_sensitivity=True)


def test_mock_generator_emits_issue23_v2_contract_columns(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, "scripts/generate_mock_analysis_data.py", "--output-dir", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    nulls = pd.read_csv(tmp_path / "mock_null_metrics.csv")
    assert {
        "mapping_scope", "pcmci_feature_count", "null_feature_count",
        "input_count_preserved", "lag_identity_preserved",
    }.issubset(nulls.columns)
    lag = pd.read_csv(tmp_path / "mock_lag_response.csv")
    assert {"same_region_identity", "same_feature_count"}.issubset(lag.columns)
    config = json.loads((tmp_path / "mock_primary_config.json").read_text(encoding="utf-8"))
    assert config["evaluation"]["primary_metric"] == "velocity_rmse"


def test_primary_only_never_resolves_reads_or_hashes_sensitivity(tmp_path, monkeypatch):
    import shutil
    import lagged_facial_graph_forecasting.analysis_pipeline as pipeline
    source = tmp_path / "input"
    shutil.copytree(FIXTURE_ROOT, source)
    (source / "mock_sensitivity.csv").unlink()
    inputs = AnalysisInputs.from_directory(source)
    assert inputs.sensitivity is None
    original_read, original_hash = pipeline._read_csv, pipeline._file_hash
    def read(path):
        assert "sensitivity" not in path.name
        return original_read(path)
    def digest(path):
        assert "sensitivity" not in path.name
        return original_hash(path)
    monkeypatch.setattr(pipeline, "_read_csv", read)
    monkeypatch.setattr(pipeline, "_file_hash", digest)
    result = generate_analysis_outputs(inputs, repository_root=tmp_path)
    registry = pd.read_csv(result.registry_path)
    assert set(registry.output_id) - {"CAPTIONS"} == ({f"T{i:02}" for i in range(1, 9)} | {f"F{i:02}" for i in range(1, 15) if i != 10})
    manifest = json.loads(result.manifest_path.read_text())
    assert not any("sensitivity" in row["name"] for row in manifest["input_artifacts"])
    with pytest.raises(AnalysisOutputError, match="sensitivity.csv"):
        AnalysisInputs.from_directory(source, include_sensitivity=True)
    # Even an existing malformed file must be ignored in Primary-only mode.
    (source / "mock_sensitivity.csv").write_text("invalid")
    assert AnalysisInputs.from_directory(source).sensitivity is None


@pytest.mark.parametrize("failure", [None, "missing", "incomplete", "tampered", "reference", "boolean"])
def test_real_analysis_validates_freeze_before_writing(tmp_path, failure):
    import shutil
    from test_sensitivity_freeze_barrier import _config as barrier_config, _write_complete_freeze
    from lagged_facial_graph_forecasting.sensitivity_execution import (
        PRIMARY_FREEZE_MANIFEST_PATH, SensitivityExecutionMode, SensitivityExecutionError,
    )
    source = tmp_path / "input"
    shutil.copytree(FIXTURE_ROOT, source)
    for path in source.glob("*.csv"):
        frame = pd.read_csv(path)
        # Isolated contract fixture, never evidence of real-data validation.
        if "subject_id" in frame:
            frame["subject_id"] = frame["subject_id"].str.replace("MOCK_S", "FIXTURE_S", regex=False)
        frame["is_synthetic"] = False
        if "synthetic_notice" in frame:
            frame["synthetic_notice"] = ""
        frame.to_csv(path, index=False)
    config_path = source / "mock_primary_config.json"
    config = json.loads(config_path.read_text())
    config["is_synthetic"] = False
    config_path.write_text(json.dumps(config))
    manifest = _write_complete_freeze(tmp_path)
    manifest_path = tmp_path / PRIMARY_FREEZE_MANIFEST_PATH
    sensitivity_path = source / "mock_sensitivity.csv"
    frame = pd.read_csv(sensitivity_path)
    frame["source_primary_freeze_manifest"] = PRIMARY_FREEZE_MANIFEST_PATH
    frame["source_primary_freeze_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if failure == "reference":
        frame["source_primary_freeze_sha256"] = "0" * 64
    frame.to_csv(sensitivity_path, index=False)
    if failure == "missing":
        manifest_path.unlink()
    elif failure == "incomplete":
        manifest["artifacts"].pop()
        manifest_path.write_text(json.dumps(manifest))
    elif failure == "tampered":
        (tmp_path / manifest["artifacts"][0]["relative_path"]).write_text("changed")
    kwargs = dict(repository_root=tmp_path, include_sensitivity=True, primary_frozen=True,
                  sensitivity_config=None if failure == "boolean" else barrier_config(SensitivityExecutionMode.REAL))
    inputs = AnalysisInputs.from_directory(source, include_sensitivity=True)
    if failure:
        with pytest.raises((AnalysisOutputError, SensitivityExecutionError)):
            generate_analysis_outputs(inputs, **kwargs)
        assert not (tmp_path / "artifacts/sensitivity/analysis").exists()
    else:
        result = generate_analysis_outputs(inputs, **kwargs)
        assert result.output_root == (tmp_path / "artifacts/sensitivity/analysis").resolve()
        assert not (tmp_path / "artifacts/analysis").exists()
        saved = json.loads(result.manifest_path.read_text())
        assert saved["primary_frozen"] is True
        assert saved["primary_freeze_reference"]["source_primary_freeze_sha256"] == frame.source_primary_freeze_sha256.iloc[0]
