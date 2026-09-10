from __future__ import annotations

import json
import hashlib
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from lagged_facial_graph_forecasting.analysis_pipeline import (
    AnalysisInputs,
    AnalysisOutputError,
    enrichment_sources,
    generate_analysis_outputs,
    landscape_aggregate_sources,
    landscape_source,
    population_sources,
    table_t04,
)
from lagged_facial_graph_forecasting.landscape import CandidateGrid, LandscapeCandidate


FIXTURE_ROOT = Path("data/mock_analysis")
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
SUPPORT_DIGEST = "d" * 64


def _config() -> dict[str, object]:
    config = json.loads((FIXTURE_ROOT / "mock_primary_config.json").read_text())
    config["landscape"] = {
        "lag_bands": {"early": [1, 1], "late": [2, 2]},
        "aggregation_id": "cell_mean_then_subject_median",
        "ci_method": "subject_percentile",
    }
    config["enrichment"] = {
        "repeat_summary_aggregation": "mean_over_matched_repeats",
        "ci_method": "subject_percentile",
    }
    config["population"] = {
        "delta_frames": [-1, 0, 1],
        "aggregation_id": "median_edge_subject",
        "ci_method": "subject_cluster_percentile",
    }
    return config


def _grid() -> CandidateGrid:
    candidates = [
        LandscapeCandidate("left_eye", "mouth", lag, "vx", "vy", "region_dimension_lag")
        for lag in (1, 2)
    ] + [
        LandscapeCandidate("right_eye", "mouth", lag, "vx", "vy", "region_dimension_lag")
        for lag in (1, 2)
    ]
    return CandidateGrid.from_candidates(candidates, protocol_sha256=HASH_A)


def _provenance(frame: pd.DataFrame) -> pd.DataFrame:
    frame.insert(0, "is_synthetic", True)
    frame.insert(1, "synthetic_notice", "MOCK DATA / NOT A SCIENTIFIC RESULT")
    return frame


def _with_config_hash(frame: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    frame = frame.copy()
    frame["config_sha256"] = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return frame


def _landscape_frame(grid: CandidateGrid) -> pd.DataFrame:
    config_sha256 = hashlib.sha256(
        json.dumps(_config(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    rows = []
    for subject_index, subject_id in enumerate(("MOCK_S01", "MOCK_S02", "MOCK_S03", "MOCK_S04")):
        for candidate_index, candidate in enumerate(grid.candidates):
            self_error = 1.0 + subject_index * 0.1
            cell_error = self_error - 0.1 * (candidate_index + 1)
            rows.append({
                "run_id": "MOCK_RUN_R16_LANDSCAPE",
                "git_sha": "a" * 40,
                "protocol_sha256": HASH_A,
                "config_sha256": config_sha256,
                "source_sha256": HASH_C,
                "outer_fold": subject_index,
                "horizon": 1,
                "seed": 20260908,
                "subject_id": subject_id,
                **candidate.to_payload(),
                "metric_name": "velocity_rmse",
                "metric_direction": "lower_is_better",
                "estimand_id": "landscape_cell_gain_v1",
                "self_error": self_error,
                "cell_error": cell_error,
                "gain": self_error - cell_error,
                "support_sha256": HASH_B,
                "candidate_grid_sha256": grid.digest,
                "status": "evaluable",
            })
    return _provenance(pd.DataFrame(rows))


def _enrichment_frame() -> pd.DataFrame:
    config_sha256 = hashlib.sha256(
        json.dumps(_config(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    rows = []
    for index, subject_id in enumerate(("MOCK_S01", "MOCK_S02", "MOCK_S03", "MOCK_S04")):
        selected = 0.5 + index * 0.1
        matched = 0.1 + index * 0.05
        rows.append({
            "run_id": "MOCK_RUN_R16_ENRICHMENT",
            "git_sha": "a" * 40,
            "protocol_sha256": HASH_A,
            "config_sha256": config_sha256,
            "source_sha256": HASH_C,
            "outer_fold": 0,
            "subject_id": subject_id,
            "target_region": "mouth",
            "horizon": 1,
            "metric_name": "velocity_rmse",
            "metric_direction": "lower_is_better",
            "repeat_id": "MOCK_R000",
            "seed": 20260908 + index,
            "selected_aggregate": selected,
            "matched_aggregate": matched,
            "difference_selected_minus_matched": selected - matched,
            "status": "evaluable",
            "estimand_id": "cell_gain_enrichment_v1",
            "aggregation_id": "arithmetic_mean",
            "candidate_grid_sha256": HASH_A,
            "support_sha256": HASH_B,
            "membership_sha256": HASH_C,
        })
    return _provenance(pd.DataFrame(rows))


def _population_frame() -> pd.DataFrame:
    config_sha256 = hashlib.sha256(
        json.dumps(_config(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    rows = []
    for index, subject_id in enumerate(("MOCK_S01", "MOCK_S02", "MOCK_S03", "MOCK_S04")):
        reference = 1.0 + index * 0.1
        for delta in (-1, 0, 1):
            shifted = reference + 0.2 * abs(delta)
            rows.append({
            "run_id": "MOCK_RUN_20260911",
            "git_sha": "a" * 40,
            "protocol_sha256": HASH_A,
            "config_sha256": config_sha256,
            "source_sha256": HASH_C,
            "metric_name": "velocity_rmse",
            "metric_direction": "lower_is_better",
            "estimand_id": "edge_centered_self_plus_edge_v1",
            "horizon": 1,
            "subject_id": subject_id,
            "outer_fold": 0,
            "edge_id": "edge_left_eye_mouth",
                "source_region": "left_eye",
                "target_region": "mouth",
                "source_dimension": "vx",
            "target_dimension": "vy",
            "context_id": "self_plus_edge_v1",
            "tau_star": 2,
            "delta": delta,
            "shifted_lag": 2 + delta,
            "sampling_rate_hz": 30.0,
                "reference_error": reference,
                "shifted_error": shifted,
                "difference": shifted - reference,
            "reference_support_sha256": HASH_B,
            "shifted_support_sha256": HASH_B,
            "status": "evaluable",
            })
    return _provenance(pd.DataFrame(rows))


@pytest.mark.parametrize("column", ["status", "estimand_id", "aggregation_id", "candidate_grid_sha256", "support_sha256"])
def test_enrichment_rejects_repeat_contract_drift(column) -> None:
    first = _enrichment_frame()
    second = first.assign(repeat_id="MOCK_R001")
    second.loc[0, column] = (
        "unevaluable_empty_selected" if column == "status"
        else HASH_C if column.endswith("sha256") else "different"
    )
    config = _config()
    config["primary"]["matched_sparsity"]["repeat_count"] = 2
    first = _with_config_hash(first, config)
    second = _with_config_hash(second, config)
    with pytest.raises(AnalysisOutputError, match=f"{column} changes across repeats"):
        enrichment_sources(pd.concat([first, second], ignore_index=True), config)


@pytest.mark.parametrize("column", ["subject_id", "target_region", "repeat_id", "estimand_id", "aggregation_id"])
@pytest.mark.parametrize("value", [None, " "])
def test_enrichment_rejects_missing_identifiers(column, value) -> None:
    frame = _enrichment_frame()
    frame.loc[0, column] = value
    with pytest.raises(AnalysisOutputError, match="non-empty identifiers"):
        enrichment_sources(frame, _config())


@pytest.mark.parametrize("column", ["estimand_id", "aggregation_id"])
def test_enrichment_rejects_mixed_subject_estimands(column) -> None:
    frame = _enrichment_frame()
    frame.loc[0, column] = "different"
    with pytest.raises(AnalysisOutputError, match="differs across subject units"):
        enrichment_sources(frame, _config())


def test_enrichment_preserves_unevaluable_units_without_numeric_effects() -> None:
    frame = _enrichment_frame()
    frame.loc[0, "status"] = "unevaluable_empty_selected"
    with pytest.raises(AnalysisOutputError, match="must be missing"):
        enrichment_sources(frame, _config())
    columns = ["selected_aggregate", "matched_aggregate", "difference_selected_minus_matched"]
    frame.loc[0, columns] = np.nan
    distribution, summary = enrichment_sources(frame, _config())
    assert len(distribution) == 4
    assert summary.iloc[0].n_subjects == 3


def test_enrichment_accepts_same_digest_case_and_subject_specific_support() -> None:
    first = _enrichment_frame()
    first.loc[0, "support_sha256"] = HASH_C
    second = first.assign(repeat_id="MOCK_R001", seed=123)
    for column in ("candidate_grid_sha256", "support_sha256", "membership_sha256"):
        second[column] = second[column].str.upper()
    config = _config()
    config["primary"]["matched_sparsity"]["repeat_count"] = 2
    first = _with_config_hash(first, config)
    second = _with_config_hash(second, config)
    distribution, summary = enrichment_sources(pd.concat([first, second]), config)
    assert len(distribution) == 8
    assert summary.iloc[0].n_subjects == 4
    assert summary.iloc[0].n_repeats_per_subject == 2


def test_enrichment_rejects_missing_repeat_record() -> None:
    first = _enrichment_frame()
    second = first.assign(repeat_id="MOCK_R001", seed=123)
    config = _config()
    config["primary"]["matched_sparsity"]["repeat_count"] = 2
    first = _with_config_hash(first, config)
    second = _with_config_hash(second, config)
    combined = pd.concat([first, second], ignore_index=True).drop(index=4)
    with pytest.raises(AnalysisOutputError, match="frozen repeat count"):
        enrichment_sources(combined, config)


@pytest.mark.parametrize("column", ["run_id", "git_sha", "protocol_sha256", "source_sha256"])
def test_enrichment_rejects_run_provenance_drift(column) -> None:
    frame = _enrichment_frame()
    frame.loc[0, column] = "different-run" if column in {"run_id"} else "f" * (40 if column == "git_sha" else 64)
    with pytest.raises(AnalysisOutputError, match=f"enrichment {column} changes across run"):
        enrichment_sources(frame, _config())


def test_enrichment_rejects_resolved_config_drift() -> None:
    frame = _enrichment_frame()
    frame["config_sha256"] = HASH_C
    with pytest.raises(AnalysisOutputError, match="does not match resolved config"):
        enrichment_sources(frame, _config())


def test_enrichment_rejects_undefined_metric() -> None:
    frame = _enrichment_frame()
    frame["metric_name"] = "position_rmse"
    with pytest.raises(AnalysisOutputError, match="undefined or not the frozen primary metric"):
        enrichment_sources(frame, _config())


@pytest.mark.parametrize("column", ["outer_fold", "tau_star", "delta", "shifted_lag"])
@pytest.mark.parametrize("value", [0.5, True, float("inf"), float("nan")])
def test_population_source_rejects_non_integer_identity(column, value) -> None:
    frame = _population_frame()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(AnalysisOutputError, match=f"population.{column}.*finite integers"):
        population_sources(frame, _config())


@pytest.mark.parametrize("column", [
    "run_id", "git_sha", "protocol_sha256", "config_sha256", "source_sha256",
    "metric_name", "metric_direction", "estimand_id", "horizon", "status",
])
def test_population_source_requires_provenance_record_fields(column) -> None:
    frame = _population_frame().drop(columns=column)
    with pytest.raises(AnalysisOutputError, match=f"population missing required columns.*{column}"):
        population_sources(frame, _config())


@pytest.mark.parametrize("column", ["run_id", "protocol_sha256", "config_sha256", "source_sha256"])
def test_population_source_rejects_mixed_run_provenance(column) -> None:
    frame = _population_frame()
    frame.loc[1, column] = "e" * (40 if column == "run_id" else 64)
    if column == "config_sha256":
        frame.loc[1, column] = "c" * 64
    with pytest.raises(AnalysisOutputError, match=f"population {column} changes across records"):
        population_sources(frame, _config())


def test_population_source_rejects_resolved_config_drift() -> None:
    frame = _population_frame()
    frame["config_sha256"] = HASH_B
    with pytest.raises(AnalysisOutputError, match="config_sha256 does not match resolved config"):
        population_sources(frame, _config())


@pytest.mark.parametrize("column,value", [
    ("metric_name", "position_rmse"),
    ("metric_direction", "higher_is_better"),
])
def test_population_source_rejects_undefined_metric(column, value) -> None:
    frame = _population_frame()
    frame.loc[0, column] = value
    with pytest.raises(AnalysisOutputError, match="configured lower_is_better primary metric"):
        population_sources(frame, _config())


def test_population_source_reconciles_difference_and_preserves_failure_counts() -> None:
    frame = _population_frame()
    frame.loc[0, "difference"] = 99.0
    with pytest.raises(AnalysisOutputError, match="difference does not reconcile"):
        population_sources(frame, _config())

    frame = _population_frame()
    frame.loc[0, "status"] = "failed"
    frame.loc[1, "status"] = "failed"
    _, summary = population_sources(frame, _config())
    assert (summary.failure_count == 2).all()
    assert (summary.n_failed_edge_subject_units == 1).all()
    assert (summary.n_unevaluable_edge_subject_units == 1).all()


def test_population_source_rejects_reference_drift() -> None:
    frame = _population_frame()
    frame.loc[0, "reference_error"] += 0.5
    with pytest.raises(AnalysisOutputError, match="complete-unit"):
        population_sources(frame, _config())


@pytest.mark.parametrize("aggregation", ["median", "mean"])
def test_population_ci_resamples_whole_subjects_with_the_point_estimand(aggregation) -> None:
    template = _population_frame().iloc[0].to_dict()
    clusters = ([0.0, 0.0, 100.0], [10.0], [20.0, 40.0])
    rows = []
    for subject, values in enumerate(clusters):
        for edge, value in enumerate(values):
            for delta in (-1, 0, 1):
                rows.append({**template, "subject_id": f"s{subject}", "edge_id": f"e{edge}",
                             "delta": delta, "shifted_lag": 2 + delta,
                             "reference_error": 1.0,
                             "shifted_error": 1.0 + (value if delta else 0.0),
                             "difference": value if delta else 0.0})
    config = _config()
    config["population"]["aggregation_id"] = f"{aggregation}_edge_subject"
    frame = pd.DataFrame(rows)
    frame["config_sha256"] = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _, summary = population_sources(frame, config)
    point = summary.loc[summary.delta_frames == 1].iloc[0]
    statistic = getattr(np, aggregation)
    draws = np.random.default_rng(config["seed"]).integers(
        0, len(clusters), size=(config["statistics"]["bootstrap_n_resamples"], len(clusters))
    )
    bootstrap_values = [statistic([v for i in draw for v in clusters[i]]) for draw in draws]
    low, high = np.percentile(bootstrap_values, [2.5, 97.5])
    assert point.point_estimate == pytest.approx(statistic(np.concatenate(clusters)))
    assert (point.ci_low, point.ci_high) == pytest.approx((low, high))
    assert point.median_edge_subject_difference == pytest.approx(15.0)
    assert point.n_subjects == 3 and point.n_edge_subject_units == 6
    _, reordered = population_sources(frame.iloc[::-1], config)
    pd.testing.assert_frame_equal(summary, reordered)


@pytest.mark.parametrize("column", ["self_error", "support_sha256"])
def test_landscape_rejects_changed_self_reference_or_support(column) -> None:
    grid = _grid()
    frame = _landscape_frame(grid)
    if column == "self_error":
        frame.loc[0, column] += 0.5
        # Per-cell arithmetic still reconciles; the shared baseline is wrong.
        frame.loc[0, "gain"] = frame.loc[0, "self_error"] - frame.loc[0, "cell_error"]
    else:
        frame.loc[0, column] = HASH_C
    with pytest.raises(AnalysisOutputError, match="same Self reference|same evaluation support"):
        landscape_source(frame, grid)


def test_landscape_reference_is_scoped_to_subject_and_target_component() -> None:
    base = _grid()
    grid = CandidateGrid.from_candidates(
        [LandscapeCandidate(c.source_region, c.target_region, c.lag,
                            c.source_dimension, dimension, c.feature_unit)
         for c in base.candidates for dimension in ("vx", "vy")],
        protocol_sha256=HASH_A,
    )
    frame = _landscape_frame(grid)
    other_target = frame.target_dimension == "vx"
    frame.loc[other_target, "self_error"] += 2
    frame.loc[other_target, "cell_error"] += 2
    frame.loc[other_target, "support_sha256"] = HASH_C
    expected = landscape_source(frame, grid)
    # DataFrame index labels are not cell identities (e.g. after concatenation).
    frame.index = [0] * len(frame)
    actual = landscape_source(frame, grid)
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.gain.tolist() == pytest.approx([0.1 * i for i in range(1, 9)] * 4)


def test_landscape_unevaluable_cells_do_not_supply_a_self_reference() -> None:
    grid = _grid()
    frame = _landscape_frame(grid)
    frame.loc[0, ["self_error", "cell_error", "gain"]] = np.nan
    frame.loc[0, "status"] = "unevaluable"
    frame.loc[0, "support_sha256"] = HASH_C
    result = landscape_source(frame, grid)
    assert len(result) == len(frame)
    assert result.loc[result.status == "unevaluable", "gain"].isna().all()


@pytest.mark.parametrize("column", [
    "run_id", "git_sha", "protocol_sha256", "config_sha256", "source_sha256",
    "outer_fold", "horizon", "seed", "metric_name", "metric_direction", "estimand_id",
])
def test_landscape_source_requires_run_provenance(column) -> None:
    frame = _landscape_frame(_grid()).drop(columns=column)
    with pytest.raises(AnalysisOutputError, match=f"landscape missing required columns.*{column}"):
        landscape_source(frame, _grid(), _config())


@pytest.mark.parametrize("column", ["run_id", "git_sha", "protocol_sha256", "source_sha256"])
def test_landscape_source_rejects_provenance_drift(column) -> None:
    frame = _landscape_frame(_grid())
    frame.loc[0, column] = "different-run" if column == "run_id" else "f" * (40 if column == "git_sha" else 64)
    with pytest.raises(AnalysisOutputError, match=f"landscape {column} changes across records"):
        landscape_source(frame, _grid(), _config())


def test_landscape_source_rejects_resolved_config_or_metric_drift() -> None:
    frame = _landscape_frame(_grid())
    frame["config_sha256"] = HASH_C
    with pytest.raises(AnalysisOutputError, match="config_sha256 does not match resolved config"):
        landscape_source(frame, _grid(), _config())

    frame = _landscape_frame(_grid())
    frame["metric_name"] = "position_rmse"
    with pytest.raises(AnalysisOutputError, match="lower_is_better primary metric"):
        landscape_source(frame, _grid(), _config())


def test_landscape_aggregation_preserves_cell_and_subject_denominators() -> None:
    grid = _grid()
    frame = _landscape_frame(grid)
    frame.loc[frame.subject_id == "MOCK_S01", ["self_error", "cell_error", "gain"]] = np.nan
    frame.loc[frame.subject_id == "MOCK_S01", "status"] = "failed"
    source = landscape_source(frame, grid, _config())
    pair, _ = landscape_aggregate_sources(source, _config())
    row = pair.iloc[0]
    assert (row.n_total_cells, row.n_evaluable_cells, row.n_failed_cells) == (8, 6, 2)
    assert (row.n_total_subjects, row.n_evaluable_subjects, row.n_unevaluable_subjects) == (4, 3, 1)
    assert row.failure_count == 2


def test_extended_sources_validate_and_reconcile() -> None:
    grid = _grid()
    source = landscape_source(_landscape_frame(grid), grid)
    assert source.gain.tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4] * 4)

    distribution, summary = enrichment_sources(_enrichment_frame(), _config())
    assert len(distribution) == 4
    assert summary.iloc[0].median_enrichment_effect == pytest.approx(0.475)

    source, response = population_sources(_population_frame(), _config())
    assert source.population_evaluable.all()
    assert response.delta_frames.tolist() == [-1, 0, 1]
    assert response.loc[response.delta_frames == 0, "median_edge_subject_difference"].iloc[0] == pytest.approx(0.0)


def test_extended_outputs_are_generated_without_overwriting_old_contract(tmp_path: Path) -> None:
    source = tmp_path / "input"
    shutil.copytree(FIXTURE_ROOT, source)
    grid = _grid()
    _landscape_frame(grid).to_csv(source / "mock_landscape.csv", index=False)
    _enrichment_frame().to_csv(source / "mock_enrichment.csv", index=False)
    _population_frame().to_csv(source / "mock_population.csv", index=False)
    (source / "mock_primary_config.json").write_text(
        json.dumps(_config(), indent=2) + "\n"
    )
    (source / "mock_candidate_grid.json").write_text(json.dumps({
        "protocol_sha256": HASH_A,
        "candidates": [candidate.to_payload() for candidate in grid.candidates],
    }))

    inputs = AnalysisInputs.from_directory(source, include_extended=True)
    result = generate_analysis_outputs(
        inputs,
        repository_root=tmp_path,
        include_extended=True,
    )
    assert result.is_synthetic
    assert (result.output_root / "tables/LANDSCAPE_lag_band_summary.csv").is_file()
    assert (result.output_root / "tables/LANDSCAPE_cell_summary.csv").is_file()
    assert (result.output_root / "tables/ENRICHMENT_subject_summary.csv").is_file()
    assert (result.output_root / "tables/POPULATION_response_summary.csv").is_file()
    assert (result.output_root / "figures/LANDSCAPE_gain_heatmap.svg").is_file()
    assert (result.output_root / "figures/ENRICHMENT_selected_vs_matched.png").is_file()
    assert (result.output_root / "figures/POPULATION_edge_centered_response.png").is_file()

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["extended_outputs_included"] is True
    assert manifest["extended_output_ids"] == ["LANDSCAPE", "ENRICHMENT", "POPULATION"]
    assert manifest["landscape_provenance"]["run_id"] == "MOCK_RUN_R16_LANDSCAPE"
    assert manifest["landscape_provenance"]["failure_count"] == 0
    assert manifest["landscape_provenance"]["n_candidate_cells"] == 4
    assert manifest["enrichment_provenance"]["run_id"] == "MOCK_RUN_R16_ENRICHMENT"
    assert manifest["enrichment_provenance"]["n_unevaluable_records"] == 0
    assert manifest["population_provenance"]["run_id"] == "MOCK_RUN_20260911"
    assert manifest["population_provenance"]["config_sha256"] == hashlib.sha256(
        json.dumps(_config(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert manifest["population_provenance"]["source_sha256"] == HASH_C
    assert manifest["population_provenance"]["failure_count"] == 0
    captions = json.loads((result.output_root / "captions.json").read_text())
    assert {"LANDSCAPE", "ENRICHMENT", "POPULATION"}.issubset(captions)
    assert "legacy F04 common-shift" in captions["POPULATION"]
    registry = pd.read_csv(result.registry_path)
    assert {"LANDSCAPE", "ENRICHMENT", "POPULATION"}.issubset(set(registry.output_id))
    population_registry = registry[registry.output_id == "POPULATION"]
    assert set(population_registry.run_id) == {"MOCK_RUN_20260911"}
    assert set(population_registry.config_sha256) == {manifest["population_provenance"]["config_sha256"]}
    landscape_registry = registry[registry.output_id == "LANDSCAPE"]
    assert set(landscape_registry.run_id) == {"MOCK_RUN_R16_LANDSCAPE"}
    assert (result.output_root / "tables/T04_pcmci_vs_self_paired_effect.csv").is_file()


def test_extended_sources_reject_missing_grid_or_incomplete_population() -> None:
    grid = _grid()
    with pytest.raises(AnalysisOutputError, match="coverage mismatch"):
        landscape_source(_landscape_frame(grid).iloc[:-1], grid)

    incomplete = _population_frame().drop(index=2)
    source, summary = population_sources(incomplete, _config())
    assert not source.loc[source.subject_id == "MOCK_S01", "population_evaluable"].any()
    assert summary.delta_frames.tolist() == [-1, 0, 1]


def test_audit_pairing_fix_rejects_n_valid_mismatch() -> None:
    metrics = pd.read_csv(FIXTURE_ROOT / "mock_metrics.csv")
    pcmci = (metrics.condition == "pcmci") & (metrics.subject_id == "MOCK_S01")
    metrics.loc[pcmci, "n_valid"] = 119
    with pytest.raises(AnalysisOutputError, match="missing paired evaluation unit"):
        table_t04(metrics, json.loads((FIXTURE_ROOT / "mock_primary_config.json").read_text()))


def test_audit_pairing_fix_rejects_support_digest_mismatch() -> None:
    metrics = pd.read_csv(FIXTURE_ROOT / "mock_metrics.csv")
    metrics["evaluation_support_sha256"] = SUPPORT_DIGEST
    pcmci = (metrics.condition == "pcmci") & (metrics.subject_id == "MOCK_S01")
    metrics.loc[pcmci, "evaluation_support_sha256"] = "e" * 64
    with pytest.raises(AnalysisOutputError, match="missing paired evaluation unit"):
        table_t04(metrics, json.loads((FIXTURE_ROOT / "mock_primary_config.json").read_text()))
