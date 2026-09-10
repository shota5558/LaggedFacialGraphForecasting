from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from lagged_facial_graph_forecasting.analysis_pipeline import (
    AnalysisInputs,
    AnalysisOutputError,
    enrichment_sources,
    generate_analysis_outputs,
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


def _landscape_frame(grid: CandidateGrid) -> pd.DataFrame:
    rows = []
    for subject_index, subject_id in enumerate(("MOCK_S01", "MOCK_S02", "MOCK_S03", "MOCK_S04")):
        for candidate_index, candidate in enumerate(grid.candidates):
            self_error = 1.0 + subject_index * 0.1
            cell_error = self_error - 0.1 * (candidate_index + 1)
            rows.append({
                "subject_id": subject_id,
                **candidate.to_payload(),
                "self_error": self_error,
                "cell_error": cell_error,
                "gain": self_error - cell_error,
                "support_sha256": HASH_B,
                "candidate_grid_sha256": grid.digest,
                "status": "evaluable",
            })
    return _provenance(pd.DataFrame(rows))


def _enrichment_frame() -> pd.DataFrame:
    rows = []
    for index, subject_id in enumerate(("MOCK_S01", "MOCK_S02", "MOCK_S03", "MOCK_S04")):
        selected = 0.5 + index * 0.1
        matched = 0.1 + index * 0.05
        rows.append({
            "subject_id": subject_id,
            "target_region": "mouth",
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
    rows = []
    for index, subject_id in enumerate(("MOCK_S01", "MOCK_S02", "MOCK_S03", "MOCK_S04")):
        reference = 1.0 + index * 0.1
        for delta in (-1, 0, 1):
            shifted = reference + 0.2 * abs(delta)
            rows.append({
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
            })
    return _provenance(pd.DataFrame(rows))


@pytest.mark.parametrize("column", ["outer_fold", "tau_star", "delta", "shifted_lag"])
@pytest.mark.parametrize("value", [0.5, True, float("inf"), float("nan")])
def test_population_source_rejects_non_integer_identity(column, value) -> None:
    frame = _population_frame()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(AnalysisOutputError, match=f"population.{column}.*finite integers"):
        population_sources(frame, _config())


def test_population_source_rejects_reference_drift() -> None:
    frame = _population_frame()
    frame.loc[0, "reference_error"] += 0.5
    with pytest.raises(AnalysisOutputError, match="complete-unit"):
        population_sources(frame, _config())


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
    assert (result.output_root / "tables/ENRICHMENT_subject_summary.csv").is_file()
    assert (result.output_root / "tables/POPULATION_response_summary.csv").is_file()
    assert (result.output_root / "figures/LANDSCAPE_gain_heatmap.svg").is_file()
    assert (result.output_root / "figures/ENRICHMENT_selected_vs_matched.png").is_file()
    assert (result.output_root / "figures/POPULATION_edge_centered_response.png").is_file()

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["extended_outputs_included"] is True
    assert manifest["extended_output_ids"] == ["LANDSCAPE", "ENRICHMENT", "POPULATION"]
    captions = json.loads((result.output_root / "captions.json").read_text())
    assert {"LANDSCAPE", "ENRICHMENT", "POPULATION"}.issubset(captions)
    registry = pd.read_csv(result.registry_path)
    assert {"LANDSCAPE", "ENRICHMENT", "POPULATION"}.issubset(set(registry.output_id))
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
