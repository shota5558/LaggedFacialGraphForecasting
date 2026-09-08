from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MOCK_DIR = ROOT / "data" / "mock_analysis"
NOTICE = "FAKE DATA - NOT FOR SCIENTIFIC CONCLUSIONS"


def _csv_rows(name: str) -> list[dict[str, str]]:
    with (MOCK_DIR / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("filename", ["mock_metrics.csv", "mock_null_metrics.csv", "mock_feature_counts.csv", "mock_lag_response.csv", "mock_edge_stability.csv", "mock_sensitivity.csv", "mock_dataset_summary.csv", "mock_prediction_trajectory.csv"])
def test_committed_mock_csvs_are_explicitly_synthetic(filename: str) -> None:
    rows = _csv_rows(filename)
    assert rows
    assert {row["is_synthetic"] for row in rows} == {"True"}
    assert {row["synthetic_notice"] for row in rows} == {NOTICE}


def test_mock_subject_ids_use_reserved_prefix() -> None:
    rows = _csv_rows("mock_metrics.csv")
    assert all(row["subject_id"].startswith("MOCK_S") for row in rows)


def test_mock_primary_conditions_cover_analysis_backbone() -> None:
    assert {"persistence", "self", "full", "pcmci"} <= {row["condition"] for row in _csv_rows("mock_metrics.csv")}


def test_mock_lag_response_has_complete_symmetric_grid_per_unit() -> None:
    grouped: dict[tuple[str, str], set[int]] = {}
    for row in _csv_rows("mock_lag_response.csv"):
        key = (row["subject_id"], row["region_id"])
        grouped.setdefault(key, set()).add(int(row["delta_frames"]))
        assert row["evaluable"] == "True"
        assert row["same_support"] == "True"
    assert grouped
    assert all(grid == {-2, -1, 0, 1, 2} for grid in grouped.values())


def test_mock_config_is_explicitly_non_scientific() -> None:
    payload = json.loads((MOCK_DIR / "mock_primary_config.json").read_text(encoding="utf-8"))
    assert payload["is_synthetic"] is True
    assert payload["synthetic_notice"] == NOTICE
    assert payload["purpose"] == "software verification only"
    assert payload["primary"]["tau_max"] == 10
    assert payload["primary"]["horizon"] == 1


def test_mock_manifest_forbids_scientific_use() -> None:
    payload = json.loads((MOCK_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert payload["is_synthetic"] is True
    assert payload["real_data"] is False
    assert payload["scientific_use_allowed"] is False


def test_generator_produces_richer_synthetic_fixture_set(tmp_path: Path) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_mock_analysis_data.py"), "--output-dir", str(tmp_path)], cwd=ROOT, check=True)
    for filename in ("mock_metrics.csv", "mock_null_metrics.csv", "mock_feature_counts.csv", "mock_lag_response.csv", "mock_edge_stability.csv", "mock_sensitivity.csv", "mock_dataset_summary.csv", "mock_prediction_trajectory.csv", "mock_primary_config.json"):
        assert (tmp_path / filename).exists()
    with (tmp_path / "mock_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(row["is_synthetic"] == "True" for row in rows)
    assert all(row["synthetic_notice"] == NOTICE for row in rows)
