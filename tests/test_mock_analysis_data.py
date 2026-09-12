from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOCK_DIR = ROOT / "data" / "mock_analysis"
NOTICE = "MOCK DATA / NOT A SCIENTIFIC RESULT"


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_committed_mock_assets_are_unambiguously_non_scientific() -> None:
    for filename in (
        "mock_metrics.csv",
        "mock_null_metrics.csv",
        "mock_feature_counts.csv",
        "mock_lag_response.csv",
        "mock_edge_stability.csv",
        "mock_sensitivity.csv",
        "mock_dataset_summary.csv",
        "mock_prediction_trajectory.csv",
    ):
        rows = _csv_rows(MOCK_DIR / filename)
        assert rows
        assert {row["is_synthetic"] for row in rows} == {"True"}
        assert {row["synthetic_notice"] for row in rows} == {NOTICE}

    config = json.loads(
        (MOCK_DIR / "mock_primary_config.json").read_text(encoding="utf-8")
    )
    assert config["is_synthetic"] is True
    assert config["synthetic_notice"] == NOTICE
    assert config["purpose"] == "software verification only"

    manifest = json.loads((MOCK_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["is_synthetic"] is True
    assert manifest["real_data"] is False
    assert manifest["scientific_use_allowed"] is False


def test_generator_emits_required_mock_analysis_inputs(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_mock_analysis_data.py"),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
    )

    required = (
        "mock_metrics.csv",
        "mock_null_metrics.csv",
        "mock_feature_counts.csv",
        "mock_lag_response.csv",
        "mock_edge_stability.csv",
        "mock_sensitivity.csv",
        "mock_dataset_summary.csv",
        "mock_prediction_trajectory.csv",
        "mock_population.csv",
        "mock_landscape.csv",
        "mock_candidate_grid.json",
        "mock_primary_config.json",
    )
    assert all((tmp_path / filename).is_file() for filename in required)

    rows = _csv_rows(tmp_path / "mock_metrics.csv")
    assert rows
    assert {row["is_synthetic"] for row in rows} == {"True"}
    assert {row["synthetic_notice"] for row in rows} == {NOTICE}
