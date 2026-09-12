"""Candidate-grid loading must preserve the frozen candidate identity."""

import json

import pytest

from lagged_facial_graph_forecasting.analysis_pipeline import (
    AnalysisOutputError,
    _load_candidate_grid,
)
from lagged_facial_graph_forecasting.landscape import CandidateGrid, LandscapeCandidate


def _payload():
    return {
        "protocol_sha256": "a" * 64,
        "candidates": [LandscapeCandidate(
            "eye", "mouth", 1, "vx", "vy", "region_dimension_lag"
        ).to_payload()],
    }


@pytest.mark.parametrize("lag", [1.9, 0])
def test_grid_loader_rejects_invalid_lags(tmp_path, lag):
    payload = _payload()
    payload["candidates"][0]["lag"] = lag
    path = tmp_path / "candidate_grid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AnalysisOutputError, match="invalid candidate_grid"):
        _load_candidate_grid(path)


@pytest.mark.parametrize("payload", [None, {"candidates": [None]}])
def test_grid_loader_reports_malformed_structure(tmp_path, payload):
    path = tmp_path / "candidate_grid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AnalysisOutputError, match="invalid candidate_grid"):
        _load_candidate_grid(path)


@pytest.mark.parametrize("wrapped", [False, True])
def test_grid_loader_preserves_candidate_and_digest(tmp_path, wrapped):
    payload = _payload()
    expected = CandidateGrid.from_candidates(
        [LandscapeCandidate(**payload["candidates"][0])],
        protocol_sha256=payload["protocol_sha256"],
    )
    path = tmp_path / "candidate_grid.json"
    path.write_text(json.dumps({"payload": payload} if wrapped else payload), encoding="utf-8")
    actual = _load_candidate_grid(path)
    assert actual == expected
    assert actual.digest == expected.digest
