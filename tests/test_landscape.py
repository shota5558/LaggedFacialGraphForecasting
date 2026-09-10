from __future__ import annotations

import pytest

from lagged_facial_graph_forecasting.landscape import (
    CandidateGrid,
    LandscapeCandidate,
    LandscapeContractError,
    compute_cell_gain,
    validate_candidate_grid_coverage,
)


PROTOCOL_SHA256 = "a" * 64
SUPPORT_SHA256 = "b" * 64


def _candidate(lag: int) -> LandscapeCandidate:
    return LandscapeCandidate(
        source_region="left_eye",
        target_region="mouth",
        lag=lag,
        source_dimension="v_x",
        target_dimension="v_y",
        feature_unit="region_dimension_lag",
    )


def test_candidate_grid_is_order_independent_and_digestable() -> None:
    first = CandidateGrid.from_candidates(
        [_candidate(2), _candidate(1)], protocol_sha256=PROTOCOL_SHA256
    )
    second = CandidateGrid.from_candidates(
        [_candidate(1), _candidate(2)], protocol_sha256=PROTOCOL_SHA256
    )

    assert first.candidates == (_candidate(1), _candidate(2))
    assert first.digest == second.digest
    assert len(first.digest) == 64


def test_candidate_grid_rejects_duplicates() -> None:
    with pytest.raises(LandscapeContractError, match="duplicate"):
        CandidateGrid.from_candidates(
            [_candidate(1), _candidate(1)], protocol_sha256=PROTOCOL_SHA256
        )


def test_candidate_grid_coverage_rejects_missing_and_extra_cells() -> None:
    expected = CandidateGrid.from_candidates(
        [_candidate(1), _candidate(2)], protocol_sha256=PROTOCOL_SHA256
    )

    with pytest.raises(LandscapeContractError, match="missing=1"):
        validate_candidate_grid_coverage(expected, [_candidate(1)])

    with pytest.raises(LandscapeContractError, match="extra=1"):
        validate_candidate_grid_coverage(
            expected, [_candidate(1), _candidate(2), _candidate(3)]
        )


def test_cell_gain_requires_identical_support() -> None:
    assert compute_cell_gain(
        3.5,
        2.0,
        self_support_sha256=SUPPORT_SHA256,
        cell_support_sha256=SUPPORT_SHA256,
    ) == pytest.approx(1.5)

    with pytest.raises(LandscapeContractError, match="support digests differ"):
        compute_cell_gain(
            3.5,
            2.0,
            self_support_sha256=SUPPORT_SHA256,
            cell_support_sha256="c" * 64,
        )


def test_cell_gain_rejects_non_finite_errors() -> None:
    with pytest.raises(LandscapeContractError, match="finite"):
        compute_cell_gain(
            float("nan"),
            2.0,
            self_support_sha256=SUPPORT_SHA256,
            cell_support_sha256=SUPPORT_SHA256,
        )

