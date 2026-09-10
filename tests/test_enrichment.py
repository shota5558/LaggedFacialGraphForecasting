from __future__ import annotations

from dataclasses import replace
from statistics import mean

import pytest

from lagged_facial_graph_forecasting.enrichment import (
    CellGainObservation,
    EnrichmentContractError,
    MatchedGainRepeat,
    reduce_cell_gain_enrichment,
)
from lagged_facial_graph_forecasting.landscape import CandidateGrid, LandscapeCandidate


PROTOCOL_SHA256 = "a" * 64
SUPPORT_SHA256 = "b" * 64


def _candidate(lag: int) -> LandscapeCandidate:
    return LandscapeCandidate(
        source_region="left_eye",
        target_region="mouth",
        lag=lag,
        source_dimension="vx",
        target_dimension="vy",
        feature_unit="region_dimension_lag",
    )


def _grid() -> CandidateGrid:
    return CandidateGrid.from_candidates(
        [_candidate(1), _candidate(2), _candidate(3)],
        protocol_sha256=PROTOCOL_SHA256,
    )


def _observation(lag: int, gain: float) -> CellGainObservation:
    return CellGainObservation(
        subject_id="s01",
        candidate=_candidate(lag),
        gain=gain,
        support_sha256=SUPPORT_SHA256,
    )


def test_enrichment_preserves_selected_and_all_matched_repeat_values() -> None:
    result = reduce_cell_gain_enrichment(
        subject_id="s01",
        target_region="mouth",
        estimand_id="cell_gain_enrichment_v1",
        aggregation_id="arithmetic_mean",
        candidate_grid=_grid(),
        support_sha256=SUPPORT_SHA256,
        selected=(_observation(1, 0.5), _observation(2, 0.2)),
        matched_repeats=(
            MatchedGainRepeat("r000", 10, (_observation(1, 0.5), _observation(3, 0.1))),
            MatchedGainRepeat("r001", 11, (_observation(2, 0.2), _observation(3, 0.1))),
        ),
        aggregate=mean,
        expected_repeat_count=2,
    )

    assert result.status == "evaluable"
    assert result.selected_aggregate == pytest.approx(0.35)
    assert result.matched_aggregates == pytest.approx((0.3, 0.15))
    assert result.differences == pytest.approx((0.05, 0.2))
    assert result.repeat_ids == ("r000", "r001")
    assert result.to_records()[0]["seed"] == 10
    assert result.to_records()[0]["membership_sha256"]


def test_empty_selected_is_unevaluable_not_zero() -> None:
    result = reduce_cell_gain_enrichment(
        subject_id="s01",
        target_region="mouth",
        estimand_id="cell_gain_enrichment_v1",
        aggregation_id="arithmetic_mean",
        candidate_grid=_grid(),
        support_sha256=SUPPORT_SHA256,
        selected=(),
        matched_repeats=(MatchedGainRepeat("r000", 10, ()),),
        aggregate=lambda _: pytest.fail("empty selected must not be aggregated"),
    )

    assert result.status == "unevaluable_empty_selected"
    assert result.selected_aggregate is None
    assert result.differences == (None,)


@pytest.mark.parametrize("conflict", ["selected_gain", "repeat_gain", "feature_unit"])
def test_enrichment_rejects_inconsistent_landscape_before_aggregation(conflict) -> None:
    selected = _observation(1, 0.5)
    first = _observation(2, 0.1)
    second = first
    grid = _grid()
    if conflict == "selected_gain":
        second = replace(selected, gain=0.2)
    elif conflict == "repeat_gain":
        second = replace(first, gain=0.2)
    else:
        second = replace(first, candidate=replace(first.candidate, feature_unit="region_block"))
        grid = CandidateGrid.from_candidates(
            (*grid.candidates, second.candidate), protocol_sha256=PROTOCOL_SHA256
        )
    with pytest.raises(EnrichmentContractError, match="feature_unit|changes gain"):
        reduce_cell_gain_enrichment(
            subject_id="s01", target_region="mouth", estimand_id="cell_gain_enrichment_v1",
            aggregation_id="arithmetic_mean", candidate_grid=grid,
            support_sha256=SUPPORT_SHA256, selected=(selected,),
            matched_repeats=(MatchedGainRepeat("r0", 10, (first,)), MatchedGainRepeat("r1", 11, (second,))),
            aggregate=lambda _: pytest.fail("invalid input must fail before aggregation"),
        )


def test_enrichment_allows_identical_memberships_in_distinct_repeats() -> None:
    selected = (_observation(1, 0.5),)
    result = reduce_cell_gain_enrichment(
        subject_id="s01", target_region="mouth", estimand_id="cell_gain_enrichment_v1",
        aggregation_id="arithmetic_mean", candidate_grid=_grid(),
        support_sha256=SUPPORT_SHA256, selected=selected,
        matched_repeats=(MatchedGainRepeat("r0", 10, selected), MatchedGainRepeat("r1", 11, selected)),
        aggregate=mean,
    )
    assert result.differences == (0.0, 0.0)
    assert result.repeat_ids == ("r0", "r1")


@pytest.mark.parametrize(
    ("selected", "matched", "message"),
    [
        ((_observation(1, 0.5),), (MatchedGainRepeat("r000", 10, ()),), "feature count"),
        ((_observation(1, 0.5),), (MatchedGainRepeat("r000", 10, (_observation(2, 0.1),)),), "support digest"),
    ],
)
def test_enrichment_rejects_invalid_matched_repeats(selected, matched, message) -> None:
    if message == "support digest":
        matched = (
            MatchedGainRepeat(
                "r000",
                10,
                (
                    CellGainObservation(
                        "s01", _candidate(2), 0.1, "c" * 64
                    ),
                ),
            ),
        )
    with pytest.raises(EnrichmentContractError, match=message):
        reduce_cell_gain_enrichment(
            subject_id="s01",
            target_region="mouth",
            estimand_id="cell_gain_enrichment_v1",
            aggregation_id="arithmetic_mean",
            candidate_grid=_grid(),
            support_sha256=SUPPORT_SHA256,
            selected=selected,
            matched_repeats=matched,
            aggregate=mean,
        )


def test_enrichment_rejects_duplicate_repeat_ids_and_out_of_grid_cells() -> None:
    with pytest.raises(EnrichmentContractError, match="repeat IDs must be unique"):
        reduce_cell_gain_enrichment(
            subject_id="s01",
            target_region="mouth",
            estimand_id="cell_gain_enrichment_v1",
            aggregation_id="arithmetic_mean",
            candidate_grid=_grid(),
            support_sha256=SUPPORT_SHA256,
            selected=(_observation(1, 0.5),),
            matched_repeats=(
                MatchedGainRepeat("r000", 10, (_observation(2, 0.1),)),
                MatchedGainRepeat("r000", 11, (_observation(3, 0.2),)),
            ),
            aggregate=mean,
        )

    outside = LandscapeCandidate(
        source_region="brow",
        target_region="mouth",
        lag=1,
        source_dimension="vx",
        target_dimension="vy",
        feature_unit="region_dimension_lag",
    )
    with pytest.raises(EnrichmentContractError, match="outside candidate_grid"):
        reduce_cell_gain_enrichment(
            subject_id="s01",
            target_region="mouth",
            estimand_id="cell_gain_enrichment_v1",
            aggregation_id="arithmetic_mean",
            candidate_grid=_grid(),
            support_sha256=SUPPORT_SHA256,
            selected=(
                CellGainObservation("s01", outside, 0.5, SUPPORT_SHA256),
            ),
            matched_repeats=(
                MatchedGainRepeat("r000", 10, (_observation(1, 0.1),)),
            ),
            aggregate=mean,
        )


def test_enrichment_requires_explicit_repeat_count_when_requested() -> None:
    with pytest.raises(EnrichmentContractError, match="expected_repeat_count"):
        reduce_cell_gain_enrichment(
            subject_id="s01",
            target_region="mouth",
            estimand_id="cell_gain_enrichment_v1",
            aggregation_id="arithmetic_mean",
            candidate_grid=_grid(),
            support_sha256=SUPPORT_SHA256,
            selected=(_observation(1, 0.5),),
            matched_repeats=(
                MatchedGainRepeat("r000", 10, (_observation(2, 0.1),)),
            ),
            aggregate=mean,
            expected_repeat_count=1000,
        )
