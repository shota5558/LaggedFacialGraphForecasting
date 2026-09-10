"""Contract-level reduction for PCMCI selection enrichment.

This module keeps the selected and matched-sparsity paths separate.  It validates
the cell-level ``G`` records and returns the full matched-repeat distribution, but
does not choose the unresolved scientific aggregation or inferential rule.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import hashlib
import json
import math
from numbers import Real
import re
from dataclasses import dataclass

from .landscape import CandidateGrid, LandscapeCandidate


class EnrichmentContractError(ValueError):
    """Raised when an enrichment input violates its frozen data contract."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EnrichmentContractError(f"{field_name} must be a non-empty string")
    return value.strip()


def _sha256(value: str, field_name: str) -> str:
    value = _text(value, field_name).lower()
    if not _SHA256_RE.fullmatch(value):
        raise EnrichmentContractError(
            f"{field_name} must be a 64-character SHA-256 digest"
        )
    return value


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise EnrichmentContractError(f"{field_name} must be a finite number")
    return float(value)


@dataclass(frozen=True, slots=True)
class CellGainObservation:
    """One subject/target cell-level gain from the frozen landscape."""

    subject_id: str
    candidate: LandscapeCandidate
    gain: float
    support_sha256: str

    def __post_init__(self) -> None:
        subject_id = _text(self.subject_id, "subject_id")
        if not isinstance(self.candidate, LandscapeCandidate):
            raise EnrichmentContractError("candidate must be a LandscapeCandidate")
        gain = _finite(self.gain, "gain")
        support_sha256 = _sha256(self.support_sha256, "support_sha256")
        object.__setattr__(self, "subject_id", subject_id)
        object.__setattr__(self, "gain", gain)
        object.__setattr__(self, "support_sha256", support_sha256)


@dataclass(frozen=True, slots=True)
class MatchedGainRepeat:
    """One explicit matched-sparsity replicate and its cell-level gains."""

    repeat_id: str
    seed: int
    observations: tuple[CellGainObservation, ...]

    def __post_init__(self) -> None:
        repeat_id = _text(self.repeat_id, "repeat_id")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise EnrichmentContractError("seed must be a non-negative integer")
        observations = tuple(self.observations)
        if any(not isinstance(item, CellGainObservation) for item in observations):
            raise EnrichmentContractError(
                "matched repeat observations must be CellGainObservation instances"
            )
        candidates = tuple(item.candidate for item in observations)
        if len(set(candidates)) != len(candidates):
            raise EnrichmentContractError(
                "matched repeat contains duplicate candidate memberships"
            )
        object.__setattr__(self, "repeat_id", repeat_id)
        object.__setattr__(self, "observations", tuple(sorted(observations, key=lambda item: item.candidate)))

    @property
    def membership_sha256(self) -> str:
        """Digest the repeat identity and canonical candidate membership."""

        payload = {
            "repeat_id": self.repeat_id,
            "seed": self.seed,
            "candidates": [item.candidate.to_payload() for item in self.observations],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class EnrichmentDistribution:
    """A subject-level selected-vs-matched cell-gain distribution.

    ``matched_aggregates`` and ``differences`` retain every repeat.  No CI,
    p-value, or repeat-as-subject inference is performed here.
    """

    subject_id: str
    target_region: str
    estimand_id: str
    aggregation_id: str
    candidate_grid_sha256: str
    support_sha256: str
    status: str
    selected: tuple[CellGainObservation, ...]
    matched_repeats: tuple[MatchedGainRepeat, ...]
    selected_aggregate: float | None
    matched_aggregates: tuple[float | None, ...]
    differences: tuple[float | None, ...]

    @property
    def repeat_ids(self) -> tuple[str, ...]:
        return tuple(repeat.repeat_id for repeat in self.matched_repeats)

    @property
    def selected_candidate_count(self) -> int:
        return len(self.selected)

    def to_records(self) -> tuple[dict[str, object], ...]:
        """Return one provenance-complete row per matched repeat."""

        selected_candidates = [item.candidate.to_payload() for item in self.selected]
        records: list[dict[str, object]] = []
        for index, repeat in enumerate(self.matched_repeats):
            records.append(
                {
                    "aggregation_id": self.aggregation_id,
                    "candidate_grid_sha256": self.candidate_grid_sha256,
                    "difference_selected_minus_matched": self.differences[index],
                    "estimand_id": self.estimand_id,
                    "matched_aggregate": self.matched_aggregates[index],
                    "matched_candidates": [
                        item.candidate.to_payload() for item in repeat.observations
                    ],
                    "membership_sha256": repeat.membership_sha256,
                    "repeat_id": repeat.repeat_id,
                    "seed": repeat.seed,
                    "selected_aggregate": self.selected_aggregate,
                    "selected_candidates": selected_candidates,
                    "status": self.status,
                    "subject_id": self.subject_id,
                    "support_sha256": self.support_sha256,
                    "target_region": self.target_region,
                }
            )
        return tuple(records)


def reduce_cell_gain_enrichment(
    *,
    subject_id: str,
    target_region: str,
    estimand_id: str,
    aggregation_id: str,
    candidate_grid: CandidateGrid,
    support_sha256: str,
    selected: Sequence[CellGainObservation],
    matched_repeats: Sequence[MatchedGainRepeat],
    aggregate: Callable[[tuple[float, ...]], Real],
    expected_repeat_count: int | None = None,
) -> EnrichmentDistribution:
    """Validate and reduce one selected/matched enrichment unit.

    ``aggregate`` is intentionally injected: D-10/D-11/D-12 have not frozen the
    scientific aggregation yet.  The function therefore records the chosen
    aggregation identity but never silently defaults to mean or median.
    """

    subject_id = _text(subject_id, "subject_id")
    target_region = _text(target_region, "target_region")
    estimand_id = _text(estimand_id, "estimand_id")
    aggregation_id = _text(aggregation_id, "aggregation_id")
    if not isinstance(candidate_grid, CandidateGrid):
        raise TypeError("candidate_grid must be a CandidateGrid")
    support_sha256 = _sha256(support_sha256, "support_sha256")
    if not callable(aggregate):
        raise TypeError("aggregate must be callable")

    selected = tuple(selected)
    matched_repeats = tuple(matched_repeats)
    if any(not isinstance(item, CellGainObservation) for item in selected):
        raise EnrichmentContractError(
            "selected observations must be CellGainObservation instances"
        )
    if any(not isinstance(item, MatchedGainRepeat) for item in matched_repeats):
        raise EnrichmentContractError(
            "matched_repeats entries must be MatchedGainRepeat instances"
        )
    if not matched_repeats:
        raise EnrichmentContractError("matched_repeats must not be empty")
    if expected_repeat_count is not None and (
        not isinstance(expected_repeat_count, int)
        or isinstance(expected_repeat_count, bool)
        or expected_repeat_count < 1
    ):
        raise EnrichmentContractError(
            "expected_repeat_count must be a positive integer when provided"
        )
    if expected_repeat_count is not None and len(matched_repeats) != expected_repeat_count:
        raise EnrichmentContractError(
            "matched repeat count does not match expected_repeat_count"
        )

    repeat_ids = tuple(repeat.repeat_id for repeat in matched_repeats)
    if len(set(repeat_ids)) != len(repeat_ids):
        raise EnrichmentContractError("matched repeat IDs must be unique")
    grid_candidates = set(candidate_grid.candidates)

    def validate_observations(
        observations: Sequence[CellGainObservation], *, label: str
    ) -> tuple[CellGainObservation, ...]:
        observations = tuple(observations)
        candidates = tuple(item.candidate for item in observations)
        if len(set(candidates)) != len(candidates):
            raise EnrichmentContractError(f"{label} contains duplicate candidate memberships")
        for item in observations:
            if item.subject_id != subject_id:
                raise EnrichmentContractError(f"{label} subject_id does not match unit")
            if item.candidate.target_region != target_region:
                raise EnrichmentContractError(
                    f"{label} candidate target_region does not match unit"
                )
            if item.candidate not in grid_candidates:
                raise EnrichmentContractError(
                    f"{label} contains a candidate outside candidate_grid"
                )
            if item.support_sha256 != support_sha256:
                raise EnrichmentContractError(
                    f"{label} support digest does not match unit support"
                )
        feature_units = {item.candidate.feature_unit for item in observations}
        if len(feature_units) > 1:
            raise EnrichmentContractError(
                f"{label} mixes feature_unit values: {sorted(feature_units)}"
            )
        return observations

    selected = validate_observations(selected, label="selected")
    for repeat in matched_repeats:
        validate_observations(repeat.observations, label=f"repeat {repeat.repeat_id}")
        if len(repeat.observations) != len(selected):
            raise EnrichmentContractError(
                f"repeat {repeat.repeat_id!r} does not match selected feature count"
            )

    # Empty selected membership is explicitly represented as unevaluable.  It is
    # not silently converted to a zero aggregate or an independent observation.
    if not selected:
        return EnrichmentDistribution(
            subject_id=subject_id,
            target_region=target_region,
            estimand_id=estimand_id,
            aggregation_id=aggregation_id,
            candidate_grid_sha256=candidate_grid.digest,
            support_sha256=support_sha256,
            status="unevaluable_empty_selected",
            selected=selected,
            matched_repeats=matched_repeats,
            selected_aggregate=None,
            matched_aggregates=tuple(None for _ in matched_repeats),
            differences=tuple(None for _ in matched_repeats),
        )

    def apply_aggregate(values: tuple[float, ...], *, label: str) -> float:
        if not values:
            raise EnrichmentContractError(f"{label} cannot aggregate an empty set")
        try:
            result = aggregate(values)
        except Exception as exc:  # pragma: no cover - preserves caller error context
            raise EnrichmentContractError(f"{label} aggregation failed") from exc
        return _finite(result, f"{label} aggregate")

    selected_aggregate = apply_aggregate(
        tuple(item.gain for item in selected), label="selected"
    )
    matched_aggregates = tuple(
        apply_aggregate(
            tuple(item.gain for item in repeat.observations),
            label=f"repeat {repeat.repeat_id}",
        )
        for repeat in matched_repeats
    )
    differences = tuple(selected_aggregate - value for value in matched_aggregates)
    return EnrichmentDistribution(
        subject_id=subject_id,
        target_region=target_region,
        estimand_id=estimand_id,
        aggregation_id=aggregation_id,
        candidate_grid_sha256=candidate_grid.digest,
        support_sha256=support_sha256,
        status="evaluable",
        selected=selected,
        matched_repeats=matched_repeats,
        selected_aggregate=selected_aggregate,
        matched_aggregates=matched_aggregates,
        differences=differences,
    )
