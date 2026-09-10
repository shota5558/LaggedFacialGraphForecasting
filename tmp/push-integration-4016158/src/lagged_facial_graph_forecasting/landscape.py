"""Contracts for a frozen predictive-gain candidate landscape.

This module deliberately stops at the contract boundary. It does not choose a
candidate grid, fit a model, or turn an outcome into a new grid. Callers must
provide the already-approved candidates and protocol digest.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
import re


class LandscapeContractError(ValueError):
    """Raised when a landscape candidate or gain violates its contract."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LandscapeContractError(f"{field_name} must be a non-empty string")
    return value.strip()


def _sha256(value: str, field_name: str) -> str:
    value = _text(value, field_name).lower()
    if not _SHA256_RE.fullmatch(value):
        raise LandscapeContractError(f"{field_name} must be a 64-character SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True, order=True)
class LandscapeCandidate:
    """One explicitly resolved source-to-target candidate cell."""

    source_region: str
    target_region: str
    lag: int
    source_dimension: str
    target_dimension: str
    feature_unit: str

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.source_region, "source_region"),
            (self.target_region, "target_region"),
            (self.source_dimension, "source_dimension"),
            (self.target_dimension, "target_dimension"),
            (self.feature_unit, "feature_unit"),
        ):
            _text(value, field_name)
        if not isinstance(self.lag, int) or isinstance(self.lag, bool) or self.lag < 1:
            raise LandscapeContractError("lag must be a positive integer")

        object.__setattr__(self, "source_region", self.source_region.strip())
        object.__setattr__(self, "target_region", self.target_region.strip())
        object.__setattr__(self, "source_dimension", self.source_dimension.strip())
        object.__setattr__(self, "target_dimension", self.target_dimension.strip())
        object.__setattr__(self, "feature_unit", self.feature_unit.strip())

    def to_payload(self) -> dict[str, object]:
        """Return the canonical JSON representation used by the grid digest."""

        return {
            "feature_unit": self.feature_unit,
            "lag": self.lag,
            "source_dimension": self.source_dimension,
            "source_region": self.source_region,
            "target_dimension": self.target_dimension,
            "target_region": self.target_region,
        }


@dataclass(frozen=True, slots=True)
class CandidateGrid:
    """A deterministic, non-empty set of approved landscape candidates."""

    candidates: tuple[LandscapeCandidate, ...]
    protocol_sha256: str

    def __post_init__(self) -> None:
        protocol_sha256 = _sha256(self.protocol_sha256, "protocol_sha256")
        candidates = tuple(self.candidates)
        if not candidates:
            raise LandscapeContractError("candidate grid must not be empty")
        if any(not isinstance(candidate, LandscapeCandidate) for candidate in candidates):
            raise LandscapeContractError(
                "candidate grid entries must be LandscapeCandidate instances"
            )

        ordered = tuple(sorted(candidates))
        if len(set(ordered)) != len(ordered):
            raise LandscapeContractError("candidate grid contains duplicate candidates")

        object.__setattr__(self, "candidates", ordered)
        object.__setattr__(self, "protocol_sha256", protocol_sha256)

    @classmethod
    def from_candidates(
        cls,
        candidates: Iterable[LandscapeCandidate],
        *,
        protocol_sha256: str,
    ) -> "CandidateGrid":
        """Build a grid while making ordering independent of caller input order."""

        return cls(tuple(candidates), protocol_sha256)

    @property
    def digest(self) -> str:
        payload = {
            "candidates": [candidate.to_payload() for candidate in self.candidates],
            "protocol_sha256": self.protocol_sha256,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()


def validate_candidate_grid_coverage(
    expected: CandidateGrid,
    observed: Iterable[LandscapeCandidate],
) -> None:
    """Reject missing, duplicate, or unexpected cells before aggregation."""

    observed = tuple(observed)
    if any(not isinstance(candidate, LandscapeCandidate) for candidate in observed):
        raise LandscapeContractError(
            "observed grid entries must be LandscapeCandidate instances"
        )
    if len(set(observed)) != len(observed):
        raise LandscapeContractError("observed candidate grid contains duplicates")

    expected_keys = set(expected.candidates)
    observed_keys = set(observed)
    missing = sorted(expected_keys - observed_keys)
    extra = sorted(observed_keys - expected_keys)
    if missing or extra:
        raise LandscapeContractError(
            f"candidate grid coverage mismatch: missing={len(missing)}, extra={len(extra)}"
        )


def compute_cell_gain(
    self_error: float,
    cell_error: float,
    *,
    self_support_sha256: str,
    cell_support_sha256: str,
) -> float:
    """Compute ``G = E_self - E_cell`` only on identical evaluation support."""

    self_support_sha256 = _sha256(self_support_sha256, "self_support_sha256")
    cell_support_sha256 = _sha256(cell_support_sha256, "cell_support_sha256")
    if self_support_sha256 != cell_support_sha256:
        raise LandscapeContractError("Self and cell evaluation support digests differ")

    for value, field_name in ((self_error, "self_error"), (cell_error, "cell_error")):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise LandscapeContractError(f"{field_name} must be a finite number")
    return float(self_error) - float(cell_error)
