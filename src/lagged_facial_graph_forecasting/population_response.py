"""Contract-level reducer for edge-centered population lag response.

The reducer keeps the edge/subject unit explicit and requires a complete,
common delta grid.  It does not select the unresolved edge context or cluster
bootstrap rule.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from numbers import Real
import re


class PopulationResponseContractError(ValueError):
    """Raised when an edge-centered lag-response record is invalid."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PopulationResponseContractError(
            f"{field_name} must be a non-empty string"
        )
    return value.strip()


def _sha256(value: str, field_name: str) -> str:
    value = _text(value, field_name).lower()
    if not _SHA256_RE.fullmatch(value):
        raise PopulationResponseContractError(
            f"{field_name} must be a 64-character SHA-256 digest"
        )
    return value


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise PopulationResponseContractError(f"{field_name} must be a finite number")
    return float(value)


@dataclass(frozen=True, slots=True)
class EdgeLagResponseObservation:
    """One edge/subject response at a fixed ``tau_star + delta`` lag."""

    outer_fold: int
    edge_id: str
    subject_id: str
    source_region: str
    target_region: str
    source_dimension: str
    target_dimension: str
    context_id: str
    tau_star: int
    delta: int
    shifted_lag: int
    sampling_rate_hz: float
    reference_error: float
    shifted_error: float
    reference_support_sha256: str
    shifted_support_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.outer_fold, int) or isinstance(self.outer_fold, bool):
            raise PopulationResponseContractError("outer_fold must be an integer")
        if self.outer_fold < 0:
            raise PopulationResponseContractError("outer_fold must be >= 0")
        normalized_text = {}
        for value, name in (
            (self.edge_id, "edge_id"),
            (self.subject_id, "subject_id"),
            (self.source_region, "source_region"),
            (self.target_region, "target_region"),
            (self.source_dimension, "source_dimension"),
            (self.target_dimension, "target_dimension"),
            (self.context_id, "context_id"),
        ):
            normalized_text[name] = _text(value, name)
        if not isinstance(self.tau_star, int) or isinstance(self.tau_star, bool) or self.tau_star < 1:
            raise PopulationResponseContractError("tau_star must be a positive integer")
        if not isinstance(self.delta, int) or isinstance(self.delta, bool):
            raise PopulationResponseContractError("delta must be an integer")
        if (
            not isinstance(self.shifted_lag, int)
            or isinstance(self.shifted_lag, bool)
            or self.shifted_lag < 1
        ):
            raise PopulationResponseContractError("shifted_lag must be a positive integer")
        if self.shifted_lag != self.tau_star + self.delta:
            raise PopulationResponseContractError(
                "shifted_lag must equal tau_star + delta; clipping is forbidden"
            )
        sampling_rate_hz = _finite(self.sampling_rate_hz, "sampling_rate_hz")
        if sampling_rate_hz <= 0:
            raise PopulationResponseContractError("sampling_rate_hz must be > 0")
        reference_error = _finite(self.reference_error, "reference_error")
        shifted_error = _finite(self.shifted_error, "shifted_error")
        reference_support = _sha256(
            self.reference_support_sha256, "reference_support_sha256"
        )
        shifted_support = _sha256(
            self.shifted_support_sha256, "shifted_support_sha256"
        )
        if reference_support != shifted_support:
            raise PopulationResponseContractError(
                "reference and shifted support digests differ"
            )
        if self.delta == 0 and shifted_error != reference_error:
            raise PopulationResponseContractError(
                "delta=0 must use the reference error and have zero difference"
            )
        object.__setattr__(self, "sampling_rate_hz", sampling_rate_hz)
        object.__setattr__(self, "reference_error", reference_error)
        object.__setattr__(self, "shifted_error", shifted_error)
        for name, value in normalized_text.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "reference_support_sha256", reference_support)
        object.__setattr__(self, "shifted_support_sha256", shifted_support)

    @property
    def difference(self) -> float:
        """Return ``E_shifted - E_reference`` for this edge/subject unit."""

        return self.shifted_error - self.reference_error

    @property
    def support_sha256(self) -> str:
        return self.reference_support_sha256

    @property
    def reference_lag_milliseconds(self) -> float:
        return 1000.0 * self.tau_star / self.sampling_rate_hz

    @property
    def shifted_lag_milliseconds(self) -> float:
        return 1000.0 * self.shifted_lag / self.sampling_rate_hz

    @property
    def unit_key(self) -> tuple[int, str, str]:
        return (self.outer_fold, self.edge_id, self.subject_id)


@dataclass(frozen=True, slots=True)
class EdgeLagResponseAggregate:
    """One point estimate over the complete edge/subject support at a delta."""

    delta: int
    context_id: str
    aggregation_id: str
    value: float
    n_edge_subject: int
    n_subjects: int
    n_edges: int
    unit_keys: tuple[tuple[int, str, str], ...]
    support_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.delta, int) or isinstance(self.delta, bool):
            raise PopulationResponseContractError("aggregate delta must be an integer")
        context_id = _text(self.context_id, "context_id")
        aggregation_id = _text(self.aggregation_id, "aggregation_id")
        value = _finite(self.value, "aggregate value")
        for name in ("n_edge_subject", "n_subjects", "n_edges"):
            count = getattr(self, name)
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise PopulationResponseContractError(
                    f"{name} must be a positive integer"
                )
        unit_keys = tuple(self.unit_keys)
        if len(unit_keys) != self.n_edge_subject or len(set(unit_keys)) != len(unit_keys):
            raise PopulationResponseContractError(
                "unit_keys must be unique and match n_edge_subject"
            )
        support_sha256 = tuple(
            _sha256(item, "support_sha256 entry") for item in self.support_sha256
        )
        if not support_sha256:
            raise PopulationResponseContractError("support_sha256 must not be empty")
        object.__setattr__(self, "context_id", context_id)
        object.__setattr__(self, "aggregation_id", aggregation_id)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "unit_keys", unit_keys)
        object.__setattr__(self, "support_sha256", support_sha256)


def aggregate_edge_lag_response(
    observations: Sequence[EdgeLagResponseObservation],
    *,
    deltas: Sequence[int],
    aggregate: Callable[[tuple[float, ...]], Real],
    aggregation_id: str,
) -> tuple[EdgeLagResponseAggregate, ...]:
    """Reduce only edge/subject units observed on the complete delta grid.

    ``aggregate`` is explicit because the approved population point estimator and
    cluster bootstrap are still scientific decisions.  Incomplete units are
    rejected instead of being clipped to a one-sided response curve.
    """

    observations = tuple(observations)
    if not observations:
        raise PopulationResponseContractError("observations must not be empty")
    if any(not isinstance(item, EdgeLagResponseObservation) for item in observations):
        raise TypeError("observations must be EdgeLagResponseObservation instances")
    deltas = tuple(deltas)
    if not deltas or any(not isinstance(delta, int) or isinstance(delta, bool) for delta in deltas):
        raise PopulationResponseContractError("deltas must be non-empty integers")
    if len(set(deltas)) != len(deltas) or 0 not in deltas:
        raise PopulationResponseContractError(
            "deltas must be unique and include the reference delta=0"
        )
    if not callable(aggregate):
        raise TypeError("aggregate must be callable")
    aggregation_id = _text(aggregation_id, "aggregation_id")

    contexts = {item.context_id for item in observations}
    if len(contexts) != 1:
        raise PopulationResponseContractError(
            "one aggregation call must not mix edge response contexts"
        )
    context_id = next(iter(contexts))

    by_key: dict[tuple[int, str, str], dict[int, EdgeLagResponseObservation]] = {}
    edge_definitions: dict[tuple[int, str], tuple[str, str, str, str, int]] = {}
    for item in observations:
        edge_key = (item.outer_fold, item.edge_id)
        definition = (
            item.source_region, item.target_region,
            item.source_dimension, item.target_dimension, item.tau_star,
        )
        if edge_definitions.setdefault(edge_key, definition) != definition:
            raise PopulationResponseContractError(
                f"edge {edge_key!r} changes identity across observations"
            )
        unit = by_key.setdefault(item.unit_key, {})
        if item.delta in unit:
            raise PopulationResponseContractError(
                f"duplicate edge/subject/delta observation for {item.unit_key!r}"
            )
        unit[item.delta] = item

    expected = set(deltas)
    for unit_key, rows in by_key.items():
        if set(rows) != expected:
            raise PopulationResponseContractError(
                f"edge/subject unit {unit_key!r} does not have the complete delta grid"
            )
        ordered = [rows[delta] for delta in deltas]
        reference = rows[0]
        for item in ordered:
            if item.reference_error != reference.reference_error:
                raise PopulationResponseContractError(
                    f"edge/subject unit {unit_key!r} changes reference error across deltas"
                )
            if (
                item.tau_star != reference.tau_star
                or item.source_region != reference.source_region
                or item.target_region != reference.target_region
                or item.source_dimension != reference.source_dimension
                or item.target_dimension != reference.target_dimension
                or item.sampling_rate_hz != reference.sampling_rate_hz
                or item.support_sha256 != reference.support_sha256
            ):
                raise PopulationResponseContractError(
                    f"edge/subject unit {unit_key!r} changes identity or support across deltas"
                )

    unit_keys = tuple(sorted(by_key))
    n_subjects = len({key[2] for key in unit_keys})
    n_edges = len({(key[0], key[1]) for key in unit_keys})
    support_digests = tuple(sorted({item.support_sha256 for item in observations}))
    results: list[EdgeLagResponseAggregate] = []
    for delta in deltas:
        values = tuple(by_key[key][delta].difference for key in unit_keys)
        try:
            value = aggregate(values)
        except Exception as exc:  # pragma: no cover - preserves caller error context
            raise PopulationResponseContractError(
                f"delta={delta} aggregation failed"
            ) from exc
        value = _finite(value, f"delta={delta} aggregate")
        results.append(
            EdgeLagResponseAggregate(
                delta=delta,
                context_id=context_id,
                aggregation_id=aggregation_id,
                value=value,
                n_edge_subject=len(unit_keys),
                n_subjects=n_subjects,
                n_edges=n_edges,
                unit_keys=unit_keys,
                support_sha256=support_digests,
            )
        )
    return tuple(results)
