"""Versioned core contracts frozen after the V0 vertical slice.

The original V0 contracts remain in ``contracts.py``. This module adds the
remaining cross-lane interfaces without coupling forecasting code to Tigramite,
Null implementations, statistics code, or orchestration internals.

Core-contract schema v3 completes the component-aware migration required by
Scientific Freeze v2. PCMCI+ + ParCorr discovery operates on scalar
``region × dimension`` nodes, so retained ParentLinks preserve source- and
target-component identity and downstream DesignMatrix / PredictionArtifact
contracts preserve the target output dimension labels as well. Region-only
projection remains a reporting operation and must not destroy canonical
component provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
import re

import numpy as np

from .contracts import ContractError

CORE_CONTRACT_SCHEMA_VERSION = 3
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _non_empty_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field_name} must be a non-empty string")
    return value.strip()


def _non_negative_int(value: int, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContractError(f"{field_name} must be a non-negative integer")
    return value


def _positive_int(value: int, field_name: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ContractError(f"{field_name} must be a positive integer")
    value = int(value)
    if value < 1:
        raise ContractError(f"{field_name} must be a positive integer")
    return value


def _relative_posix_path(value: str, field_name: str) -> str:
    value = _non_empty_text(value, field_name)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ContractError(f"{field_name} must be a repository-relative POSIX path")
    return path.as_posix()


def _schema_version(value: int) -> int:
    if value != CORE_CONTRACT_SCHEMA_VERSION:
        raise ContractError(
            f"schema_version must be {CORE_CONTRACT_SCHEMA_VERSION}; got {value!r}"
        )
    return value


@dataclass(frozen=True, slots=True, order=True)
class ParentLink:
    """One canonical scalar component-to-component lagged parent link.

    ``source_dimension`` and ``target_dimension`` preserve the exact Tigramite
    scalar-node provenance required by Scientific Freeze v2. Their ``"value"``
    defaults keep legacy scalar (D=1) fixtures source-compatible; multicomponent
    discovery must always populate the actual dimension labels explicitly.
    """

    source_region: str
    lag: int
    source_dimension: str = "value"
    target_dimension: str = "value"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_region", _non_empty_text(self.source_region, "source_region")
        )
        object.__setattr__(self, "lag", _positive_int(self.lag, "lag"))
        object.__setattr__(
            self,
            "source_dimension",
            _non_empty_text(self.source_dimension, "source_dimension"),
        )
        object.__setattr__(
            self,
            "target_dimension",
            _non_empty_text(self.target_dimension, "target_dimension"),
        )


@dataclass(frozen=True, slots=True)
class ParentSet:
    """Canonical Tigramite-independent boundary between discovery and forecasting."""

    outer_fold: int
    target_region: str
    parents: tuple[ParentLink, ...]
    discovery_method: str = "pcmci_plus"
    schema_version: int = CORE_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "outer_fold", _non_negative_int(self.outer_fold, "outer_fold")
        )
        object.__setattr__(
            self,
            "target_region",
            _non_empty_text(self.target_region, "target_region"),
        )
        object.__setattr__(
            self,
            "discovery_method",
            _non_empty_text(self.discovery_method, "discovery_method"),
        )
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))

        parents = tuple(self.parents)
        if any(not isinstance(parent, ParentLink) for parent in parents):
            raise ContractError("parents entries must be ParentLink instances")
        if len(set(parents)) != len(parents):
            raise ContractError(
                "parents must not contain duplicate "
                "(source_region, lag, source_dimension, target_dimension) links"
            )
        object.__setattr__(self, "parents", parents)


@dataclass(frozen=True, slots=True)
class NullMapping:
    """Frozen falsification mapping constructed without outer-test information.

    Region/lag/component based Nulls use equal-length ``source_parents`` and
    ``mapped_parents``. Temporal Nulls can additionally carry a deterministic row
    permutation. Component identity is retained so Null generation cannot silently
    change the scientific feature-selection rule.
    """

    outer_fold: int
    target_region: str
    condition: str
    seed: int
    source_parents: tuple[ParentLink, ...]
    mapped_parents: tuple[ParentLink, ...]
    permutation: tuple[int, ...] = ()
    schema_version: int = CORE_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "outer_fold", _non_negative_int(self.outer_fold, "outer_fold")
        )
        object.__setattr__(
            self,
            "target_region",
            _non_empty_text(self.target_region, "target_region"),
        )
        object.__setattr__(
            self, "condition", _non_empty_text(self.condition, "condition")
        )
        object.__setattr__(self, "seed", _non_negative_int(self.seed, "seed"))
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))

        source_parents = tuple(self.source_parents)
        mapped_parents = tuple(self.mapped_parents)
        if any(
            not isinstance(parent, ParentLink)
            for parent in source_parents + mapped_parents
        ):
            raise ContractError("NullMapping parents entries must be ParentLink instances")
        if len(source_parents) != len(mapped_parents):
            raise ContractError("source_parents and mapped_parents must have equal length")
        if len(set(source_parents)) != len(source_parents):
            raise ContractError("source_parents must not contain duplicates")
        if len(set(mapped_parents)) != len(mapped_parents):
            raise ContractError("mapped_parents must not contain duplicates")

        permutation = tuple(self.permutation)
        if any(
            not isinstance(index, (int, np.integer))
            or isinstance(index, (bool, np.bool_))
            or int(index) < 0
            for index in permutation
        ):
            raise ContractError("permutation entries must be non-negative integers")
        permutation = tuple(int(index) for index in permutation)
        if permutation and set(permutation) != set(range(len(permutation))):
            raise ContractError("permutation must contain each row index exactly once")

        object.__setattr__(self, "source_parents", source_parents)
        object.__setattr__(self, "mapped_parents", mapped_parents)
        object.__setattr__(self, "permutation", permutation)


@dataclass(frozen=True, slots=True)
class MetricsResult:
    """One subject-region-condition metric suitable for paired aggregation."""

    outer_fold: int
    subject_id: str
    region_id: str
    condition: str
    metric_name: str
    value: float
    n_valid: int
    schema_version: int = CORE_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "outer_fold", _non_negative_int(self.outer_fold, "outer_fold")
        )
        for field_name in ("subject_id", "region_id", "condition", "metric_name"):
            object.__setattr__(
                self,
                field_name,
                _non_empty_text(getattr(self, field_name), field_name),
            )
        value = float(self.value)
        if not np.isfinite(value):
            raise ContractError("value must be finite")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "n_valid", _positive_int(self.n_valid, "n_valid"))
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))


@dataclass(frozen=True, slots=True)
class ExperimentArtifact:
    """Content-addressed registry entry for a reproducible experiment artifact."""

    experiment_id: str
    artifact_type: str
    relative_path: str
    sha256: str
    outer_fold: int | None = None
    condition: str | None = None
    schema_version: int = CORE_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "experiment_id",
            _non_empty_text(self.experiment_id, "experiment_id"),
        )
        object.__setattr__(
            self,
            "artifact_type",
            _non_empty_text(self.artifact_type, "artifact_type"),
        )
        object.__setattr__(
            self,
            "relative_path",
            _relative_posix_path(self.relative_path, "relative_path"),
        )
        sha256 = _non_empty_text(self.sha256, "sha256").lower()
        if not _SHA256_RE.fullmatch(sha256):
            raise ContractError(
                "sha256 must contain exactly 64 lowercase hexadecimal characters"
            )
        object.__setattr__(self, "sha256", sha256)
        if self.outer_fold is not None:
            object.__setattr__(
                self,
                "outer_fold",
                _non_negative_int(self.outer_fold, "outer_fold"),
            )
        if self.condition is not None:
            object.__setattr__(
                self,
                "condition",
                _non_empty_text(self.condition, "condition"),
            )
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Stable envelope pointing to versioned scientific and run configuration."""

    experiment_id: str
    scientific_config_path: str
    run_config_path: str
    artifact_root: str
    seed: int
    schema_version: int = CORE_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "experiment_id",
            _non_empty_text(self.experiment_id, "experiment_id"),
        )
        object.__setattr__(
            self,
            "scientific_config_path",
            _relative_posix_path(
                self.scientific_config_path, "scientific_config_path"
            ),
        )
        object.__setattr__(
            self,
            "run_config_path",
            _relative_posix_path(self.run_config_path, "run_config_path"),
        )
        object.__setattr__(
            self,
            "artifact_root",
            _relative_posix_path(self.artifact_root, "artifact_root"),
        )
        object.__setattr__(self, "seed", _non_negative_int(self.seed, "seed"))
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
