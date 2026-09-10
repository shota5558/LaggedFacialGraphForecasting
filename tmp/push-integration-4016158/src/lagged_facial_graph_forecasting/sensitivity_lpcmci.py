"""LPCMCI Sensitivity adapter and canonical PAG evidence (S-IMPL-03).

LPCMCI is used only as a latent-confounding Sensitivity.  Primary remains
PCMCI+ + ParCorr.  All LPCMCI edge marks are preserved in a Sensitivity-only
canonical graph artifact; only definitely directed lagged ``-->`` cross-region
links are projected into the forecasting ParentSet.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.lpcmci import LPCMCI

from .contracts import SplitManifest
from .core_contracts import ParentLink, ParentSet
from .leakage_guard import assert_discovery_fit_scope
from .parcorr_factory import PRIMARY_PARCORR_MASK_TYPE, PRIMARY_PARCORR_SIGNIFICANCE
from .pcmci_pc_alpha import PRIMARY_PC_ALPHA
from .pcmci_tau_max import PRIMARY_TAU_MAX
from .sensitivity_execution import SensitivityExperimentConfig, assert_sensitivity_execution_allowed
from .tigramite_adapter import TigramiteDataFrameBundle


LPCMCI_SCHEMA_VERSION = 1
LPCMCI_DISCOVERY_METHOD = "lpcmci"
LPCMCI_DEFINITE_PARENT_MARK = "-->"


class SensitivityLPCMCIError(ValueError):
    """Raised when the LPCMCI Sensitivity contract is violated."""


@dataclass(frozen=True, slots=True)
class LPCMCIConfig:
    schema_version: int = LPCMCI_SCHEMA_VERSION
    tau_min: int = 0
    tau_max: int = PRIMARY_TAU_MAX
    pc_alpha: float = PRIMARY_PC_ALPHA
    n_preliminary_iterations: int = 1
    max_cond_px: int = 0

    def __post_init__(self) -> None:
        if self.schema_version != LPCMCI_SCHEMA_VERSION:
            raise SensitivityLPCMCIError("unsupported LPCMCI config schema_version")
        if self.tau_min != 0:
            raise SensitivityLPCMCIError("LPCMCI Sensitivity tau_min is frozen to 0")
        if self.tau_max != PRIMARY_TAU_MAX:
            raise SensitivityLPCMCIError(
                f"LPCMCI Sensitivity tau_max must remain {PRIMARY_TAU_MAX}"
            )
        if not np.isclose(self.pc_alpha, PRIMARY_PC_ALPHA, rtol=0.0, atol=0.0):
            raise SensitivityLPCMCIError(
                f"LPCMCI Sensitivity pc_alpha must remain {PRIMARY_PC_ALPHA}"
            )
        if self.n_preliminary_iterations < 1:
            raise SensitivityLPCMCIError("n_preliminary_iterations must be >= 1")
        if self.max_cond_px < 0:
            raise SensitivityLPCMCIError("max_cond_px must be >= 0")

    def tigramite_kwargs(self) -> dict[str, Any]:
        return {
            "tau_min": self.tau_min,
            "tau_max": self.tau_max,
            "pc_alpha": self.pc_alpha,
            "n_preliminary_iterations": self.n_preliminary_iterations,
            "max_cond_px": self.max_cond_px,
        }


@dataclass(frozen=True, slots=True, order=True)
class LPCMCIAdjacency:
    source_region: str
    source_dimension: str
    target_region: str
    target_dimension: str
    lag: int
    mark: str
    p_value: float
    test_statistic: float

    def __post_init__(self) -> None:
        if self.lag < 0 or self.lag > PRIMARY_TAU_MAX:
            raise SensitivityLPCMCIError("LPCMCI adjacency lag is outside the frozen domain")
        if not self.mark:
            raise SensitivityLPCMCIError("LPCMCI adjacency mark must be non-empty")


@dataclass(frozen=True, slots=True)
class LPCMCIGraphArtifact:
    outer_fold: int
    seed: int
    adjacencies: tuple[LPCMCIAdjacency, ...]
    method: str = LPCMCI_DISCOVERY_METHOD
    schema_version: int = LPCMCI_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.outer_fold < 0:
            raise SensitivityLPCMCIError("outer_fold must be non-negative")
        if self.seed < 0:
            raise SensitivityLPCMCIError("seed must be non-negative")
        if self.method != LPCMCI_DISCOVERY_METHOD:
            raise SensitivityLPCMCIError("LPCMCI method provenance is immutable")
        if self.schema_version != LPCMCI_SCHEMA_VERSION:
            raise SensitivityLPCMCIError("unsupported LPCMCI graph schema_version")
        if any(not isinstance(edge, LPCMCIAdjacency) for edge in self.adjacencies):
            raise SensitivityLPCMCIError("adjacencies must contain LPCMCIAdjacency records")


def load_lpcmci_config(path: str | Path) -> LPCMCIConfig:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SensitivityLPCMCIError(f"cannot load LPCMCI config: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise SensitivityLPCMCIError("LPCMCI config root must be a mapping")
    expected = {
        "schema_version",
        "tau_min",
        "tau_max",
        "pc_alpha",
        "n_preliminary_iterations",
        "max_cond_px",
    }
    if set(raw) != expected:
        raise SensitivityLPCMCIError("LPCMCI config keys differ from the frozen schema")
    return LPCMCIConfig(**dict(raw))


def make_sensitivity_lpcmci(bundle: TigramiteDataFrameBundle, *, seed: int) -> LPCMCI:
    if not isinstance(bundle, TigramiteDataFrameBundle):
        raise SensitivityLPCMCIError("bundle must be a TigramiteDataFrameBundle")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise SensitivityLPCMCIError("seed must be a non-negative integer")
    parcorr = ParCorr(
        seed=seed,
        significance=PRIMARY_PARCORR_SIGNIFICANCE,
        mask_type=PRIMARY_PARCORR_MASK_TYPE,
        recycle_residuals=False,
        confidence=None,
        verbosity=0,
    )
    return LPCMCI(dataframe=bundle.dataframe, cond_ind_test=parcorr, verbosity=0)


def run_sensitivity_lpcmci(
    execution_config: SensitivityExperimentConfig,
    manifest: SplitManifest,
    bundle: TigramiteDataFrameBundle,
    method_config: LPCMCIConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    assert_sensitivity_execution_allowed(execution_config)
    assert_discovery_fit_scope(manifest, bundle.subject_ids)
    adapter = make_sensitivity_lpcmci(bundle, seed=seed)
    result = adapter.run_lpcmci(**method_config.tigramite_kwargs())
    if not isinstance(result, dict):
        raise SensitivityLPCMCIError("Tigramite run_lpcmci must return a dictionary")
    return result


def lpcmci_result_to_graph_artifact(
    result: Mapping[str, Any],
    bundle: TigramiteDataFrameBundle,
    *,
    outer_fold: int,
    seed: int,
) -> LPCMCIGraphArtifact:
    graph = np.asarray(result.get("graph"))
    p_matrix = np.asarray(result.get("p_matrix"), dtype=float)
    val_matrix = np.asarray(result.get("val_matrix"), dtype=float)
    n = len(bundle.variable_components)
    expected = (n, n, PRIMARY_TAU_MAX + 1)
    if graph.shape != expected or p_matrix.shape != expected or val_matrix.shape != expected:
        raise SensitivityLPCMCIError(
            f"LPCMCI result matrices must have shape {expected}; got "
            f"graph={graph.shape}, p={p_matrix.shape}, val={val_matrix.shape}"
        )

    edges: list[LPCMCIAdjacency] = []
    for source_index in range(n):
        source_region, source_dimension = bundle.variable_components[source_index]
        for target_index in range(n):
            target_region, target_dimension = bundle.variable_components[target_index]
            for lag in range(PRIMARY_TAU_MAX + 1):
                mark = str(graph[source_index, target_index, lag])
                if not mark:
                    continue
                edges.append(
                    LPCMCIAdjacency(
                        source_region=source_region,
                        source_dimension=source_dimension,
                        target_region=target_region,
                        target_dimension=target_dimension,
                        lag=lag,
                        mark=mark,
                        p_value=float(p_matrix[source_index, target_index, lag]),
                        test_statistic=float(val_matrix[source_index, target_index, lag]),
                    )
                )
    return LPCMCIGraphArtifact(
        outer_fold=outer_fold,
        seed=seed,
        adjacencies=tuple(sorted(edges)),
    )


def lpcmci_graph_to_parent_set(
    artifact: LPCMCIGraphArtifact,
    *,
    target_region: str,
) -> ParentSet:
    parents = {
        ParentLink(
            source_region=edge.source_region,
            lag=edge.lag,
            source_dimension=edge.source_dimension,
            target_dimension=edge.target_dimension,
        )
        for edge in artifact.adjacencies
        if edge.target_region == target_region
        and edge.source_region != target_region
        and edge.lag >= 1
        and edge.mark == LPCMCI_DEFINITE_PARENT_MARK
    }
    return ParentSet(
        outer_fold=artifact.outer_fold,
        target_region=target_region,
        parents=tuple(sorted(parents)),
        discovery_method=LPCMCI_DISCOVERY_METHOD,
    )


def dumps_lpcmci_graph(artifact: LPCMCIGraphArtifact) -> str:
    payload = {
        "schema_version": artifact.schema_version,
        "method": artifact.method,
        "outer_fold": artifact.outer_fold,
        "seed": artifact.seed,
        "adjacencies": [asdict(edge) for edge in artifact.adjacencies],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def loads_lpcmci_graph(text: str) -> LPCMCIGraphArtifact:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SensitivityLPCMCIError("invalid LPCMCI graph JSON") from exc
    expected = {"schema_version", "method", "outer_fold", "seed", "adjacencies"}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise SensitivityLPCMCIError("LPCMCI graph fields differ from the frozen schema")
    if not isinstance(raw["adjacencies"], list):
        raise SensitivityLPCMCIError("adjacencies must be a list")
    edges = tuple(LPCMCIAdjacency(**item) for item in raw["adjacencies"])
    return LPCMCIGraphArtifact(
        schema_version=raw["schema_version"],
        method=raw["method"],
        outer_fold=raw["outer_fold"],
        seed=raw["seed"],
        adjacencies=edges,
    )
