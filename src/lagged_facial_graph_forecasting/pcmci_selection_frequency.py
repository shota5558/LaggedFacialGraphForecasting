"""Component-level PCMCI+ bootstrap selection frequency aggregation (D-13).

D-13 is descriptive stability analysis only.  It consumes Tigramite's raw
``boot_results['graph']`` produced by D-12 and never feeds frequencies back into
the frozen Primary ParentSet or discovery configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .pcmci_lagged_filter import PRIMARY_LAGGED_LINK_MARK, PRIMARY_LAG_MIN
from .pcmci_tau_max import PRIMARY_TAU_MAX
from .tigramite_adapter import TigramiteDataFrameBundle


class SelectionFrequencyError(ValueError):
    """Raised when bootstrap graph results violate the D-13 contract."""


@dataclass(frozen=True, slots=True)
class PCMCISelectionFrequency:
    """Selection frequency for one exact source-component → target-component lag."""

    source_node_index: int
    target_node_index: int
    lag: int
    selected_count: int
    bootstrap_count: int
    frequency: float
    source_variable_name: str
    target_variable_name: str
    source_region: str
    source_dimension: str
    target_region: str
    target_dimension: str

    def __post_init__(self) -> None:
        if self.lag < PRIMARY_LAG_MIN or self.lag > PRIMARY_TAU_MAX:
            raise SelectionFrequencyError(
                f"lag must satisfy {PRIMARY_LAG_MIN} <= tau <= {PRIMARY_TAU_MAX}"
            )
        if self.bootstrap_count <= 0:
            raise SelectionFrequencyError("bootstrap_count must be positive")
        if self.selected_count < 0 or self.selected_count > self.bootstrap_count:
            raise SelectionFrequencyError(
                "selected_count must lie between zero and bootstrap_count"
            )
        expected = self.selected_count / self.bootstrap_count
        if not np.isclose(self.frequency, expected, rtol=0.0, atol=1e-15):
            raise SelectionFrequencyError(
                "frequency must equal selected_count / bootstrap_count"
            )

    @property
    def is_interregional(self) -> bool:
        return self.source_region != self.target_region


def aggregate_primary_selection_frequency(
    bootstrap_result: dict[str, Any],
    bundle: TigramiteDataFrameBundle,
) -> tuple[PCMCISelectionFrequency, ...]:
    """Aggregate exact component/lag selection frequency from raw bootstrap graphs.

    All possible lagged component pairs for ``tau=1..10`` are returned, including
    zero-frequency entries, so absence is explicit rather than encoded by a missing
    row.  Contemporaneous ``tau=0`` entries are intentionally outside this Primary
    direct-h=1 stability contract.  At positive lags Tigramite PCMCI+ must encode a
    retained directed link as ``'-->'``; any other non-empty mark fails closed.

    ``summary_results`` is deliberately ignored.  D-13 therefore cannot turn
    Tigramite's majority aggregation into a new selection rule for Primary.
    """

    if not isinstance(bootstrap_result, dict):
        raise SelectionFrequencyError("bootstrap_result must be a dictionary")
    if not isinstance(bundle, TigramiteDataFrameBundle):
        raise SelectionFrequencyError("bundle must be a TigramiteDataFrameBundle")

    boot_results = bootstrap_result.get("boot_results")
    if not isinstance(boot_results, dict):
        raise SelectionFrequencyError("bootstrap_result.boot_results must be a dictionary")
    if "graph" not in boot_results:
        raise SelectionFrequencyError("bootstrap_result.boot_results is missing 'graph'")

    graph = np.asarray(boot_results["graph"])
    node_count = len(bundle.variable_names)
    if len(bundle.variable_components) != node_count:
        raise SelectionFrequencyError(
            "variable_components length must match variable_names length"
        )
    if graph.ndim != 4:
        raise SelectionFrequencyError(
            "bootstrap graph must have shape (B, N, N, tau_max + 1)"
        )
    bootstrap_count = graph.shape[0]
    expected_shape = (bootstrap_count, node_count, node_count, PRIMARY_TAU_MAX + 1)
    if bootstrap_count <= 0 or graph.shape != expected_shape:
        raise SelectionFrequencyError(
            f"bootstrap graph shape must be {expected_shape} with B > 0; got {graph.shape}"
        )

    rows: list[PCMCISelectionFrequency] = []
    for target_node in range(node_count):
        target_region, target_dimension = bundle.variable_components[target_node]
        for lag in range(PRIMARY_LAG_MIN, PRIMARY_TAU_MAX + 1):
            for source_node in range(node_count):
                source_region, source_dimension = bundle.variable_components[source_node]
                marks = graph[:, source_node, target_node, lag]
                normalized: list[str] = []
                for raw_mark in marks:
                    if not isinstance(raw_mark, (str, np.str_)):
                        raise SelectionFrequencyError(
                            "bootstrap graph entries must be Tigramite link-mark strings"
                        )
                    mark = str(raw_mark)
                    if mark not in ("", PRIMARY_LAGGED_LINK_MARK):
                        raise SelectionFrequencyError(
                            "positive-lag bootstrap graph entries must be empty or '-->'; "
                            f"got {mark!r} at source={source_node}, target={target_node}, tau={lag}"
                        )
                    normalized.append(mark)

                selected_count = sum(
                    mark == PRIMARY_LAGGED_LINK_MARK for mark in normalized
                )
                rows.append(
                    PCMCISelectionFrequency(
                        source_node_index=source_node,
                        target_node_index=target_node,
                        lag=lag,
                        selected_count=selected_count,
                        bootstrap_count=bootstrap_count,
                        frequency=selected_count / bootstrap_count,
                        source_variable_name=bundle.variable_names[source_node],
                        target_variable_name=bundle.variable_names[target_node],
                        source_region=source_region,
                        source_dimension=source_dimension,
                        target_region=target_region,
                        target_dimension=target_dimension,
                    )
                )

    return tuple(rows)
