"""Raw significant-link extraction from Tigramite PCMCI+ results (D-07).

D-07 deliberately does not decide whether contemporaneous links are forecastable
and does not construct the canonical ParentSet.  It only converts non-empty entries
of Tigramite's authoritative PCMCI+ ``graph`` output into deterministic, provenance-
carrying records for D-08/D-09/D-10.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .pcmci_tau_max import PRIMARY_TAU_MAX
from .tigramite_adapter import TigramiteDataFrameBundle


class PCMCIPlusLinkExtractionError(ValueError):
    """Raised when a Tigramite result cannot satisfy the frozen D-07 contract."""


@dataclass(frozen=True, slots=True)
class SignificantPCMCIPlusLink:
    """One retained raw PCMCI+ graph entry with exact component provenance."""

    source_node_index: int
    target_node_index: int
    lag: int
    link_mark: str
    p_value: float
    test_statistic: float
    source_variable_name: str
    target_variable_name: str
    source_region: str
    source_dimension: str
    target_region: str
    target_dimension: str


def _result_matrix(result: dict[str, Any], key: str) -> np.ndarray:
    if key not in result:
        raise PCMCIPlusLinkExtractionError(f"PCMCI+ result is missing {key!r}")
    return np.asarray(result[key])


def extract_significant_pcmciplus_links(
    result: dict[str, Any],
    bundle: TigramiteDataFrameBundle,
) -> tuple[SignificantPCMCIPlusLink, ...]:
    """Extract all non-empty Tigramite ``graph`` entries without re-thresholding.

    Tigramite PCMCI+ encodes retained adjacencies/orientations in ``graph``. D-07
    therefore treats ``graph != ''`` as authoritative and carries ``p_matrix`` and
    ``val_matrix`` only as diagnostics. A small p-value with an empty graph entry is
    *not* independently promoted to a link here.

    Lag zero is intentionally preserved. Its forecasting disposition belongs to D-08,
    while D-09 owns the lagged-only filtering policy.
    """

    if not isinstance(result, dict):
        raise PCMCIPlusLinkExtractionError("result must be the Tigramite result dictionary")
    if not isinstance(bundle, TigramiteDataFrameBundle):
        raise PCMCIPlusLinkExtractionError("bundle must be a TigramiteDataFrameBundle")

    graph = _result_matrix(result, "graph")
    p_matrix = _result_matrix(result, "p_matrix")
    val_matrix = _result_matrix(result, "val_matrix")

    node_count = len(bundle.variable_names)
    expected_shape = (node_count, node_count, PRIMARY_TAU_MAX + 1)
    for key, matrix in (
        ("graph", graph),
        ("p_matrix", p_matrix),
        ("val_matrix", val_matrix),
    ):
        if matrix.shape != expected_shape:
            raise PCMCIPlusLinkExtractionError(
                f"{key} shape must be {expected_shape}; got {matrix.shape}"
            )

    if len(bundle.variable_components) != node_count:
        raise PCMCIPlusLinkExtractionError(
            "variable_components length must match variable_names length"
        )

    links: list[SignificantPCMCIPlusLink] = []
    for source_node in range(node_count):
        source_region, source_dimension = bundle.variable_components[source_node]
        for target_node in range(node_count):
            target_region, target_dimension = bundle.variable_components[target_node]
            for lag in range(PRIMARY_TAU_MAX + 1):
                raw_mark = graph[source_node, target_node, lag]
                if not isinstance(raw_mark, (str, np.str_)):
                    raise PCMCIPlusLinkExtractionError(
                        "graph entries must be Tigramite link-mark strings"
                    )
                link_mark = str(raw_mark)
                if link_mark == "":
                    continue

                p_value = float(p_matrix[source_node, target_node, lag])
                test_statistic = float(val_matrix[source_node, target_node, lag])
                if not np.isfinite(p_value) or not np.isfinite(test_statistic):
                    raise PCMCIPlusLinkExtractionError(
                        "retained graph entries require finite p-value and test statistic"
                    )

                links.append(
                    SignificantPCMCIPlusLink(
                        source_node_index=source_node,
                        target_node_index=target_node,
                        lag=lag,
                        link_mark=link_mark,
                        p_value=p_value,
                        test_statistic=test_statistic,
                        source_variable_name=bundle.variable_names[source_node],
                        target_variable_name=bundle.variable_names[target_node],
                        source_region=source_region,
                        source_dimension=source_dimension,
                        target_region=target_region,
                        target_dimension=target_dimension,
                    )
                )

    links.sort(
        key=lambda link: (
            link.target_node_index,
            link.lag,
            link.source_node_index,
            link.link_mark,
        )
    )
    return tuple(links)
