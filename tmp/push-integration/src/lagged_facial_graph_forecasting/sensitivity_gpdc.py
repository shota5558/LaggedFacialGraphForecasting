"""Synthetic-only pre-implementation adapter for PCMCI+ + GPDC (S-IMPL-02).

This module changes only the conditional-independence test relative to Primary.
PCMCI+ itself remains Tigramite OSS, the frozen h=1/tau/pc_alpha discovery options
are reused, and discovery input remains protected by the canonical outer-train
leakage guard. Real execution is subject to the Sensitivity execution barrier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tigramite.independence_tests.gpdc import GPDC
from tigramite.pcmci import PCMCI

from .contracts import SplitManifest
from .leakage_guard import assert_discovery_fit_scope
from .pcmci_contemporaneous import apply_primary_contemporaneous_policy
from .pcmci_lagged_filter import filter_primary_lagged_links
from .pcmci_link_extraction import extract_significant_pcmciplus_links
from .pcmci_parent_set import convert_lagged_links_to_parent_set
from .pcmci_runner import primary_run_pcmciplus_kwargs
from .core_contracts import ParentSet
from .sensitivity_execution import (
    SensitivityExperimentConfig,
    assert_sensitivity_execution_allowed,
)
from .tigramite_adapter import TigramiteDataFrameBundle


SENSITIVITY_GPDC_DISCOVERY_METHOD = "pcmci_plus_gpdc"
SENSITIVITY_GPDC_MASK_TYPE = "xyz"
SENSITIVITY_GPDC_SIGNIFICANCE = "analytic"


class SensitivityGPDCError(ValueError):
    """Raised when the GPDC Sensitivity adapter contract is violated."""


@dataclass(frozen=True, slots=True)
class GPDCDiscoveryProvenance:
    """Minimal deterministic provenance for one GPDC discovery invocation."""

    seed: int
    discovery_method: str = SENSITIVITY_GPDC_DISCOVERY_METHOD
    ci_test: str = "gpdc"

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise SensitivityGPDCError("seed must be a non-negative integer")
        if self.discovery_method != SENSITIVITY_GPDC_DISCOVERY_METHOD:
            raise SensitivityGPDCError("GPDC discovery method provenance is immutable")
        if self.ci_test != "gpdc":
            raise SensitivityGPDCError("GPDC ci_test provenance is immutable")


def make_sensitivity_gpdc(*, seed: int) -> GPDC:
    """Construct Tigramite's GPDC with explicit deterministic RNG and mask policy."""

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise SensitivityGPDCError("seed must be a non-negative integer")
    return GPDC(
        seed=seed,
        significance=SENSITIVITY_GPDC_SIGNIFICANCE,
        mask_type=SENSITIVITY_GPDC_MASK_TYPE,
        recycle_residuals=False,
        confidence=None,
        verbosity=0,
    )


def make_sensitivity_pcmci_plus_gpdc(
    bundle: TigramiteDataFrameBundle,
    *,
    seed: int,
) -> PCMCI:
    """Bind the canonical outer-train Tigramite DataFrame to OSS GPDC + PCMCI+."""

    if not isinstance(bundle, TigramiteDataFrameBundle):
        raise SensitivityGPDCError("bundle must be a TigramiteDataFrameBundle")
    return PCMCI(
        dataframe=bundle.dataframe,
        cond_ind_test=make_sensitivity_gpdc(seed=seed),
        verbosity=0,
    )


def run_sensitivity_pcmciplus_gpdc(
    config: SensitivityExperimentConfig,
    manifest: SplitManifest,
    bundle: TigramiteDataFrameBundle,
    *,
    seed: int,
) -> tuple[dict[str, Any], GPDCDiscoveryProvenance]:
    """Run the GPDC Sensitivity discovery path under the execution and leakage guards."""

    if not isinstance(bundle, TigramiteDataFrameBundle):
        raise SensitivityGPDCError("bundle must be a TigramiteDataFrameBundle")
    assert_sensitivity_execution_allowed(config)
    assert_discovery_fit_scope(manifest, bundle.subject_ids)

    pcmci = make_sensitivity_pcmci_plus_gpdc(bundle, seed=seed)
    result = pcmci.run_pcmciplus(**dict(primary_run_pcmciplus_kwargs()))
    if not isinstance(result, dict):
        raise SensitivityGPDCError("Tigramite run_pcmciplus must return a result dictionary")
    return result, GPDCDiscoveryProvenance(seed=seed)


def gpdc_result_to_parent_set(
    result: dict[str, Any],
    bundle: TigramiteDataFrameBundle,
    *,
    outer_fold: int,
    target_region: str,
) -> ParentSet:
    """Reuse the frozen PCMCI+ link pipeline and emit canonical GPDC ParentSet."""

    raw_links = extract_significant_pcmciplus_links(result, bundle)
    disposition = apply_primary_contemporaneous_policy(raw_links)
    lagged_links = filter_primary_lagged_links(disposition)
    return convert_lagged_links_to_parent_set(
        lagged_links,
        outer_fold=outer_fold,
        target_region=target_region,
        discovery_method=SENSITIVITY_GPDC_DISCOVERY_METHOD,
    )
