"""Outer-train-only PCMCI+ bootstrap discovery wrapper (D-12).

Bootstrap mechanics are delegated to Tigramite ``PCMCI.run_bootstrap_of``.  D-12
only validates scope/reproducibility inputs and injects the already-frozen Primary
``run_pcmciplus`` arguments.  Selection-frequency aggregation is intentionally left
to D-13.
"""

from __future__ import annotations

from typing import Any

from .contracts import SplitManifest
from .leakage_guard import assert_discovery_fit_scope
from .pcmci_plus_factory import make_primary_pcmci_plus
from .pcmci_runner import primary_run_pcmciplus_kwargs
from .tigramite_adapter import TigramiteDataFrameBundle


class PCMCIPlusBootstrapError(ValueError):
    """Raised when the D-12 bootstrap execution contract is violated."""


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PCMCIPlusBootstrapError(f"{name} must be a positive integer")
    return value


def _seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PCMCIPlusBootstrapError("seed must be a non-negative integer")
    return value


def run_primary_pcmciplus_bootstrap(
    manifest: SplitManifest,
    bundle: TigramiteDataFrameBundle,
    *,
    boot_samples: int,
    boot_blocklength: int,
    seed: int,
) -> dict[str, Any]:
    """Run Tigramite block-bootstrap PCMCI+ on outer-train data only.

    ``boot_samples``, ``boot_blocklength``, and ``seed`` are explicit execution
    inputs because the research plan requires bootstrap stability analysis but does
    not yet freeze a single study-level resampling count/block length.  They are
    recorded by callers/artifacts and cannot be inferred from outer-test outcomes.

    D-13 must consume ``boot_results`` directly when computing selection frequency;
    Tigramite's bootstrap summary/majority aggregation is not used to redefine the
    Primary ParentSet.
    """

    if not isinstance(bundle, TigramiteDataFrameBundle):
        raise PCMCIPlusBootstrapError("bundle must be a TigramiteDataFrameBundle")

    boot_samples = _positive_int(boot_samples, name="boot_samples")
    boot_blocklength = _positive_int(boot_blocklength, name="boot_blocklength")
    seed = _seed(seed)

    # Re-check at the actual discovery boundary so an altered/manually constructed
    # bundle cannot bypass the outer-test lock enforced by the converter.
    assert_discovery_fit_scope(manifest, bundle.subject_ids)

    pcmci = make_primary_pcmci_plus(bundle)
    result = pcmci.run_bootstrap_of(
        method="run_pcmciplus",
        method_args=dict(primary_run_pcmciplus_kwargs()),
        boot_samples=boot_samples,
        boot_blocklength=boot_blocklength,
        seed=seed,
    )
    if not isinstance(result, dict):
        raise PCMCIPlusBootstrapError(
            "Tigramite run_bootstrap_of must return a result dictionary"
        )
    if set(("summary_results", "boot_results")) - set(result):
        raise PCMCIPlusBootstrapError(
            "Tigramite bootstrap result must contain summary_results and boot_results"
        )
    if not isinstance(result["boot_results"], dict):
        raise PCMCIPlusBootstrapError("boot_results must be a dictionary")
    return result
