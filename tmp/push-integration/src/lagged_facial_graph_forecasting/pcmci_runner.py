"""Frozen Primary Tigramite PCMCI+ execution path (D-06).

This module is intentionally thin: it validates discovery scope, constructs the
already-frozen Tigramite PCMCI object, combines the independently frozen D-03,
D-04, and D-05 arguments, and delegates discovery to Tigramite's
``PCMCI.run_pcmciplus`` without reimplementing the algorithm.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

from .contracts import SplitManifest
from .leakage_guard import assert_discovery_fit_scope
from .pcmci_pc_alpha import freeze_primary_pc_alpha
from .pcmci_plus_config import make_primary_pcmci_plus_config
from .pcmci_plus_factory import make_primary_pcmci_plus
from .pcmci_tau_max import freeze_primary_tau_max
from .tigramite_adapter import TigramiteDataFrameBundle


class PCMCIPlusRunError(ValueError):
    """Raised when the frozen Primary PCMCI+ execution contract is violated."""


def primary_run_pcmciplus_kwargs() -> Mapping[str, object]:
    """Return the complete immutable Primary ``run_pcmciplus`` argument set.

    D-03 owns static Tigramite options, D-04 owns ``tau_max``, and D-05 owns
    ``pc_alpha``. Keeping those contracts independent prevents a later runtime
    caller from silently overriding a scientific freeze parameter.
    """

    kwargs = dict(make_primary_pcmci_plus_config().tigramite_kwargs())
    for frozen_kwargs in (
        freeze_primary_tau_max().tigramite_kwargs(),
        freeze_primary_pc_alpha().tigramite_kwargs(),
    ):
        overlap = set(kwargs).intersection(frozen_kwargs)
        if overlap:
            raise PCMCIPlusRunError(
                f"duplicate frozen run_pcmciplus parameters: {sorted(overlap)}"
            )
        kwargs.update(frozen_kwargs)

    return MappingProxyType(kwargs)


def run_primary_pcmciplus(
    manifest: SplitManifest,
    bundle: TigramiteDataFrameBundle,
) -> dict[str, Any]:
    """Run frozen Primary PCMCI+ on an outer-train-only Tigramite bundle.

    The D-01 converter already guards subject scope, but this execution boundary
    deliberately validates it again. Therefore a manually constructed or altered
    bundle cannot be used to bypass the outer-test lock at the discovery call.

    No ``tau_max``, ``pc_alpha``, CI-test, or other discovery option is accepted
    from the caller. The only algorithmic operation here is the OSS call to
    Tigramite ``PCMCI.run_pcmciplus``.
    """

    if not isinstance(bundle, TigramiteDataFrameBundle):
        raise PCMCIPlusRunError("bundle must be a TigramiteDataFrameBundle")

    assert_discovery_fit_scope(manifest, bundle.subject_ids)

    pcmci = make_primary_pcmci_plus(bundle)
    result = pcmci.run_pcmciplus(**dict(primary_run_pcmciplus_kwargs()))
    if not isinstance(result, dict):
        raise PCMCIPlusRunError("Tigramite run_pcmciplus must return a result dictionary")
    return result
