"""Primary random-region falsification mapping (F-20).

Random-region changes source-region identity while keeping the canonical
component-level ParentLink representation. Candidate regions are supplied from an
already-frozen region definition; this module never infers candidates from
outer-test values or prediction performance.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .contracts import SplitManifest
from .core_contracts import NullMapping, ParentLink, ParentSet
from .leakage_guard import assert_null_construction_scope


class RandomRegionMappingError(ValueError):
    """Raised when a random-region mapping cannot satisfy the Primary contract."""


def _normalize_candidate_regions(candidate_regions: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(candidate_regions)
    if not normalized:
        raise RandomRegionMappingError("candidate_regions must not be empty")
    if any(not isinstance(region, str) or not region.strip() for region in normalized):
        raise RandomRegionMappingError("candidate regions must be non-empty strings")
    normalized = tuple(region.strip() for region in normalized)
    if len(set(normalized)) != len(normalized):
        raise RandomRegionMappingError("candidate_regions must be unique")
    return normalized


def construct_random_region_mapping(
    manifest: SplitManifest,
    parent_set: ParentSet,
    *,
    construction_subject_ids: tuple[str, ...],
    candidate_regions: Sequence[str],
) -> NullMapping:
    """Construct one deterministic random-region control from outer-train only.

    The RNG is seeded from the SplitManifest. Target-region self history is not a
    valid replacement because Primary PCMCI forecasting already contains the fixed
    Self block. The source region itself is also excluded. If multiple source links
    could collapse onto the same component-level ParentLink, the sampler selects a
    different eligible region rather than silently deduplicating features.
    """

    assert_null_construction_scope(manifest, construction_subject_ids)
    if not isinstance(parent_set, ParentSet):
        raise TypeError("parent_set must be a ParentSet")
    if parent_set.outer_fold != manifest.outer_fold:
        raise RandomRegionMappingError("ParentSet outer_fold must match SplitManifest")

    candidates = _normalize_candidate_regions(candidate_regions)
    rng = np.random.default_rng(manifest.seed)
    mapped: list[ParentLink] = []
    used: set[ParentLink] = set()

    for parent in parent_set.parents:
        eligible: list[ParentLink] = []
        for region in candidates:
            if region == parent.source_region or region == parent_set.target_region:
                continue
            proposal = ParentLink(
                source_region=region,
                lag=parent.lag,
                source_dimension=parent.source_dimension,
                target_dimension=parent.target_dimension,
            )
            if proposal in used:
                continue
            eligible.append(proposal)

        if not eligible:
            raise RandomRegionMappingError(
                "no eligible replacement region preserves the random-region contract "
                f"for parent {parent!r}"
            )

        selected = eligible[int(rng.integers(0, len(eligible)))]
        mapped.append(selected)
        used.add(selected)

    return NullMapping(
        outer_fold=manifest.outer_fold,
        target_region=parent_set.target_region,
        condition="random-region",
        seed=manifest.seed,
        source_parents=parent_set.parents,
        mapped_parents=tuple(mapped),
    )
