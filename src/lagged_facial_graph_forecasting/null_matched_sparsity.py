"""Primary matched-sparsity Null utilities (F-30/F-31/F-32).

F-30 freezes already-selected equal-cardinality ParentLinks into the canonical
``NullMapping`` boundary. F-31 defines the feasible cross-region candidate universe.
F-32 draws one deterministic, without-replacement random sparse subset with the same
feature count per target component as the PCMCI ParentSet. Repeated samples remain
reserved for F-33.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from .contracts import SplitManifest
from .core_contracts import NullMapping, ParentLink, ParentSet
from .leakage_guard import assert_null_construction_scope
from .pcmci_lagged_filter import PRIMARY_LAG_MIN
from .pcmci_tau_max import PRIMARY_TAU_MAX


class MatchedSparsityMappingError(ValueError):
    """Raised when a matched-sparsity mapping violates its frozen contract."""


def _normalize_ids(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    normalized = tuple(values)
    if not normalized:
        raise MatchedSparsityMappingError(f"{field_name} must not be empty")
    if any(not isinstance(value, str) or not value.strip() for value in normalized):
        raise MatchedSparsityMappingError(
            f"{field_name} entries must be non-empty strings"
        )
    if len(set(normalized)) != len(normalized):
        raise MatchedSparsityMappingError(f"{field_name} entries must be unique")
    return normalized


def build_matched_sparsity_candidate_space(
    manifest: SplitManifest,
    parent_set: ParentSet,
    *,
    construction_subject_ids: tuple[str, ...],
    region_ids: Sequence[str],
    dimension_ids: Sequence[str],
) -> tuple[ParentLink, ...]:
    """Build the frozen feasible universe for Primary matched-sparsity sampling.

    The candidate universe mirrors the *cross-region* feature block available to the
    Primary PCMCI forecasting condition: every non-target region, every scalar source
    component, and every directly observable Primary lag ``1..tau_max``. The target
    component strata are taken from the canonical PCMCI ParentSet so F-32 can preserve
    the selected feature count separately for each scalar prediction target.

    Existing PCMCI-selected links are intentionally *not* excluded. Matched sparsity
    is a random sparse-subset control from the same feasible feature universe, not an
    adversarial "anything except PCMCI" control. This function performs no sampling.
    """

    assert_null_construction_scope(manifest, construction_subject_ids)
    if not isinstance(parent_set, ParentSet):
        raise TypeError("parent_set must be a ParentSet")
    if parent_set.outer_fold != manifest.outer_fold:
        raise MatchedSparsityMappingError(
            "ParentSet outer_fold must match SplitManifest"
        )

    regions = _normalize_ids(region_ids, field_name="region_ids")
    dimensions = _normalize_ids(dimension_ids, field_name="dimension_ids")
    if parent_set.target_region not in regions:
        raise MatchedSparsityMappingError(
            "target_region must be present in region_ids"
        )

    target_dimensions = tuple(
        sorted({parent.target_dimension for parent in parent_set.parents})
    )
    if not target_dimensions:
        return ()
    unknown_targets = set(target_dimensions) - set(dimensions)
    if unknown_targets:
        raise MatchedSparsityMappingError(
            "ParentSet target dimensions must be present in dimension_ids: "
            f"{sorted(unknown_targets)}"
        )

    candidates = tuple(
        sorted(
            ParentLink(
                source_region=source_region,
                lag=lag,
                source_dimension=source_dimension,
                target_dimension=target_dimension,
            )
            for target_dimension in target_dimensions
            for source_region in regions
            if source_region != parent_set.target_region
            for source_dimension in dimensions
            for lag in range(PRIMARY_LAG_MIN, PRIMARY_TAU_MAX + 1)
        )
    )
    missing_selected = tuple(sorted(set(parent_set.parents) - set(candidates)))
    if missing_selected:
        raise MatchedSparsityMappingError(
            "candidate universe must contain every selected ParentLink; "
            "check frozen region_ids, dimension_ids, cross-region semantics, and Primary lag domain"
        )
    return candidates


def _sample_equal_size_by_target_dimension(
    parent_set: ParentSet,
    candidates: Sequence[ParentLink],
    *,
    rng: np.random.Generator,
) -> tuple[ParentLink, ...]:
    """Sample exact per-target-component counts from canonical candidate strata."""

    normalized_candidates = tuple(candidates)
    if any(not isinstance(parent, ParentLink) for parent in normalized_candidates):
        raise MatchedSparsityMappingError(
            "candidate entries must be ParentLink instances"
        )
    if len(set(normalized_candidates)) != len(normalized_candidates):
        raise MatchedSparsityMappingError("candidates must not contain duplicates")

    required_counts = Counter(parent.target_dimension for parent in parent_set.parents)
    if not required_counts:
        return ()

    unexpected_targets = {
        parent.target_dimension for parent in normalized_candidates
    } - set(required_counts)
    if unexpected_targets:
        raise MatchedSparsityMappingError(
            "candidates contain target dimensions absent from PCMCI ParentSet: "
            f"{sorted(unexpected_targets)}"
        )

    sampled: list[ParentLink] = []
    for target_dimension in sorted(required_counts):
        required = required_counts[target_dimension]
        stratum = tuple(
            sorted(
                parent
                for parent in normalized_candidates
                if parent.target_dimension == target_dimension
            )
        )
        if len(stratum) < required:
            raise MatchedSparsityMappingError(
                "matched-sparsity candidate stratum is smaller than required PCMCI "
                f"feature count for target_dimension={target_dimension!r}: "
                f"required={required}, available={len(stratum)}"
            )
        if required == 0:
            continue
        indices = np.sort(rng.choice(len(stratum), size=required, replace=False))
        sampled.extend(stratum[int(index)] for index in indices)

    return tuple(sorted(sampled))


def sample_matched_sparsity_parents(
    manifest: SplitManifest,
    parent_set: ParentSet,
    *,
    construction_subject_ids: tuple[str, ...],
    candidates: Sequence[ParentLink],
) -> tuple[ParentLink, ...]:
    """Draw one exact-size Primary matched-sparsity control without replacement.

    Sampling uses only the frozen ``SplitManifest.seed`` and a canonicalized candidate
    ordering. Equal size is enforced separately for each target component because the
    D=2 Primary forecasts scalar target components independently. F-33 is responsible
    for generating multiple distinct repeated samples; this function returns one.
    """

    assert_null_construction_scope(manifest, construction_subject_ids)
    if not isinstance(parent_set, ParentSet):
        raise TypeError("parent_set must be a ParentSet")
    if parent_set.outer_fold != manifest.outer_fold:
        raise MatchedSparsityMappingError(
            "ParentSet outer_fold must match SplitManifest"
        )

    return _sample_equal_size_by_target_dimension(
        parent_set,
        candidates,
        rng=np.random.default_rng(manifest.seed),
    )


def construct_matched_sparsity_mapping(
    manifest: SplitManifest,
    parent_set: ParentSet,
    *,
    construction_subject_ids: tuple[str, ...],
    sampled_parents: Sequence[ParentLink],
) -> NullMapping:
    """Freeze one preselected equal-size matched-sparsity control.

    The caller must supply already-selected ParentLinks. This function performs no
    candidate discovery or sampling and therefore cannot inspect outer-test outcomes.
    """

    assert_null_construction_scope(manifest, construction_subject_ids)
    if not isinstance(parent_set, ParentSet):
        raise TypeError("parent_set must be a ParentSet")
    if parent_set.outer_fold != manifest.outer_fold:
        raise MatchedSparsityMappingError("ParentSet outer_fold must match SplitManifest")

    sampled = tuple(sampled_parents)
    if any(not isinstance(parent, ParentLink) for parent in sampled):
        raise MatchedSparsityMappingError("sampled_parents entries must be ParentLink instances")
    if len(sampled) != len(parent_set.parents):
        raise MatchedSparsityMappingError(
            "matched-sparsity sampled_parents must exactly match PCMCI ParentSet cardinality"
        )
    if len(set(sampled)) != len(sampled):
        raise MatchedSparsityMappingError("sampled_parents must not contain duplicates")

    source_counts = Counter(parent.target_dimension for parent in parent_set.parents)
    sampled_counts = Counter(parent.target_dimension for parent in sampled)
    if sampled_counts != source_counts:
        raise MatchedSparsityMappingError(
            "matched-sparsity sampled_parents must preserve PCMCI feature count per "
            "target_dimension"
        )

    return NullMapping(
        outer_fold=manifest.outer_fold,
        target_region=parent_set.target_region,
        condition="matched-sparsity",
        seed=manifest.seed,
        source_parents=parent_set.parents,
        mapped_parents=sampled,
    )
