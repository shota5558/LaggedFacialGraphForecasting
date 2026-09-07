"""Primary matched-sparsity Null utilities (F-30/F-31).

F-30 freezes already-selected equal-cardinality ParentLinks into the canonical
``NullMapping`` boundary. F-31 defines the feasible cross-region candidate universe
for that control. Random equal-size sampling and repeated samples remain deferred to
F-32 and F-33.
"""

from __future__ import annotations

from collections.abc import Sequence

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
    return candidates


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

    return NullMapping(
        outer_fold=manifest.outer_fold,
        target_region=parent_set.target_region,
        condition="matched-sparsity",
        seed=manifest.seed,
        source_parents=parent_set.parents,
        mapped_parents=sampled,
    )
