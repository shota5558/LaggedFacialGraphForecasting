"""Primary matched-sparsity Null mapping contract (F-30).

F-30 only freezes already-selected equal-cardinality ParentLinks into the canonical
``NullMapping`` boundary. Candidate-space construction, random equal-size sampling,
and repeated samples are intentionally deferred to F-31, F-32, and F-33.
"""

from __future__ import annotations

from collections.abc import Sequence

from .contracts import SplitManifest
from .core_contracts import NullMapping, ParentLink, ParentSet
from .leakage_guard import assert_null_construction_scope


class MatchedSparsityMappingError(ValueError):
    """Raised when a matched-sparsity mapping violates its frozen contract."""


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
