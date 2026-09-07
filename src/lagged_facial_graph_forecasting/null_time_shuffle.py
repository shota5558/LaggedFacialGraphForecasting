"""Primary coarse time-shuffle Null contract (F-40).

F-40 freezes only the scientific mapping rule: keep the exact PCMCI-selected
cross-region ParentLinks and later destroy their temporal row correspondence.  It does
not inspect outer-test row counts or values and does not realize a permutation yet.
F-41 owns permutation generation and F-42 owns deterministic application.
"""

from __future__ import annotations

from .contracts import SplitManifest
from .core_contracts import NullMapping, ParentSet
from .leakage_guard import assert_null_construction_scope


class TimeShuffleMappingError(ValueError):
    """Raised when a Primary time-shuffle mapping violates its frozen contract."""


def construct_time_shuffle_mapping(
    manifest: SplitManifest,
    parent_set: ParentSet,
    *,
    construction_subject_ids: tuple[str, ...],
) -> NullMapping:
    """Freeze the coarse Primary time-shuffle rule without touching outer-test data.

    Time-shuffle is a temporal falsification of the *additional PCMCI-selected
    cross-region block*. Therefore region/lag/component identities are unchanged here;
    only row correspondence will be permuted downstream. The concrete permutation is
    intentionally empty at this construction boundary because its length depends on
    the evaluation DesignMatrix row support, which is unavailable until evaluation.
    The frozen SplitManifest seed records the stochastic provenance required by the
    later generator.
    """

    assert_null_construction_scope(manifest, construction_subject_ids)
    if not isinstance(parent_set, ParentSet):
        raise TypeError("parent_set must be a ParentSet")
    if parent_set.outer_fold != manifest.outer_fold:
        raise TimeShuffleMappingError(
            "ParentSet outer_fold must match SplitManifest"
        )

    return NullMapping(
        outer_fold=manifest.outer_fold,
        target_region=parent_set.target_region,
        condition="time-shuffle",
        seed=manifest.seed,
        source_parents=parent_set.parents,
        mapped_parents=parent_set.parents,
        permutation=(),
    )
