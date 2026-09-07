"""Primary coarse time-shuffle Null utilities (F-40/F-41).

F-40 freezes only the scientific mapping rule: keep the exact PCMCI-selected
cross-region ParentLinks and later destroy their temporal row correspondence. It does
not inspect outer-test row counts or values and does not realize a permutation yet.
F-41 generates a deterministic non-identity row permutation from only the frozen seed
and an explicit row count. F-42 owns deterministic application.
"""

from __future__ import annotations

import numpy as np

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


def generate_time_shuffle_permutation(
    mapping: NullMapping,
    *,
    row_count: int,
) -> tuple[int, ...]:
    """Generate one deterministic non-identity permutation for a row support.

    Only ``mapping.seed`` and ``row_count`` influence the result. No predictor values,
    target values, metrics, or outer-test outcomes are accepted by this API. The input
    must be the unrealized F-40 time-shuffle rule. For two or more rows, identity is
    rejected and redrawn so the coarse Null always destroys temporal correspondence.
    Redrawing conditions uniformly on the set of non-identity permutations rather than
    mapping an identity draw to a special fixed permutation.
    """

    if not isinstance(mapping, NullMapping):
        raise TypeError("mapping must be a NullMapping")
    if mapping.condition != "time-shuffle":
        raise TimeShuffleMappingError(
            "permutation generation requires condition='time-shuffle'"
        )
    if mapping.source_parents != mapping.mapped_parents:
        raise TimeShuffleMappingError(
            "time-shuffle must preserve ParentLink identity before row permutation"
        )
    if mapping.permutation:
        raise TimeShuffleMappingError(
            "F-41 expects an unrealized time-shuffle mapping with empty permutation"
        )
    if (
        not isinstance(row_count, (int, np.integer))
        or isinstance(row_count, (bool, np.bool_))
        or int(row_count) < 2
    ):
        raise TimeShuffleMappingError("row_count must be an integer >= 2")

    row_count = int(row_count)
    identity = np.arange(row_count, dtype=int)
    rng = np.random.default_rng(mapping.seed)
    while True:
        permutation = rng.permutation(row_count)
        if not np.array_equal(permutation, identity):
            return tuple(int(index) for index in permutation)
