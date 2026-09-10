"""Primary coarse time-shuffle Null utilities (F-40/F-41/F-42).

F-40 freezes the scientific rule: keep the exact PCMCI-selected cross-region
ParentLinks and later destroy their temporal row correspondence. F-41 generates a
deterministic non-identity row permutation from only the frozen seed and an explicit
row count. F-42 applies that permutation only to the additional PCMCI cross-region
feature block on the valid evaluation support, leaving Self-history, targets, row
provenance, and missingness support unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .contracts import DesignMatrix, SplitManifest
from .core_contracts import NullMapping, ParentSet
from .design_matrix import decode_feature_name, encode_feature_name
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
    the evaluation DesignMatrix valid-row support, which is unavailable until
    evaluation. The frozen SplitManifest seed records the stochastic provenance
    required by the later generator.
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


def _validate_unrealized_time_shuffle_mapping(mapping: NullMapping) -> None:
    if not isinstance(mapping, NullMapping):
        raise TypeError("mapping must be a NullMapping")
    if mapping.condition != "time-shuffle":
        raise TimeShuffleMappingError(
            "time-shuffle operation requires condition='time-shuffle'"
        )
    if mapping.source_parents != mapping.mapped_parents:
        raise TimeShuffleMappingError(
            "time-shuffle must preserve ParentLink identity before row permutation"
        )
    if mapping.permutation:
        raise TimeShuffleMappingError(
            "time-shuffle operation expects the frozen unrealized mapping rule"
        )


def _normalize_permutation(
    permutation: Sequence[int], *, expected_length: int
) -> tuple[int, ...]:
    normalized = tuple(permutation)
    if expected_length < 2:
        raise TimeShuffleMappingError(
            "time-shuffle requires at least two valid evaluation rows"
        )
    if len(normalized) != expected_length:
        raise TimeShuffleMappingError(
            "permutation length must equal the number of valid evaluation rows"
        )
    if any(
        not isinstance(index, (int, np.integer))
        or isinstance(index, (bool, np.bool_))
        for index in normalized
    ):
        raise TimeShuffleMappingError("permutation entries must be integers")
    normalized = tuple(int(index) for index in normalized)
    if set(normalized) != set(range(expected_length)):
        raise TimeShuffleMappingError(
            "permutation must contain every valid-row index exactly once"
        )
    if normalized == tuple(range(expected_length)):
        raise TimeShuffleMappingError(
            "identity permutation does not implement the time-shuffle falsification"
        )
    return normalized


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

    _validate_unrealized_time_shuffle_mapping(mapping)
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


def apply_time_shuffle_to_design_matrix(
    matrix: DesignMatrix,
    mapping: NullMapping,
    *,
    permutation: Sequence[int] | None = None,
) -> DesignMatrix:
    """Apply the frozen-seed time shuffle to PCMCI-added cross-region features.

    The same F-41 permutation is applied jointly to every selected cross-region
    column. This preserves the selected feature vector's internal configuration while
    breaking its temporal correspondence with the target and fixed Self-history block.
    Invalid rows are not used as permutation donors or recipients, so missingness
    support and the compared sample set remain identical across Self/PCMCI/time-shuffle
    conditions.

    ``permutation`` is optional and exists only for explicit replay/audit. When given,
    it must exactly equal the deterministic F-41 permutation derived from the frozen
    ``mapping.seed`` and current valid-row count. This prevents callers from bypassing
    the frozen stochastic rule with an arbitrary non-identity permutation.

    This function is intentionally single-subject and scalar-target. Canonical PCMCI
    design matrices are built that way, and permitting cross-subject shuffling would
    violate subject independence rather than merely falsify temporal alignment.
    """

    if not isinstance(matrix, DesignMatrix):
        raise TypeError("matrix must be a DesignMatrix")
    _validate_unrealized_time_shuffle_mapping(mapping)

    subjects = set(matrix.subject_id)
    if len(subjects) != 1:
        raise TimeShuffleMappingError(
            "time-shuffle must be applied within one subject at a time"
        )
    regions = set(matrix.region_id)
    if regions != {mapping.target_region}:
        raise TimeShuffleMappingError(
            "DesignMatrix region_id must match the time-shuffle target_region"
        )
    if len(matrix.target_dimensions) != 1:
        raise TimeShuffleMappingError(
            "time-shuffle expects a scalar target-component PCMCI DesignMatrix"
        )

    valid_rows = np.flatnonzero(matrix.valid_mask)
    expected_permutation = generate_time_shuffle_permutation(
        mapping, row_count=len(valid_rows)
    )
    if permutation is None:
        normalized_permutation = expected_permutation
    else:
        normalized_permutation = _normalize_permutation(
            permutation, expected_length=len(valid_rows)
        )
        if normalized_permutation != expected_permutation:
            raise TimeShuffleMappingError(
                "explicit permutation must equal the frozen-seed F-41 permutation"
            )

    target_dimension = matrix.target_dimensions[0]
    expected_cross_keys = {
        (
            encode_feature_name(parent.source_region, parent.source_dimension),
            parent.lag,
        )
        for parent in mapping.mapped_parents
        if parent.source_region != mapping.target_region
        and parent.target_dimension == target_dimension
    }
    if not expected_cross_keys:
        raise TimeShuffleMappingError(
            "time-shuffle is unevaluable without cross-region ParentLinks for the "
            f"target_dimension={target_dimension!r}"
        )

    cross_columns: list[int] = []
    actual_cross_keys: set[tuple[str, int]] = set()
    for column, (feature_name, lag) in enumerate(
        zip(matrix.feature_names, matrix.feature_lags, strict=True)
    ):
        source_region, _ = decode_feature_name(feature_name)
        if source_region == mapping.target_region:
            continue
        cross_columns.append(column)
        actual_cross_keys.add((feature_name, lag))

    if actual_cross_keys != expected_cross_keys:
        missing = sorted(expected_cross_keys - actual_cross_keys)
        unexpected = sorted(actual_cross_keys - expected_cross_keys)
        raise TimeShuffleMappingError(
            "DesignMatrix cross-region provenance must exactly match the mapped "
            f"ParentLinks for target_dimension={target_dimension!r}; "
            f"missing={missing}, unexpected={unexpected}"
        )

    shuffled_X = np.array(matrix.X, copy=True)
    valid_cross_block = np.array(
        shuffled_X[np.ix_(valid_rows, np.asarray(cross_columns, dtype=int))],
        copy=True,
    )
    shuffled_X[np.ix_(valid_rows, np.asarray(cross_columns, dtype=int))] = (
        valid_cross_block[np.asarray(normalized_permutation, dtype=int), :]
    )

    return DesignMatrix(
        X=shuffled_X,
        y=np.array(matrix.y, copy=True),
        subject_id=matrix.subject_id,
        region_id=matrix.region_id,
        target_dimensions=matrix.target_dimensions,
        forecast_origin=np.array(matrix.forecast_origin, copy=True),
        target_time=np.array(matrix.target_time, copy=True),
        feature_names=matrix.feature_names,
        feature_lags=matrix.feature_lags,
        valid_mask=np.array(matrix.valid_mask, copy=True),
    )
