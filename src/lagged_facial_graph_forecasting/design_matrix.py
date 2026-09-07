"""Design-matrix builders for frozen Primary forecasting conditions."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .alignment import aligned_indices
from .contracts import DesignMatrix, FaceTimeSeries
from .core_contracts import ParentLink, ParentSet


def _escape_feature_component(value: str) -> str:
    return value.replace("\\", "\\\\").replace(".", "\\.")


def encode_feature_name(source_region: str, dimension: str) -> str:
    """Encode source-region and dimension provenance into one reversible name.

    Simple identifiers retain the historical ``region.dimension`` representation.
    Dots and backslashes inside either component are escaped so every valid
    FaceTimeSeries identifier remains losslessly recoverable.
    """

    if not isinstance(source_region, str) or not source_region.strip():
        raise ValueError("source_region must be a non-empty string")
    if not isinstance(dimension, str) or not dimension.strip():
        raise ValueError("dimension must be a non-empty string")
    return f"{_escape_feature_component(source_region)}.{_escape_feature_component(dimension)}"


def decode_feature_name(feature_name: str) -> tuple[str, str]:
    """Recover ``(source_region, dimension)`` from ``encode_feature_name`` output."""

    if not isinstance(feature_name, str) or not feature_name:
        raise ValueError("feature_name must be a non-empty string")

    components: list[str] = []
    current: list[str] = []
    escaped = False
    for character in feature_name:
        if escaped:
            if character not in (".", "\\"):
                raise ValueError("feature_name contains an unsupported escape sequence")
            current.append(character)
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == ".":
            components.append("".join(current))
            current = []
            continue
        current.append(character)

    if escaped:
        raise ValueError("feature_name contains a dangling escape")
    components.append("".join(current))
    if len(components) != 2 or any(not component for component in components):
        raise ValueError("feature_name must encode exactly source_region and dimension")
    return components[0], components[1]


def _normalize_lags(lags: Iterable[int]) -> tuple[int, ...]:
    normalized: list[int] = []
    for lag in lags:
        if not isinstance(lag, (int, np.integer)) or isinstance(lag, (bool, np.bool_)):
            raise ValueError("lags must contain integers")
        normalized.append(int(lag))
    if not normalized:
        raise ValueError("lags must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("lags must be unique")
    if any(lag < 1 for lag in normalized):
        raise ValueError("lags must be positive")
    return tuple(sorted(normalized))


def _resolve_alignment_lag(
    feature_lags: Iterable[int], alignment_lag: int | None
) -> int:
    """Return a common row-support lag without permitting unavailable features.

    ``alignment_lag`` is not a predictor. It only fixes the earliest target row so
    conditions with different feature sets can be evaluated on the exact same
    forecast origins/targets.
    """

    required_lag = max(feature_lags)
    if alignment_lag is None:
        return required_lag
    if (
        not isinstance(alignment_lag, (int, np.integer))
        or isinstance(alignment_lag, (bool, np.bool_))
    ):
        raise ValueError("alignment_lag must be an integer")
    alignment_lag = int(alignment_lag)
    if alignment_lag < required_lag:
        raise ValueError(
            "alignment_lag must be >= every feature lag to preserve valid row support"
        )
    return alignment_lag


def _validate_target_dimension(
    series: FaceTimeSeries, target_dimension: str | None
) -> tuple[int | None, str | None]:
    if target_dimension is None:
        return None, None
    if target_dimension not in series.dimension:
        raise ValueError(f"unknown target_dimension: {target_dimension!r}")
    return series.dimension.index(target_dimension), target_dimension


def _target_values(
    series: FaceTimeSeries,
    target_indices: np.ndarray,
    target_region_index: int,
    target_dimension: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    dimension_index, _ = _validate_target_dimension(series, target_dimension)
    if dimension_index is None:
        return (
            series.X[target_indices, target_region_index, :],
            np.all(series.valid_mask[target_indices, target_region_index, :], axis=1),
        )
    return (
        series.X[target_indices, target_region_index, dimension_index],
        series.valid_mask[target_indices, target_region_index, dimension_index],
    )


def _resolve_link_dimension(
    series: FaceTimeSeries,
    value: str,
    *,
    field_name: str,
) -> str:
    """Resolve legacy ``value`` only when the series is provably scalar.

    Component-aware v2 artifacts must carry real dimension labels for D>1. The legacy
    default exists only for source compatibility with certified scalar fixtures and is
    never interpreted as a component choice in a multicomponent series.
    """

    if value == "value":
        if len(series.dimension) != 1:
            raise ValueError(
                f"{field_name}='value' is ambiguous for multicomponent FaceTimeSeries"
            )
        return series.dimension[0]
    if value not in series.dimension:
        raise ValueError(f"unknown {field_name}: {value!r}")
    return value


def _resolve_pcmci_target_dimension(
    series: FaceTimeSeries,
    parent_set: ParentSet,
    requested: str | None,
) -> str:
    if requested is not None:
        if requested not in series.dimension:
            raise ValueError(f"unknown target_dimension: {requested!r}")
        # Validate every stored target label even though links for other target
        # components are filtered below. This prevents lossy/ambiguous v2 artifacts.
        for parent in parent_set.parents:
            _resolve_link_dimension(
                series, parent.target_dimension, field_name="target_dimension"
            )
        return requested

    if not parent_set.parents:
        if len(series.dimension) == 1:
            return series.dimension[0]
        raise ValueError(
            "target_dimension is required for a multicomponent empty ParentSet"
        )

    target_dimensions = {
        _resolve_link_dimension(
            series, parent.target_dimension, field_name="target_dimension"
        )
        for parent in parent_set.parents
    }
    if len(target_dimensions) != 1:
        raise ValueError(
            "target_dimension is required when ParentSet contains multiple target components"
        )
    return next(iter(target_dimensions))


def build_self_history_design_matrix(
    series: FaceTimeSeries,
    *,
    target_region: str,
    lags: Iterable[int] = (1,),
    horizon: int = 1,
    alignment_lag: int | None = None,
    target_dimension: str | None = None,
) -> DesignMatrix:
    """Build Self history for one target region.

    The predictor set remains the full past vector of the target region. When
    ``target_dimension`` is supplied, only that scalar output component is predicted;
    this is the fair baseline for component-level PCMCI+ discovery.
    """

    normalized_lags = _normalize_lags(lags)
    if target_region not in series.region_id:
        raise ValueError(f"unknown target_region: {target_region!r}")
    _validate_target_dimension(series, target_dimension)

    support_lag = _resolve_alignment_lag(normalized_lags, alignment_lag)
    common = aligned_indices(len(series.time_index), lag=support_lag, horizon=horizon)
    target_indices = common.target_index
    origin_indices = common.forecast_origin
    region_index = series.region_id.index(target_region)

    feature_columns: list[np.ndarray] = []
    feature_valid_columns: list[np.ndarray] = []
    feature_names: list[str] = []
    feature_lags: list[int] = []

    for lag in normalized_lags:
        source_indices = target_indices - lag
        for dimension_index, dimension_name in enumerate(series.dimension):
            feature_columns.append(
                series.X[source_indices, region_index, dimension_index]
            )
            feature_valid_columns.append(
                series.valid_mask[source_indices, region_index, dimension_index]
            )
            feature_names.append(encode_feature_name(target_region, dimension_name))
            feature_lags.append(lag)

    X = np.column_stack(feature_columns)
    feature_valid = np.column_stack(feature_valid_columns)
    y, target_valid = _target_values(
        series, target_indices, region_index, target_dimension
    )
    valid_mask = np.all(feature_valid, axis=1) & target_valid

    row_count = len(target_indices)
    return DesignMatrix(
        X=X,
        y=y,
        subject_id=(series.subject_id,) * row_count,
        region_id=(target_region,) * row_count,
        forecast_origin=series.time_index[origin_indices],
        target_time=series.time_index[target_indices],
        feature_names=tuple(feature_names),
        feature_lags=tuple(feature_lags),
        valid_mask=valid_mask.astype(bool, copy=False),
    )


def build_persistence_design_matrix(
    series: FaceTimeSeries,
    *,
    target_region: str,
    horizon: int = 1,
    alignment_lag: int | None = None,
    target_dimension: str | None = None,
) -> DesignMatrix:
    """Build the frozen Persistence condition on identical target support."""

    return build_self_history_design_matrix(
        series,
        target_region=target_region,
        lags=(1,),
        horizon=horizon,
        alignment_lag=alignment_lag,
        target_dimension=target_dimension,
    )


def build_full_history_design_matrix(
    series: FaceTimeSeries,
    *,
    target_region: str,
    lags: Iterable[int] = (1,),
    horizon: int = 1,
    alignment_lag: int | None = None,
    target_dimension: str | None = None,
) -> DesignMatrix:
    """Build Full using every region component at every requested lag."""

    normalized_lags = _normalize_lags(lags)
    if target_region not in series.region_id:
        raise ValueError(f"unknown target_region: {target_region!r}")
    _validate_target_dimension(series, target_dimension)

    support_lag = _resolve_alignment_lag(normalized_lags, alignment_lag)
    common = aligned_indices(
        len(series.time_index), lag=support_lag, horizon=horizon
    )
    target_indices = common.target_index
    origin_indices = common.forecast_origin
    target_region_index = series.region_id.index(target_region)

    feature_columns: list[np.ndarray] = []
    feature_valid_columns: list[np.ndarray] = []
    feature_names: list[str] = []
    feature_lags: list[int] = []

    for lag in normalized_lags:
        source_indices = target_indices - lag
        for region_index, region_name in enumerate(series.region_id):
            for dimension_index, dimension_name in enumerate(series.dimension):
                feature_columns.append(
                    series.X[source_indices, region_index, dimension_index]
                )
                feature_valid_columns.append(
                    series.valid_mask[source_indices, region_index, dimension_index]
                )
                feature_names.append(encode_feature_name(region_name, dimension_name))
                feature_lags.append(lag)

    X = np.column_stack(feature_columns)
    feature_valid = np.column_stack(feature_valid_columns)
    y, target_valid = _target_values(
        series, target_indices, target_region_index, target_dimension
    )
    valid_mask = np.all(feature_valid, axis=1) & target_valid

    row_count = len(target_indices)
    return DesignMatrix(
        X=X,
        y=y,
        subject_id=(series.subject_id,) * row_count,
        region_id=(target_region,) * row_count,
        forecast_origin=series.time_index[origin_indices],
        target_time=series.time_index[target_indices],
        feature_names=tuple(feature_names),
        feature_lags=tuple(feature_lags),
        valid_mask=valid_mask.astype(bool, copy=False),
    )


def build_pcmci_parent_design_matrix(
    series: FaceTimeSeries,
    *,
    parent_set: ParentSet,
    self_lags: Iterable[int] = (1,),
    horizon: int = 1,
    alignment_lag: int | None = None,
    target_dimension: str | None = None,
) -> DesignMatrix:
    """Build one target-component PCMCI matrix from exact selected components.

    Primary ParCorr discovery nodes are scalar ``region × dimension`` components.
    Accordingly, this builder predicts one scalar target component at a time and adds
    only ParentLinks retained for that target component. Each selected inter-regional
    link contributes exactly its ``source_dimension``; selecting one component never
    expands to all dimensions of the source region.

    The fixed Self block remains the target region's full past vector so the
    incremental comparison is Self-region history plus selected cross-region
    components versus the identical Self baseline. Same-region ParentLinks are not
    duplicated. A mixed-target ParentSet therefore requires ``target_dimension`` to be
    supplied explicitly.
    """

    normalized_self_lags = _normalize_lags(self_lags)
    target_region = parent_set.target_region
    if target_region not in series.region_id:
        raise ValueError(f"unknown target_region from ParentSet: {target_region!r}")

    resolved_target_dimension = _resolve_pcmci_target_dimension(
        series, parent_set, target_dimension
    )

    selected_parents: list[tuple[ParentLink, str]] = []
    for parent in parent_set.parents:
        parent_target = _resolve_link_dimension(
            series, parent.target_dimension, field_name="target_dimension"
        )
        if parent_target != resolved_target_dimension:
            continue
        if parent.source_region == target_region:
            continue
        if parent.source_region not in series.region_id:
            raise ValueError(
                f"unknown source_region from ParentSet: {parent.source_region!r}"
            )
        source_dimension = _resolve_link_dimension(
            series, parent.source_dimension, field_name="source_dimension"
        )
        selected_parents.append((parent, source_dimension))

    all_lags = normalized_self_lags + tuple(
        parent.lag for parent, _ in selected_parents
    )
    support_lag = _resolve_alignment_lag(all_lags, alignment_lag)
    common = aligned_indices(
        len(series.time_index), lag=support_lag, horizon=horizon
    )
    target_indices = common.target_index
    origin_indices = common.forecast_origin
    target_region_index = series.region_id.index(target_region)

    feature_columns: list[np.ndarray] = []
    feature_valid_columns: list[np.ndarray] = []
    feature_names: list[str] = []
    feature_lags: list[int] = []

    for lag in normalized_self_lags:
        source_indices = target_indices - lag
        for dimension_index, dimension_name in enumerate(series.dimension):
            feature_columns.append(
                series.X[source_indices, target_region_index, dimension_index]
            )
            feature_valid_columns.append(
                series.valid_mask[source_indices, target_region_index, dimension_index]
            )
            feature_names.append(encode_feature_name(target_region, dimension_name))
            feature_lags.append(lag)

    for parent, source_dimension in selected_parents:
        source_indices = target_indices - parent.lag
        source_region_index = series.region_id.index(parent.source_region)
        source_dimension_index = series.dimension.index(source_dimension)
        feature_columns.append(
            series.X[source_indices, source_region_index, source_dimension_index]
        )
        feature_valid_columns.append(
            series.valid_mask[source_indices, source_region_index, source_dimension_index]
        )
        feature_names.append(
            encode_feature_name(parent.source_region, source_dimension)
        )
        feature_lags.append(parent.lag)

    X = np.column_stack(feature_columns)
    feature_valid = np.column_stack(feature_valid_columns)
    y, target_valid = _target_values(
        series,
        target_indices,
        target_region_index,
        resolved_target_dimension,
    )
    valid_mask = np.all(feature_valid, axis=1) & target_valid

    row_count = len(target_indices)
    return DesignMatrix(
        X=X,
        y=y,
        subject_id=(series.subject_id,) * row_count,
        region_id=(target_region,) * row_count,
        forecast_origin=series.time_index[origin_indices],
        target_time=series.time_index[target_indices],
        feature_names=tuple(feature_names),
        feature_lags=tuple(feature_lags),
        valid_mask=valid_mask.astype(bool, copy=False),
    )
