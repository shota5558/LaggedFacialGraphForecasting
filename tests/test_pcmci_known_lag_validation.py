from __future__ import annotations

import numpy as np

from lagged_facial_graph_forecasting.contracts import FaceTimeSeries, InnerFold, SplitManifest
from lagged_facial_graph_forecasting.core_contract_io import dumps_core_contract, loads_core_contract
from lagged_facial_graph_forecasting.core_contracts import ParentLink, ParentSet
from lagged_facial_graph_forecasting.pcmci_contemporaneous import (
    apply_primary_contemporaneous_policy,
)
from lagged_facial_graph_forecasting.pcmci_lagged_filter import filter_primary_lagged_links
from lagged_facial_graph_forecasting.pcmci_link_extraction import (
    extract_significant_pcmciplus_links,
)
from lagged_facial_graph_forecasting.pcmci_parent_set import (
    convert_primary_lagged_links_to_parent_set,
)
from lagged_facial_graph_forecasting.pcmci_runner import run_primary_pcmciplus
from lagged_facial_graph_forecasting.tigramite_adapter import (
    face_time_series_to_tigramite_dataframe,
)


TRUE_SOURCE_REGION = "left_cheek"
TRUE_SOURCE_DIMENSION = "vx"
TRUE_TARGET_REGION = "mouth"
TRUE_TARGET_DIMENSION = "vy"
TRUE_LAG = 2


def _manifest() -> SplitManifest:
    return SplitManifest(
        outer_fold=0,
        train_subject_ids=("train_1", "train_2", "train_3"),
        test_subject_ids=("outer_test",),
        inner_folds=(
            InnerFold(
                inner_train_subject_ids=("train_1", "train_2"),
                inner_val_subject_ids=("train_3",),
            ),
        ),
        seed=1402,
    )


def _known_lag_series(subject_id: str, seed: int, length: int = 500) -> FaceTimeSeries:
    rng = np.random.default_rng(seed)
    values = rng.normal(scale=0.35, size=(length, 2, 2))

    source = rng.normal(size=length)
    values[:, 0, 0] = source
    target_noise = rng.normal(scale=0.08, size=length)
    values[:TRUE_LAG, 1, 1] = target_noise[:TRUE_LAG]
    values[TRUE_LAG:, 1, 1] = (
        1.8 * source[:-TRUE_LAG] + target_noise[TRUE_LAG:]
    )

    return FaceTimeSeries(
        X=values,
        subject_id=subject_id,
        time_index=np.arange(length, dtype=float) / 25.0,
        region_id=(TRUE_SOURCE_REGION, TRUE_TARGET_REGION),
        dimension=("vx", "vy"),
        valid_mask=np.ones_like(values, dtype=bool),
        sampling_rate=25.0,
    )


def test_primary_thin_path_recovers_and_serializes_known_cross_region_lag() -> None:
    manifest = _manifest()
    series = tuple(
        _known_lag_series(subject_id, 8000 + index)
        for index, subject_id in enumerate(manifest.train_subject_ids)
    )
    bundle = face_time_series_to_tigramite_dataframe(manifest, series)

    result = run_primary_pcmciplus(manifest, bundle)
    raw_links = extract_significant_pcmciplus_links(result, bundle)
    disposition = apply_primary_contemporaneous_policy(raw_links)
    lagged_links = filter_primary_lagged_links(disposition)
    parent_set = convert_primary_lagged_links_to_parent_set(
        lagged_links,
        outer_fold=manifest.outer_fold,
        target_region=TRUE_TARGET_REGION,
    )
    restored = loads_core_contract(dumps_core_contract(parent_set))

    expected = ParentLink(
        source_region=TRUE_SOURCE_REGION,
        lag=TRUE_LAG,
        source_dimension=TRUE_SOURCE_DIMENSION,
        target_dimension=TRUE_TARGET_DIMENSION,
    )

    assert isinstance(restored, ParentSet)
    assert restored == parent_set
    assert expected in restored.parents
    assert all(parent.source_region != TRUE_TARGET_REGION for parent in restored.parents)
    assert all(parent.lag >= 1 for parent in restored.parents)
