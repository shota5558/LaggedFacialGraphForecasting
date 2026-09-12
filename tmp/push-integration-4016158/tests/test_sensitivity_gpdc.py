from __future__ import annotations

import numpy as np

from lagged_facial_graph_forecasting.contracts import FaceTimeSeries, InnerFold, SplitManifest
from lagged_facial_graph_forecasting.core_contract_io import dumps_core_contract, loads_core_contract
from lagged_facial_graph_forecasting.core_contracts import ParentSet
from lagged_facial_graph_forecasting.pcmci_link_extraction import SignificantPCMCIPlusLink
from lagged_facial_graph_forecasting.pcmci_parent_set import convert_lagged_links_to_parent_set
from lagged_facial_graph_forecasting.sensitivity_gpdc import (
    SENSITIVITY_GPDC_DISCOVERY_METHOD,
    GPDCDiscoveryProvenance,
    make_sensitivity_gpdc,
    make_sensitivity_pcmci_plus_gpdc,
)
from lagged_facial_graph_forecasting.tigramite_adapter import face_time_series_to_tigramite_dataframe


def _manifest() -> SplitManifest:
    return SplitManifest(
        outer_fold=2,
        train_subject_ids=("train_a", "train_b"),
        test_subject_ids=("held_out",),
        inner_folds=(
            InnerFold(
                inner_train_subject_ids=("train_a",),
                inner_val_subject_ids=("train_b",),
            ),
        ),
        seed=2202,
    )


def _series(subject_id: str, seed: int, length: int = 80) -> FaceTimeSeries:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(length, 2, 2))
    return FaceTimeSeries(
        X=values,
        subject_id=subject_id,
        time_index=np.arange(length, dtype=float) / 25.0,
        region_id=("left_cheek", "mouth"),
        dimension=("vx", "vy"),
        valid_mask=np.ones_like(values, dtype=bool),
        sampling_rate=25.0,
    )


def test_gpdc_factory_is_oss_gpdc_with_deterministic_seed() -> None:
    first = make_sensitivity_gpdc(seed=1701)
    second = make_sensitivity_gpdc(seed=1701)

    assert first.measure == "gp_dc"
    assert first.mask_type == "xyz"
    np.testing.assert_allclose(
        first.random_state.random(8),
        second.random_state.random(8),
    )
    assert GPDCDiscoveryProvenance(seed=1701) == GPDCDiscoveryProvenance(seed=1701)


def test_gpdc_detects_strong_synthetic_nonlinear_dependence() -> None:
    rng = np.random.default_rng(991)
    source = rng.normal(size=300)
    target = source**2 + rng.normal(scale=0.03, size=300)
    array = np.vstack([source, target])
    xyz = np.array([0, 1], dtype=int)

    gpdc = make_sensitivity_gpdc(seed=991)
    dependence = gpdc.get_dependence_measure(array.copy(), xyz)

    assert np.isfinite(dependence)
    assert dependence > 0.45


def test_gpdc_pcmci_factory_reuses_canonical_outer_train_bundle() -> None:
    manifest = _manifest()
    bundle = face_time_series_to_tigramite_dataframe(
        manifest,
        tuple(_series(subject, 400 + index) for index, subject in enumerate(manifest.train_subject_ids)),
    )
    pcmci = make_sensitivity_pcmci_plus_gpdc(bundle, seed=88)

    assert pcmci.cond_ind_test.measure == "gp_dc"
    assert pcmci.cond_ind_test.dataframe is bundle.dataframe


def test_gpdc_parent_set_conversion_and_serialization_round_trip() -> None:
    link = SignificantPCMCIPlusLink(
        source_node_index=0,
        target_node_index=3,
        lag=2,
        link_mark="-->",
        p_value=0.001,
        test_statistic=0.7,
        source_variable_name="left_cheek::vx",
        target_variable_name="mouth::vy",
        source_region="left_cheek",
        source_dimension="vx",
        target_region="mouth",
        target_dimension="vy",
    )
    parent_set = convert_lagged_links_to_parent_set(
        (link,),
        outer_fold=2,
        target_region="mouth",
        discovery_method=SENSITIVITY_GPDC_DISCOVERY_METHOD,
    )
    restored = loads_core_contract(dumps_core_contract(parent_set))

    assert isinstance(restored, ParentSet)
    assert restored == parent_set
    assert restored.discovery_method == SENSITIVITY_GPDC_DISCOVERY_METHOD
    assert restored.parents[0].lag == 2
    assert restored.parents[0].source_region == "left_cheek"
