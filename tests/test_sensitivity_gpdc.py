from __future__ import annotations

import numpy as np

from lagged_facial_graph_forecasting.contracts import FaceTimeSeries, InnerFold, SplitManifest
from lagged_facial_graph_forecasting.sensitivity_gpdc import make_sensitivity_pcmci_plus_gpdc
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


def test_gpdc_pcmci_factory_reuses_canonical_outer_train_bundle() -> None:
    manifest = _manifest()
    bundle = face_time_series_to_tigramite_dataframe(
        manifest,
        tuple(
            _series(subject, 400 + index)
            for index, subject in enumerate(manifest.train_subject_ids)
        ),
    )
    pcmci = make_sensitivity_pcmci_plus_gpdc(bundle, seed=88)

    assert pcmci.cond_ind_test.measure == "gp_dc"
    assert pcmci.cond_ind_test.dataframe is bundle.dataframe
