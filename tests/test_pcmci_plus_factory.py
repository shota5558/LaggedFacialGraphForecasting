from __future__ import annotations

import numpy as np
import pytest
from tigramite.pcmci import PCMCI

from lagged_facial_graph_forecasting.contracts import FaceTimeSeries, InnerFold, SplitManifest
from lagged_facial_graph_forecasting.pcmci_plus_factory import (
    PCMCIPlusConfigError,
    make_primary_pcmci_plus,
)
from lagged_facial_graph_forecasting.tigramite_adapter import (
    face_time_series_to_tigramite_dataframe,
)


def _bundle():
    manifest = SplitManifest(
        outer_fold=0,
        train_subject_ids=("s1", "s2"),
        test_subject_ids=("outer_test",),
        inner_folds=(
            InnerFold(
                inner_train_subject_ids=("s1",),
                inner_val_subject_ids=("s2",),
            ),
        ),
        seed=17,
    )
    series = []
    for subject_id in manifest.train_subject_ids:
        # D-01 deliberately accepts exactly one scalar component per region until
        # the real-data region-state mapping is scientifically frozen. D-03 tests
        # object construction only and must not bypass that fail-closed contract.
        values = np.arange(12, dtype=float).reshape(6, 2, 1)
        series.append(
            FaceTimeSeries(
                X=values,
                subject_id=subject_id,
                time_index=np.arange(6, dtype=float) / 25.0,
                region_id=("left_eye", "mouth"),
                dimension=("scalar_motion",),
                valid_mask=np.ones_like(values, dtype=bool),
                sampling_rate=25.0,
            )
        )
    return face_time_series_to_tigramite_dataframe(manifest, tuple(series))


def test_primary_pcmci_plus_factory_binds_guarded_dataframe_and_frozen_parcorr() -> None:
    bundle = _bundle()
    pcmci = make_primary_pcmci_plus(bundle)

    assert isinstance(pcmci, PCMCI)
    assert pcmci.dataframe is bundle.dataframe
    assert pcmci.cond_ind_test.measure == "par_corr"
    assert pcmci.cond_ind_test.mask_type == "xyz"
    assert pcmci.cond_ind_test.significance == "analytic"
    assert callable(pcmci.run_pcmciplus)


def test_primary_pcmci_plus_factory_does_not_run_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    def _unexpected_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("run_pcmciplus must not be called during D-03 construction")

    monkeypatch.setattr(PCMCI, "run_pcmciplus", _unexpected_run)
    pcmci = make_primary_pcmci_plus(_bundle())

    assert isinstance(pcmci, PCMCI)


def test_primary_pcmci_plus_factory_rejects_noncanonical_input() -> None:
    with pytest.raises(PCMCIPlusConfigError, match="TigramiteDataFrameBundle"):
        make_primary_pcmci_plus(object())  # type: ignore[arg-type]
