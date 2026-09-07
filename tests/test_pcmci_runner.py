from __future__ import annotations

from dataclasses import replace
from inspect import signature

import numpy as np
import pytest
from tigramite.pcmci import PCMCI

from lagged_facial_graph_forecasting.contracts import FaceTimeSeries, InnerFold, SplitManifest
from lagged_facial_graph_forecasting.leakage_guard import LeakageGuardError
from lagged_facial_graph_forecasting.pcmci_runner import (
    PCMCIPlusRunError,
    primary_run_pcmciplus_kwargs,
    run_primary_pcmciplus,
)
from lagged_facial_graph_forecasting.tigramite_adapter import (
    face_time_series_to_tigramite_dataframe,
)


def _manifest() -> SplitManifest:
    return SplitManifest(
        outer_fold=0,
        train_subject_ids=("train_1", "train_2"),
        test_subject_ids=("outer_test",),
        inner_folds=(
            InnerFold(
                inner_train_subject_ids=("train_1",),
                inner_val_subject_ids=("train_2",),
            ),
        ),
        seed=31,
    )


def _bundle():
    manifest = _manifest()
    series = []
    for index, subject_id in enumerate(manifest.train_subject_ids):
        rng = np.random.default_rng(100 + index)
        noise = rng.normal(scale=0.25, size=(80, 2))
        values = np.zeros((80, 1, 2), dtype=float)
        for t in range(1, values.shape[0]):
            values[t, 0, 0] = 0.55 * values[t - 1, 0, 0] + noise[t, 0]
            values[t, 0, 1] = (
                0.40 * values[t - 1, 0, 1]
                + 0.30 * values[t - 1, 0, 0]
                + noise[t, 1]
            )
        series.append(
            FaceTimeSeries(
                X=values,
                subject_id=subject_id,
                time_index=np.arange(values.shape[0], dtype=float) / 25.0,
                region_id=("mouth",),
                dimension=("vx", "vy"),
                valid_mask=np.ones_like(values, dtype=bool),
                sampling_rate=25.0,
            )
        )
    return face_time_series_to_tigramite_dataframe(manifest, tuple(series))


def test_primary_run_kwargs_combine_only_frozen_d03_d04_d05_contracts() -> None:
    kwargs = primary_run_pcmciplus_kwargs()

    assert dict(kwargs) == {
        "link_assumptions": None,
        "tau_min": 0,
        "contemp_collider_rule": "majority",
        "conflict_resolution": True,
        "reset_lagged_links": False,
        "max_conds_dim": None,
        "max_combinations": 1,
        "max_conds_py": None,
        "max_conds_px": None,
        "max_conds_px_lagged": None,
        "fdr_method": "none",
        "tau_max": 10,
        "pc_alpha": 0.01,
    }

    with pytest.raises(TypeError):
        kwargs["pc_alpha"] = 0.05  # type: ignore[index]


def test_run_api_exposes_no_runtime_scientific_parameter_override() -> None:
    assert tuple(signature(run_primary_pcmciplus).parameters) == ("manifest", "bundle")


def test_run_primary_pcmciplus_delegates_exactly_once_to_tigramite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    def _fake_run(self, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(dict(kwargs))
        return {"graph": "sentinel"}

    monkeypatch.setattr(PCMCI, "run_pcmciplus", _fake_run)

    result = run_primary_pcmciplus(_manifest(), _bundle())

    assert result == {"graph": "sentinel"}
    assert captured == [dict(primary_run_pcmciplus_kwargs())]
    assert captured[0]["tau_max"] == 10
    assert captured[0]["pc_alpha"] == 0.01


def test_run_primary_pcmciplus_rechecks_outer_train_scope_before_oss_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unexpected_run(self, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("Tigramite must not run for an outer-test bundle")

    monkeypatch.setattr(PCMCI, "run_pcmciplus", _unexpected_run)
    altered_bundle = replace(_bundle(), subject_ids=("outer_test",))

    with pytest.raises(LeakageGuardError, match="outer-test"):
        run_primary_pcmciplus(_manifest(), altered_bundle)


def test_run_primary_pcmciplus_rejects_noncanonical_bundle() -> None:
    with pytest.raises(PCMCIPlusRunError, match="TigramiteDataFrameBundle"):
        run_primary_pcmciplus(_manifest(), object())  # type: ignore[arg-type]


def test_run_primary_pcmciplus_real_tigramite_smoke() -> None:
    result = run_primary_pcmciplus(_manifest(), _bundle())

    assert {"graph", "p_matrix", "val_matrix"}.issubset(result)
    assert result["graph"].shape == (2, 2, 11)
    assert result["p_matrix"].shape == (2, 2, 11)
    assert result["val_matrix"].shape == (2, 2, 11)
