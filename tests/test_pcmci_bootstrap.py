from __future__ import annotations

from dataclasses import replace
from inspect import signature

import numpy as np
import pytest
from tigramite.pcmci import PCMCI

from lagged_facial_graph_forecasting.contracts import FaceTimeSeries, InnerFold, SplitManifest
from lagged_facial_graph_forecasting.leakage_guard import LeakageGuardError
from lagged_facial_graph_forecasting.pcmci_bootstrap import (
    PCMCIPlusBootstrapError,
    run_primary_pcmciplus_bootstrap,
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
        seed=1201,
    )


def _bundle(length: int = 90):
    manifest = _manifest()
    series = []
    for index, subject_id in enumerate(manifest.train_subject_ids):
        rng = np.random.default_rng(500 + index)
        values = rng.normal(scale=0.2, size=(length, 1, 2))
        for t in range(2, length):
            values[t, 0, 1] += 0.8 * values[t - 2, 0, 0]
        series.append(
            FaceTimeSeries(
                X=values,
                subject_id=subject_id,
                time_index=np.arange(length, dtype=float) / 25.0,
                region_id=("mouth",),
                dimension=("vx", "vy"),
                valid_mask=np.ones_like(values, dtype=bool),
                sampling_rate=25.0,
            )
        )
    return face_time_series_to_tigramite_dataframe(manifest, tuple(series))


def test_bootstrap_delegates_to_tigramite_with_frozen_primary_method_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    def _fake_bootstrap(self, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(dict(kwargs))
        return {"summary_results": {}, "boot_results": {"graph": np.empty((2, 2, 2, 11))}}

    monkeypatch.setattr(PCMCI, "run_bootstrap_of", _fake_bootstrap)

    result = run_primary_pcmciplus_bootstrap(
        _manifest(),
        _bundle(),
        boot_samples=2,
        boot_blocklength=5,
        seed=77,
    )

    assert "boot_results" in result
    assert len(captured) == 1
    call = captured[0]
    assert call["method"] == "run_pcmciplus"
    assert call["boot_samples"] == 2
    assert call["boot_blocklength"] == 5
    assert call["seed"] == 77
    method_args = call["method_args"]
    assert isinstance(method_args, dict)
    assert method_args["tau_max"] == 10
    assert method_args["pc_alpha"] == 0.01
    assert method_args["tau_min"] == 0
    assert method_args["fdr_method"] == "none"


def test_bootstrap_api_exposes_no_tau_max_or_pc_alpha_override() -> None:
    assert tuple(signature(run_primary_pcmciplus_bootstrap).parameters) == (
        "manifest",
        "bundle",
        "boot_samples",
        "boot_blocklength",
        "seed",
    )


def test_bootstrap_rechecks_outer_train_scope_before_tigramite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unexpected(self, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("bootstrap must not run with an outer-test subject")

    monkeypatch.setattr(PCMCI, "run_bootstrap_of", _unexpected)
    altered = replace(_bundle(), subject_ids=("outer_test",))

    with pytest.raises(LeakageGuardError, match="outer-test"):
        run_primary_pcmciplus_bootstrap(
            _manifest(), altered, boot_samples=2, boot_blocklength=5, seed=1
        )


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_rejects_invalid_boot_samples(value: object) -> None:
    with pytest.raises(PCMCIPlusBootstrapError, match="boot_samples"):
        run_primary_pcmciplus_bootstrap(
            _manifest(), _bundle(), boot_samples=value, boot_blocklength=5, seed=1  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [0, -1, False, 1.5, "5"])
def test_rejects_invalid_boot_blocklength(value: object) -> None:
    with pytest.raises(PCMCIPlusBootstrapError, match="boot_blocklength"):
        run_primary_pcmciplus_bootstrap(
            _manifest(), _bundle(), boot_samples=2, boot_blocklength=value, seed=1  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [-1, True, 1.5, "7"])
def test_rejects_invalid_seed(value: object) -> None:
    with pytest.raises(PCMCIPlusBootstrapError, match="seed"):
        run_primary_pcmciplus_bootstrap(
            _manifest(), _bundle(), boot_samples=2, boot_blocklength=5, seed=value  # type: ignore[arg-type]
        )


def test_rejects_noncanonical_bundle() -> None:
    with pytest.raises(PCMCIPlusBootstrapError, match="TigramiteDataFrameBundle"):
        run_primary_pcmciplus_bootstrap(
            _manifest(), object(), boot_samples=2, boot_blocklength=5, seed=1  # type: ignore[arg-type]
        )


def test_rejects_malformed_tigramite_bootstrap_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_bootstrap(self, **kwargs):  # type: ignore[no-untyped-def]
        return {"summary_results": {}}

    monkeypatch.setattr(PCMCI, "run_bootstrap_of", _fake_bootstrap)

    with pytest.raises(PCMCIPlusBootstrapError, match="summary_results and boot_results"):
        run_primary_pcmciplus_bootstrap(
            _manifest(), _bundle(), boot_samples=2, boot_blocklength=5, seed=1
        )


def test_real_tigramite_bootstrap_smoke_is_reproducible() -> None:
    first = run_primary_pcmciplus_bootstrap(
        _manifest(), _bundle(length=70), boot_samples=2, boot_blocklength=5, seed=123
    )
    second = run_primary_pcmciplus_bootstrap(
        _manifest(), _bundle(length=70), boot_samples=2, boot_blocklength=5, seed=123
    )

    first_graph = first["boot_results"]["graph"]
    second_graph = second["boot_results"]["graph"]
    assert first_graph.shape == (2, 2, 2, 11)
    assert np.array_equal(first_graph, second_graph)
