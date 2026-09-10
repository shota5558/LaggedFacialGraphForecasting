from __future__ import annotations

from pathlib import Path

import numpy as np

from lagged_facial_graph_forecasting.contracts import FaceTimeSeries, InnerFold, SplitManifest
from lagged_facial_graph_forecasting.sensitivity_execution import load_sensitivity_experiment_config
from lagged_facial_graph_forecasting.sensitivity_lpcmci import (
    LPCMCIAdjacency,
    LPCMCIGraphArtifact,
    LPCMCIConfig,
    dumps_lpcmci_graph,
    load_lpcmci_config,
    loads_lpcmci_graph,
    lpcmci_graph_to_parent_set,
    lpcmci_result_to_graph_artifact,
    run_sensitivity_lpcmci,
)
from lagged_facial_graph_forecasting.tigramite_adapter import face_time_series_to_tigramite_dataframe


ROOT = Path(__file__).resolve().parents[1]


def _manifest() -> SplitManifest:
    return SplitManifest(
        outer_fold=0,
        train_subject_ids=("train_a", "train_b"),
        test_subject_ids=("held_out",),
        inner_folds=(InnerFold(inner_train_subject_ids=("train_a",), inner_val_subject_ids=("train_b",)),),
        seed=303,
    )


def _latent_confounded_series(subject_id: str, seed: int, length: int = 90) -> FaceTimeSeries:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=length)
    x = 1.8 * latent + rng.normal(scale=0.08, size=length)
    y = -1.5 * latent + rng.normal(scale=0.08, size=length)
    values = np.stack([x, y], axis=1)[:, :, None]
    return FaceTimeSeries(
        X=values,
        subject_id=subject_id,
        time_index=np.arange(length, dtype=float) / 25.0,
        region_id=("left_cheek", "mouth"),
        dimension=("value",),
        valid_mask=np.ones_like(values, dtype=bool),
        sampling_rate=25.0,
    )


def test_lpcmci_config_schema_is_frozen() -> None:
    config = load_lpcmci_config(ROOT / "configs/sensitivity_lpcmci.yaml")
    assert config == LPCMCIConfig()
    assert config.tau_max == 10
    assert config.pc_alpha == 0.01


def test_lpcmci_graph_round_trip_preserves_latent_edge_mark_and_parent_policy() -> None:
    artifact = LPCMCIGraphArtifact(
        outer_fold=1,
        seed=91,
        adjacencies=(
            LPCMCIAdjacency(
                source_region="left_cheek",
                source_dimension="vx",
                target_region="mouth",
                target_dimension="vy",
                lag=0,
                mark="<->",
                p_value=0.001,
                test_statistic=0.8,
            ),
            LPCMCIAdjacency(
                source_region="left_cheek",
                source_dimension="vx",
                target_region="mouth",
                target_dimension="vy",
                lag=2,
                mark="-->",
                p_value=0.002,
                test_statistic=0.7,
            ),
        ),
    )
    restored = loads_lpcmci_graph(dumps_lpcmci_graph(artifact))
    parent_set = lpcmci_graph_to_parent_set(restored, target_region="mouth")

    assert restored == artifact
    assert any(edge.mark == "<->" for edge in restored.adjacencies)
    assert len(parent_set.parents) == 1
    assert parent_set.parents[0].lag == 2
    assert parent_set.discovery_method == "lpcmci"


def test_lpcmci_latent_confounding_fixture_runs_deterministically_on_synthetic_only() -> None:
    manifest = _manifest()
    series = tuple(
        _latent_confounded_series(subject, 900 + index)
        for index, subject in enumerate(manifest.train_subject_ids)
    )
    bundle = face_time_series_to_tigramite_dataframe(manifest, series)
    execution = load_sensitivity_experiment_config(
        ROOT / "configs/sensitivity_preimplementation.yaml", repository_root=ROOT
    )
    method = load_lpcmci_config(ROOT / "configs/sensitivity_lpcmci.yaml")

    first = run_sensitivity_lpcmci(execution, manifest, bundle, method, seed=404)
    second = run_sensitivity_lpcmci(execution, manifest, bundle, method, seed=404)

    np.testing.assert_array_equal(first["graph"], second["graph"])
    np.testing.assert_allclose(first["p_matrix"], second["p_matrix"])
    artifact = lpcmci_result_to_graph_artifact(
        first, bundle, outer_fold=manifest.outer_fold, seed=404
    )
    assert artifact.method == "lpcmci"
    assert len(artifact.adjacencies) > 0
