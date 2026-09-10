#!/usr/bin/env python3
"""Generate deterministic synthetic analysis inputs for software verification.

WARNING: THIS SCRIPT GENERATES FAKE / SYNTHETIC DATA ONLY.
The outputs are not observations, not experimental results, and MUST NOT be used
for scientific conclusions, effect-size reporting, manuscript claims, or model
selection. They exist only to exercise the analysis table/figure pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260908
NOTICE = "MOCK DATA / NOT A SCIENTIFIC RESULT"
SUBJECTS = tuple(f"MOCK_S{i:02d}" for i in range(1, 9))
REGIONS = ("mouth", "jaw", "left_cheek", "right_cheek")


def _common() -> dict[str, object]:
    return {"is_synthetic": True, "synthetic_notice": NOTICE}


def _evaluation_support_sha256(outer_fold: int, subject_id: str, region_id: str) -> str:
    payload = f"mock-evaluation-support|{outer_fold}|{subject_id}|{region_id}|240"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _metrics(rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs = {
        "velocity_rmse": ("lower_is_better", 1.0),
        "position_rmse": ("lower_is_better", 0.8),
        "acceleration_rmse": ("lower_is_better", 1.2),
        "temporal_correlation": ("higher_is_better", 0.7),
    }
    error_adjust = {"persistence": 0.30, "self": 0.15, "full": 0.04, "pcmci": 0.02}
    corr_adjust = {"persistence": -0.16, "self": -0.06, "full": 0.02, "pcmci": 0.035}
    for subject in SUBJECTS:
        sid = int(subject[-2:])
        subject_offset = (sid - 4.5) * 0.012
        fold = (sid - 1) % 4
        for region_index, region in enumerate(REGIONS):
            region_offset = region_index * 0.015
            for metric_name, (direction, base) in specs.items():
                for condition in ("persistence", "self", "full", "pcmci"):
                    noise = rng.normal(0.0, 0.008)
                    if direction == "lower_is_better":
                        value = base + error_adjust[condition] + subject_offset + region_offset + noise
                    else:
                        value = base + corr_adjust[condition] - 0.3 * subject_offset - 0.2 * region_offset + noise
                    rows.append({
                        **_common(),
                        "outer_fold": fold,
                        "subject_id": subject,
                        "region_id": region,
                        "condition": condition,
                        "metric_name": metric_name,
                        "metric_direction": direction,
                        "value": round(float(value), 6),
                        "n_valid": 240,
                        "evaluation_support_sha256": _evaluation_support_sha256(
                            fold, subject, region
                        ),
                    })
    return pd.DataFrame(rows)


def _pcmci_feature_count(subject_id: str, region: str) -> int:
    sid = int(subject_id[-2:])
    region_index = REGIONS.index(region)
    return 8 + ((sid + region_index) % 5)


def _null_metrics(rng: np.random.Generator, metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    penalties = {
        "lag_shift_-2": 0.10,
        "lag_shift_-1": 0.055,
        "lag_shift_+1": 0.060,
        "lag_shift_+2": 0.105,
        "random_region": 0.12,
        "matched_sparsity": 0.09,
        "time_shuffle": 0.18,
    }
    for subject in SUBJECTS:
        fold = (int(subject[-2:]) - 1) % 4
        for region in REGIONS:
            ref = float(metrics.loc[
                (metrics["subject_id"] == subject)
                & (metrics["region_id"] == region)
                & (metrics["condition"] == "pcmci")
                & (metrics["metric_name"] == "velocity_rmse"),
                "value",
            ].iloc[0])
            pcmci_count = _pcmci_feature_count(subject, region)
            for condition, penalty in penalties.items():
                rows.append({
                    **_common(),
                    "outer_fold": fold,
                    "subject_id": subject,
                    "region_id": region,
                    "condition": condition,
                    "metric_name": "velocity_rmse",
                    "metric_direction": "lower_is_better",
                    "value": round(float(ref + penalty + rng.normal(0.0, 0.006)), 6),
                    "n_valid": 240,
                    "replicate_id": "MOCK_R000",
                    "seed": SEED,
                    "mapping_scope": "outer_train_only",
                    "pcmci_feature_count": pcmci_count,
                    "null_feature_count": pcmci_count,
                    "input_count_preserved": True,
                    "lag_identity_preserved": not condition.startswith("lag_shift"),
                })
    return pd.DataFrame(rows)


def _feature_counts() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject in SUBJECTS:
        sid = int(subject[-2:])
        fold = (sid - 1) % 4
        for region in REGIONS:
            pcmci_count = _pcmci_feature_count(subject, region)
            for condition, count in (("persistence", 1), ("self", 10), ("full", 80), ("pcmci", pcmci_count)):
                rows.append({
                    **_common(),
                    "outer_fold": fold,
                    "subject_id": subject,
                    "region_id": region,
                    "condition": condition,
                    "feature_count": count,
                })
    return pd.DataFrame(rows)


def _lag_response(rng: np.random.Generator, metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    penalties = {-2: 0.11, -1: 0.05, 0: 0.0, 1: 0.055, 2: 0.115}
    for subject in SUBJECTS:
        fold = (int(subject[-2:]) - 1) % 4
        for region in REGIONS:
            ref = float(metrics.loc[
                (metrics["subject_id"] == subject)
                & (metrics["region_id"] == region)
                & (metrics["condition"] == "pcmci")
                & (metrics["metric_name"] == "velocity_rmse"),
                "value",
            ].iloc[0])
            for delta in (-2, -1, 0, 1, 2):
                error = ref + penalties[delta] + (rng.normal(0.0, 0.004) if delta else 0.0)
                rows.append({
                    **_common(),
                    "outer_fold": fold,
                    "subject_id": subject,
                    "region_id": region,
                    "delta_frames": delta,
                    "delta_ms": round(delta * 1000.0 / 30.0, 3),
                    "reference_delta": 0,
                    "error": round(float(error), 6),
                    "reference_error": round(ref, 6),
                    "delta_error": round(float(error - ref), 6),
                    "evaluable": True,
                    "same_support": True,
                    "same_region_identity": True,
                    "same_feature_count": True,
                    "n_valid": 240,
                })
    return pd.DataFrame(rows)


def _edge_stability() -> pd.DataFrame:
    """Create internally reconciled stability counts and frequencies.

    Fold-level selection opportunities are discrete (4 outer folds), so arbitrary
    decimal frequencies such as 0.88 cannot be represented exactly. Counts are the
    source of truth and frequencies are derived from count / opportunities.
    """
    edges = (
        ("left_cheek", "vx", "mouth", "vx", 2, 4, 84),
        ("right_cheek", "vx", "mouth", "vx", 3, 3, 79),
        ("jaw", "vy", "mouth", "vy", 1, 4, 90),
        ("mouth", "vx", "jaw", "vx", 2, 3, 66),
        ("left_cheek", "vy", "right_cheek", "vy", 4, 2, 52),
        ("right_cheek", "vy", "left_cheek", "vy", 4, 2, 50),
    )
    fold_opportunities = 4
    bootstrap_opportunities = 100
    rows = []
    for (
        source_region,
        source_dimension,
        target_region,
        target_dimension,
        lag,
        fold_selected_count,
        bootstrap_selected_count,
    ) in edges:
        rows.append({
            **_common(),
            "source_region": source_region,
            "source_dimension": source_dimension,
            "target_region": target_region,
            "target_dimension": target_dimension,
            "lag": lag,
            "fold_selected_count": fold_selected_count,
            "fold_opportunities": fold_opportunities,
            "outer_fold_selection_frequency": fold_selected_count / fold_opportunities,
            "bootstrap_selected_count": bootstrap_selected_count,
            "bootstrap_opportunities": bootstrap_opportunities,
            "bootstrap_selection_frequency": bootstrap_selected_count / bootstrap_opportunities,
            "evaluable": True,
        })
    return pd.DataFrame(rows)


def _sensitivity() -> pd.DataFrame:
    specs = (
        ("primary_parcorr_ridge", "Primary ParCorr + Ridge", 0.130, 0.085, 0.176),
        ("gpdc_ridge", "GPDC + Ridge", 0.118, 0.071, 0.165),
        ("lpcmci_ridge", "LPCMCI + Ridge", 0.101, 0.052, 0.151),
        ("phase_surrogate", "Phase-shuffled surrogate", 0.022, -0.018, 0.061),
        ("circular_shift", "Circular-shift surrogate", 0.028, -0.011, 0.068),
        ("h2_parcorr_ridge", "h=2", 0.086, 0.041, 0.132),
        ("gru_confirmation", "GRU confirmation", 0.121, 0.069, 0.171),
    )
    return pd.DataFrame([
        {
            **_common(),
            "analysis_id": analysis_id,
            "label": label,
            "effect_self_minus_pcmci": effect,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "confidence_level": 0.95,
            "metric_name": "velocity_rmse",
            "effect_direction": "positive_is_pcmci_better",
        }
        for analysis_id, label, effect, ci_low, ci_high in specs
    ])


def _dataset_summary() -> pd.DataFrame:
    rows = []
    for subject in SUBJECTS:
        sid = int(subject[-2:])
        frame_count = 3600 + sid * 60
        missing_ratio = 0.01 + sid * 0.001
        rows.append({
            **_common(),
            "subject_id": subject,
            "outer_fold": (sid - 1) % 4,
            "frame_count": frame_count,
            "valid_frame_count": round(frame_count * (1.0 - missing_ratio)),
            "missing_ratio": round(missing_ratio, 4),
            "sampling_rate_hz": 30.0,
            "sequence_length": frame_count,
            "mean_motion_magnitude": round(0.12 + sid * 0.004, 4),
            "active_interval_ratio": round(0.42 + sid * 0.01, 4),
        })
    return pd.DataFrame(rows)


def _trajectory() -> pd.DataFrame:
    rows = []
    for time_index in range(60):
        y_true = 0.4 * math.sin(time_index / 5.0) + 0.15 * math.sin(time_index / 2.4)
        self_prediction = 0.88 * y_true + 0.10 * math.sin((time_index - 2) / 4.0)
        pcmci_prediction = 0.96 * y_true + 0.04 * math.sin((time_index - 1) / 4.0)
        rows.append({
            **_common(),
            "subject_id": "MOCK_S01",
            "region_id": "mouth",
            "time_index": time_index,
            "y_true": round(y_true, 6),
            "self_prediction": round(self_prediction, 6),
            "pcmci_prediction": round(pcmci_prediction, 6),
        })
    return pd.DataFrame(rows)


def _primary_config() -> dict[str, object]:
    return {
        **_common(),
        "purpose": "software verification only",
        "seed": SEED,
        "primary": {
            "discovery": "pcmci_plus",
            "ci_test": "parcorr",
            "tau_max": 10,
            "pc_alpha": 0.01,
            "forecaster": "ridge",
            "horizon": 1,
            "matched_sparsity": {
                "repeat_count": 1,
                "seed_source": "split_manifest_seed",
                "repeat_aggregation": "median_error_across_repeats_per_subject_region",
            },
        },
        "evaluation": {"primary_metric": "velocity_rmse"},
        "lag_response": {
            "delta_frames": [-2, -1, 0, 1, 2],
            "reference_delta": 0,
            "shift_mode": "common_shift_all_selected_parents",
            "require_complete_symmetric_grid": True,
            "clipping": "forbidden",
            "wrapping": "forbidden",
            "feature_dropping": "forbidden",
        },
        "statistics": {
            "paired_unit": "outer_test_subject",
            "bootstrap_confidence_level": 0.95,
            "bootstrap_n_resamples": 10000,
            "bootstrap_method": "percentile",
        },
    }


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    metrics = _metrics(rng)
    frames = {
        "mock_metrics.csv": metrics,
        "mock_null_metrics.csv": _null_metrics(rng, metrics),
        "mock_feature_counts.csv": _feature_counts(),
        "mock_lag_response.csv": _lag_response(rng, metrics),
        "mock_edge_stability.csv": _edge_stability(),
        "mock_sensitivity.csv": _sensitivity(),
        "mock_dataset_summary.csv": _dataset_summary(),
        "mock_prediction_trajectory.csv": _trajectory(),
    }
    for filename, frame in frames.items():
        frame.to_csv(output_dir / filename, index=False, lineterminator="\n")
    (output_dir / "mock_primary_config.json").write_text(
        json.dumps(_primary_config(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mock_analysis"),
        help="Output directory for synthetic verification inputs.",
    )
    args = parser.parse_args()
    generate(args.output_dir)
    print(f"Generated SYNTHETIC verification data in {args.output_dir}")
    print(NOTICE)


if __name__ == "__main__":
    main()
