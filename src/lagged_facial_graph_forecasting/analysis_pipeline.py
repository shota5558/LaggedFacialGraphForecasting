"""Reproducible analysis-output pipeline for Issue #23 (T01-T09, F01-F14).

This module is downstream-only. It consumes already-frozen experiment artifacts and
never performs discovery, tuning, split selection, Null construction, or example
selection from outer-test outcomes. Canonical CSV source tables are serialized and
read back before every figure is rendered.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import bootstrap as scipy_bootstrap

ANALYSIS_SCHEMA_VERSION = 2
ANALYSIS_CODE_VERSION = "issue23-analysis-v2"
SYNTHETIC_NOTICE = "FAKE DATA - NOT FOR SCIENTIFIC CONCLUSIONS"
PRIMARY_CONDITIONS = ("persistence", "self", "full", "pcmci")
LAG_GRID = (-2, -1, 0, 1, 2)
REFERENCE_DELTA = 0
TAU_MAX = 10
PRIMARY_PC_ALPHA = 0.01
SENSITIVITY_ORDER = (
    "primary_parcorr_ridge",
    "gpdc_ridge",
    "lpcmci_ridge",
    "phase_surrogate",
    "circular_shift",
    "h2_parcorr_ridge",
    "gru_confirmation",
)


class AnalysisOutputError(ValueError):
    """Raised when a frozen analysis/output contract is violated."""


@dataclass(frozen=True, slots=True)
class AnalysisInputs:
    dataset_summary: Path
    primary_config: Path
    metrics: Path
    feature_counts: Path
    null_metrics: Path
    lag_response: Path
    edge_stability: Path
    sensitivity: Path
    prediction_trajectory: Path

    @classmethod
    def from_directory(cls, root: str | Path) -> "AnalysisInputs":
        root = Path(root)

        def pick(stem: str, suffix: str = ".csv") -> Path:
            for candidate in (root / f"{stem}{suffix}", root / f"mock_{stem}{suffix}"):
                if candidate.is_file():
                    return candidate
            raise AnalysisOutputError(f"required analysis input missing: {stem}{suffix}")

        return cls(
            dataset_summary=pick("dataset_summary"),
            primary_config=pick("primary_config", ".json"),
            metrics=pick("metrics"),
            feature_counts=pick("feature_counts"),
            null_metrics=pick("null_metrics"),
            lag_response=pick("lag_response"),
            edge_stability=pick("edge_stability"),
            sensitivity=pick("sensitivity"),
            prediction_trajectory=pick("prediction_trajectory"),
        )

    def paths(self) -> tuple[Path, ...]:
        return (
            self.dataset_summary,
            self.primary_config,
            self.metrics,
            self.feature_counts,
            self.null_metrics,
            self.lag_response,
            self.edge_stability,
            self.sensitivity,
            self.prediction_trajectory,
        )


@dataclass(frozen=True, slots=True)
class AnalysisRunResult:
    output_root: Path
    manifest_path: Path
    registry_path: Path
    artifact_paths: tuple[Path, ...]
    is_synthetic: bool
    publication_ready: bool


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise AnalysisOutputError(f"analysis input is empty: {path}")
    return frame


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise AnalysisOutputError(f"{name} missing required columns: {missing}")


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False})


def _synthetic_status(frame: pd.DataFrame, name: str) -> bool:
    _require_columns(frame, ("is_synthetic",), name)
    flags = _as_bool(frame["is_synthetic"])
    if flags.isna().any() or flags.nunique(dropna=False) != 1:
        raise AnalysisOutputError(f"{name} mixes or has invalid synthetic provenance")
    synthetic = bool(flags.iloc[0])
    if synthetic:
        _require_columns(frame, ("synthetic_notice",), name)
        if not (frame.synthetic_notice.astype(str) == SYNTHETIC_NOTICE).all():
            raise AnalysisOutputError(f"{name} has invalid synthetic_notice")
        if "subject_id" in frame.columns:
            ids = frame.subject_id.dropna().astype(str)
            if not ids.str.startswith("MOCK_S").all():
                raise AnalysisOutputError(f"{name} synthetic subject IDs must use MOCK_S*")
    elif "subject_id" in frame.columns:
        if frame.subject_id.dropna().astype(str).str.startswith("MOCK_S").any():
            raise AnalysisOutputError(f"{name} real input contains reserved MOCK_S* subject")
    return synthetic


def _validate_provenance(frames: Mapping[str, pd.DataFrame], config: Mapping[str, object]) -> bool:
    statuses = {_synthetic_status(frame, name) for name, frame in frames.items()}
    statuses.add(bool(config.get("is_synthetic", False)))
    if len(statuses) != 1:
        raise AnalysisOutputError("synthetic and real analysis inputs must never be mixed")
    return statuses.pop()


def _json_hash(payload: Mapping[str, object]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AnalysisOutputError(f"{name} config must be an object")
    return value


def validate_primary_config(config: Mapping[str, object]) -> None:
    primary = _mapping(config.get("primary", {}), "primary")
    expected = {
        "discovery": "pcmci_plus",
        "ci_test": "parcorr",
        "tau_max": TAU_MAX,
        "pc_alpha": PRIMARY_PC_ALPHA,
        "forecaster": "ridge",
        "horizon": 1,
    }
    for key, expected_value in expected.items():
        if primary.get(key) != expected_value:
            raise AnalysisOutputError(
                f"Primary {key} must remain frozen at {expected_value!r}; got {primary.get(key)!r}"
            )

    lag = _mapping(config.get("lag_response", {}), "lag_response")
    if tuple(int(x) for x in lag.get("delta_frames", ())) != LAG_GRID:
        raise AnalysisOutputError("Primary lag-response grid must be [-2,-1,0,1,2]")
    if int(lag.get("reference_delta", 999)) != REFERENCE_DELTA:
        raise AnalysisOutputError("Primary lag-response reference_delta must be 0")
    if lag.get("shift_mode") != "common_shift_all_selected_parents":
        raise AnalysisOutputError("Primary lag-response shift_mode changed")
    if lag.get("require_complete_symmetric_grid") is not True:
        raise AnalysisOutputError("Primary lag-response requires complete symmetric grid")
    for key in ("clipping", "wrapping", "feature_dropping"):
        if lag.get(key) != "forbidden":
            raise AnalysisOutputError(f"Primary lag-response {key} must be forbidden")

    stats = _mapping(config.get("statistics", {}), "statistics")
    if stats.get("paired_unit") != "outer_test_subject":
        raise AnalysisOutputError("Primary paired unit must be outer_test_subject")
    if float(stats.get("bootstrap_confidence_level", -1)) != 0.95:
        raise AnalysisOutputError("Primary bootstrap confidence_level must be 0.95")
    if int(stats.get("bootstrap_n_resamples", 0)) != 10000:
        raise AnalysisOutputError("Primary bootstrap_n_resamples must be 10000")
    if stats.get("bootstrap_method") != "percentile":
        raise AnalysisOutputError("Primary bootstrap method must be percentile")


def _statistics_config(config: Mapping[str, object]) -> tuple[int, int, str]:
    validate_primary_config(config)
    stats = _mapping(config["statistics"], "statistics")
    return int(stats["bootstrap_n_resamples"]), int(config.get("seed", 0)), str(stats["bootstrap_method"])


def _primary_metric(metrics: pd.DataFrame, config: Mapping[str, object]) -> str:
    evaluation = config.get("evaluation", {})
    configured = evaluation.get("primary_metric") if isinstance(evaluation, Mapping) else None
    available = set(metrics.metric_name.astype(str))
    if configured is not None:
        configured = str(configured)
        if configured not in available:
            raise AnalysisOutputError(f"configured primary_metric absent from metrics: {configured}")
        directions = set(metrics.loc[metrics.metric_name == configured, "metric_direction"].astype(str))
        if directions != {"lower_is_better"}:
            raise AnalysisOutputError("Primary effect metric must be a lower-is-better error metric")
        return configured
    if "velocity_rmse" in available:
        return "velocity_rmse"
    candidates = sorted(
        metric
        for metric in available
        if set(metrics.loc[metrics.metric_name == metric, "metric_direction"].astype(str))
        == {"lower_is_better"}
    )
    if not candidates:
        raise AnalysisOutputError("no lower-is-better metric available for Primary paired effect")
    return candidates[0]


def _bootstrap_median(
    values: Iterable[float], *, n_resamples: int, seed: int, method: str
) -> tuple[float, float, float]:
    arr = np.asarray(tuple(float(v) for v in values), dtype=float)
    if len(arr) < 2 or not np.isfinite(arr).all():
        raise AnalysisOutputError("bootstrap requires at least two finite subject-level values")
    median = float(np.median(arr))
    result = scipy_bootstrap(
        (arr,),
        np.median,
        confidence_level=0.95,
        n_resamples=n_resamples,
        method=method,
        vectorized=False,
        random_state=np.random.default_rng(seed),
    )
    return median, float(result.confidence_interval.low), float(result.confidence_interval.high)


def _mark_provenance(frame: pd.DataFrame, synthetic: bool) -> pd.DataFrame:
    out = frame.copy()
    for column in ("is_synthetic", "synthetic_notice"):
        if column in out.columns:
            out = out.drop(columns=column)
    out.insert(0, "is_synthetic", synthetic)
    if synthetic:
        out.insert(1, "synthetic_notice", SYNTHETIC_NOTICE)
    return out


def _write_table(
    frame: pd.DataFrame,
    csv_path: Path,
    md_path: Path | None,
    *,
    synthetic: bool,
) -> list[Path]:
    out = _mark_provenance(frame, synthetic)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False, lineterminator="\n", float_format="%.12g")
    written = [csv_path]
    if md_path is not None:
        warning = f"> **{SYNTHETIC_NOTICE}**\n\n" if synthetic else ""
        md_path.write_text(warning + out.to_markdown(index=False) + "\n", encoding="utf-8")
        written.append(md_path)
    return written


def _exact_pair(metrics: pd.DataFrame, reference: str, comparison: str, metric: str) -> pd.DataFrame:
    keys = ["outer_fold", "subject_id", "region_id", "metric_name", "metric_direction"]
    subset = metrics[metrics.metric_name == metric]
    left = subset[subset.condition == reference][keys + ["value"]].rename(columns={"value": "reference_value"})
    right = subset[subset.condition == comparison][keys + ["value"]].rename(columns={"value": "comparison_value"})
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise AnalysisOutputError("paired metrics contain duplicate exact evaluation units")
    merged = left.merge(right, on=keys, how="outer", indicator=True, validate="one_to_one")
    if not (merged._merge == "both").all():
        example = merged.loc[merged._merge != "both", keys + ["_merge"]].head().to_dict("records")
        raise AnalysisOutputError(f"missing paired evaluation unit; no silent dropping allowed: {example}")
    merged = merged.drop(columns="_merge")
    merged["difference"] = merged.reference_value - merged.comparison_value
    merged["reference_condition"] = reference
    merged["comparison_condition"] = comparison
    return merged.sort_values(keys, kind="stable").reset_index(drop=True)


def _metric_subject(metrics: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        metrics,
        ("outer_fold", "subject_id", "region_id", "condition", "metric_name", "metric_direction", "value"),
        "metrics",
    )
    values = pd.to_numeric(metrics.value, errors="coerce")
    if not np.isfinite(values).all():
        raise AnalysisOutputError("metrics.value must be finite")
    if (metrics.groupby("metric_name").metric_direction.nunique() != 1).any():
        raise AnalysisOutputError("each metric must have exactly one metric_direction")
    return (
        metrics.groupby(["subject_id", "condition", "metric_name", "metric_direction"], as_index=False)
        .agg(value=("value", "mean"), evaluable_unit_count=("value", "size"))
        .sort_values(["metric_name", "condition", "subject_id"], kind="stable")
        .reset_index(drop=True)
    )


def table_t01(dataset: pd.DataFrame) -> pd.DataFrame:
    required = (
        "subject_id", "outer_fold", "frame_count", "valid_frame_count", "missing_ratio",
        "sampling_rate_hz", "sequence_length", "mean_motion_magnitude", "active_interval_ratio",
    )
    _require_columns(dataset, required, "dataset_summary")
    base = dataset[list(required) + (["region_id"] if "region_id" in dataset.columns else [])].copy()
    base["region_id"] = base.get("region_id", "ALL")
    base["aggregation_level"] = "subject"
    base["valid_frame_ratio"] = base.valid_frame_count / base.frame_count
    base["static_interval_ratio"] = 1.0 - base.active_interval_ratio
    base["evaluable_unit_count"] = np.where(base.valid_frame_count > 0, 1, 0)
    base["unevaluable_unit_count"] = np.where(base.valid_frame_count > 0, 0, 1)

    fold_rows = []
    for fold, group in base.groupby("outer_fold", sort=True):
        frames = float(group.frame_count.sum())
        valid = float(group.valid_frame_count.sum())
        fold_rows.append({
            "subject_id": "ALL",
            "outer_fold": fold,
            "region_id": "ALL",
            "frame_count": int(frames),
            "valid_frame_count": int(valid),
            "missing_ratio": 1.0 - valid / frames if frames else np.nan,
            "sampling_rate_hz": float(group.sampling_rate_hz.median()),
            "sequence_length": int(group.sequence_length.sum()),
            "mean_motion_magnitude": float(group.mean_motion_magnitude.mean()),
            "active_interval_ratio": float(group.active_interval_ratio.mean()),
            "aggregation_level": "outer_fold",
            "valid_frame_ratio": valid / frames if frames else np.nan,
            "static_interval_ratio": 1.0 - float(group.active_interval_ratio.mean()),
            "evaluable_unit_count": int(group.evaluable_unit_count.sum()),
            "unevaluable_unit_count": int(group.unevaluable_unit_count.sum()),
        })
    out = pd.concat([base, pd.DataFrame(fold_rows)], ignore_index=True)
    return out.sort_values(["aggregation_level", "outer_fold", "subject_id", "region_id"], kind="stable")


def table_t02(config: Mapping[str, object]) -> pd.DataFrame:
    validate_primary_config(config)
    rows: list[dict[str, object]] = []

    def walk(prefix: str, value: object) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value):
                walk(f"{prefix}.{key}" if prefix else str(key), value[key])
        elif isinstance(value, list):
            rows.append({"parameter": prefix, "value": json.dumps(value, separators=(",", ":"))})
        else:
            rows.append({"parameter": prefix, "value": value})

    walk("", config)
    rows.extend((
        {"parameter": "analysis_schema_version", "value": ANALYSIS_SCHEMA_VERSION},
        {"parameter": "analysis_code_version", "value": ANALYSIS_CODE_VERSION},
        {"parameter": "config_sha256", "value": _json_hash(config)},
    ))
    return pd.DataFrame(rows)


def table_t03(metrics: pd.DataFrame, config: Mapping[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject = _metric_subject(metrics)
    n_resamples, seed, method = _statistics_config(config)
    rows = []
    for (metric, direction, condition), group in subject.groupby(
        ["metric_name", "metric_direction", "condition"], sort=True
    ):
        if condition not in PRIMARY_CONDITIONS:
            continue
        median, low, high = _bootstrap_median(group.value, n_resamples=n_resamples, seed=seed, method=method)
        raw = metrics[(metrics.metric_name == metric) & (metrics.condition == condition)]
        rows.append({
            "condition": condition,
            "metric_name": metric,
            "metric_direction": direction,
            "median": median,
            "mean": float(group.value.mean()),
            "ci_low": low,
            "ci_high": high,
            "confidence_level": 0.95,
            "subject_count": int(group.subject_id.nunique()),
            "evaluable_unit_count": int(len(raw)),
        })
    out = pd.DataFrame(rows)
    expected = set(PRIMARY_CONDITIONS)
    for metric, group in out.groupby("metric_name"):
        if set(group.condition) != expected:
            raise AnalysisOutputError(f"T03 metric {metric} does not contain all four Primary conditions")
    return out.sort_values(["metric_name", "condition"], kind="stable"), subject


def table_t04(metrics: pd.DataFrame, config: Mapping[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric = _primary_metric(metrics, config)
    pair = _exact_pair(metrics, "self", "pcmci", metric)
    if set(pair.metric_direction) != {"lower_is_better"}:
        raise AnalysisOutputError("T04 Primary DeltaE requires lower-is-better error metric")
    n_resamples, seed, method = _statistics_config(config)
    subject = pair.groupby("subject_id", as_index=False).agg(
        difference=("difference", "mean"),
        self_error=("reference_value", "mean"),
        pcmci_error=("comparison_value", "mean"),
        evaluable_unit_count=("difference", "size"),
    )
    median, low, high = _bootstrap_median(subject.difference, n_resamples=n_resamples, seed=seed, method=method)
    summary = pd.DataFrame([{
        "metric_name": metric,
        "metric_direction": "lower_is_better",
        "effect_definition": "Self - PCMCI",
        "median_paired_difference": median,
        "mean_paired_difference": float(subject.difference.mean()),
        "ci_low": low,
        "ci_high": high,
        "confidence_level": 0.95,
        "n_subjects": int(subject.subject_id.nunique()),
        "evaluable_unit_count": int(len(pair)),
        "positive_interpretation": "PCMCI improvement",
    }])
    return summary, subject.sort_values("subject_id", kind="stable")


def table_t05(metrics: pd.DataFrame, features: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    metric = _primary_metric(metrics, config)
    pair = _exact_pair(metrics, "full", "pcmci", metric)
    _require_columns(features, ("outer_fold", "subject_id", "region_id", "condition", "feature_count"), "feature_counts")
    keys = ["outer_fold", "subject_id", "region_id"]
    full = features[features.condition == "full"][keys + ["feature_count"]].rename(columns={"feature_count": "full_feature_count"})
    pcmci = features[features.condition == "pcmci"][keys + ["feature_count"]].rename(columns={"feature_count": "pcmci_feature_count"})
    feature_pair = full.merge(pcmci, on=keys, how="outer", indicator=True, validate="one_to_one")
    if not (feature_pair._merge == "both").all():
        raise AnalysisOutputError("T05 requires exact Full/PCMCI feature-count pairs")
    feature_pair = feature_pair.drop(columns="_merge")
    joined = pair.merge(feature_pair, on=keys, how="left", validate="one_to_one")
    if joined[["full_feature_count", "pcmci_feature_count"]].isna().any().any():
        raise AnalysisOutputError("T05 feature provenance missing for a paired performance unit")
    if (joined.full_feature_count <= 0).any() or (joined.pcmci_feature_count < 0).any():
        raise AnalysisOutputError("feature counts must be non-negative and Full must be positive")
    joined["feature_ratio_pcmci_over_full"] = joined.pcmci_feature_count / joined.full_feature_count
    n_resamples, seed, method = _statistics_config(config)
    rows = []
    for region, group in joined.groupby("region_id", sort=True):
        subject = group.groupby("subject_id", as_index=False).agg(
            full_error=("reference_value", "mean"),
            pcmci_error=("comparison_value", "mean"),
            full_minus_pcmci=("difference", "mean"),
            full_features=("full_feature_count", "median"),
            pcmci_features=("pcmci_feature_count", "median"),
            feature_ratio=("feature_ratio_pcmci_over_full", "median"),
        )
        median, low, high = _bootstrap_median(subject.full_minus_pcmci, n_resamples=n_resamples, seed=seed, method=method)
        rows.append({
            "target_region": region,
            "metric_name": metric,
            "median_full_error": float(subject.full_error.median()),
            "median_pcmci_error": float(subject.pcmci_error.median()),
            "median_full_minus_pcmci": median,
            "ci_low": low,
            "ci_high": high,
            "median_full_feature_count": float(subject.full_features.median()),
            "median_pcmci_feature_count": float(subject.pcmci_features.median()),
            "median_feature_ratio_pcmci_over_full": float(subject.feature_ratio.median()),
            "n_subjects": int(subject.subject_id.nunique()),
        })
    return pd.DataFrame(rows).sort_values("target_region", kind="stable")


def table_t06(metrics: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    metric = _primary_metric(metrics, config)
    pair = _exact_pair(metrics, "self", "pcmci", metric)
    n_resamples, seed, method = _statistics_config(config)
    subject_region = pair.groupby(["subject_id", "region_id"], as_index=False).agg(difference=("difference", "mean"))
    rows = []
    for region, group in subject_region.groupby("region_id", sort=True):
        median, low, high = _bootstrap_median(group.difference, n_resamples=n_resamples, seed=seed, method=method)
        rows.append({
            "target_region": region,
            "metric_name": metric,
            "metric_direction": "lower_is_better",
            "median_self_minus_pcmci": median,
            "ci_low": low,
            "ci_high": high,
            "confidence_level": 0.95,
            "n_subjects": int(group.subject_id.nunique()),
            "evaluable_count": int(len(group)),
        })
    return pd.DataFrame(rows).sort_values("target_region", kind="stable")


def null_distribution_source(metrics: pd.DataFrame, nulls: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    metric = _primary_metric(metrics, config)
    required = (
        "outer_fold", "subject_id", "region_id", "condition", "metric_name", "metric_direction",
        "value", "replicate_id", "seed", "mapping_scope", "pcmci_feature_count", "null_feature_count",
        "input_count_preserved", "lag_identity_preserved",
    )
    _require_columns(nulls, required, "null_metrics")
    selected = nulls[nulls.metric_name == metric].copy()
    if selected.empty:
        raise AnalysisOutputError(f"null_metrics has no rows for Primary metric {metric}")
    if not (selected.mapping_scope.astype(str) == "outer_train_only").all():
        raise AnalysisOutputError("Null mapping provenance must be outer_train_only")
    if not _as_bool(selected.input_count_preserved).all():
        raise AnalysisOutputError("Primary Null controls must preserve specified input count")
    random_rows = selected[selected.condition.astype(str).str.startswith("random_region")]
    if not random_rows.empty and not _as_bool(random_rows.lag_identity_preserved).all():
        raise AnalysisOutputError("random-region Null must preserve lag identity")
    matched = selected[selected.condition == "matched_sparsity"]
    if matched.empty:
        raise AnalysisOutputError("matched_sparsity Null is required")
    if not np.array_equal(matched.pcmci_feature_count.astype(int), matched.null_feature_count.astype(int)):
        raise AnalysisOutputError("matched_sparsity must preserve PCMCI feature count exactly")
    dup_keys = ["outer_fold", "subject_id", "region_id", "condition", "metric_name", "replicate_id"]
    if selected.duplicated(dup_keys).any():
        raise AnalysisOutputError("Null replicate identity must be unique within an evaluation unit")

    keys = ["outer_fold", "subject_id", "region_id", "metric_name", "metric_direction"]
    base = metrics[(metrics.condition == "pcmci") & (metrics.metric_name == metric)][keys + ["value"]].rename(columns={"value": "pcmci_error"})
    if base.duplicated(keys).any():
        raise AnalysisOutputError("PCMCI baseline contains duplicate exact units")
    joined = selected.merge(base, on=keys, how="left", validate="many_to_one")
    if joined.pcmci_error.isna().any():
        raise AnalysisOutputError("Null condition lacks matched PCMCI outer-test support")
    joined["null_error"] = joined.value.astype(float)
    joined["null_minus_pcmci"] = joined.null_error - joined.pcmci_error
    columns = list(required) + ["pcmci_error", "null_error", "null_minus_pcmci"]
    return joined[columns].sort_values(
        ["condition", "subject_id", "region_id", "replicate_id"], kind="stable"
    ).reset_index(drop=True)


def table_t07(metrics: pd.DataFrame, nulls: pd.DataFrame, config: Mapping[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    distribution = null_distribution_source(metrics, nulls, config)
    n_resamples, seed, method = _statistics_config(config)
    rows = []
    for condition, group in distribution.groupby("condition", sort=True):
        subject = group.groupby("subject_id", as_index=False).agg(
            effect=("null_minus_pcmci", "mean"),
            replicate_observations=("replicate_id", "size"),
        )
        median, low, high = _bootstrap_median(subject.effect, n_resamples=n_resamples, seed=seed, method=method)
        rows.append({
            "null_condition": condition,
            "metric_name": str(group.metric_name.iloc[0]),
            "effect_definition": "Null - PCMCI",
            "median_null_minus_pcmci": median,
            "mean_null_minus_pcmci": float(subject.effect.mean()),
            "ci_low": low,
            "ci_high": high,
            "confidence_level": 0.95,
            "n_subjects": int(subject.subject_id.nunique()),
            "evaluable_unit_count": int(group[["outer_fold", "subject_id", "region_id"]].drop_duplicates().shape[0]),
            "replicate_observations": int(len(group)),
            "replicate_count": int(group.replicate_id.nunique()),
            "seed_count": int(group.seed.nunique()),
            "mapping_scope": "outer_train_only",
        })
    return pd.DataFrame(rows).sort_values("null_condition", kind="stable"), distribution


def lag_response_source(lag: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    validate_primary_config(config)
    required = (
        "outer_fold", "subject_id", "region_id", "delta_frames", "delta_ms", "reference_delta",
        "error", "reference_error", "delta_error", "evaluable", "same_support", "same_region_identity",
        "same_feature_count", "n_valid",
    )
    _require_columns(lag, required, "lag_response")
    kept: list[pd.DataFrame] = []
    total = 0
    unevaluable = 0
    for _, group in lag.groupby(["outer_fold", "subject_id", "region_id"], sort=True):
        total += 1
        deltas = tuple(sorted(pd.to_numeric(group.delta_frames).astype(int).tolist()))
        complete = (
            len(group) == len(LAG_GRID)
            and deltas == LAG_GRID
            and _as_bool(group.evaluable).all()
            and _as_bool(group.same_support).all()
            and _as_bool(group.same_region_identity).all()
            and _as_bool(group.same_feature_count).all()
            and group.n_valid.nunique() == 1
        )
        if not complete:
            unevaluable += 1
            continue
        reference = group[group.delta_frames.astype(int) == REFERENCE_DELTA]
        if len(reference) != 1:
            raise AnalysisOutputError("lag-response reference delta must occur exactly once")
        ref_error = float(reference.error.iloc[0])
        if not np.allclose(group.reference_error.astype(float), ref_error):
            raise AnalysisOutputError("lag-response reference_error does not reconcile")
        if not np.allclose(group.delta_error.astype(float), group.error.astype(float) - ref_error):
            raise AnalysisOutputError("lag-response delta_error does not reconcile")
        kept.append(group.copy())
    if not kept:
        raise AnalysisOutputError("no evaluable complete symmetric lag-response target-folds")
    valid = pd.concat(kept, ignore_index=True)
    n_resamples, seed, method = _statistics_config(config)
    rows = []
    for delta in LAG_GRID:
        group = valid[valid.delta_frames.astype(int) == delta]
        subject = group.groupby("subject_id", as_index=False).agg(delta_error=("delta_error", "mean"))
        median, low, high = _bootstrap_median(subject.delta_error, n_resamples=n_resamples, seed=seed, method=method)
        rows.append({
            "delta_frames": delta,
            "delta_ms": float(np.median(group.delta_ms.astype(float))),
            "median_delta_error": median,
            "ci_low": low,
            "ci_high": high,
            "confidence_level": 0.95,
            "n_subjects": int(subject.subject_id.nunique()),
            "evaluable_target_folds": total - unevaluable,
            "unevaluable_target_folds": unevaluable,
            "same_support_across_deltas": True,
            "same_region_identity": True,
            "same_feature_count": True,
            "reference_delta": REFERENCE_DELTA,
        })
    return pd.DataFrame(rows)


def table_t08(edges: pd.DataFrame) -> pd.DataFrame:
    required = (
        "source_region", "target_region", "lag", "fold_selected_count", "fold_opportunities",
        "outer_fold_selection_frequency", "bootstrap_selected_count", "bootstrap_opportunities",
        "bootstrap_selection_frequency", "evaluable",
    )
    _require_columns(edges, required, "edge_stability")
    out = edges.copy()
    evaluable = _as_bool(out.evaluable)
    if out.loc[evaluable, "lag"].astype(int).lt(1).any() or out.loc[evaluable, "lag"].astype(int).gt(TAU_MAX).any():
        raise AnalysisOutputError(f"T08/F08 lag must be in 1..{TAU_MAX}")
    for selected, opportunities, frequency in (
        ("fold_selected_count", "fold_opportunities", "outer_fold_selection_frequency"),
        ("bootstrap_selected_count", "bootstrap_opportunities", "bootstrap_selection_frequency"),
    ):
        rows = out.loc[evaluable]
        if (rows[opportunities].astype(float) <= 0).any():
            raise AnalysisOutputError(f"{opportunities} must be positive for evaluable edges")
        calculated = rows[selected].astype(float) / rows[opportunities].astype(float)
        if not np.allclose(calculated, rows[frequency].astype(float)):
            raise AnalysisOutputError(f"{frequency} does not reconcile with selected/opportunities")
    columns = [c for c in (
        "source_region", "source_dimension", "target_region", "target_dimension", "lag",
        "fold_selected_count", "fold_opportunities", "outer_fold_selection_frequency",
        "bootstrap_selected_count", "bootstrap_opportunities", "bootstrap_selection_frequency", "evaluable",
    ) if c in out.columns]
    return out[columns].sort_values(["source_region", "target_region", "lag"], kind="stable").reset_index(drop=True)


def table_t09(
    sensitivity: pd.DataFrame, *, primary_frozen: bool, synthetic: bool, allow_mock_sensitivity: bool
) -> pd.DataFrame:
    required = (
        "analysis_id", "label", "effect_self_minus_pcmci", "ci_low", "ci_high",
        "confidence_level", "metric_name", "effect_direction",
    )
    _require_columns(sensitivity, required, "sensitivity")
    if not primary_frozen and not (synthetic and allow_mock_sensitivity):
        raise AnalysisOutputError("Sensitivity outputs are forbidden before Primary freeze")
    out = sensitivity[list(required)].copy()
    primary = out[out.analysis_id == "primary_parcorr_ridge"]
    if len(primary) != 1:
        raise AnalysisOutputError("Sensitivity registry requires exactly one Primary reference")
    if not (out.confidence_level.astype(float) == 0.95).all():
        raise AnalysisOutputError("Sensitivity CI confidence_level must be explicit 0.95")
    primary_effect = float(primary.effect_self_minus_pcmci.iloc[0])
    order = {name: i for i, name in enumerate(SENSITIVITY_ORDER)}
    out["primary_reference_effect"] = primary_effect
    out["direction_consistent_with_primary"] = np.sign(out.effect_self_minus_pcmci.astype(float)) == np.sign(primary_effect)
    out["validation_scope"] = "software_only_mock" if synthetic else "post_primary_freeze_sensitivity"
    out["display_order"] = out.analysis_id.map(order).fillna(len(order)).astype(int)
    return out.sort_values(["display_order", "analysis_id"], kind="stable").reset_index(drop=True)


def edge_relation_source(t08: pd.DataFrame) -> pd.DataFrame:
    rows = []
    eligible = t08[_as_bool(t08.evaluable)]
    for (source, target), group in eligible.groupby(["source_region", "target_region"], sort=True):
        fold_opp = float(group.fold_opportunities.sum())
        boot_opp = float(group.bootstrap_opportunities.sum())
        rows.append({
            "source_region": source,
            "target_region": target,
            "fold_selection_frequency": float(group.fold_selected_count.sum()) / fold_opp,
            "bootstrap_selection_frequency": float(group.bootstrap_selected_count.sum()) / boot_opp,
            "observed": True,
        })
    return pd.DataFrame(rows)


def edge_lag_source(t08: pd.DataFrame) -> pd.DataFrame:
    eligible = t08[_as_bool(t08.evaluable)]
    relations = sorted({f"{r.source_region}→{r.target_region}" for r in eligible.itertuples()})
    observed = {
        (f"{r.source_region}→{r.target_region}", int(r.lag)):
        (float(r.outer_fold_selection_frequency), float(r.bootstrap_selection_frequency))
        for r in eligible.itertuples()
    }
    rows = []
    for relation in relations:
        for lag in range(1, TAU_MAX + 1):
            key = (relation, lag)
            frequencies = observed.get(key)
            rows.append({
                "relation": relation,
                "lag": lag,
                "fold_selection_frequency": frequencies[0] if frequencies else np.nan,
                "bootstrap_selection_frequency": frequencies[1] if frequencies else np.nan,
                "observed": frequencies is not None,
            })
    return pd.DataFrame(rows)


def metric_concordance_source(metrics: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    subject = _metric_subject(metrics)
    n_resamples, seed, method = _statistics_config(config)
    rows = []
    for (metric, direction), group in subject.groupby(["metric_name", "metric_direction"], sort=True):
        self_rows = group[group.condition == "self"][["subject_id", "value"]].rename(columns={"value": "self_value"})
        pcmci_rows = group[group.condition == "pcmci"][["subject_id", "value"]].rename(columns={"value": "pcmci_value"})
        paired = self_rows.merge(pcmci_rows, on="subject_id", how="outer", indicator=True, validate="one_to_one")
        if not (paired._merge == "both").all():
            raise AnalysisOutputError(f"F14 metric {metric} has unmatched subject support")
        raw = paired.self_value - paired.pcmci_value
        normalized = raw if direction == "lower_is_better" else -raw
        median, low, high = _bootstrap_median(normalized, n_resamples=n_resamples, seed=seed, method=method)
        rows.append({
            "metric_name": metric,
            "metric_direction": direction,
            "effect_definition": "positive = PCMCI better",
            "median_direction_normalized_effect": median,
            "ci_low": low,
            "ci_high": high,
            "n_subjects": int(len(paired)),
        })
    return pd.DataFrame(rows).sort_values("metric_name", kind="stable")


def _import_pyplot():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "pcmci-analysis-issue23-v2"
    import matplotlib.pyplot as plt
    return plt


def _title(title: str, synthetic: bool) -> str:
    return f"{title} — SYNTHETIC / NOT SCIENTIFIC" if synthetic else title


def _save_figure(fig, path: Path, *, synthetic: bool) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if synthetic:
        fig.text(0.5, 0.01, SYNTHETIC_NOTICE, ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.04 if synthetic else 0, 1, 1))
    fig.savefig(path, dpi=180)
    svg = path.with_suffix(".svg")
    fig.savefig(svg, metadata={"Date": None})
    _import_pyplot().close(fig)
    return [path, svg]


def _plot_f01(source: pd.DataFrame, path: Path, synthetic: bool) -> list[Path]:
    plt = _import_pyplot(); fig, ax = plt.subplots(figsize=(8, 5))
    conditions = [c for c in PRIMARY_CONDITIONS if c in set(source.condition)]
    for x, condition in enumerate(conditions):
        subjects = source[(source.row_type == "subject") & (source.condition == condition)]
        summary = source[(source.row_type == "summary") & (source.condition == condition)].iloc[0]
        ax.scatter(np.full(len(subjects), x), subjects.value.astype(float), alpha=0.45, s=18)
        median = float(summary.value)
        ax.errorbar([x], [median], yerr=[[median - float(summary.ci_low)], [float(summary.ci_high) - median]], fmt="o", capsize=4)
    ax.set_xticks(range(len(conditions)), [c.title() for c in conditions]); ax.set_ylabel(str(source.metric_name.iloc[0])); ax.set_title(_title("F01 Primary condition comparison", synthetic))
    return _save_figure(fig, path, synthetic=synthetic)


def _plot_f02(source: pd.DataFrame, path: Path, synthetic: bool) -> list[Path]:
    plt = _import_pyplot(); fig, ax = plt.subplots(figsize=(8, 5))
    subjects = source[source.row_type == "subject"].sort_values("subject_id", kind="stable")
    summary = source[source.row_type == "summary"].iloc[0]
    x = np.arange(len(subjects)); ax.scatter(x, subjects.value.astype(float)); ax.axhline(0, linewidth=1)
    median = float(summary.value); ax.axhline(median, linestyle="--", linewidth=1); ax.axhspan(float(summary.ci_low), float(summary.ci_high), alpha=0.15)
    ax.set_xticks(x, subjects.subject_id, rotation=45, ha="right"); ax.set_ylabel("Self - PCMCI error"); ax.set_title(_title("F02 Subject-level paired PCMCI vs Self", synthetic))
    return _save_figure(fig, path, synthetic=synthetic)


def _plot_effect(source: pd.DataFrame, label: str, effect: str, path: Path, title: str, synthetic: bool) -> list[Path]:
    plt = _import_pyplot(); fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(source) + 2)))
    s = source.reset_index(drop=True); y = np.arange(len(s)); x = s[effect].astype(float).to_numpy(); low = s.ci_low.astype(float).to_numpy(); high = s.ci_high.astype(float).to_numpy()
    ax.errorbar(x, y, xerr=[x - low, high - x], fmt="o", capsize=4); ax.axvline(0, linewidth=1); ax.set_yticks(y, s[label].astype(str)); ax.set_xlabel(effect.replace("_", " ")); ax.set_title(_title(title, synthetic))
    return _save_figure(fig, path, synthetic=synthetic)


def _plot_f04(source: pd.DataFrame, path: Path, synthetic: bool) -> list[Path]:
    plt = _import_pyplot(); fig, ax = plt.subplots(figsize=(8, 5)); s = source.sort_values("delta_frames")
    x = s.delta_frames.astype(float).to_numpy(); y = s.median_delta_error.astype(float).to_numpy(); low = s.ci_low.astype(float).to_numpy(); high = s.ci_high.astype(float).to_numpy()
    ax.errorbar(x, y, yerr=[y - low, high - y], marker="o", capsize=4); ax.axvline(0, linestyle="--", linewidth=1); ax.axhline(0, linewidth=1); ax.set_xticks(list(LAG_GRID)); ax.set_xlabel("Common lag shift Δ (frames)"); ax.set_ylabel("Error(τ*+Δ) - Error(τ*)"); ax.set_title(_title("F04 Lag-response curve", synthetic))
    return _save_figure(fig, path, synthetic=synthetic)


def _plot_f06(source: pd.DataFrame, path: Path, synthetic: bool) -> list[Path]:
    plt = _import_pyplot(); fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter(source.median_full_feature_count, source.median_full_error, label="Full"); ax.scatter(source.median_pcmci_feature_count, source.median_pcmci_error, label="PCMCI"); ax.set_xlabel("Feature count"); ax.set_ylabel("Prediction error"); ax.legend(); ax.set_title(_title("F06 Performance–sparsity", synthetic)); return _save_figure(fig, path, synthetic=synthetic)


def _plot_heatmap(source: pd.DataFrame, row: str, column: str, value: str, path: Path, title: str, synthetic: bool) -> list[Path]:
    plt = _import_pyplot(); pivot = source.pivot(index=row, columns=column, values=value); fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(pivot.columns) + 3), max(4, 0.45 * len(pivot.index) + 2))); masked = np.ma.masked_invalid(pivot.to_numpy(dtype=float)); image = ax.imshow(masked, aspect="auto", vmin=0, vmax=1); ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45, ha="right"); ax.set_yticks(np.arange(len(pivot.index)), pivot.index); fig.colorbar(image, ax=ax, label="selection frequency"); ax.set_title(_title(title, synthetic)); return _save_figure(fig, path, synthetic=synthetic)


def _plot_f09(source: pd.DataFrame, path: Path, synthetic: bool) -> list[Path]:
    plt = _import_pyplot(); fig, ax = plt.subplots(figsize=(8, 5)); s = source.sort_values("outer_fold"); ax.scatter(s.outer_fold, s.effect_self_minus_pcmci); ax.axhline(float(s.effect_self_minus_pcmci.median()), linestyle="--", linewidth=1); ax.axhline(0, linewidth=1); ax.set_xlabel("Outer fold"); ax.set_ylabel("Self - PCMCI error"); ax.set_title(_title("F09 Outer-fold effect distribution", synthetic)); return _save_figure(fig, path, synthetic=synthetic)


def _plot_f11(source: pd.DataFrame, path: Path, synthetic: bool) -> list[Path]:
    plt = _import_pyplot(); fig, ax = plt.subplots(figsize=(9, 5)); ax.plot(source.time_index, source.y_true, label="Observed"); ax.plot(source.time_index, source.self_prediction, label="Self"); ax.plot(source.time_index, source.pcmci_prediction, label="PCMCI"); ax.legend(); ax.set_xlabel("Time index"); ax.set_ylabel("Target"); ax.set_title(_title(f"F11 Forecast trajectory — {source.subject_id.iloc[0]} / {source.region_id.iloc[0]}", synthetic)); return _save_figure(fig, path, synthetic=synthetic)


def _plot_f12(source: pd.DataFrame, directory: Path, synthetic: bool) -> list[Path]:
    plt = _import_pyplot(); written: list[Path] = []; directory.mkdir(parents=True, exist_ok=True); s = source[source.aggregation_level == "subject"].sort_values("subject_id", kind="stable")
    specs = (
        ("frame_count", "F12a_subject_frame_counts.png", "Frame count"),
        ("sequence_length", "F12b_sequence_length.png", "Sequence length"),
        ("missing_ratio", "F12c_missing_ratio.png", "Missing ratio"),
        ("valid_frame_ratio", "F12d_valid_frame_ratio.png", "Valid-frame ratio"),
        ("mean_motion_magnitude", "F12e_motion_magnitude.png", "Mean motion magnitude"),
        ("active_interval_ratio", "F12f_active_interval_ratio.png", "Active interval ratio"),
    )
    for column, filename, ylabel in specs:
        fig, ax = plt.subplots(figsize=(8, 4)); x = np.arange(len(s)); ax.bar(x, s[column].astype(float)); ax.set_xticks(x, s.subject_id, rotation=45, ha="right"); ax.set_ylabel(ylabel); ax.set_title(_title(f"F12 {ylabel}", synthetic)); written.extend(_save_figure(fig, directory / filename, synthetic=synthetic))
    return written


def _plot_f13(source: pd.DataFrame, path: Path, synthetic: bool) -> list[Path]:
    plt = _import_pyplot(); fig, ax = plt.subplots(figsize=(8, 5)); labels = []; data = []
    for condition in ("self", "pcmci"):
        values = source[source.condition == condition].value.astype(float).to_numpy()
        if len(values): labels.append(condition.title()); data.append(values)
    ax.boxplot(data, tick_labels=labels, showfliers=True)
    for x, values in enumerate(data, start=1): ax.scatter(np.full(len(values), x), values, alpha=0.35, s=15)
    ax.set_ylabel(str(source.metric_name.iloc[0])); ax.set_title(_title("F13 Prediction-error distribution", synthetic)); return _save_figure(fig, path, synthetic=synthetic)


def _caption_contract(synthetic: bool) -> dict[str, str]:
    prefix = "SYNTHETIC SOFTWARE-VERIFICATION ONLY. " if synthetic else ""
    return {
        **{f"T{i:02d}": prefix + "Canonical analysis table. Report aggregation unit, evaluable support, frozen metric definition, and 95% CI where applicable." for i in range(1, 10)},
        **{f"F{i:02d}": prefix + "Rendered only from the registered canonical source CSV; plotting choices are frozen independently of outcome." for i in range(1, 15)},
    }


def generate_analysis_outputs(
    inputs: AnalysisInputs,
    *,
    repository_root: str | Path = ".",
    output_root: str | Path | None = None,
    publication_ready: bool = False,
    include_sensitivity: bool = False,
    primary_frozen: bool = False,
    allow_mock_sensitivity: bool = False,
) -> AnalysisRunResult:
    repository_root = Path(repository_root).resolve()
    with inputs.primary_config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_primary_config(config)
    frames = {
        "dataset_summary": _read_csv(inputs.dataset_summary),
        "metrics": _read_csv(inputs.metrics),
        "feature_counts": _read_csv(inputs.feature_counts),
        "null_metrics": _read_csv(inputs.null_metrics),
        "lag_response": _read_csv(inputs.lag_response),
        "edge_stability": _read_csv(inputs.edge_stability),
        "sensitivity": _read_csv(inputs.sensitivity),
        "prediction_trajectory": _read_csv(inputs.prediction_trajectory),
    }
    synthetic = _validate_provenance(frames, config)
    if synthetic and publication_ready:
        raise AnalysisOutputError("synthetic analysis outputs can never be publication-ready")
    required_root = (repository_root / ("artifacts/mock_analysis" if synthetic else "artifacts/analysis")).resolve()
    root = Path(output_root).resolve() if output_root is not None else required_root
    try:
        root.relative_to(required_root)
    except ValueError as exc:
        raise AnalysisOutputError(f"output_root must remain inside {required_root}") from exc
    tables, figures = root / "tables", root / "figures"
    primary_metric = _primary_metric(frames["metrics"], config)
    config_hash = _json_hash(config)
    seed = int(config.get("seed", 0))
    input_records = [{"name": path.name, "sha256": _file_hash(path)} for path in inputs.paths()]
    generated: list[Path] = []
    records: list[dict[str, object]] = []

    def source_names(*names: str) -> list[str]:
        mapping = {p.stem.removeprefix("mock_"): p.name for p in inputs.paths()}
        return [mapping.get(name, name) for name in names]

    def register(paths: Sequence[Path], output_id: str, sources: Sequence[str], metric: str | None, aggregation: str) -> None:
        for path in paths:
            path = Path(path)
            generated.append(path)
            records.append({
                "output_id": output_id,
                "relative_path": path.resolve().relative_to(repository_root).as_posix(),
                "sha256": _file_hash(path),
                "source_artifacts": list(sources),
                "config_sha256": config_hash,
                "seed": seed,
                "metric": metric,
                "aggregation_unit": aggregation,
                "code_version": ANALYSIS_CODE_VERSION,
                "schema_version": ANALYSIS_SCHEMA_VERSION,
                "is_synthetic": synthetic,
            })

    def write_source(frame: pd.DataFrame, path: Path, output_id: str, sources: Sequence[str], metric: str | None, aggregation: str) -> pd.DataFrame:
        paths = _write_table(frame, path, None, synthetic=synthetic); register(paths, output_id, sources, metric, aggregation); return pd.read_csv(path)

    # T01-T09 canonical tables
    t01 = table_t01(frames["dataset_summary"]); paths = _write_table(t01, tables / "T01_dataset_outer_fold_summary.csv", tables / "T01_dataset_outer_fold_summary.md", synthetic=synthetic); register(paths, "T01", source_names("dataset_summary"), None, "subject_and_outer_fold")
    t02 = table_t02(config); paths = _write_table(t02, tables / "T02_primary_frozen_configuration.csv", tables / "T02_primary_frozen_configuration.md", synthetic=synthetic); register(paths, "T02", source_names("primary_config"), None, "configuration")
    t03, t03_subject = table_t03(frames["metrics"], config); paths = _write_table(t03, tables / "T03_primary_condition_performance.csv", tables / "T03_primary_condition_performance.md", synthetic=synthetic); register(paths, "T03", source_names("metrics"), None, "outer_test_subject")
    t04, t04_subject = table_t04(frames["metrics"], config); paths = _write_table(t04, tables / "T04_pcmci_vs_self_paired_effect.csv", tables / "T04_pcmci_vs_self_paired_effect.md", synthetic=synthetic); register(paths, "T04", source_names("metrics"), primary_metric, "outer_test_subject_exact_pair")
    t05 = table_t05(frames["metrics"], frames["feature_counts"], config); paths = _write_table(t05, tables / "T05_pcmci_vs_full_sparsity.csv", tables / "T05_pcmci_vs_full_sparsity.md", synthetic=synthetic); register(paths, "T05", source_names("metrics", "feature_counts"), primary_metric, "target_region_and_outer_test_subject")
    t06 = table_t06(frames["metrics"], config); paths = _write_table(t06, tables / "T06_region_wise_effect.csv", tables / "T06_region_wise_effect.md", synthetic=synthetic); register(paths, "T06", source_names("metrics"), primary_metric, "target_region_outer_test_subject")
    t07, t07_distribution = table_t07(frames["metrics"], frames["null_metrics"], config); paths = _write_table(t07, tables / "T07_falsification_summary.csv", tables / "T07_falsification_summary.md", synthetic=synthetic); register(paths, "T07", source_names("metrics", "null_metrics"), primary_metric, "outer_test_subject"); paths = _write_table(t07_distribution, tables / "T07_falsification_distribution.csv", None, synthetic=synthetic); register(paths, "T07", source_names("metrics", "null_metrics"), primary_metric, "outer_test_unit_replicate")
    t08 = table_t08(frames["edge_stability"]); paths = _write_table(t08, tables / "T08_edge_lag_stability.csv", tables / "T08_edge_lag_stability.md", synthetic=synthetic); register(paths, "T08", source_names("edge_stability"), None, "edge_lag_opportunity")
    t09 = None
    if include_sensitivity:
        t09 = table_t09(frames["sensitivity"], primary_frozen=primary_frozen, synthetic=synthetic, allow_mock_sensitivity=allow_mock_sensitivity); paths = _write_table(t09, tables / "T09_sensitivity_summary.csv", tables / "T09_sensitivity_summary.md", synthetic=synthetic); register(paths, "T09", source_names("sensitivity"), primary_metric, "sensitivity_analysis")

    # F01: subject points + summary in one source table.
    subj = t03_subject[(t03_subject.metric_name == primary_metric) & t03_subject.condition.isin(PRIMARY_CONDITIONS)].copy()
    f01_subject = pd.DataFrame({"row_type": "subject", "condition": subj.condition, "subject_id": subj.subject_id, "metric_name": subj.metric_name, "value": subj.value, "ci_low": np.nan, "ci_high": np.nan})
    summ = t03[t03.metric_name == primary_metric].copy(); f01_summary = pd.DataFrame({"row_type": "summary", "condition": summ.condition, "subject_id": "", "metric_name": summ.metric_name, "value": summ["median"], "ci_low": summ.ci_low, "ci_high": summ.ci_high})
    f01 = write_source(pd.concat([f01_subject, f01_summary], ignore_index=True), tables / "F01_primary_condition_comparison_source.csv", "F01", source_names("metrics"), primary_metric, "outer_test_subject_plus_summary"); paths = _plot_f01(f01, figures / "F01_primary_condition_comparison.png", synthetic); register(paths, "F01", ["F01_primary_condition_comparison_source.csv"], primary_metric, "outer_test_subject_plus_summary")

    f02_subject = pd.DataFrame({"row_type": "subject", "subject_id": t04_subject.subject_id, "value": t04_subject.difference, "ci_low": np.nan, "ci_high": np.nan}); f02_summary = pd.DataFrame({"row_type": ["summary"], "subject_id": [""], "value": [float(t04.median_paired_difference.iloc[0])], "ci_low": [float(t04.ci_low.iloc[0])], "ci_high": [float(t04.ci_high.iloc[0])]}); f02 = write_source(pd.concat([f02_subject, f02_summary], ignore_index=True), tables / "F02_subject_paired_pcmci_vs_self_source.csv", "F02", source_names("metrics"), primary_metric, "outer_test_subject_exact_pair"); paths = _plot_f02(f02, figures / "F02_subject_paired_pcmci_vs_self.png", synthetic); register(paths, "F02", ["F02_subject_paired_pcmci_vs_self_source.csv"], primary_metric, "outer_test_subject_exact_pair")

    f03 = write_source(t06, tables / "F03_incremental_gain_by_region_source.csv", "F03", source_names("metrics"), primary_metric, "target_region_outer_test_subject"); paths = _plot_effect(f03, "target_region", "median_self_minus_pcmci", figures / "F03_incremental_gain_by_region.png", "F03 Incremental Gain by target region", synthetic); register(paths, "F03", ["F03_incremental_gain_by_region_source.csv"], primary_metric, "target_region_outer_test_subject")
    f04 = write_source(lag_response_source(frames["lag_response"], config), tables / "F04_lag_response_source.csv", "F04", source_names("lag_response"), primary_metric, "complete_grid_target_fold_then_subject"); paths = _plot_f04(f04, figures / "F04_lag_response_curve.png", synthetic); register(paths, "F04", ["F04_lag_response_source.csv"], primary_metric, "complete_grid_target_fold_then_subject")
    f05 = write_source(t07, tables / "F05_falsification_effect_comparison_source.csv", "F05", source_names("metrics", "null_metrics"), primary_metric, "outer_test_subject"); paths = _plot_effect(f05, "null_condition", "median_null_minus_pcmci", figures / "F05_falsification_effect_comparison.png", "F05 Null / falsification effect comparison", synthetic); register(paths, "F05", ["F05_falsification_effect_comparison_source.csv", "T07_falsification_distribution.csv"], primary_metric, "outer_test_subject")
    f06 = write_source(t05, tables / "F06_performance_sparsity_source.csv", "F06", source_names("metrics", "feature_counts"), primary_metric, "target_region_outer_test_subject"); paths = _plot_f06(f06, figures / "F06_performance_sparsity.png", synthetic); register(paths, "F06", ["F06_performance_sparsity_source.csv"], primary_metric, "target_region_outer_test_subject")
    f07 = write_source(edge_relation_source(t08), tables / "F07_region_edge_stability_source.csv", "F07", source_names("edge_stability"), None, "region_relation_opportunity"); paths = _plot_heatmap(f07, "source_region", "target_region", "fold_selection_frequency", figures / "F07_region_edge_stability_heatmap.png", "F07 Region→region edge stability", synthetic); register(paths, "F07", ["F07_region_edge_stability_source.csv"], None, "region_relation_opportunity")
    f08 = write_source(edge_lag_source(t08), tables / "F08_region_lag_stability_source.csv", "F08", source_names("edge_stability"), None, "region_relation_lag"); paths = _plot_heatmap(f08, "relation", "lag", "fold_selection_frequency", figures / "F08_region_lag_stability_heatmap.png", "F08 Region × lag stability", synthetic); register(paths, "F08", ["F08_region_lag_stability_source.csv"], None, "region_relation_lag")

    pair = _exact_pair(frames["metrics"], "self", "pcmci", primary_metric); f09_raw = pair.groupby("outer_fold", as_index=False).agg(effect_self_minus_pcmci=("difference", "mean"), n_units=("difference", "size")); f09 = write_source(f09_raw, tables / "F09_outer_fold_effect_distribution_source.csv", "F09", source_names("metrics"), primary_metric, "outer_fold"); paths = _plot_f09(f09, figures / "F09_outer_fold_effect_distribution.png", synthetic); register(paths, "F09", ["F09_outer_fold_effect_distribution_source.csv"], primary_metric, "outer_fold")

    if include_sensitivity and t09 is not None:
        f10 = write_source(t09, tables / "F10_sensitivity_forest_source.csv", "F10", source_names("sensitivity"), primary_metric, "sensitivity_analysis"); paths = _plot_effect(f10, "label", "effect_self_minus_pcmci", figures / "F10_sensitivity_forest_plot.png", "F10 Sensitivity forest plot", synthetic); register(paths, "F10", ["F10_sensitivity_forest_source.csv"], primary_metric, "sensitivity_analysis")

    trajectory = frames["prediction_trajectory"].copy(); _require_columns(trajectory, ("subject_id", "region_id", "time_index", "y_true", "self_prediction", "pcmci_prediction"), "prediction_trajectory"); choice = trajectory[["subject_id", "region_id"]].drop_duplicates().sort_values(["subject_id", "region_id"], kind="stable").iloc[0]; f11_raw = trajectory[(trajectory.subject_id == choice.subject_id) & (trajectory.region_id == choice.region_id)].sort_values("time_index", kind="stable"); f11 = write_source(f11_raw, tables / "F11_forecast_trajectory_source.csv", "F11", source_names("prediction_trajectory"), primary_metric, "predeclared_lexicographic_example"); paths = _plot_f11(f11, figures / "F11_forecast_trajectory_example.png", synthetic); register(paths, "F11", ["F11_forecast_trajectory_source.csv"], primary_metric, "predeclared_lexicographic_example")
    f12 = write_source(t01, tables / "F12_data_quality_source.csv", "F12", source_names("dataset_summary"), None, "subject_and_outer_fold"); paths = _plot_f12(f12, figures / "F12_data_quality", synthetic); register(paths, "F12", ["F12_data_quality_source.csv"], None, "subject")
    f13_raw = t03_subject[(t03_subject.metric_name == primary_metric) & t03_subject.condition.isin(["self", "pcmci"])].copy(); f13 = write_source(f13_raw, tables / "F13_prediction_error_distribution_source.csv", "F13", source_names("metrics"), primary_metric, "outer_test_subject"); paths = _plot_f13(f13, figures / "F13_prediction_error_distribution.png", synthetic); register(paths, "F13", ["F13_prediction_error_distribution_source.csv"], primary_metric, "outer_test_subject")
    f14_raw = metric_concordance_source(frames["metrics"], config); f14 = write_source(f14_raw, tables / "F14_metric_concordance_source.csv", "F14", source_names("metrics"), None, "outer_test_subject_by_metric"); paths = _plot_effect(f14, "metric_name", "median_direction_normalized_effect", figures / "F14_metric_concordance.png", "F14 Metric concordance — positive means PCMCI better", synthetic); register(paths, "F14", ["F14_metric_concordance_source.csv"], None, "outer_test_subject_by_metric")

    captions = root / "captions.json"; captions.write_text(json.dumps(_caption_contract(synthetic), indent=2, sort_keys=True) + "\n", encoding="utf-8"); register([captions], "CAPTIONS", source_names("primary_config"), None, "caption_contract")

    registry_path = root / "analysis_artifact_registry.csv"
    registry_frame = pd.DataFrame(records).sort_values(["output_id", "relative_path"], kind="stable")
    registry_frame["source_artifacts"] = registry_frame.source_artifacts.map(lambda x: json.dumps(x, separators=(",", ":")))
    registry_frame.to_csv(registry_path, index=False, lineterminator="\n")
    generated.append(registry_path)

    manifest = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "code_version": ANALYSIS_CODE_VERSION,
        "analysis_issue": 23,
        "config_sha256": config_hash,
        "seed": seed,
        "primary_metric": primary_metric,
        "input_artifacts": input_records,
        "is_synthetic": synthetic,
        "synthetic_notice": SYNTHETIC_NOTICE if synthetic else None,
        "publication_ready": bool(publication_ready and not synthetic),
        "primary_frozen": bool(primary_frozen),
        "sensitivity_included": bool(include_sensitivity),
        "sensitivity_validation_scope": "software_only_mock" if (include_sensitivity and synthetic) else ("post_primary_freeze" if include_sensitivity else "not_generated"),
        "figure_source_reconciliation": "every figure reads its serialized canonical source CSV before rendering",
        "example_selection_rule": "lexicographically first subject_id,region_id; independent of outcome",
        "artifact_registry": {"relative_path": registry_path.relative_to(repository_root).as_posix(), "sha256": _file_hash(registry_path), "record_count": len(records)},
    }
    manifest_path = root / "analysis_manifest.json"; manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"); generated.append(manifest_path)
    return AnalysisRunResult(root, manifest_path, registry_path, tuple(sorted(set(generated))), synthetic, bool(publication_ready and not synthetic))
