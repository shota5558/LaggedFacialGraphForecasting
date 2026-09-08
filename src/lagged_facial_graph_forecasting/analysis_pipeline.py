"""Canonical analysis table/figure generation for Issue #23 (T01-T09, F01-F14).

The module is intentionally downstream-only: it consumes frozen experiment artifacts,
never performs discovery/tuning, and never changes Primary configuration. Every
figure is rendered from a canonical CSV source table written by the same run.
Synthetic fixtures are provenance-isolated under artifacts/mock_analysis and cannot
be marked publication-ready.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import bootstrap as scipy_bootstrap

SYNTHETIC_NOTICE = "FAKE DATA - NOT FOR SCIENTIFIC CONCLUSIONS"
PRIMARY_CONDITIONS = ("persistence", "self", "full", "pcmci")
LAG_GRID = (-2, -1, 0, 1, 2)
REFERENCE_DELTA = 0
TAU_MAX = 10


class AnalysisOutputError(ValueError):
    """Raised when analysis artifacts violate a frozen analysis contract."""


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
            candidates = (root / f"{stem}{suffix}", root / f"mock_{stem}{suffix}")
            for candidate in candidates:
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


@dataclass(frozen=True, slots=True)
class AnalysisRunResult:
    output_root: Path
    manifest_path: Path
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


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False})


def _synthetic_status(frame: pd.DataFrame, name: str) -> bool:
    _require_columns(frame, ("is_synthetic",), name)
    flags = _bool_series(frame["is_synthetic"])
    if flags.isna().any() or flags.nunique(dropna=False) != 1:
        raise AnalysisOutputError(f"{name} mixes or has invalid synthetic provenance")
    is_synthetic = bool(flags.iloc[0])
    if is_synthetic:
        _require_columns(frame, ("synthetic_notice",), name)
        if not (frame["synthetic_notice"].astype(str) == SYNTHETIC_NOTICE).all():
            raise AnalysisOutputError(f"{name} has invalid synthetic_notice")
        if "subject_id" in frame.columns:
            subject_ids = frame["subject_id"].dropna().astype(str)
            if not subject_ids.str.startswith("MOCK_S").all():
                raise AnalysisOutputError(f"{name} synthetic subjects must use MOCK_S* IDs")
    elif "subject_id" in frame.columns:
        if frame["subject_id"].dropna().astype(str).str.startswith("MOCK_S").any():
            raise AnalysisOutputError(f"{name} real input contains reserved MOCK_S* subject")
    return is_synthetic


def _validate_provenance(frames: Mapping[str, pd.DataFrame], config: Mapping[str, object]) -> bool:
    statuses = {_synthetic_status(frame, name) for name, frame in frames.items()}
    statuses.add(bool(config.get("is_synthetic", False)))
    if len(statuses) != 1:
        raise AnalysisOutputError("synthetic and real analysis inputs must never be mixed")
    return statuses.pop()


def _canonical_json_hash(payload: Mapping[str, object]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _bootstrap_median(values: Iterable[float], *, n_resamples: int, seed: int, method: str) -> tuple[float, float, float]:
    arr = np.asarray(tuple(float(v) for v in values), dtype=float)
    if len(arr) < 2 or not np.isfinite(arr).all():
        raise AnalysisOutputError("bootstrap requires at least two finite subject-level values")
    med = float(np.median(arr))
    result = scipy_bootstrap(
        (arr,), np.median, confidence_level=0.95, n_resamples=int(n_resamples),
        method=method, vectorized=False, random_state=np.random.default_rng(int(seed)),
    )
    return med, float(result.confidence_interval.low), float(result.confidence_interval.high)


def _statistics_config(config: Mapping[str, object]) -> tuple[int, int, str]:
    stats = config.get("statistics", {})
    if not isinstance(stats, Mapping):
        raise AnalysisOutputError("statistics config must be an object")
    n = int(stats.get("bootstrap_n_resamples", 10000))
    seed = int(config.get("seed", 0))
    method = str(stats.get("bootstrap_method", "percentile"))
    if float(stats.get("bootstrap_confidence_level", 0.95)) != 0.95:
        raise AnalysisOutputError("Primary analysis requires 95% bootstrap CI")
    if n < 1 or method not in {"percentile", "basic", "BCa"}:
        raise AnalysisOutputError("invalid bootstrap settings")
    return n, seed, method


def _mark_provenance(frame: pd.DataFrame, is_synthetic: bool) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, "is_synthetic", bool(is_synthetic))
    if is_synthetic:
        out.insert(1, "synthetic_notice", SYNTHETIC_NOTICE)
    return out


def _write_table(frame: pd.DataFrame, csv_path: Path, md_path: Path | None, *, is_synthetic: bool) -> list[Path]:
    frame = _mark_provenance(frame, is_synthetic)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False, lineterminator="\n", float_format="%.12g")
    written = [csv_path]
    if md_path is not None:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        warning = f"> **{SYNTHETIC_NOTICE}**\n\n" if is_synthetic else ""
        md_path.write_text(warning + frame.to_markdown(index=False) + "\n", encoding="utf-8")
        written.append(md_path)
    return written


def _metric_subject(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, ("outer_fold", "subject_id", "region_id", "condition", "metric_name", "metric_direction", "value"), "metrics")
    if not np.isfinite(pd.to_numeric(frame["value"], errors="coerce")).all():
        raise AnalysisOutputError("metrics.value must be finite")
    directions = frame.groupby("metric_name")["metric_direction"].nunique()
    if (directions != 1).any():
        raise AnalysisOutputError("each metric must have one metric_direction")
    return (
        frame.groupby(["subject_id", "condition", "metric_name", "metric_direction"], as_index=False)
        .agg(value=("value", "mean"), evaluable_unit_count=("value", "size"))
        .sort_values(["metric_name", "condition", "subject_id"], kind="stable")
        .reset_index(drop=True)
    )


def table_t03(metrics: pd.DataFrame, config: Mapping[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject = _metric_subject(metrics)
    n_resamples, seed, method = _statistics_config(config)
    rows = []
    for (metric, direction, condition), group in subject.groupby(["metric_name", "metric_direction", "condition"], sort=True):
        if condition not in PRIMARY_CONDITIONS:
            continue
        med, low, high = _bootstrap_median(group["value"], n_resamples=n_resamples, seed=seed, method=method)
        raw_count = int(metrics[(metrics.metric_name == metric) & (metrics.condition == condition)].shape[0])
        rows.append({"condition": condition, "metric_name": metric, "metric_direction": direction,
                     "median": med, "mean": float(group.value.mean()), "ci_low": low, "ci_high": high,
                     "confidence_level": 0.95, "subject_count": int(group.subject_id.nunique()),
                     "evaluable_unit_count": raw_count})
    out = pd.DataFrame(rows)
    expected = set(PRIMARY_CONDITIONS)
    for metric, group in out.groupby("metric_name"):
        if set(group.condition) != expected:
            raise AnalysisOutputError(f"T03 metric {metric} does not contain all four Primary conditions")
    return out.sort_values(["metric_name", "condition"], kind="stable"), subject


def _exact_pair(metrics: pd.DataFrame, reference: str, comparison: str) -> pd.DataFrame:
    keys = ["outer_fold", "subject_id", "region_id", "metric_name", "metric_direction"]
    left = metrics[metrics.condition == reference][keys + ["value"]].rename(columns={"value": "reference_error"})
    right = metrics[metrics.condition == comparison][keys + ["value"]].rename(columns={"value": "comparison_error"})
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise AnalysisOutputError("paired metrics contain duplicate exact units")
    merged = left.merge(right, on=keys, how="outer", indicator=True, validate="one_to_one")
    if not (merged["_merge"] == "both").all():
        missing = merged.loc[merged._merge != "both", keys + ["_merge"]].to_dict("records")[:5]
        raise AnalysisOutputError(f"missing paired evaluation unit; no silent dropping allowed: {missing}")
    merged = merged.drop(columns="_merge")
    merged["difference"] = merged.reference_error - merged.comparison_error
    merged["reference_condition"] = reference
    merged["comparison_condition"] = comparison
    return merged.sort_values(keys, kind="stable").reset_index(drop=True)


def table_t04(metrics: pd.DataFrame, config: Mapping[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair = _exact_pair(metrics, "self", "pcmci")
    n_resamples, seed, method = _statistics_config(config)
    subject = (
        pair.groupby(["subject_id", "metric_name", "metric_direction"], as_index=False)
        .agg(difference=("difference", "mean"), reference_error=("reference_error", "mean"), comparison_error=("comparison_error", "mean"), evaluable_unit_count=("difference", "size"))
    )
    rows = []
    for (metric, direction), group in subject.groupby(["metric_name", "metric_direction"], sort=True):
        med, low, high = _bootstrap_median(group.difference, n_resamples=n_resamples, seed=seed, method=method)
        rows.append({"metric_name": metric, "metric_direction": direction, "effect_definition": "Self - PCMCI",
                     "median_paired_difference": med, "mean_paired_difference": float(group.difference.mean()),
                     "ci_low": low, "ci_high": high, "confidence_level": 0.95,
                     "n_subjects": int(group.subject_id.nunique()),
                     "evaluable_unit_count": int(pair[pair.metric_name == metric].shape[0]),
                     "positive_interpretation": "PCMCI improvement" if direction == "lower_is_better" else "CHECK_DIRECTION"})
    return pd.DataFrame(rows).sort_values("metric_name", kind="stable"), subject.sort_values(["metric_name", "subject_id"], kind="stable")


def table_t06(metrics: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    pair = _exact_pair(metrics, "self", "pcmci")
    n_resamples, seed, method = _statistics_config(config)
    subject_region = pair.groupby(["subject_id", "region_id", "metric_name", "metric_direction"], as_index=False).agg(difference=("difference", "mean"))
    rows = []
    for (region, metric, direction), group in subject_region.groupby(["region_id", "metric_name", "metric_direction"], sort=True):
        med, low, high = _bootstrap_median(group.difference, n_resamples=n_resamples, seed=seed, method=method)
        rows.append({"target_region": region, "metric_name": metric, "metric_direction": direction,
                     "median_self_minus_pcmci": med, "ci_low": low, "ci_high": high,
                     "confidence_level": 0.95, "n_subjects": int(group.subject_id.nunique()),
                     "evaluable_count": int(len(group))})
    return pd.DataFrame(rows).sort_values(["metric_name", "target_region"], kind="stable")


def table_t05(metrics: pd.DataFrame, features: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    pair = _exact_pair(metrics, "full", "pcmci")
    _require_columns(features, ("outer_fold", "subject_id", "region_id", "condition", "feature_count"), "feature_counts")
    fkeys = ["outer_fold", "subject_id", "region_id"]
    fullf = features[features.condition == "full"][fkeys + ["feature_count"]].rename(columns={"feature_count": "full_feature_count"})
    pcmcif = features[features.condition == "pcmci"][fkeys + ["feature_count"]].rename(columns={"feature_count": "pcmci_feature_count"})
    fj = fullf.merge(pcmcif, on=fkeys, how="outer", indicator=True, validate="one_to_one")
    if not (fj._merge == "both").all():
        raise AnalysisOutputError("T05 requires exact Full/PCMCI feature-count pairs")
    fj = fj.drop(columns="_merge")
    joined = pair.merge(fj, on=fkeys, how="left", validate="many_to_one")
    if joined[["full_feature_count", "pcmci_feature_count"]].isna().any().any():
        raise AnalysisOutputError("T05 feature provenance missing for a paired performance unit")
    joined["feature_ratio_pcmci_over_full"] = joined.pcmci_feature_count / joined.full_feature_count
    n_resamples, seed, method = _statistics_config(config)
    rows = []
    for (region, metric), group in joined.groupby(["region_id", "metric_name"], sort=True):
        subj = group.groupby("subject_id", as_index=False).agg(full_error=("reference_error", "mean"), pcmci_error=("comparison_error", "mean"), difference=("difference", "mean"), full_features=("full_feature_count", "median"), pcmci_features=("pcmci_feature_count", "median"), feature_ratio=("feature_ratio_pcmci_over_full", "median"))
        med, low, high = _bootstrap_median(subj.difference, n_resamples=n_resamples, seed=seed, method=method)
        rows.append({"target_region": region, "metric_name": metric, "median_full_error": float(subj.full_error.median()),
                     "median_pcmci_error": float(subj.pcmci_error.median()), "median_full_minus_pcmci": med,
                     "ci_low": low, "ci_high": high, "median_full_feature_count": float(subj.full_features.median()),
                     "median_pcmci_feature_count": float(subj.pcmci_features.median()),
                     "median_feature_ratio_pcmci_over_full": float(subj.feature_ratio.median()),
                     "n_subjects": int(subj.subject_id.nunique())})
    return pd.DataFrame(rows).sort_values(["metric_name", "target_region"], kind="stable")


def lag_response_source(lag: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    req = ("outer_fold", "subject_id", "region_id", "delta_frames", "delta_ms", "reference_delta", "error", "reference_error", "delta_error", "evaluable", "same_support", "n_valid")
    _require_columns(lag, req, "lag_response")
    cfg = config.get("lag_response", {})
    if not isinstance(cfg, Mapping):
        raise AnalysisOutputError("lag_response config must be an object")
    if tuple(int(x) for x in cfg.get("delta_frames", LAG_GRID)) != LAG_GRID or int(cfg.get("reference_delta", 0)) != 0:
        raise AnalysisOutputError("Primary lag-response grid/reference differs from frozen [-2,-1,0,1,2]/0")
    for key in ("clipping", "wrapping", "feature_dropping"):
        if str(cfg.get(key, "forbidden")) != "forbidden":
            raise AnalysisOutputError(f"Primary lag-response {key} must be forbidden")
    keys = ["outer_fold", "subject_id", "region_id"]
    valid_groups = []
    total = 0
    invalid = 0
    for _, group in lag.groupby(keys, sort=True):
        total += 1
        deltas = tuple(sorted(pd.to_numeric(group.delta_frames).astype(int).tolist()))
        flags = _bool_series(group.evaluable)
        support_flags = _bool_series(group.same_support)
        complete = deltas == LAG_GRID and len(group) == len(LAG_GRID) and flags.all() and support_flags.all() and group.n_valid.nunique() == 1
        if complete:
            ref = group[group.delta_frames == REFERENCE_DELTA]
            if len(ref) != 1:
                raise AnalysisOutputError("lag-response reference delta must occur exactly once per target-fold")
            ref_error = float(ref.error.iloc[0])
            if not np.allclose(group.reference_error.astype(float), ref_error) or not np.allclose(group.delta_error.astype(float), group.error.astype(float) - ref_error):
                raise AnalysisOutputError("lag-response delta_error/reference_error reconciliation failed")
            valid_groups.append(group.copy())
        else:
            invalid += 1
    if not valid_groups:
        raise AnalysisOutputError("no evaluable complete symmetric lag-response target-folds")
    kept = pd.concat(valid_groups, ignore_index=True)
    n_resamples, seed, method = _statistics_config(config)
    rows = []
    for delta in LAG_GRID:
        group = kept[kept.delta_frames.astype(int) == delta]
        subject = group.groupby("subject_id", as_index=False).agg(delta_error=("delta_error", "mean"))
        med, low, high = _bootstrap_median(subject.delta_error, n_resamples=n_resamples, seed=seed, method=method)
        rows.append({"delta_frames": delta, "delta_ms": float(np.median(group.delta_ms.astype(float))),
                     "median_delta_error": med, "ci_low": low, "ci_high": high, "confidence_level": 0.95,
                     "n_subjects": int(subject.subject_id.nunique()), "evaluable_target_folds": total - invalid,
                     "unevaluable_target_folds": invalid, "same_support_across_deltas": True,
                     "reference_delta": REFERENCE_DELTA})
    return pd.DataFrame(rows)


def table_t07(metrics: pd.DataFrame, nulls: pd.DataFrame, config: Mapping[str, object]) -> pd.DataFrame:
    _require_columns(nulls, ("outer_fold", "subject_id", "region_id", "condition", "metric_name", "metric_direction", "value"), "null_metrics")
    base = metrics[metrics.condition == "pcmci"][["outer_fold", "subject_id", "region_id", "metric_name", "metric_direction", "value"]].rename(columns={"value": "pcmci_error"})
    keys = ["outer_fold", "subject_id", "region_id", "metric_name", "metric_direction"]
    if base.duplicated(keys).any():
        raise AnalysisOutputError("PCMCI baseline contains duplicate units")
    agg = nulls.groupby(keys + ["condition"], as_index=False).agg(null_error=("value", "mean"), repeat_count=("value", "size"))
    joined = agg.merge(base, on=keys, how="left", validate="many_to_one")
    if joined.pcmci_error.isna().any():
        raise AnalysisOutputError("Null condition lacks matched PCMCI evaluation support")
    joined["null_minus_pcmci"] = joined.null_error - joined.pcmci_error
    n_resamples, seed, method = _statistics_config(config)
    rows = []
    for (condition, metric), group in joined.groupby(["condition", "metric_name"], sort=True):
        subject = group.groupby("subject_id", as_index=False).agg(effect=("null_minus_pcmci", "mean"), repeat_count=("repeat_count", "sum"))
        med, low, high = _bootstrap_median(subject.effect, n_resamples=n_resamples, seed=seed, method=method)
        rows.append({"null_condition": condition, "metric_name": metric, "effect_definition": "Null - PCMCI",
                     "median_null_minus_pcmci": med, "mean_null_minus_pcmci": float(subject.effect.mean()),
                     "ci_low": low, "ci_high": high, "confidence_level": 0.95,
                     "n_subjects": int(subject.subject_id.nunique()), "evaluable_unit_count": int(len(group)),
                     "repeat_observations": int(group.repeat_count.sum())})
    return pd.DataFrame(rows).sort_values(["metric_name", "null_condition"], kind="stable")


def table_t08(edges: pd.DataFrame) -> pd.DataFrame:
    req = ("source_region", "target_region", "lag", "fold_selected_count", "fold_opportunities", "outer_fold_selection_frequency", "bootstrap_selected_count", "bootstrap_opportunities", "bootstrap_selection_frequency", "evaluable")
    _require_columns(edges, req, "edge_stability")
    out = edges.copy()
    if ((out.lag.astype(int) < 1) | (out.lag.astype(int) > TAU_MAX)).any():
        raise AnalysisOutputError(f"T08/F08 lag must be in 1..{TAU_MAX}")
    for selected, opp, freq in (("fold_selected_count", "fold_opportunities", "outer_fold_selection_frequency"), ("bootstrap_selected_count", "bootstrap_opportunities", "bootstrap_selection_frequency")):
        calc = out[selected].astype(float) / out[opp].astype(float)
        if not np.allclose(calc, out[freq].astype(float)):
            raise AnalysisOutputError(f"{freq} does not reconcile with selected/opportunities")
    cols = [c for c in ("source_region", "source_dimension", "target_region", "target_dimension", "lag", "fold_selected_count", "fold_opportunities", "outer_fold_selection_frequency", "bootstrap_selected_count", "bootstrap_opportunities", "bootstrap_selection_frequency", "evaluable") if c in out.columns]
    return out[cols].sort_values(["source_region", "target_region", "lag"], kind="stable").reset_index(drop=True)


def table_t09(sensitivity: pd.DataFrame, *, primary_frozen: bool, is_synthetic: bool, allow_mock_sensitivity: bool) -> pd.DataFrame:
    req = ("analysis_id", "label", "effect_self_minus_pcmci", "ci_low", "ci_high", "confidence_level", "metric_name", "effect_direction")
    _require_columns(sensitivity, req, "sensitivity")
    if not primary_frozen and not (is_synthetic and allow_mock_sensitivity):
        raise AnalysisOutputError("Sensitivity outputs are forbidden before Primary freeze")
    out = sensitivity[list(req)].copy()
    primary = out[out.analysis_id == "primary_parcorr_ridge"]
    if len(primary) != 1:
        raise AnalysisOutputError("sensitivity registry must contain exactly one Primary ParCorr + Ridge reference")
    p = float(primary.effect_self_minus_pcmci.iloc[0])
    out["primary_reference_effect"] = p
    out["direction_consistent_with_primary"] = np.sign(out.effect_self_minus_pcmci.astype(float)) == np.sign(p)
    out["validation_scope"] = "software_only_mock" if is_synthetic else "post_primary_freeze_sensitivity"
    return out.sort_values(["analysis_id"], kind="stable").reset_index(drop=True)


def table_t01(dataset: pd.DataFrame) -> pd.DataFrame:
    req = ("subject_id", "outer_fold", "frame_count", "valid_frame_count", "missing_ratio", "sampling_rate_hz", "sequence_length", "mean_motion_magnitude", "active_interval_ratio")
    _require_columns(dataset, req, "dataset_summary")
    out = dataset[list(req)].copy().sort_values(["outer_fold", "subject_id"], kind="stable")
    out["valid_frame_ratio"] = out.valid_frame_count / out.frame_count
    return out


def table_t02(config: Mapping[str, object]) -> pd.DataFrame:
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
    rows.append({"parameter": "config_sha256", "value": _canonical_json_hash(config)})
    return pd.DataFrame(rows)


def _import_pyplot():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "pcmci-analysis-issue23-v1"
    import matplotlib.pyplot as plt
    return plt


def _synthetic_title(title: str, is_synthetic: bool) -> str:
    return f"{title} — SYNTHETIC / NOT SCIENTIFIC" if is_synthetic else title


def _save_figure(fig, path: Path, *, is_synthetic: bool) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_synthetic:
        fig.text(0.5, 0.01, SYNTHETIC_NOTICE, ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.04 if is_synthetic else 0, 1, 1))
    fig.savefig(path, dpi=180)
    svg = path.with_suffix(".svg")
    fig.savefig(svg, metadata={"Date": None})
    plt = _import_pyplot()
    plt.close(fig)
    return [path, svg]


def _figure_f01(source: pd.DataFrame, path: Path, is_synthetic: bool):
    plt = _import_pyplot(); fig, ax = plt.subplots(figsize=(8, 5))
    order = [c for c in PRIMARY_CONDITIONS if c in set(source.condition)]
    for x, condition in enumerate(order):
        row = source[source.condition == condition].iloc[0]
        ax.errorbar([x], [row["median"]], yerr=[[row["median"] - row.ci_low], [row.ci_high - row["median"]]], fmt="o", capsize=4)
    ax.set_xticks(range(len(order)), [c.title() for c in order]); ax.set_ylabel(str(source.metric_name.iloc[0])); ax.set_title(_synthetic_title("F01 Primary condition comparison", is_synthetic))
    return _save_figure(fig, path, is_synthetic=is_synthetic)


def _figure_f02(subject: pd.DataFrame, summary: pd.DataFrame, path: Path, is_synthetic: bool):
    plt = _import_pyplot(); fig, ax = plt.subplots(figsize=(8, 5)); s = subject.sort_values("subject_id")
    ax.scatter(np.arange(len(s)), s.difference); ax.axhline(0, linewidth=1); ax.axhline(float(summary.median_paired_difference.iloc[0]), linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(s)), s.subject_id, rotation=45, ha="right"); ax.set_ylabel("Self - PCMCI error"); ax.set_title(_synthetic_title("F02 Subject-level paired PCMCI vs Self", is_synthetic))
    return _save_figure(fig, path, is_synthetic=is_synthetic)


def _figure_f03(source: pd.DataFrame, path: Path, is_synthetic: bool):
    plt = _import_pyplot(); fig, ax = plt.subplots(figsize=(8, 5)); s = source.sort_values("target_region")
    x = np.arange(len(s)); y = s.median_self_minus_pcmci.to_numpy(); lo = s.ci_low.to_numpy(); hi = s.ci_high.to_numpy(); ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o", capsize=4); ax.axhline(0, linewidth=1)
    ax.set_xticks(x, s.target_region, rotation=45, ha="right"); ax.set_ylabel("Self - PCMCI error"); ax.set_title(_synthetic_title("F03 Incremental Gain by target region", is_synthetic)); return _save_figure(fig, path, is_synthetic=is_synthetic)


def _figure_f04(source: pd.DataFrame, path: Path, is_synthetic: bool):
    plt = _import_pyplot(); fig, ax = plt.subplots(figsize=(8, 5)); s = source.sort_values("delta_frames"); x = s.delta_frames.to_numpy(); y = s.median_delta_error.to_numpy(); lo = s.ci_low.to_numpy(); hi = s.ci_high.to_numpy(); ax.errorbar(x, y, yerr=[y - lo, hi - y], marker="o", capsize=4); ax.axvline(0, linestyle="--", linewidth=1); ax.axhline(0, linewidth=1); ax.set_xticks(list(LAG_GRID)); ax.set_xlabel("Common lag shift Δ (frames)"); ax.set_ylabel("Error(τ*+Δ) - Error(τ*)"); ax.set_title(_synthetic_title("F04 Lag-response curve", is_synthetic)); return _save_figure(fig, path, is_synthetic=is_synthetic)


def _figure_effect(source: pd.DataFrame, label_col: str, effect_col: str, path: Path, title: str, is_synthetic: bool):
    plt = _import_pyplot(); fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(source) + 2))); s = source.reset_index(drop=True); y = np.arange(len(s)); x = s[effect_col].astype(float).to_numpy(); ax.errorbar(x, y, xerr=[x - s.ci_low.to_numpy(), s.ci_high.to_numpy() - x], fmt="o", capsize=4); ax.axvline(0, linewidth=1); ax.set_yticks(y, s[label_col].astype(str)); ax.set_xlabel(effect_col.replace("_", " ")); ax.set_title(_synthetic_title(title, is_synthetic)); return _save_figure(fig, path, is_synthetic=is_synthetic)


def _figure_f06(source: pd.DataFrame, path: Path, is_synthetic: bool):
    plt = _import_pyplot(); fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter(source.median_full_feature_count, source.median_full_error, label="Full"); ax.scatter(source.median_pcmci_feature_count, source.median_pcmci_error, label="PCMCI"); ax.set_xlabel("Feature count"); ax.set_ylabel("Prediction error"); ax.legend(); ax.set_title(_synthetic_title("F06 Performance–sparsity", is_synthetic)); return _save_figure(fig, path, is_synthetic=is_synthetic)


def _edge_relation_source(t08: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (src, tgt), g in t08.groupby(["source_region", "target_region"], sort=True):
        fopp = float(g.fold_opportunities.sum()); bopp = float(g.bootstrap_opportunities.sum())
        rows.append({"source_region": src, "target_region": tgt, "fold_selection_frequency": float(g.fold_selected_count.sum()) / fopp if fopp else np.nan, "bootstrap_selection_frequency": float(g.bootstrap_selected_count.sum()) / bopp if bopp else np.nan, "observed": True})
    return pd.DataFrame(rows)


def _edge_lag_source(t08: pd.DataFrame) -> pd.DataFrame:
    relations = sorted({f"{r.source_region}→{r.target_region}" for r in t08.itertuples()})
    observed = {(f"{r.source_region}→{r.target_region}", int(r.lag)): float(r.outer_fold_selection_frequency) for r in t08.itertuples()}
    rows = []
    for rel in relations:
        for lag in range(1, TAU_MAX + 1):
            key = (rel, lag); rows.append({"relation": rel, "lag": lag, "fold_selection_frequency": observed.get(key, np.nan), "observed": key in observed})
    return pd.DataFrame(rows)


def _figure_heatmap(source: pd.DataFrame, row_col: str, col_col: str, value_col: str, path: Path, title: str, is_synthetic: bool):
    plt = _import_pyplot(); pivot = source.pivot(index=row_col, columns=col_col, values=value_col); fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(pivot.columns) + 3), max(4, 0.45 * len(pivot.index) + 2))); masked = np.ma.masked_invalid(pivot.to_numpy(dtype=float)); im = ax.imshow(masked, aspect="auto", vmin=0, vmax=1); ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45, ha="right"); ax.set_yticks(np.arange(len(pivot.index)), pivot.index); fig.colorbar(im, ax=ax, label="selection frequency"); ax.set_title(_synthetic_title(title, is_synthetic)); return _save_figure(fig, path, is_synthetic=is_synthetic)


def _figure_f09(source: pd.DataFrame, path: Path, is_synthetic: bool):
    plt = _import_pyplot(); fig, ax = plt.subplots(figsize=(8, 5)); s = source.sort_values("outer_fold"); ax.scatter(s.outer_fold, s.effect_self_minus_pcmci); ax.axhline(float(s.effect_self_minus_pcmci.median()), linestyle="--", linewidth=1); ax.axhline(0, linewidth=1); ax.set_xlabel("Outer fold"); ax.set_ylabel("Self - PCMCI error"); ax.set_title(_synthetic_title("F09 Outer-fold effect distribution", is_synthetic)); return _save_figure(fig, path, is_synthetic=is_synthetic)


def _figure_f11(source: pd.DataFrame, path: Path, is_synthetic: bool):
    plt = _import_pyplot(); fig, ax = plt.subplots(figsize=(9, 5)); ax.plot(source.time_index, source.y_true, label="Observed"); ax.plot(source.time_index, source.self_prediction, label="Self"); ax.plot(source.time_index, source.pcmci_prediction, label="PCMCI"); ax.legend(); ax.set_xlabel("Time index"); ax.set_ylabel("Target"); ax.set_title(_synthetic_title(f"F11 Forecast trajectory — {source.subject_id.iloc[0]} / {source.region_id.iloc[0]}", is_synthetic)); return _save_figure(fig, path, is_synthetic=is_synthetic)


def _figure_f12(dataset: pd.DataFrame, dirpath: Path, is_synthetic: bool):
    plt = _import_pyplot(); written = []; dirpath.mkdir(parents=True, exist_ok=True)
    specs = [("frame_count", "F12a_subject_frame_counts.png", "Frame count"), ("missing_ratio", "F12b_missing_ratio.png", "Missing ratio"), ("mean_motion_magnitude", "F12c_motion_magnitude.png", "Mean motion magnitude"), ("active_interval_ratio", "F12d_active_interval_ratio.png", "Active interval ratio")]
    for col, name, ylabel in specs:
        fig, ax = plt.subplots(figsize=(8, 4)); s = dataset.sort_values("subject_id"); ax.bar(np.arange(len(s)), s[col]); ax.set_xticks(np.arange(len(s)), s.subject_id, rotation=45, ha="right"); ax.set_ylabel(ylabel); ax.set_title(_synthetic_title(f"F12 {ylabel}", is_synthetic)); written.extend(_save_figure(fig, dirpath / name, is_synthetic=is_synthetic))
    return written


def _figure_f13(source: pd.DataFrame, path: Path, is_synthetic: bool):
    plt = _import_pyplot(); fig, ax = plt.subplots(figsize=(8, 5)); labels = []; data = []
    for condition in ("self", "pcmci"):
        values = source[source.condition == condition].value.astype(float).to_numpy()
        if len(values):
            labels.append(condition.title()); data.append(values)
    ax.boxplot(data, tick_labels=labels, showfliers=True); ax.set_ylabel(str(source.metric_name.iloc[0])); ax.set_title(_synthetic_title("F13 Prediction-error distribution", is_synthetic)); return _save_figure(fig, path, is_synthetic=is_synthetic)


def _caption_contract(is_synthetic: bool) -> dict[str, str]:
    prefix = "SYNTHETIC SOFTWARE-VERIFICATION ONLY. " if is_synthetic else ""
    return {f"T{i:02d}": prefix + "Canonical source table; report evaluable support and frozen statistical definitions." for i in range(1, 10)} | {f"F{i:02d}": prefix + "Figure rendered only from its registered canonical source table; no post-outcome selection." for i in range(1, 15)}


def generate_analysis_outputs(inputs: AnalysisInputs, *, repository_root: str | Path = ".", output_root: str | Path | None = None, publication_ready: bool = False, include_sensitivity: bool = False, primary_frozen: bool = False, allow_mock_sensitivity: bool = False) -> AnalysisRunResult:
    """Generate T01-T09 and F01-F14 deterministically from frozen artifacts."""
    repository_root = Path(repository_root).resolve()
    with inputs.primary_config.open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    frames = {
        "dataset_summary": _read_csv(inputs.dataset_summary), "metrics": _read_csv(inputs.metrics),
        "feature_counts": _read_csv(inputs.feature_counts), "null_metrics": _read_csv(inputs.null_metrics),
        "lag_response": _read_csv(inputs.lag_response), "edge_stability": _read_csv(inputs.edge_stability),
        "sensitivity": _read_csv(inputs.sensitivity), "prediction_trajectory": _read_csv(inputs.prediction_trajectory),
    }
    is_synthetic = _validate_provenance(frames, config)
    if is_synthetic and publication_ready:
        raise AnalysisOutputError("synthetic analysis outputs can never be publication-ready")
    default_root = repository_root / ("artifacts/mock_analysis" if is_synthetic else "artifacts/analysis")
    root = Path(output_root).resolve() if output_root is not None else default_root.resolve()
    required_root = (repository_root / ("artifacts/mock_analysis" if is_synthetic else "artifacts/analysis")).resolve()
    try:
        root.relative_to(required_root)
    except ValueError as exc:
        raise AnalysisOutputError(f"output_root must remain inside {required_root}") from exc
    tables = root / "tables"; figures = root / "figures"; artifacts: list[Path] = []

    t01 = table_t01(frames["dataset_summary"]); artifacts += _write_table(t01, tables / "T01_dataset_outer_fold_summary.csv", tables / "T01_dataset_outer_fold_summary.md", is_synthetic=is_synthetic)
    t02 = table_t02(config); artifacts += _write_table(t02, tables / "T02_primary_frozen_configuration.csv", tables / "T02_primary_frozen_configuration.md", is_synthetic=is_synthetic)
    t03, t03_subject = table_t03(frames["metrics"], config); artifacts += _write_table(t03, tables / "T03_primary_condition_performance.csv", tables / "T03_primary_condition_performance.md", is_synthetic=is_synthetic)
    t04, t04_subject = table_t04(frames["metrics"], config); artifacts += _write_table(t04, tables / "T04_pcmci_vs_self_paired_effect.csv", tables / "T04_pcmci_vs_self_paired_effect.md", is_synthetic=is_synthetic)
    t05 = table_t05(frames["metrics"], frames["feature_counts"], config); artifacts += _write_table(t05, tables / "T05_pcmci_vs_full_sparsity.csv", tables / "T05_pcmci_vs_full_sparsity.md", is_synthetic=is_synthetic)
    t06 = table_t06(frames["metrics"], config); artifacts += _write_table(t06, tables / "T06_region_wise_effect.csv", tables / "T06_region_wise_effect.md", is_synthetic=is_synthetic)
    t07 = table_t07(frames["metrics"], frames["null_metrics"], config); artifacts += _write_table(t07, tables / "T07_falsification_summary.csv", tables / "T07_falsification_summary.md", is_synthetic=is_synthetic)
    t08 = table_t08(frames["edge_stability"]); artifacts += _write_table(t08, tables / "T08_edge_lag_stability.csv", tables / "T08_edge_lag_stability.md", is_synthetic=is_synthetic)
    t09 = None
    if include_sensitivity:
        t09 = table_t09(frames["sensitivity"], primary_frozen=primary_frozen, is_synthetic=is_synthetic, allow_mock_sensitivity=allow_mock_sensitivity); artifacts += _write_table(t09, tables / "T09_sensitivity_summary.csv", tables / "T09_sensitivity_summary.md", is_synthetic=is_synthetic)

    metric = sorted(t03.metric_name.unique())[0]
    f01s = t03[t03.metric_name == metric].copy(); artifacts += _write_table(f01s, tables / "F01_primary_condition_comparison_source.csv", None, is_synthetic=is_synthetic); artifacts += _figure_f01(f01s, figures / "F01_primary_condition_comparison.png", is_synthetic)
    f02s = t04_subject[t04_subject.metric_name == metric].copy(); f02sum = t04[t04.metric_name == metric]; artifacts += _write_table(f02s, tables / "F02_subject_paired_pcmci_vs_self_source.csv", None, is_synthetic=is_synthetic); artifacts += _figure_f02(f02s, f02sum, figures / "F02_subject_paired_pcmci_vs_self.png", is_synthetic)
    f03s = t06[t06.metric_name == metric].copy(); artifacts += _write_table(f03s, tables / "F03_incremental_gain_by_region_source.csv", None, is_synthetic=is_synthetic); artifacts += _figure_f03(f03s, figures / "F03_incremental_gain_by_region.png", is_synthetic)
    f04s = lag_response_source(frames["lag_response"], config); artifacts += _write_table(f04s, tables / "F04_lag_response_source.csv", None, is_synthetic=is_synthetic); artifacts += _figure_f04(f04s, figures / "F04_lag_response_curve.png", is_synthetic)
    f05s = t07[t07.metric_name == metric].copy(); artifacts += _write_table(f05s, tables / "F05_falsification_effect_comparison_source.csv", None, is_synthetic=is_synthetic); artifacts += _figure_effect(f05s, "null_condition", "median_null_minus_pcmci", figures / "F05_falsification_effect_comparison.png", "F05 Null / falsification effect comparison", is_synthetic)
    f06s = t05[t05.metric_name == metric].copy(); artifacts += _write_table(f06s, tables / "F06_performance_sparsity_source.csv", None, is_synthetic=is_synthetic); artifacts += _figure_f06(f06s, figures / "F06_performance_sparsity.png", is_synthetic)
    f07s = _edge_relation_source(t08); artifacts += _write_table(f07s, tables / "F07_region_edge_stability_source.csv", None, is_synthetic=is_synthetic); artifacts += _figure_heatmap(f07s, "source_region", "target_region", "fold_selection_frequency", figures / "F07_region_edge_stability_heatmap.png", "F07 Region→region edge stability", is_synthetic)
    f08s = _edge_lag_source(t08); artifacts += _write_table(f08s, tables / "F08_region_lag_stability_source.csv", None, is_synthetic=is_synthetic); artifacts += _figure_heatmap(f08s, "relation", "lag", "fold_selection_frequency", figures / "F08_region_lag_stability_heatmap.png", "F08 Region × lag stability", is_synthetic)
    pair_raw = _exact_pair(frames["metrics"], "self", "pcmci"); f09s = pair_raw[pair_raw.metric_name == metric].groupby("outer_fold", as_index=False).agg(effect_self_minus_pcmci=("difference", "mean"), n_units=("difference", "size")); artifacts += _write_table(f09s, tables / "F09_outer_fold_effect_distribution_source.csv", None, is_synthetic=is_synthetic); artifacts += _figure_f09(f09s, figures / "F09_outer_fold_effect_distribution.png", is_synthetic)
    if include_sensitivity and t09 is not None:
        f10s = t09[t09.metric_name == metric].copy(); artifacts += _write_table(f10s, tables / "F10_sensitivity_forest_source.csv", None, is_synthetic=is_synthetic); artifacts += _figure_effect(f10s, "label", "effect_self_minus_pcmci", figures / "F10_sensitivity_forest_plot.png", "F10 Sensitivity forest plot", is_synthetic)
    traj = frames["prediction_trajectory"].copy(); _require_columns(traj, ("subject_id", "region_id", "time_index", "y_true", "self_prediction", "pcmci_prediction"), "prediction_trajectory")
    choice = traj[["subject_id", "region_id"]].drop_duplicates().sort_values(["subject_id", "region_id"], kind="stable").iloc[0]
    f11s = traj[(traj.subject_id == choice.subject_id) & (traj.region_id == choice.region_id)].sort_values("time_index", kind="stable")
    artifacts += _write_table(f11s.drop(columns=[c for c in ("is_synthetic", "synthetic_notice") if c in f11s.columns]), tables / "F11_forecast_trajectory_source.csv", None, is_synthetic=is_synthetic); artifacts += _figure_f11(f11s, figures / "F11_forecast_trajectory_example.png", is_synthetic)
    f12s = t01.copy(); artifacts += _write_table(f12s, tables / "F12_data_quality_source.csv", None, is_synthetic=is_synthetic); artifacts += _figure_f12(f12s, figures / "F12_data_quality", is_synthetic)
    f13s = t03_subject[(t03_subject.metric_name == metric) & (t03_subject.condition.isin(["self", "pcmci"]))].copy(); artifacts += _write_table(f13s, tables / "F13_prediction_error_distribution_source.csv", None, is_synthetic=is_synthetic); artifacts += _figure_f13(f13s, figures / "F13_prediction_error_distribution.png", is_synthetic)
    f14s = t04.copy(); artifacts += _write_table(f14s, tables / "F14_metric_concordance_source.csv", None, is_synthetic=is_synthetic); artifacts += _figure_effect(f14s, "metric_name", "median_paired_difference", figures / "F14_metric_concordance.png", "F14 Metric concordance", is_synthetic)

    captions = root / "captions.json"; captions.write_text(json.dumps(_caption_contract(is_synthetic), indent=2, sort_keys=True) + "\n", encoding="utf-8"); artifacts.append(captions)
    entries = []
    for path in sorted(set(Path(x).resolve() for x in artifacts)):
        entries.append({"relative_path": path.relative_to(repository_root).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "synthetic": is_synthetic})
    manifest = {
        "schema_version": 1, "analysis_issue": 23, "config_sha256": _canonical_json_hash(config),
        "is_synthetic": is_synthetic, "synthetic_notice": SYNTHETIC_NOTICE if is_synthetic else None,
        "publication_ready": bool(publication_ready and not is_synthetic), "primary_frozen": bool(primary_frozen),
        "sensitivity_included": bool(include_sensitivity),
        "sensitivity_validation_scope": "software_only_mock" if (include_sensitivity and is_synthetic) else ("post_primary_freeze" if include_sensitivity else "not_generated"),
        "figure_source_reconciliation": "figures rendered from registered canonical source CSVs",
        "example_selection_rule": "lexicographically first subject_id,region_id; independent of outcome",
        "artifacts": entries,
    }
    manifest_path = root / "analysis_manifest.json"; manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return AnalysisRunResult(root, manifest_path, tuple(sorted(set(Path(x) for x in artifacts))), is_synthetic, bool(publication_ready and not is_synthetic))
