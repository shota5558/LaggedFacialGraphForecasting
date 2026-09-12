#!/usr/bin/env python3
"""Reproduce the Japanese new-metric Markdown draft with explicit mock provenance.

No PCMCI or Ridge fitting is performed. Synthetic predictions are reconstructible
from coordinates.npz and the recorded residual scales; membership is created
before any held-out predictions. The legacy scientific freeze is never changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

from lagged_facial_graph_forecasting.displacement_analysis import (
    BANDS, CELL, CONDITIONS, NOTICE, POINTS, PROTOCOL, REGIONS, UNIT,
    band_gains, bootstrap_indices, centered_response, digest, enrichment,
    matched_sets, model_comparisons, summarize, validate_inputs, write_csv,
)
from lagged_facial_graph_forecasting.metrics import (
    DISPLACEMENT_EUCLIDEAN_NAME, displacement_mean_euclidean_error,
)
from lagged_facial_graph_forecasting.seed_registry import SeedRegistry


def generate_inputs(root: Path, seeds: SeedRegistry) -> dict:
    config = dict(protocol_id=PROTOCOL, is_synthetic=True, publication_ready=False,
                  primary_metric=DISPLACEMENT_EUCLIDEAN_NAME,
                  unit="interocular_midpoint_distance_ratio", dataset="SYNTHETIC_ONLY",
                  seed=seeds.root_seed, fps=30, horizon=1, self_history=30, lag_max=15,
                  delta_radius=3, matched_repeats=1000, bootstrap_repeats=10000,
                  discovery_repeats=100, n_subjects=30, n_groups=15,
                  n_evaluation_frames=900, bands=list(BANDS),
                  point_mapping=dict(zip(REGIONS, POINTS)),
                  selection_scope="synthetic_train_recipe_before_test_generation",
                  prediction_recipe="truth + residual_scale * unit_residual; persistence = previous truth",
                  fitting_performed=False, real_protocol_frozen=False)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    subjects = pd.DataFrame([dict(subject_id=f"MOCK_S{s+1:02d}", group_id=f"MOCK_G{s//2+1:02d}",
                                  outer_fold=(s//2)//3) for s in range(30)])
    write_csv(subjects, root / "subjects.csv")
    selected, discovery = [], {}
    for condition in CONDITIONS:
        for fold in range(5):
            rng = np.random.default_rng(seeds.seed_for(f"discovery/{condition}/{fold}"))
            groups = subjects[subjects.outer_fold != fold].group_id.unique()
            discovery[f"group_draws_{condition}_{fold}"] = rng.choice(groups, (100, len(groups)))
            statuses = np.ones(100, dtype=bool)
            if fold == 4 and condition == "non_speaking":
                statuses[[3, 47]] = False
            discovery[f"success_{condition}_{fold}"] = statuses
            for j, target in enumerate(REGIONS):
                for i, source in enumerate(REGIONS):
                    if i == j:
                        continue
                    available = not (fold == 4 and "right_cheek" in (target, source))
                    for lag in range(1, 16):
                        primary = i == (j+1) % 8 and lag in (5+(j % 3), 6+(j % 3))
                        boundary = i == (j+3) % 8 and lag == (1 if j % 2 else 15)
                        choose = available and (primary or boundary)
                        if condition == "non_speaking" and fold == 1 and target == "left_brow":
                            choose = False
                        probability = (0.72 if primary else 0.38 if boundary else 0.08) if available else 0
                        history = rng.random(100) < probability
                        history = np.where(statuses, history.astype(np.int8), -1)
                        if not available:
                            history[:] = -2
                        discovery[f"{condition}_{fold}_{target}_{source}_{lag}"] = history
                        n = 100 if available else 0
                        k = int((history == 1).sum())
                        failures = int((history == -1).sum())
                        selected.append(dict(condition=condition, outer_fold=fold, target=target, source=source,
                                             lag=lag, available=available, selected=choose,
                                             scalar_count=len(POINTS[i])*2, planned=100,
                                             opportunities=n, successes=n-failures, selected_count=k, failures=failures,
                                             frequency=k/n if n and not failures else np.nan,
                                             frequency_low=k/n if n else np.nan,
                                             frequency_high=(k+failures)/n if n else np.nan))
    selection = pd.DataFrame(selected)
    write_csv(selection, root / "selection.csv")
    np.savez_compressed(root / "mock_discovery_history.npz", **discovery)
    mappings = matched_sets(selection, 1000, seeds)
    write_csv(mappings, root / "matched_memberships.csv.gz")
    # All random memberships above are fixed before synthetic test values exist.
    coordinates, metrics, cells, quality = {}, [], [], []
    for s, subject in subjects.iterrows():
        for c, condition in enumerate(CONDITIONS):
            meta = {**subject.to_dict(), "condition": condition}
            support = digest(dict(meta, frames=list(range(31, 931)), fps=30,
                                  unavailable=["right_cheek"] if subject.outer_fold == 4 else []))
            for j, target in enumerate(REGIONS):
                prefix = f"{subject.subject_id}_{condition}_{target}"
                rng = np.random.default_rng(seeds.seed_for(f"coordinates/{prefix}"))
                n, p = config["n_evaluation_frames"], len(POINTS[j])
                t = np.arange(n+1) / 30
                phase = rng.uniform(0, 2*np.pi, (p, 2))
                truth_all = (0.025 * np.sin(t[:, None, None]*(2.2+0.2*j)+phase)
                             + rng.normal(0, 0.0035, (n+1, p, 2)))
                truth, previous = truth_all[1:], truth_all[:-1]
                residual = rng.normal(size=(n, p, 2))
                residual /= displacement_mean_euclidean_error(np.zeros_like(residual), residual)
                coordinates[prefix+"_truth"] = truth
                coordinates[prefix+"_previous"] = previous
                coordinates[prefix+"_unit_residual"] = residual
                available = not (subject.outer_fold == 4 and target == "right_cheek")
                pick = selection[(selection.condition == condition) & (selection.outer_fold == subject.outer_fold)
                                 & (selection.target == target)]
                chosen = pick[pick.selected]
                self_scale = 0.006 + 0.0003*j + 0.0004*c + 0.0003*np.sin(s)
                scales = {"self": self_scale,
                          "full": self_scale - 0.00055 + 0.0009*np.sin(0.6*s+j),
                          "pcmci": self_scale - 0.00045 + 0.00085*np.cos(0.5*s+j+c)}
                if chosen.empty:
                    scales["pcmci"] = scales["self"]
                for model in ("persistence", "self", "full", "pcmci"):
                    scale = scales.get(model, np.nan)
                    predicted = previous if model == "persistence" else truth + residual*scale
                    error = displacement_mean_euclidean_error(truth, predicted) if available else np.nan
                    metrics.append(dict(meta, target=target, model=model, error=error, residual_scale=scale,
                                        coordinates_key=prefix, n_points=p, n_valid=n if available else 0,
                                        support_sha256=support, metric_name=DISPLACEMENT_EUCLIDEAN_NAME,
                                        status="evaluable" if available else "unevaluable", reason="" if available else "mock_train_region_unavailable",
                                        self_scalar_count=2*p*30,
                                        added_blocks=(int(pick.available.sum()) if model == "full" else len(chosen) if model == "pcmci" else 0),
                                        added_scalar_count=(int(pick.loc[pick.available, "scalar_count"].sum()) if model == "full" else int(chosen.scalar_count.sum()) if model == "pcmci" else 0)))
                for i, source in enumerate(REGIONS):
                    if i == j:
                        continue
                    for lag in range(1, 16):
                        usable = available and not (subject.outer_fold == 4 and source == "right_cheek")
                        # Positive, negative, flat, heterogeneous and boundary-peaking fixtures.
                        location = 5+(j % 3) if i == (j+1) % 8 else 15 if i == (j+3) % 8 else 9
                        gain = (0.00075*np.cos((i-j)*0.65+c)*np.exp(-((lag-location)/4)**2)
                                + 0.0008*np.sin(0.35*s+i+j)*np.exp(-((lag-7)/6)**2) - 0.0001)
                        if source == "mouth" and target == "jaw":
                            gain = 0.00015*np.sin(s)  # deliberately flat lag curve
                        failed = s == 0 and c == 0 and target == "jaw" and source == "mouth" and lag == 8
                        error = displacement_mean_euclidean_error(truth, truth+residual*(self_scale-gain)) if usable and not failed else np.nan
                        cells.append(dict(meta, target=target, source=source, lag=lag, lag_ms=lag*1000/30,
                                          self_error=self_scale if available else np.nan, cell_error=error,
                                          gain=self_scale-error, residual_scale=self_scale-gain,
                                          coordinates_key=prefix, n_points=p, n_valid=n if usable else 0,
                                          support_sha256=support, metric_name=DISPLACEMENT_EUCLIDEAN_NAME,
                                          status="failed" if failed else "evaluable" if usable else "unevaluable",
                                          reason="injected_mock_cell_failure" if failed else "" if usable else "mock_train_region_unavailable"))
            quality.append(dict(meta, original_seconds=90., after_calibration_seconds=80.,
                                after_qc_seconds=72., after_boundary_history_seconds=60., common_support_seconds=30.,
                                missing_fraction=0.03+0.005*(s % 5), available_targets=7 if subject.outer_fold == 4 else 8,
                                observed_displacement_amplitude=float(np.linalg.norm(truth, axis=-1).mean()),
                                amplitude_target="jaw", provenance="simulated_QC_counts_not_video_measurements"))
    np.savez_compressed(root / "coordinates.npz", **coordinates)
    for name, rows in (("metrics", metrics), ("cells", cells), ("quality", quality)):
        write_csv(pd.DataFrame(rows), root / (name+".csv.gz" if name == "cells" else name+".csv"))
    return config


def generate_report(output: Path, *, seed: int = 20260911) -> Path:
    repo = Path(__file__).resolve().parents[1]
    output = output.resolve()
    if not output.is_relative_to(repo / "artifacts" / "mock_analysis"):
        raise ValueError("mock reports must remain under artifacts/mock_analysis")
    seeds = SeedRegistry(seed)
    input_dir, tables = output / "input", output / "tables"
    print("Generating new-metric mock predictions and train-fixed memberships", flush=True)
    config = generate_inputs(input_dir, seeds)
    subjects = pd.read_csv(input_dir / "subjects.csv")
    metrics = pd.read_csv(input_dir / "metrics.csv")
    cells = pd.read_csv(input_dir / "cells.csv.gz")
    selection = pd.read_csv(input_dir / "selection.csv")
    mappings = pd.read_csv(input_dir / "matched_memberships.csv.gz")
    validate_inputs(config, subjects, metrics, cells, selection)
    print("Computing paired effects, enrichment and complete centered grids", flush=True)
    draws = bootstrap_indices(subjects, 10000, seeds.seed_for("group_bootstrap"))
    np.save(output / "bootstrap_subject_indices.npy", draws)
    band = band_gains(cells, config["fps"])
    models, differences = model_comparisons(metrics)
    enrich, repeats = enrichment(cells, selection, mappings, 1000)
    aggregate = []
    for identity, part in enrich.groupby(UNIT, sort=True):
        intended = part[part.n_blocks.gt(0)]
        valid = intended.status.eq("evaluable").all() and len(intended) > 0
        aggregate.append(dict(zip(UNIT, identity), target="selected_targets", n_targets=len(intended),
                              target_set=";".join(sorted(intended.target)),
                              enrichment=float(intended.enrichment.mean()) if valid else np.nan,
                              selected_mean=float(intended.selected_mean.mean()) if valid else np.nan,
                              random_mean=float(intended.random_mean.mean()) if valid else np.nan,
                              status="evaluable" if valid else "failed_or_empty"))
    enrich = pd.concat([enrich, pd.DataFrame(aggregate)], ignore_index=True)
    response_raw, response, boundary = centered_response(cells, selection, 15, 3)
    counts = response_raw.assign(bad=response_raw.status.ne("evaluable")).groupby(["condition", "outer_fold"]).agg(
        failed_edge_subjects=("bad", lambda x: int(x.sum()/7)), n_subjects=("subject_id", "nunique"))
    boundary = boundary.merge(counts, on=["condition", "outer_fold"], how="left")
    source_tables = {"N_ST2_band_subject": band, "N_ST2_model_subject": models,
                     "N_ST2_paired_subject": differences, "N_ST2_enrichment_subject": enrich,
                     "N_ST2_centered_subject": response, "N_ST2_centered_edge_subject": response_raw,
                     "N_T3_boundary": boundary}
    for name, frame in source_tables.items():
        write_csv(frame, tables / (name+".csv"))
    write_csv(repeats, tables / "N_ST2_enrichment_repeats.csv.gz")
    print("Computing nominal group intervals with shared 10000 draws", flush=True)
    summaries = {"N_ST2_band_summary": summarize(band, ["condition", "target", "source", "band"], "gain", subjects, draws),
                 "N_ST2_cell_summary": summarize(cells, ["condition"]+CELL, "gain", subjects, draws),
                 "N_T2_model_errors": summarize(models, ["condition", "target", "model"], "error", subjects, draws),
                 "N_T2_paired_effects": summarize(differences, ["condition", "target", "contrast"], "difference", subjects, draws),
                 "N_T3_enrichment": summarize(enrich, ["condition", "target"], "enrichment", subjects, draws),
                 "N_ST2_centered_summary": summarize(response, ["condition", "delta"], "response", subjects, draws)}
    for name, frame in summaries.items():
        write_csv(frame, tables / (name+".csv"))
    quality = pd.read_csv(input_dir / "quality.csv")
    t1 = quality.groupby("condition", as_index=False).agg(n_subjects=("subject_id", "nunique"), n_groups=("group_id", "nunique"),
          original_seconds=("original_seconds", "sum"), after_calibration_seconds=("after_calibration_seconds", "sum"),
          after_qc_seconds=("after_qc_seconds", "sum"), after_boundary_history_seconds=("after_boundary_history_seconds", "sum"),
          common_support_seconds=("common_support_seconds", "sum"))
    write_csv(t1, tables / "N_T1_dataset_support.csv")
    for column in ("selected_mean", "random_mean"):
        extra = summarize(enrich, ["condition", "target"], column, subjects, draws)
        summaries["N_T3_enrichment"] = summaries["N_T3_enrichment"].merge(extra[["condition", "target", "median"]].rename(columns={"median": column+"_median"}), on=["condition", "target"])
    write_csv(summaries["N_T3_enrichment"], tables / "N_T3_enrichment.csv")
    audit = pd.concat([f.groupby(["condition", "status", "reason"], dropna=False).size().rename("n_records").reset_index().assign(analysis=a)
                       for a, f in (("model", metrics), ("cell", cells))], ignore_index=True)
    write_csv(audit, tables / "N_ST2_status_audit.csv")
    seeds.write_json(output / "seed_registry.json")
    from lagged_facial_graph_forecasting.displacement_report import render_report
    print("Rendering 5 main figure families and supplementary figures", flush=True)
    report = render_report(output, config)
    code_paths = [Path(__file__), repo / "src/lagged_facial_graph_forecasting/displacement_analysis.py",
                  repo / "src/lagged_facial_graph_forecasting/displacement_report.py", repo / "src/lagged_facial_graph_forecasting/metrics.py"]
    code_hashes = {p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    files = [p for p in output.rglob("*") if p.is_file() and p.name not in {"analysis_manifest.json", "analysis_artifact_registry.csv"}]
    registry = pd.DataFrame([dict(path=p.relative_to(output).as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                 bytes=p.stat().st_size, metric_name=DISPLACEMENT_EUCLIDEAN_NAME,
                                 config_sha256=hashlib.sha256((input_dir/"config.json").read_bytes()).hexdigest(),
                                 seed=seed, git_sha=git_sha, code_sha256=digest(code_hashes)) for p in sorted(files)])
    write_csv(registry, output / "analysis_artifact_registry.csv")
    manifest = dict(config, report=report.name, code_hashes=code_hashes, git_sha=git_sha,
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    expected_failure_fixture=True, report_generated=True,
                    real_data_validated=False, artifact_count=len(registry),
                    registry_sha256=hashlib.sha256((output/"analysis_artifact_registry.csv").read_bytes()).hexdigest())
    (output/"analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/mock_analysis/displacement_draft"))
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()
    print(generate_report(args.output_root, seed=args.seed))
    print(NOTICE)
