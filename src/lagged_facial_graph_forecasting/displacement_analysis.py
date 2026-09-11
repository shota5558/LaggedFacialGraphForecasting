"""New-metric report reducers, isolated from the frozen legacy v6 runner.

The draft input contract is synthetic-only until the real protocol migration.
No discovery, tuning, or selection is inferred from held-out error values here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .landscape import compute_cell_gain
from .metrics import DISPLACEMENT_EUCLIDEAN_NAME

PROTOCOL = "displacement-report-draft-v1"
NOTICE = "MOCK DATA / NOT A SCIENTIFIC RESULT"
REGIONS = ("left_brow", "right_brow", "left_eyelid", "right_eyelid",
           "left_cheek", "right_cheek", "mouth", "jaw")
LABELS = ("左眉", "右眉", "左眼瞼", "右眼瞼", "左頬", "右頬", "口", "顎")
POINTS = ((336, 296, 300), (107, 66, 70), (385, 386, 380, 374),
          (158, 159, 153, 145), (425, 280), (205, 50),
          (61, 291, 13, 14, 37, 267, 84, 314), (176, 152, 400))
CONDITIONS = ("speaking", "non_speaking")
BANDS = ("0–100 ms", "100–250 ms", "250–500 ms")
UNIT = ["condition", "outer_fold", "group_id", "subject_id"]
CELL = ["target", "source", "lag"]


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    out["is_synthetic"] = True
    out["synthetic_notice"] = NOTICE
    out["protocol_id"] = PROTOCOL
    compression = {"method": "gzip", "mtime": 0} if str(path).endswith(".gz") else None
    out.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n", compression=compression)
    return path


def bootstrap_indices(subjects: pd.DataFrame, count: int, seed: int) -> np.ndarray:
    """Resample groups within fold, retaining all members and subject order."""
    if subjects.subject_id.duplicated().any():
        raise ValueError("each OOF subject must appear once")
    if subjects.groupby("group_id").outer_fold.nunique().max() != 1:
        raise ValueError("a group cannot cross folds")
    rng = np.random.default_rng(seed)
    blocks = []
    for _, fold in subjects.groupby("outer_fold", sort=True):
        groups = [g.index.to_numpy() for _, g in fold.groupby("group_id", sort=True)]
        # Ragged groups are retained without truncation; draws are stored as
        # a padded array so variable group sizes preserve subject equal weight.
        sampled = rng.integers(len(groups), size=(count, len(groups)))
        blocks.append([np.concatenate([groups[i] for i in row]) for row in sampled])
    rows = [np.concatenate([block[r] for block in blocks]) for r in range(count)]
    result = np.full((count, max(map(len, rows))), -1, dtype=int)
    for r, row in enumerate(rows):
        result[r, :len(row)] = row
    return result


def summarize(frame: pd.DataFrame, keys: list[str], value: str,
              subjects: pd.DataFrame, draws: np.ndarray) -> pd.DataFrame:
    """Pointwise nominal percentile intervals; missing/failed are not zero."""
    rows = []
    for identity, part in frame.groupby(keys, sort=True, dropna=False):
        identity = identity if isinstance(identity, tuple) else (identity,)
        if part.subject_id.duplicated().any():
            raise ValueError("aggregate within each subject before population summary")
        values = part.set_index("subject_id")[value].reindex(subjects.subject_id).to_numpy(float)
        valid = np.isfinite(values)
        n = int(valid.sum())
        groups = int(subjects.loc[valid, "group_id"].nunique())
        row = dict(zip(keys, identity))
        row.update(n_subjects=n, n_groups=groups, n_missing=len(part)-n,
                   median=float(np.median(values[valid])) if n else np.nan,
                   mean=float(np.mean(values[valid])) if n else np.nan,
                   n_positive=int((values[valid] > 0).sum()),
                   fraction_positive=float((values[valid] > 0).mean()) if n else np.nan,
                   ci_low=np.nan, ci_high=np.nan,
                   ci_status="fewer_than_10_groups", estimand=value)
        if groups >= 10:
            padded = np.append(values, np.nan)
            samples = padded[np.where(draws >= 0, draws, len(values))]
            # This boundary is explicit: do not silently drop undefined draws.
            if np.any(np.all(~np.isfinite(samples), axis=1)):
                row["ci_status"] = "undefined_resample"
            else:
                low, high = np.quantile(np.nanmedian(samples, axis=1), [0.025, 0.975], method="linear")
                row.update(ci_low=low, ci_high=high, ci_status="nominal_fixed_oof_group")
        rows.append(row)
    return pd.DataFrame(rows)


def validate_inputs(config: dict, subjects: pd.DataFrame, metrics: pd.DataFrame,
                    cells: pd.DataFrame, selection: pd.DataFrame) -> None:
    if (config.get("protocol_id") != PROTOCOL or config.get("is_synthetic") is not True
            or config.get("primary_metric") != DISPLACEMENT_EUCLIDEAN_NAME
            or config.get("publication_ready") is not False):
        raise ValueError("only the explicit new-metric synthetic draft protocol is accepted")
    if config["matched_repeats"] != 1000 or config["bootstrap_repeats"] != 10000:
        raise ValueError("draft repeat contract requires 1000 matched and 10000 bootstrap")
    for frame in (subjects, metrics, cells, selection):
        if not frame.is_synthetic.eq(True).all() or not frame.protocol_id.eq(PROTOCOL).all():
            raise ValueError("mixed or legacy provenance")
    if not subjects.subject_id.str.startswith("MOCK_").all():
        raise ValueError("mock subject identifiers required")
    if subjects.subject_id.duplicated().any() or subjects.groupby("group_id").outer_fold.nunique().max() != 1:
        raise ValueError("subject/group fold identity is not unique")
    identities = subjects.set_index("subject_id")[["group_id", "outer_fold"]]
    for frame in (metrics, cells):
        observed = frame[["subject_id", "group_id", "outer_fold"]].drop_duplicates()
        for item in observed.itertuples():
            if item.subject_id not in identities.index or tuple(identities.loc[item.subject_id]) != (item.group_id, item.outer_fold):
                raise ValueError("metric subject/group provenance mismatch")
        if not frame.metric_name.eq(DISPLACEMENT_EUCLIDEAN_NAME).all():
            raise ValueError("wrong metric")
        if not frame.status.isin(["evaluable", "unevaluable", "failed"]).all():
            raise ValueError("unknown evaluation status")
        good = frame.status.eq("evaluable")
        error_col = "error" if frame is metrics else "cell_error"
        if not np.isfinite(frame.loc[good, error_col]).all() or (frame.loc[good, error_col] < 0).any():
            raise ValueError("evaluable error must be finite and nonnegative")
        if frame.loc[~good, error_col].notna().any():
            raise ValueError("unevaluable/failed errors must remain missing")
    expected_units = {(c, s) for c in CONDITIONS for s in subjects.subject_id}
    expected_cells = {(t, s, lag) for t in REGIONS for s in REGIONS if s != t
                      for lag in range(1, config["lag_max"] + 1)}
    if set(zip(cells.condition, cells.subject_id)) != expected_units:
        raise ValueError("missing subject-condition candidate grid")
    for _, part in cells.groupby(["condition", "subject_id"]):
        if len(part) != len(expected_cells) or set(part[CELL].itertuples(index=False, name=None)) != expected_cells:
            raise ValueError("missing, duplicate, or unexpected candidate cell")
    expected_models = {(r, m) for r in REGIONS for m in ("persistence", "self", "full", "pcmci")}
    if set(zip(metrics.condition, metrics.subject_id)) != expected_units:
        raise ValueError("missing subject-condition models")
    for _, part in metrics.groupby(["condition", "subject_id"]):
        if len(part) != len(expected_models) or set(part[["target", "model"]].itertuples(index=False, name=None)) != expected_models:
            raise ValueError("missing, duplicate, or unexpected model")
    support = pd.concat([metrics[UNIT + ["support_sha256"]], cells[UNIT + ["support_sha256"]]])
    if support.groupby(UNIT).support_sha256.nunique().max() != 1:
        raise ValueError("common support differs across models/cells")
    if not support.support_sha256.str.fullmatch("[a-f0-9]{64}").all():
        raise ValueError("invalid support hash")
    if selection.duplicated(["condition", "outer_fold"] + CELL).any():
        raise ValueError("duplicate selection candidate")
    for _, part in selection.groupby(["condition", "outer_fold"]):
        if len(part) != len(expected_cells) or set(part[CELL].itertuples(index=False, name=None)) != expected_cells:
            raise ValueError("incomplete selection grid")
    if len(selection.groupby(["condition", "outer_fold"])) != len(CONDITIONS) * subjects.outer_fold.nunique():
        raise ValueError("missing selection fold/condition")
    if (selection.selected & ~selection.available).any():
        raise ValueError("an unavailable block cannot be selected")
    sizes = {r: 2 * len(p) for r, p in zip(REGIONS, POINTS)}
    if not selection.scalar_count.eq(selection.source.map(sizes)).all():
        raise ValueError("block scalar dimensions do not match point mapping")
    self_rows = metrics[metrics.model.eq("self")].set_index(UNIT + ["target"])
    for row in cells[cells.status.eq("evaluable")].itertuples():
        base = self_rows.loc[tuple(getattr(row, k) for k in UNIT) + (row.target,)]
        if not np.isclose(row.self_error, base.error, atol=1e-14, rtol=0):
            raise ValueError("cell Self error differs from 4-condition Self")
        gain = compute_cell_gain(base.error, row.cell_error,
                                 self_support_sha256=base.support_sha256,
                                 cell_support_sha256=row.support_sha256)
        if not np.isclose(row.gain, gain, atol=1e-14, rtol=0):
            raise ValueError("cell gain is inconsistent")


def band_gains(cells: pd.DataFrame, fps: float) -> pd.DataFrame:
    out = cells.copy()
    out["band"] = pd.cut(out.lag * 1000 / fps, [0, 100, 250, 500], labels=BANDS).astype(str)
    rows = []
    for key, part in out.groupby(UNIT + ["target", "source", "band"], sort=True):
        valid = part.status.eq("evaluable").all()
        rows.append(dict(zip(UNIT + ["target", "source", "band"], key),
                         gain=part.gain.mean() if valid else np.nan,
                         status="evaluable" if valid else ("failed" if part.status.eq("failed").any() else "unevaluable"),
                         n_lags=len(part), support_sha256=part.support_sha256.iloc[0]))
    return pd.DataFrame(rows)


def matched_sets(selection: pd.DataFrame, repeat_count: int, seeds) -> pd.DataFrame:
    """Draw on train-fixed membership and dimensional strata, never test gains."""
    rows = []
    for key, part in selection.groupby(["condition", "outer_fold", "target"], sort=True):
        chosen = part[part.selected]
        if chosen.empty:
            continue
        strata = [(part[part.available & part.scalar_count.eq(size)].index.to_numpy(), len(group))
                  for size, group in chosen.groupby("scalar_count", sort=True)]
        members = {i: f"{row.source}:{row.lag}" for i, row in part.iterrows()}
        for r in range(repeat_count):
            seed = seeds.seed_for(f"matched/{key}/{r}")
            rng = np.random.default_rng(seed)
            indices = []
            for pool, count in strata:
                indices.extend(rng.choice(pool, count, replace=False).tolist())
            membership = ";".join(sorted(members[i] for i in indices))
            rows.append(dict(zip(["condition", "outer_fold", "target"], key),
                             repeat=r, seed=seed, membership=membership,
                             membership_sha256=digest(membership), n_blocks=len(chosen),
                             scalar_count=int(chosen.scalar_count.sum())))
    return pd.DataFrame(rows)


def enrichment(cells: pd.DataFrame, selection: pd.DataFrame,
               mappings: pd.DataFrame, repeat_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects, repeats, cache = [], [], {}
    for key, part in cells.groupby(UNIT + ["target"], sort=True):
        meta = dict(zip(UNIT + ["target"], key))
        pick = selection[(selection.condition == key[0]) & (selection.outer_fold == key[1]) & (selection.target == key[-1])]
        chosen = pick[pick.selected]
        row = dict(meta, selected_mean=np.nan, random_mean=np.nan, enrichment=np.nan,
                   status="unevaluable_empty_selected", n_blocks=len(chosen),
                   scalar_count=int(chosen.scalar_count.sum()), n_repeats=repeat_count)
        if len(chosen):
            cache_key = (key[0], key[1], key[-1])
            if cache_key not in cache:
                maps = mappings[(mappings.condition == key[0]) & (mappings.outer_fold == key[1]) & (mappings.target == key[-1])].sort_values("repeat")
                if len(maps) != repeat_count or set(maps.repeat) != set(range(repeat_count)):
                    raise ValueError("matched repeat grid is incomplete")
                ordered = pick.sort_values(["source", "lag"])
                lookup = {(r.source, r.lag): i for i, r in enumerate(ordered.itertuples())}
                sizes, available = ordered.scalar_count.to_numpy(), ordered.available.to_numpy()
                indices = []
                for mapping in maps.itertuples():
                    members = [(s, int(lag)) for s, lag in (x.split(":") for x in mapping.membership.split(";"))]
                    if len(set(members)) != len(members) or len(members) != len(chosen):
                        raise ValueError("invalid matched cardinality")
                    positions = [lookup[m] for m in members]
                    if not available[positions].all() or sorted(sizes[positions]) != sorted(chosen.scalar_count):
                        raise ValueError("matched dimensional strata differ")
                    indices.append(positions)
                cache[cache_key] = (ordered.selected.to_numpy(), np.asarray(indices), maps)
            selected_mask, random_indices, maps = cache[cache_key]
            gains = part.sort_values(["source", "lag"]).gain.to_numpy()
            selected_mean = np.mean(gains[selected_mask])
            random_values = np.mean(gains[random_indices], axis=1)
            for mapping, random_mean in zip(maps.itertuples(), random_values):
                repeats.append(dict(meta, repeat=mapping.repeat, selected_mean=selected_mean,
                                    random_mean=random_mean, membership_sha256=mapping.membership_sha256,
                                    status="evaluable" if np.isfinite(random_mean) and np.isfinite(selected_mean) else "failed",
                                    support_sha256=part.support_sha256.iloc[0]))
            ok = np.isfinite(selected_mean) and np.isfinite(random_values).all()
            row.update(selected_mean=selected_mean, random_mean=float(np.mean(random_values)),
                       enrichment=selected_mean - np.mean(random_values) if ok else np.nan,
                       status="evaluable" if ok else "failed")
        elif not pick.available.any():
            row["status"] = "unevaluable_region"
        subjects.append(row)
    return pd.DataFrame(subjects), pd.DataFrame(repeats)


def centered_response(cells: pd.DataFrame, selection: pd.DataFrame,
                      lag_max: int, delta_radius: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    edges, audit = [], []
    for (condition, fold), part in selection.groupby(["condition", "outer_fold"], sort=True):
        chosen = part[part.selected].copy()
        interior = chosen[chosen.lag.between(1 + delta_radius, lag_max - delta_radius)]
        audit.append(dict(condition=condition, outer_fold=fold, n_selected=len(chosen),
                          n_interior=len(interior), n_boundary=len(chosen)-len(interior),
                          boundary_fraction=1-len(interior)/len(chosen) if len(chosen) else np.nan))
        for subject, subject_cells in cells[(cells.condition == condition) & (cells.outer_fold == fold)].groupby("subject_id", sort=True):
            lookup = subject_cells.set_index(CELL)
            for e in interior.itertuples():
                shifted = lookup.loc[[(e.target, e.source, e.lag+d) for d in range(-delta_radius, delta_radius+1)]]
                ok = shifted.status.eq("evaluable").all()
                center = lookup.loc[(e.target, e.source, e.lag)].cell_error
                for d, shifted_row in zip(range(-delta_radius, delta_radius+1), shifted.itertuples()):
                    edges.append(dict(condition=condition, outer_fold=fold,
                                      group_id=subject_cells.group_id.iloc[0], subject_id=subject,
                                      target=e.target, source=e.source, tau_star=e.lag, delta=d,
                                      response=shifted_row.cell_error-center if ok else np.nan,
                                      status="evaluable" if ok else "failed_complete_grid",
                                      support_sha256=shifted_row.support_sha256))
    raw = pd.DataFrame(edges)
    if raw.empty:
        return raw, pd.DataFrame(columns=UNIT + ["delta", "response", "n_pairs", "n_edges"]), pd.DataFrame(audit)
    valid = raw[raw.status.eq("evaluable")]
    # Crucial hierarchy: selected lags -> region pairs -> subject -> population.
    pairs = valid.groupby(UNIT + ["target", "source", "delta"], as_index=False).response.mean()
    subject = pairs.groupby(UNIT + ["delta"], as_index=False).agg(response=("response", "mean"), n_pairs=("response", "size"))
    counts = valid.groupby(UNIT + ["delta"], as_index=False).size().rename(columns={"size": "n_edges"})
    return raw, subject.merge(counts, on=UNIT + ["delta"], validate="one_to_one"), pd.DataFrame(audit)


def model_comparisons(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    good = metrics[metrics.status.eq("evaluable")]
    face = good.groupby(UNIT + ["model"], as_index=False).agg(error=("error", "mean"), n_targets=("target", "nunique"))
    face = face[face.n_targets.eq(len(REGIONS))].assign(target="whole_face")
    combined = pd.concat([good, face], ignore_index=True)
    wide = combined.pivot(index=UNIT + ["target"], columns="model", values="error")
    rows = []
    for reference, comparison in (("persistence", "self"), ("self", "full"), ("self", "pcmci"), ("full", "pcmci")):
        effect = (wide[reference]-wide[comparison]).rename("difference").reset_index()
        effect["contrast"] = f"{reference}-{comparison}"
        rows.append(effect)
    return combined, pd.concat(rows, ignore_index=True)
