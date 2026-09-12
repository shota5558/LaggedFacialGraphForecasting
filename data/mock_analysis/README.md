# Synthetic Mock Analysis Data

> **WARNING — FAKE / SYNTHETIC DATA ONLY**
>
> Everything under this directory is intentionally fabricated for software verification.
> It is **not real facial-motion data**, **not an experiment result**, and **must not be used for scientific conclusions, effect-size reporting, model selection, manuscript claims, or Primary/Sensitivity decisions**.

## Purpose

These committed smoke fixtures plus the deterministic generator allow Issue #23 (T01–T09 / F01–F14) to be implemented and exercised before real-data Primary artifacts exist.

Every CSV row contains:

```text
is_synthetic=True
synthetic_notice=MOCK DATA / NOT A SCIENTIFIC RESULT
```

Synthetic subject IDs use the reserved prefix `MOCK_S*`.

## Committed smoke fixtures

| File | Intended verification |
|---|---|
| `mock_dataset_summary.csv` | T01 / F12 |
| `mock_primary_config.json` | T02 |
| `mock_metrics.csv` | T03 / T04 / T06 / F01 / F02 / F03 / F13 |
| `mock_feature_counts.csv` | T05 / F06 |
| `mock_null_metrics.csv` | T07 / F05 |
| `mock_lag_response.csv` | F04 |
| `mock_edge_stability.csv` | T08 / F07 / F08 |
| `mock_sensitivity.csv` | T09 / F10 |
| `mock_prediction_trajectory.csv` | F11 |

For broader F14 / multi-region checks, regenerate the richer deterministic fixture set:

```bash
python scripts/generate_mock_analysis_data.py --output-dir artifacts/mock_analysis/input
```

The generator also emits `mock_landscape.csv`, `mock_candidate_grid.json`, `mock_population.csv`, and explicit `landscape`/`population` configurations for extended-output software verification. These fixtures are intentionally generated on demand rather than committed as scientific artifacts.

Generator seed:

```text
20260908
```

## Designed synthetic behavior

The fake values are deliberately shaped so expected software behavior is easy to inspect:

- mock PCMCI error < mock Self error,
- mock PCMCI is close to mock Full with fewer features,
- lag-response has its minimum at `delta_frames=0`,
- lag-shift/random-region/matched-sparsity/time-shuffle worsen the mock score,
- several mock region-lag edges have non-zero stability,
- sensitivity contains positive and near-zero fake effects.

These are **fixture properties, not findings**.

## Required safeguards

1. Preserve synthetic provenance through all derived outputs.
2. Write mock-derived outputs under `artifacts/mock_analysis/`, never the real Primary output root.
3. Never relabel `MOCK_S*` as real subjects.
4. Never merge synthetic rows into real Primary artifacts.
5. Generated plots/tables must be visibly marked synthetic or otherwise rejected by publication-ready export.
6. Passing this fixture may mark software tests as passing, but must not mark real-data `Validated` or `publication_ready` status.
