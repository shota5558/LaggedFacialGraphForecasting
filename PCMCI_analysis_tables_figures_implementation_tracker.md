# PCMCI+ Facial Motion Research
## Analysis Tables & Figures Implementation Tracker

**Scope:** Issue #23 — T01–T09 / F01–F14 analysis-output generation pipeline  
**Branch:** `dev`  
**Implementation contract for the legacy outputs recorded below:** Issue #23 + Scientific Freeze v6.
**Last implementation audit:** 2026-09-11

**Current redesign (2026-09-11):** [新指標に基づく解析図表の再構成案](docs/analysis_tables_figures_redesign_2026-09-11.md)
reorganizes the report around pointwise mean Euclidean displacement error, full-candidate G,
subject commonality, and absolute/centered lag response. Its `N-*` outputs are a design proposal,
not implemented or scientifically validated by the checkboxes below. The current output pipeline
still fixes `velocity_rmse`; do not relabel legacy artifacts as new-metric results.

This tracker records software implementation status separately from real-data scientific validation.
Synthetic/mock success may complete `Code` and `Test`, but **must not** be used to mark real-data
`Artifact Generated`, `Validated`, `Final Status=COMPLETE`, or `publication_ready=true`.

---

# 1. Status Convention

- `[x] Code` = generation path implemented.
- `[x] Test` = synthetic/contract tests pass in repository CI.
- `[ ] Artifact Generated` = real experiment artifact has not yet been generated.
- `[ ] Validated` = real-data numerical/content validation has not yet been completed.
- `IMPLEMENTED / REAL-DATA PENDING` = software implementation complete; scientific output pending.
- `IMPLEMENTED / POST-PRIMARY PENDING` = Sensitivity software path complete; execution is intentionally deferred until Primary Freeze.
- `COMPLETE` is reserved for a real-data artifact that satisfies the publication-level completion contract.

---

# 2. Completion Contract

## Software implementation gate

- [x] T01–T09 generation paths implemented.
- [x] F01–F14 generation paths implemented.
- [x] Input schemas validated fail-closed.
- [x] Synthetic / fixture tests implemented.
- [x] Same config + same seed determinism tested.
- [x] Missing / unevaluable conditions handled explicitly.
- [x] Canonical source CSV is serialized before figure rendering.
- [x] Artifact registry records output ID, source artifact(s), config hash, seed, metric, aggregation unit, code version, schema version, synthetic provenance, and SHA256.
- [x] Analysis manifest records config/input provenance and hashes the artifact registry.
- [x] Synthetic outputs are forced under `artifacts/mock_analysis/`.
- [x] Synthetic publication-ready export is rejected.
- [x] Primary / Sensitivity execution boundary is guarded.
- [x] Caption / interpretation contract is emitted.
- [x] LANDSCAPE full-cell source, source→target/lag-band summaries, exploratory cell rank, and provenance gate are implemented.
- [x] ENRICHMENT selected/matched distribution, subject effect/CI, and provenance gate are implemented.
- [x] POPULATION edge-centered source/summary/figure, subject-cluster CI, and provenance gate are implemented.

## Real-data publication gate

- [ ] Real Primary experiment artifacts available.
- [ ] Real T01–T08 / F01–F09 / F11–F14 artifacts generated as applicable.
- [ ] Primary Freeze completed before real T09 / F10 Sensitivity generation.
- [ ] Real generated tables/figures numerically reconciled and reviewed.
- [ ] Real artifact registry / manifest archived with experiment provenance.
- [ ] Publication-ready status explicitly approved.

---

# 3. Master Status Table

| ID | Type | Deliverable | Code | Test | Artifact Generated (real) | Validated (real) | Final Status |
|---|---|---|---|---|---|---|---|
| T01 | Table | Dataset / outer-fold summary | [x] | [x] | [ ] | [ ] | IMPLEMENTED / REAL-DATA PENDING |
| T02 | Table | Primary frozen configuration | [x] | [x] | [ ] | [ ] | IMPLEMENTED / REAL-DATA PENDING |
| T03 | Table | Primary 4-condition performance | [x] | [x] | [ ] | [ ] | IMPLEMENTED / REAL-DATA PENDING |
| T04 | Table | PCMCI vs Self paired effect | [x] | [x] | [ ] | [ ] | IMPLEMENTED / REAL-DATA PENDING |
| T05 | Table | PCMCI vs Full + sparsity | [x] | [x] | [ ] | [ ] | IMPLEMENTED / REAL-DATA PENDING |
| T06 | Table | Region-wise effect | [x] | [x] | [ ] | [ ] | IMPLEMENTED / REAL-DATA PENDING |
| T07 | Table | Null / falsification summary | [x] | [x] | [ ] | [ ] | IMPLEMENTED / REAL-DATA PENDING |
| T08 | Table | Edge / lag stability | [x] | [x] | [ ] | [ ] | IMPLEMENTED / REAL-DATA PENDING |
| T09 | Table | Sensitivity summary | [x] | [x] | [ ] | [ ] | IMPLEMENTED / POST-PRIMARY PENDING |
| F01 | Figure | Primary condition comparison | [x] | [x] | [ ] | [ ] | IMPLEMENTED / REAL-DATA PENDING |
| F02 | Figure | Subject-level paired PCMCI vs Self | [x] | [x] | [ ] | [ ] | IMPLEMENTED / REAL-DATA PENDING |
| F03 | Figure | Incremental Gain by target region | [x] | [x] | [ ] | [ ] | IMPLEMENTED / REAL-DATA PENDING |
| F04 | Figure | Lag-response curve | [x] | [x] | [ ] | [ ] | IMPLEMENTED / REAL-DATA PENDING |
| F05 | Figure | Null / falsification effect comparison | [x] | [x] | [ ] | [ ] | IMPLEMENTED / REAL-DATA PENDING |
| F06 | Figure | Performance–sparsity plot | [x] | [x] | [ ] | [ ] | IMPLEMENTED / REAL-DATA PENDING |
| F07 | Figure | Region→region edge stability heatmap | [x] | [x] | [ ] | [ ] | IMPLEMENTED / REAL-DATA PENDING |
| F08 | Figure | Region × lag stability heatmap | [x] | [x] | [ ] | [ ] | IMPLEMENTED / REAL-DATA PENDING |
| F09 | Figure | Outer-fold effect distribution | [x] | [x] | [ ] | [ ] | IMPLEMENTED / REAL-DATA PENDING |
| F10 | Figure | Sensitivity forest plot | [x] | [x] | [ ] | [ ] | IMPLEMENTED / POST-PRIMARY PENDING |
| F11 | Figure | Forecast trajectory example | [x] | [x] | [ ] | [ ] | IMPLEMENTED / REAL-DATA PENDING |
| F12 | Figure | Data-quality / motion diagnostics | [x] | [x] | [ ] | [ ] | IMPLEMENTED / REAL-DATA PENDING |
| F13 | Figure | Prediction-error distribution | [x] | [x] | [ ] | [ ] | IMPLEMENTED / REAL-DATA PENDING |
| F14 | Figure | Metric concordance plot | [x] | [x] | [ ] | [ ] | IMPLEMENTED / REAL-DATA PENDING |
| LANDSCAPE | Extended | Full-cell gain, source→target/lag-band summaries, exploratory rank, provenance | [x] | [x] | [ ] | [ ] | IMPLEMENTED / REAL-DATA PENDING |
| ENRICHMENT | Extended | Selected/matched enrichment distribution, subject effect, CI | [x] | [x] | [ ] | [ ] | IMPLEMENTED / REAL-DATA PENDING |
| POPULATION | Extended | Edge-centered response curve, CI, denominators, provenance | [x] | [x] | [ ] | [ ] | IMPLEMENTED / REAL-DATA PENDING |

---

# 4. Canonical Output Contract

Real-data output root:

```text
artifacts/analysis/
```

Synthetic verification output root:

```text
artifacts/mock_analysis/
```

The implementation writes canonical source tables before plotting. Major outputs include:

```text
T01  tables/T01_dataset_outer_fold_summary.csv
T02  tables/T02_primary_frozen_configuration.csv
T03  tables/T03_primary_condition_performance.csv
T04  tables/T04_pcmci_vs_self_paired_effect.csv
T05  tables/T05_pcmci_vs_full_sparsity.csv
T06  tables/T06_region_wise_effect.csv
T07  tables/T07_falsification_summary.csv
     tables/T07_falsification_distribution.csv
T08  tables/T08_edge_lag_stability.csv
T09  tables/T09_sensitivity_summary.csv

F01  tables/F01_primary_condition_comparison_source.csv
F02  tables/F02_subject_paired_pcmci_vs_self_source.csv
F03  tables/F03_incremental_gain_by_region_source.csv
F04  tables/F04_lag_response_source.csv
F05  tables/F05_falsification_effect_comparison_source.csv
F06  tables/F06_performance_sparsity_source.csv
F07  tables/F07_region_edge_stability_source.csv
F08  tables/F08_region_lag_stability_source.csv
F09  tables/F09_outer_fold_effect_distribution_source.csv
F10  tables/F10_sensitivity_forest_source.csv
F11  tables/F11_forecast_trajectory_source.csv
F12  tables/F12_data_quality_source.csv
F13  tables/F13_prediction_error_distribution_source.csv
F14  tables/F14_metric_concordance_source.csv

LANDSCAPE  tables/LANDSCAPE_cell_gain_source.csv
           tables/LANDSCAPE_cell_summary.csv
           tables/LANDSCAPE_source_target_summary.csv
           tables/LANDSCAPE_lag_band_summary.csv
ENRICHMENT  tables/ENRICHMENT_distribution_source.csv
            tables/ENRICHMENT_subject_summary.csv
POPULATION  tables/POPULATION_edge_subject_response_source.csv
            tables/POPULATION_response_summary.csv

registry: analysis_artifact_registry.csv
manifest: analysis_manifest.json
captions: captions.json
```

PNG and deterministic SVG renderings are generated for figures. F12 is a diagnostic figure directory containing multiple component plots.

Extended ENRICHMENT output is software-complete only: real artifact generation and numerical review remain blocked until the Primary run/statistics dependency (#19) supplies a validated input set.

Extended POPULATION output is software-complete only: real artifact generation and numerical review remain blocked until the Primary run/statistics dependency (#19) supplies a validated input set.

Extended LANDSCAPE output is software-complete only: real artifact generation and numerical review remain blocked until the Primary run/statistics dependency (#19) supplies a validated input set.

---

# 5. Primary Statistical Contract — Gate A

Primary paired effect:

```text
DeltaE = Error_Self - Error_PCMCI
```

Interpretation:

```text
DeltaE > 0  -> PCMCI improvement
DeltaE = 0  -> no difference
DeltaE < 0  -> Self better
```

- [x] Exact outer-test subject/unit pairing.
- [x] Unmatched pair rejection; no silent dropping.
- [x] Median paired difference.
- [x] Mean paired difference where required.
- [x] 95% bootstrap CI.
- [x] Shared sign convention across T04 / F02 / F03 / F10.
- [x] F01 contains subject points + median + CI.
- [x] F02 contains subject effects + zero reference + median + CI band.

**Gate A: SOFTWARE PASS**

---

# 6. Lag-Response Contract — Gate B

Frozen Primary grid:

```yaml
delta_frames: [-2, -1, 0, 1, 2]
reference_delta: 0
shift_mode: common_shift_all_selected_parents
```

- [x] Complete symmetric grid required.
- [x] Same target-region identity required.
- [x] Same feature count required.
- [x] Same support / `n_valid` required across deltas.
- [x] Clipping forbidden.
- [x] Wrapping forbidden.
- [x] Feature dropping forbidden.
- [x] Incomplete target-fold is excluded as unevaluable rather than clipped.
- [x] Unevaluable target-fold count exported.
- [x] Frame and millisecond lag values exported.
- [x] `DeltaError(delta) = Error(tau* + delta) - Error(tau*)` reconciled against reference error.

**Gate B: SOFTWARE PASS**

---

# 7. Falsification Contract — Gate C

Primary Null controls:

```text
lag-shift
random-region
matched-sparsity
time-shuffle
```

Sensitivity-only temporal controls remain outside Primary.

- [x] `mapping_scope=outer_train_only` required.
- [x] Matched-sparsity requires `null_feature_count == pcmci_feature_count` exactly.
- [x] Random-region requires lag identity preservation.
- [x] Required input-count preservation validated.
- [x] Replicate identity retained.
- [x] Seed retained.
- [x] Distribution source artifact retained in addition to aggregate statistic.
- [x] Null outer-test support must match PCMCI support.

**Gate C: SOFTWARE PASS**

---

# 8. Region, Stability, Sparsity — Gates D/E/F

## Gate D — Region-level effect

- [x] T06 region-wise paired effect.
- [x] F03 region-wise effect with CI and zero reference.

**Gate D: SOFTWARE PASS**

## Gate E — Discovery stability

- [x] Source region / target region / lag retained.
- [x] Selected count and opportunity denominator retained.
- [x] Outer-fold frequency reconciled from counts.
- [x] Bootstrap frequency reconciled from counts.
- [x] `tau_max=10` enforced.
- [x] Missing/not-observed cells represented as missing (`NaN`, `observed=False`).
- [x] Observed zero-frequency cells remain explicit zero (`observed=True`).
- [x] Deterministic region / relation / lag ordering.

**Gate E: SOFTWARE PASS**

## Gate F — Sparsity

- [x] Full vs PCMCI performance exact pairing.
- [x] Full / PCMCI feature-count provenance join.
- [x] PCMCI/Full feature ratio.
- [x] Performance–sparsity source and plot.

**Gate F: SOFTWARE PASS**

---

# 9. Sensitivity — Gate G

- [x] T09 generation path implemented.
- [x] F10 generation path implemented.
- [x] Primary reference is explicit.
- [x] Sensitivity order deterministic.
- [x] Effect/CI extraction explicit.
- [x] Direction consistency field emitted.
- [x] Real Sensitivity generation fails before `primary_frozen=True`.
- [x] Synthetic software verification requires explicit `allow_mock_sensitivity=True` override.
- [x] Primary outputs can be generated with `include_sensitivity=False` and no T09/F10 output is produced.

**Gate G: SOFTWARE PASS / REAL EXECUTION DEFERRED UNTIL PRIMARY FREEZE**

---

# 10. Diagnostics / Supporting — Gate H

- [x] T01 dataset / fold summary.
- [x] T02 frozen configuration + config hash.
- [x] F09 outer-fold effect distribution.
- [x] F11 example selection is lexicographic (`subject_id`, `region_id`) and outcome-independent.
- [x] F12 frame count / sequence length / missingness / valid ratio / motion / activity diagnostics.
- [x] F13 subject-level Self vs PCMCI error distribution.
- [x] F14 metric registry through input metrics, metric-direction normalization, effect + CI.
- [x] Positive F14 effect is normalized to mean `PCMCI better` for both lower- and higher-is-better metrics.

**Gate H: SOFTWARE PASS**

## Extended ENRICHMENT — Gate I

- [x] Selected and matched repeat distribution is retained with seed and membership hash.
- [x] Subject-level selected-minus-matched effect uses the configured repeat aggregation.
- [x] Subject bootstrap CI and evaluable/unevaluable denominators are emitted.
- [x] Run/config/protocol/source/support provenance and failure counts are preserved.
- [x] Figure is rendered from the serialized canonical source tables and visibly marks mock data.
- [ ] Real artifact generated and reviewed after #19 / Primary statistics.

**Gate I: SOFTWARE PASS / REAL-DATA PENDING**

## Extended POPULATION — Gate J

- [x] Edge/subject raw records require run/config/source/support and frozen metric provenance.
- [x] Exact lag response, Δ=0 reference, frame/ms units, edge/subject denominators, and failed/unevaluable counts are emitted.
- [x] Subject-cluster percentile CI resamples all edges attached to each subject and preserves the configured point estimator.
- [x] Canonical CSVs precede PNG/SVG rendering; registry, manifest, caption, and synthetic safeguards use independent `POPULATION` output ID.
- [ ] Real artifact generated, numerically reconciled, visually reviewed, and accepted after #19 / Primary statistics.

**Gate J: SOFTWARE PASS / REAL-DATA PENDING**

## Extended LANDSCAPE — Gate K

- [x] Every approved candidate cell is retained as evaluable, unevaluable, or failed and `G = E_self − E_cell` is reconciled.
- [x] Explicit non-overlapping lag bands, source→target summaries, subject/cell denominators, and exploratory exact-cell rank are emitted.
- [x] Run/config/protocol/source/support hashes, full git SHA, seed, metric, estimand, and failure counts are preserved in canonical outputs, registry, and manifest.
- [x] Canonical CSVs precede the landscape heatmap; mock outputs carry the synthetic warning and use independent `LANDSCAPE` output ID.
- [ ] Real artifact generated, numerically reconciled, visually reviewed, and accepted after #19 / Primary statistics.

**Gate K: SOFTWARE PASS / REAL-DATA PENDING**

---

# 11. Synthetic Mock Verification Dataset

Source directory:

```text
data/mock_analysis/
```

Mandatory provenance:

```text
is_synthetic=True
synthetic_notice=MOCK DATA / NOT A SCIENTIFIC RESULT
```

- [x] Reserved `MOCK_S*` subject IDs.
- [x] Deterministic generator seed `20260908`.
- [x] Mock primary metric explicitly `velocity_rmse`.
- [x] Mock Null schema contains outer-train mapping, feature-count, input-count, lag-identity, replicate, and seed provenance.
- [x] Mock lag-response schema contains `same_support`, `same_region_identity`, and `same_feature_count`.
- [x] Generator emits the same Issue #23 v2 contract fields.
- [x] Mock-derived tables preserve synthetic provenance.
- [x] Mock-derived plots are visibly marked synthetic.
- [x] Mock outputs are restricted to `artifacts/mock_analysis/`.
- [x] Publication-ready mock export rejected.

Mock results are **software verification only** and are not scientific evidence.

---

# 12. CI Evidence

Latest repository-wide Issue #23 verification on `dev`:

```text
commit: 2d8df71725601b617944090e4771d975ccc4c806
workflow: tests #639
result: PASS
pytest: 766 passed, 2 skipped, 3 warnings
```

Warnings are the existing Tigramite single-dataset warning in sensitivity/converter tests; no Issue #23 failure remains.

Dedicated Sensitivity GRU smoke/integration workflow previously passed after analysis dependencies were introduced. The repository-wide suite above also includes the existing GPDC/LPCMCI sensitivity integration tests.

---

# 13. Implementation Commits

Key Issue #23 implementation sequence:

```text
b13d2db  synthetic mock fixtures
b07100a  analysis pipeline foundation
3c7a4c9  analysis CLI / integration
6e97331  analysis contract tests
0f1c2ab  matplotlib test dependency
6a00b01  tabulate markdown dependency
7069fae  complete v2 analysis-output contracts
5c23411  Null mock schema synchronization
8d0c8ba  lag-response mock schema synchronization
c5feec7  mock generator v2 synchronization
fd72830  explicit mock primary metric
2d8df71  strengthened provenance/stability tests
```

---

# 14. Final Completion Summary

```yaml
analysis_outputs:
  total_tables: 9
  total_figures: 14
  total_outputs: 23

  software:
    generation_paths_implemented: 23
    synthetic_contract_tested: 23
    repository_ci: PASS

  mock_verification:
    generated: 23
    scientific_validation: false
    publication_ready: false

  real_data:
    artifact_generated: 0
    validated: 0
    publication_ready: false

  sensitivity:
    T09_code: implemented
    F10_code: implemented
    real_execution: deferred_until_primary_freeze

  issue_23_implementation_acceptance: PASS
  publication_level_completion: PENDING_REAL_DATA
```

---

# 15. Update Rule for Future Real Runs

When real experiment artifacts become available, update each row independently:

1. Generate the canonical source table / figure from the frozen experiment artifacts.
2. Confirm registry + manifest provenance and SHA256.
3. Reconcile source numerical values with the rendered output.
4. Set `Artifact Generated` to `[x]` only for the real artifact.
5. Set `Validated` to `[x]` only after content review.
6. Set `Final Status=COMPLETE` only when all completion requirements for that output are satisfied.
7. Never promote mock-derived evidence into the real-data columns.
8. Do not execute or promote T09/F10 before Primary Freeze.
