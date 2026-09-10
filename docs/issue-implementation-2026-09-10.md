# Issue implementation — 2026-09-10

## Enrichment図表・実成果受入ゲート — 2026-09-11 (#75 / R-16-ENRICHMENT)

- ENRICHMENTの入力にrun ID、full git SHA、protocol/config/source hash、fold、horizon、metric/directionを必須化し、run/config/grid/supportの不一致、未定義metric、差分不整合を拒否するようにした。subjectごとのsupport差は許容し、repeat内の一致は維持する。
- canonical distribution/subject summaryへseed、estimand、集約規則、評価可能・評価不能件数を保持し、analysis manifestの`enrichment_provenance`へ実行監査情報を保存する。
- ENRICHMENT図をselected vs matched分布に加え、subject-level enrichment effectと95% CIを表示する構成へ更新した。図はcanonical CSV再読込後に描画し、mockでは`MOCK DATA / NOT A SCIENTIFIC RESULT`を表示する。
- 回帰・手計算・図表生成検証: `.venv/Scripts/python.exe -m pytest tests/test_extended_analysis_outputs.py tests/test_enrichment.py tests/test_landscape.py -o addopts='' -q -p no:cacheprovider --basetemp tmp/pytest-r16-enrichment` — 84 passed。生成PNGを目視確認し、`git diff --check`も通過。
- `artifacts/`には現時点で実験解析入力／real Primary成果がなく、#19依存と実成果受入は未完了。mock/software PASSをreal-data COMPLETEへ昇格していない。

## Population図表・実成果受入ゲート — 2026-09-11 (#76 / R-16-POPULATION)

- POPULATIONのraw recordにrun ID、full git SHA、protocol/config/source hash、metric/direction、estimand、horizonを必須化し、run/config/source/metricの混在とsupport不整合を拒否するようにした。
- `E_shifted - E_reference`を再計算し、完全なedge/subject Δ grid、Δ=0、frame/ms、edge/subject/evaluable/unevaluable/failure分母をcanonical summaryへ保存する。CIは設定済みestimatorでsubject clusterを再標本化する。
- canonical source/summary CSVを先に保存し、独立した`POPULATION` registry ID、manifest provenance、caption、PNG/SVGを生成する。captionでは旧F04 common-shift曲線と区別する。
- 回帰・手計算・mock図表生成検証: `.venv/Scripts/python.exe -m pytest tests/test_mock_analysis_data.py tests/test_extended_analysis_outputs.py tests/test_population_response.py -o addopts='' -q -p no:cacheprovider --basetemp tmp/pytest-population-generator` — 99 passed。`git diff --check`も通過。
- `artifacts/`にはreal Primary成果がなく、R-08/#19依存の実成果生成・数値受入・目視reviewは未完了。mock/software PASSをreal-data COMPLETEへ昇格していない。

## Landscape lag-band configuration validation — 2026-09-11 (#74 / R-16-LANDSCAPE)

- Lag-band aggregation requires explicit nonempty bands with integer boundaries and nonempty string labels. Null no longer generates outcome-dependent singleton bands; fractional, boolean and string boundaries are rejected instead of coerced. Overlapping intervals are rejected even outside currently evaluable lags.
- Coverage is checked against all supplied candidate cells, including failed/unevaluable cells, before numerical aggregation. Existing subject-first means/medians and evaluable denominators are preserved.
- The LANDSCAPE raw contract now preserves run ID, full git SHA, protocol/config/source/support hashes, seed, metric/direction, estimand, fold/horizon, and failure status. Canonical cell summaries include subject/cell denominators and an outcome-independent-in-execution exploratory exact-cell rank; registry and manifest carry the same provenance.
- Validation: `.venv/Scripts/python.exe -m pytest tests/test_landscape_aggregation.py tests/test_landscape.py tests/test_extended_analysis_outputs.py tests/test_mock_analysis_data.py -o addopts='' -q -p no:cacheprovider` — 126 passed, including hand-calculated aggregates, provenance rejection, denominator preservation, canonical CSV/figure generation, and mock fixture generation. `git diff --check` passed.
- This is a software/mock verification pass only. Scientific band selection, raw export from R-08/R-17, and real-result acceptance remain open; no real experiment was run.

## Population fold-level edge identity — 2026-09-11 (#35 / R-19)

- The individual population reducer now binds `(outer_fold, edge_id)` to one source/target region, source/target component and reference lag across all observations. Reusing an ID for different edges can no longer undercount the edge denominator.
- Subject-specific reference errors and support remain allowed. Identical edge IDs in different folds may represent different fitted selections.
- Validation: `.venv/Scripts/python.exe -m pytest tests/test_population_response.py tests/test_extended_analysis_outputs.py -o addopts='' -q -p no:cacheprovider --basetemp tmp/pytest-edge-identity` — 63 passed, including six new edge identity/denominator regression cases. No runner changes or real experiment; scientific decisions and real-run acceptance remain open.

## Candidate-grid JSON identity — 2026-09-11 (#33 / R-17)

- Removed integer coercion from candidate-grid loading. The existing `LandscapeCandidate` contract now validates the original JSON lag; fractional numbers, booleans and numeric strings cannot silently become another frozen candidate.
- Non-object JSON roots now produce `AnalysisOutputError` rather than an incidental attribute error. Direct and payload-wrapped valid grids preserve candidate identity and digest.
- Validation: candidate-grid input and landscape tests — 20 passed; extended analysis output tests — 47 passed, including synthetic table/figure generation. Commands used `.venv/Scripts/python.exe -m pytest` with `-o addopts='' -q -p no:cacheprovider` and isolated temporary directories. This is input correctness only; no runner changes, scientific decisions or real experiment.

## Landscape shared-reference validation — 2026-09-11 (#33 / R-17)

- The individual landscape source validator now requires the same Self error and evaluation-support digest across evaluable candidates for each subject/target component and supplied fold/horizon/analysis identity. Correct per-cell subtraction alone no longer permits a changing baseline.
- Different subject/component references remain valid; failed/unevaluable cells remain accounted for without contributing reference errors or gains. DataFrame index labels are reset before candidate coverage checks so concatenated input cannot overwrite candidate identities.
- Regression fixtures cover reconciled-but-inconsistent Self errors, support drift, distinct target-component references, duplicate DataFrame indices, and unevaluable cells. This is software validation only; model fitting, scientific aggregation decisions and real-run acceptance remain open. No runner changes.
- Validation: `.venv/Scripts/python.exe -m pytest tests/test_extended_analysis_outputs.py tests/test_landscape.py tests/test_analysis_pipeline.py tests/test_enrichment.py -o addopts='' -q -p no:cacheprovider --basetemp tmp/pytest-landscape-final` — 84 passed, including concurrent enrichment changes present in the shared workspace. `git diff --check` passed.

## Enrichment CSV aggregation guards — 2026-09-11 (#34 / R-18)

- Reject missing/blank identifiers before pandas grouping, repeat-level changes in status/estimand/aggregation/grid/support, and mixed estimands or cell aggregation rules within a target's subject summary.
- Unevaluable rows must carry missing numerical aggregates; they remain in the distribution but do not contribute to subject counts. SHA-256 values are normalized to lowercase; support may legitimately differ between subjects.
- Validation: the extended-output and enrichment test files passed (52 tests), followed by the enrichment-focused extended tests after digest normalization (19 passed, 28 deselected). New cases cover metadata drift, missing identities, mixed subject estimands, unevaluable records, and valid two-repeat input with subject-specific support and equivalent digest casing.
- This corrects the existing CSV analysis boundary. Raw-export integration, scientific decisions and real-run acceptance remain open; no runner changes or real experiment.

## Enrichment landscape consistency — 2026-09-11 (#34 / R-18)

- The individual cell-gain reducer now rejects changed gains for the same candidate within one subject/support unit, across selected membership and all matched repeats. It also rejects selected/matched feature-unit mismatches even when candidate counts match.
- Validation precedes aggregation. Distinct repeat IDs may still contain identical memberships and selected/matched overlap is preserved; these are not duplicate records.
- Corrected the existing numerical fixture, which assigned different gains to the same frozen cell, and recalculated its expected aggregates. Added regressions for selected/repeat and repeat/repeat gain conflicts, feature-unit mismatch, and valid identical memberships.
- Validation: `.venv/Scripts/python.exe -m pytest tests/test_enrichment.py tests/test_landscape.py tests/test_null_matched_sparsity.py tests/test_null_matched_sparsity_repeats.py -o addopts='' -q -p no:cacheprovider` — 30 passed. This is reducer correctness only; scientific aggregation choices, canonical export and real-run acceptance remain open. No runner changes or real experiments.

## Population subject-cluster CI correction — 2026-09-11 (#35 / R-19)

- Replaced bootstrap of subject medians with bootstrap of subject IDs carrying every associated edge. Each resample uses the configured edge-subject point estimator (median or mean), preserving unequal cluster sizes and repeated cluster draws.
- Added `point_estimate` for the configured estimator while keeping the median/mean columns numerically literal. Population figures now display that point estimate with its corresponding interval. Interval lines do not assume that percentile CIs contain the point estimate.
- Summary records include bootstrap seed/count and fixed-discovery/fitted-model conditioning; the caption states this scope. Scientific selection of the estimator/context and real-run approval remain unresolved; this corrects the existing configurable software path only.
- Validation: `.venv/Scripts/python.exe -m pytest tests/test_extended_analysis_outputs.py tests/test_population_response.py -o addopts='' -q -p no:cacheprovider --basetemp tmp/pytest-cluster-ci` — 34 passed. Unequal-edge fixtures verify both estimators against independent seeded resampling and check row-order invariance; existing synthetic table/figure generation passes. No runner changes or real experiments.

## Individual population analysis correction — 2026-09-11 (#35 / R-19)

- The edge-response reducer compares every delta's reference error against delta zero. A changed baseline is rejected even when identity and support digests match, regardless of delta ordering.
- Population CSV validation rejects fractional, boolean, missing and non-finite fold/lag/delta fields before integer conversion; truncation can no longer change a response's identity or lag.
- Validation: `.venv/Scripts/python.exe -m pytest tests/test_population_response.py tests/test_extended_analysis_outputs.py -q -p no:cacheprovider --basetemp tmp/pytest-population-validation` — passed, including 23 new regression cases and existing synthetic output generation.
- This is individual software validation only. It does not resolve the scientific context/cluster-CI decisions, run real data, or complete #35. No runner changes.

## Individual analysis correction — 2026-09-11 (#29 / #30)

- The legacy v6 lag-response reducer now rejects differing valid-row counts across deltas, including drift at delta zero.
- Each metric must cover every evaluable target fold. A fold missing from all deltas can no longer inflate the reported evaluable-fold denominator.
- Validation: `.venv/Scripts/python.exe -m pytest tests/test_lag_response_statistics.py tests/test_population_response.py tests/test_primary_statistics_orchestration.py -q -p no:cacheprovider` — 21 passed, including 7 new regression cases.
- Scope: individual reducer correctness only. Counts do not prove exact timestamp/dimension support; raw prediction support validation remains required. This does not implement the new edge-centered estimand, change scientific decisions, or complete #29/#30. No runner changes or real experiments.

## Implemented locally: #25 / R-01

- Added a single authority registry and migration gate document linking the source versions and SHA-256 values to the repository plans, decision record, requirement matrix, posted Issues and acceptance evidence.
- Recorded explicit dispositions for the v6/new-plan conflicts: 100 vs 1,000 repeats, common-shift vs edge-centered response, component vs region-block cells, cell aggregation vs joint-set gain, and optional nonlinear/Sensitivity methods.
- Replaced the placeholder README with the repository entry points and a warning that Scientific Freeze v6 remains the execution contract until R-02/R-03 decisions and a versioned migration are complete.

Validation: source hashes, issue links, conflict ownership and migration gates were reconciled against `docs/experimental_plan.md`, `docs/detailed_design.md`, `docs/audit-report.md`, `docs/requirement-matrix.md`, and the current GitHub open-Issue list. This is documentation completion only; it does not authorize a real run or close downstream decision/experiment Issues.

## Partially implemented locally: #29 / R-05 raw prediction support

- Primary paired statistics now hashes the exact valid support tuple `(subject, region, forecast_origin, target_time, target_dimensions)` for every outer fold before metric aggregation.
- Pairing rejects equal-count but different timestamps, mask drift, target-dimension reordering, duplicate support rows, missing folds and all-invalid artifacts. Successful results retain the per-fold SHA-256 support digests.
- The existing subject-level metric pairing and effect values are unchanged after exact raw support validation.

Validation: `uv run --project . --extra test python -m pytest tests/test_subject_level_paired_difference.py tests/test_primary_condition_alignment.py tests/test_analysis_pipeline.py tests/test_primary_statistics_orchestration.py -q` — 35 passed. R-05 remains partial until R-08 connects raw prediction export to the analysis CSV schema and carries the same support digest through that boundary; `n_valid` alone is not promoted as sufficient evidence.

## Implemented locally: #31 / R-07a and #32 / R-07b

- Primary-only input discovery, CSV loading, provenance validation and input hashing no longer depend on Sensitivity. The 21 Primary output IDs exclude T09/F10.
- Real Sensitivity analysis requires `SensitivityExperimentConfig` with `execution_mode=real`, passes the existing completeness/file/hash validator, and checks every input row against the validated Primary freeze path and SHA-256. The boolean `primary_frozen` no longer authorizes execution.
- Real Sensitivity output lives under `artifacts/sensitivity/analysis`, separate from Primary output. Analysis manifest schema v3 records the validated freeze reference. Explicit mock verification retains its software-only label and cannot be publication-ready.
- CLI: use `--include-sensitivity --sensitivity-config <descriptor.yaml>` for real Sensitivity; `--primary-frozen` was removed. Sensitivity CSV must contain `source_primary_freeze_manifest` and `source_primary_freeze_sha256`. Primary-only CLI needs neither the descriptor nor the CSV.
- Python callers selecting Sensitivity must use `AnalysisInputs.from_directory(path, include_sensitivity=True)` and pass the validated descriptor via `sensitivity_config` to `generate_analysis_outputs`.

Validation: `python -m pytest tests/test_analysis_pipeline.py tests/test_sensitivity_freeze_barrier.py` — 24 passed. Tests cover missing Sensitivity input, no reads/hashes of an existing Sensitivity file in Primary-only mode, 21 output IDs, missing/incomplete/tampered freeze evidence, mismatched references, boolean bypass rejection and separate output roots. Post-freeze fixtures are isolated software contract tests, not real experimental evidence.

Full regression: `python -m pytest` — 773 passed, 2 skipped, 3 Tigramite single-dataset warnings. Local Python 3.12 uses the project dependency ranges installed under `tmp/test-deps`; this directory is ignored by Git. No experimental data were used.

## Implemented locally: #30 / R-06 statistics contract (software portion)

- Analysis input validation now requires an explicit frozen `velocity_rmse` primary metric and a non-negative experiment seed; metric and seed fallback selection is rejected.
- Matched-sparsity input requires the configured repeat count, non-empty replicate IDs and seeds, exact PCMCI evaluation-unit coverage, and no duplicate or missing repeats.
- T07 aggregates repeat medians within each subject-region before the subject summary, and records the aggregation order in its output. A skewed `[0, 0, 9]` fixture guards against reverting to the previous mean behavior.
- The synthetic fixture config now declares its one-repeat software-test contract explicitly; it is not a replacement for the v6 real-run count or the pending 1,000-repeat migration.

Validation: `.venv\Scripts\python.exe -m pytest --basetemp tmp\pytest-r06-2 -q tests/test_analysis_pipeline.py tests/test_primary_statistics_orchestration.py tests/test_subject_level_paired_difference.py tests/test_scientific_config.py tests/test_runner_config.py tests/test_mock_analysis_data.py` — passed. This closes only the software validation portion; R-03 still owns unresolved scientific aggregation and migration decisions. No GitHub Issue was closed and no real experiment was run.

## Remaining dependencies

## Partially implemented locally: #33 / R-17 landscape contract

- Added an explicit `LandscapeCandidate`/`CandidateGrid` contract requiring source/target regions, component dimensions, lag, feature unit, and protocol SHA-256.
- Candidate order is canonicalized and the full grid digest is deterministic; duplicate, missing, and unexpected cells are rejected before aggregation.
- Added `G = E_self - E_cell` with mandatory identical support digests and finite-error validation.

Validation: `python -m pytest tests/test_landscape.py -q` — passed. This is the specification-independent contract portion only; it does not choose Ω, fit Ridge models, run real data, or implement the unresolved D-11 aggregation rules.

## Partially implemented locally: #34 / R-18 enrichment contract

- Added cell-level `G` observations and explicit matched-repeat records with seed and membership SHA-256 provenance.
- The reducer validates candidate-grid membership, target identity, feature-count matching, support digest equality, duplicate repeat/cell rejection, and preserves every repeat-level aggregate and selected-minus-matched difference.
- Empty selected sets are marked unevaluable rather than silently converted to zero. The aggregation callable and estimand ID are explicit, so unresolved D-10/D-11/D-12 rules are not chosen by the software.

Validation: `.venv\Scripts\python.exe -m pytest --basetemp tmp\pytest-r18 -q tests\test_enrichment.py tests\test_landscape.py tests\test_null_matched_sparsity.py tests\test_null_matched_sparsity_repeats.py` — 26 passed; full suite — 800 passed, 2 skipped, 3 Tigramite warnings. This is contract/reducer software only; it does not generate matched sets, choose the approved scientific aggregate/CI, or run real data.

## Partially implemented locally: #35 / R-19 population-response contract

- Added edge/subject response records with explicit context, `tau_star`, `delta`, shifted lag, frame/ms conversion, exact common support, and the `E_shifted - E_reference` difference.
- The reducer rejects clipping, Δ=0 drift, support/identity changes, mixed edge contexts, duplicate units, and incomplete Δ grids; output keeps edge-subject, subject, and edge denominators separate.
- Point aggregation is injected and cluster bootstrap is intentionally not implemented until D-13/D-08 are approved.

Validation: `.venv\Scripts\python.exe -m pytest --basetemp tmp\pytest-r19 -q tests\test_population_response.py tests\test_lag_response_statistics.py` — passed. This is contract/reducer software only; it does not construct edge-centered predictions or produce real population results.

## Executed locally: audit corrections and optional output lanes

- T04/T05/T06/F09 exact metric pairing now requires positive `n_valid` and an evaluation-support digest or support ID; mismatched counts or digests are rejected instead of being silently paired. The mock metric fixture and generator carry a deterministic synthetic support digest.
- Mock inputs, the mock generator, and generated analysis tables/figures use `MOCK DATA / NOT A SCIENTIFIC RESULT`. The loader retains compatibility with the former notice only for already-existing legacy inputs.
- `scripts/generate_analysis_outputs.py --include-extended` can render independent `LANDSCAPE`, `ENRICHMENT`, and `POPULATION` tables/figures from canonical source CSVs. Extended lanes require explicit aggregation, lag-grid, lag-band, context, common-support, and CI settings; they do not silently resolve the still-open R-17/R-18/R-19 scientific decisions or overwrite T/F outputs.

Validation: `.venv\Scripts\python.exe -m pytest --basetemp tmp\pytest-final -q` — passed; the run emitted only the existing Tigramite single-dataset warnings and a local pytest cache permission warning. This remains software/mock verification; no real experiment or scientific result was produced.

The untracked decision record `adopted_decisions_2026-09-10.md` changes the planned main representation/metric and several aggregation/null rules. Neither those decisions nor provisional numerical candidates have been silently written into the frozen v6 protocol.

R-01/R-02/R-03 still require migration and resolution of the dataset, extractor/landmark topology, calibration/QC thresholds, usable time, candidate/history ranges, component projection and inferential rules. R-06 remains software-complete only until those approved aggregation and migration decisions are connected to the real protocol. Implementations that depend on those choices and real runs (#16–21, #33–39 and real table/figure acceptance) must retain their respective prerequisites. Existing software-only and mock evidence is not promoted to experimental completion. No GitHub Issue was closed and no PR or result was published by this change.
