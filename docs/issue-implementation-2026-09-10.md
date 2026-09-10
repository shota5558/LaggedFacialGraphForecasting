# Issue implementation — 2026-09-10

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

The untracked decision record `adopted_decisions_2026-09-10.md` changes the planned main representation/metric and several aggregation/null rules. Neither those decisions nor provisional numerical candidates have been silently written into the frozen v6 protocol.

R-01/R-02/R-03 still require migration and resolution of the dataset, extractor/landmark topology, calibration/QC thresholds, usable time, candidate/history ranges, component projection and inferential rules. R-06 remains software-complete only until those approved aggregation and migration decisions are connected to the real protocol. Implementations that depend on those choices and real runs (#16–21, #33–39 and real table/figure acceptance) must retain their respective prerequisites. Existing software-only and mock evidence is not promoted to experimental completion. No GitHub Issue was closed and no PR or result was published by this change.
