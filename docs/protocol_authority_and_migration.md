# Protocol authority registry and executable migration plan

Updated: 2026-09-12
Protocol: `primary-2026-09-12-adopted`
Executable schema: `7`

This index implements Issue #25 / R-01. The scientific design is adopted; the
remaining gate is materialization and verification of real-data facts. It does
not claim a real experiment or a Primary result freeze.

| 優先 | 役割 | repository representation |
|---|---|---|
| 1 | 正式な Primary specification declaration | `docs/authoritative_primary_experiment_spec_2026-09-12.md` |
| 2 | 採用対象となった詳細設計本文 | `docs/primary_experiment_recommendation_2026-09-11.md` |
| 3 | 読みやすい統合実験設計 | `docs/experiment_design.md` |
| 4 | 採用判断・履歴 | `docs/adopted_decisions_2026-09-10.md` および 2026-09-12 の adoption record |
| 5 | 移行対象の旧研究計画・詳細設計 | `docs/experimental_plan.md`, `docs/detailed_design.md` |
| 6 | Legacy executable protocol | `configs/scientific_freeze.yaml`, `schemas/scientific_freeze.schema.json` |

| Priority | Authority | Repository representation |
|---|---|---|
| 1 | 2026-09-12 adopted Primary specification | `docs/authoritative_primary_experiment_spec_2026-09-12.md` |
| 2 | Adopted decision record | `docs/adopted_decisions_2026-09-12.md` |
| 3 | Adopted detailed recommendation | `docs/primary_experiment_recommendation_2026-09-11.md` |
| 4 | Integrated experiment design | `docs/experiment_design.md` |
| 5 | Migration implementation | `configs/scientific_freeze.yaml`, `configs/primary_run.yaml`, `schemas/scientific_freeze.schema.json` |
| 6 | Legacy migration source | Scientific Freeze v6; superseded where conflicting |

The adopted specification and decision record are repository documents. Their
content hashes are recorded by the executable freeze when real preflight is
complete; null hash fields in the checked-in config intentionally mean
“preflight not materialized”, not “hash unknown but accepted”.

## Adopted dispositions

| Area | Adopted rule | Legacy disposition |
|---|---|---|
| Representation/metric | normalized 2D displacement; pointwise mean Euclidean error | velocity primary metric superseded |
| Discovery/forecast | scalar PCMCI+ links OR-projected to region blocks; all usable block scalars forecast | exact-selected-scalar rule superseded |
| Lag | `h=1`; `L=floor(0.5*fps)`; integer lags and fixed ms bands | fixed `tau_max=10` superseded |
| Enrichment | cell-G mean against 1,000 random sets | v6 joint Null 100 is a different protocol |
| Lag response | single-block selected-lag response; `M=floor(0.1*fps)` symmetric grid | fixed common-shift `[-2..2]` superseded |
| Inference | subject medians; dependency-group nominal OOF bootstrap, 10,000 | held-out significance/FDR not a completion gate |
| Optional lanes | sensitivity/optional analyses after Primary freeze only | they do not block Primary adoption |

## Migration gates

1. The adopted schema and protocol identity must load exactly.
2. Preflight must provide the real dataset, extractor/model, cadence, inventory,
   realized group split, Self history, and common-support artifacts.
3. Dynamic `L`, lag bands, `M`, and the final alpha grid must be materialized
   from preflight without reading formal outer-test outcomes.
4. An executable-freeze JSON must bind the protocol ID and scientific-config
   hash. Legacy v6 freeze artifacts are rejected.
5. Downstream implementation and dry-run issues must pass before a formal
   outer-test evaluation is started.

## State semantics

`preflight_required` is the checked-in repository state. It permits contract and
synthetic tests but does not authorize a real run. `executable` is set only by
an explicit, hash-backed preflight update. It is separate from the result
`PRIMARY FREEZE` owned by Issue #20.

## Traceability

- Adopted specification: `docs/authoritative_primary_experiment_spec_2026-09-12.md`
- Decision record: `docs/adopted_decisions_2026-09-12.md`
- Requirement matrix: `docs/requirement-matrix.md`
- Issue implementation log: `docs/issue-implementation-2026-09-10.md`
- Published Issue snapshot: `docs/issue-drafts/POSTED.md`
