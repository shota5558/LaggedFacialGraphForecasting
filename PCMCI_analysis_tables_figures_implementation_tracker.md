# PCMCI+ Facial Motion Research
## Analysis Tables & Figures Implementation Tracker

**Purpose**  
本ファイルは、Primary / Falsification / Stability / Sensitivity 解析で必要となる表・グラフの
**実装済み / 未実装 / 作成済み / 検証済み**を一元管理するためのチェックリストである。

**Status convention**

- `[ ]` = 未実装 / 未完了
- `[x]` = 実装済み / 完了
- `N/A` = 対象外
- 実装済みでも、実データ artifact の生成・内容検証が終わるまでは別列を `[ ]` のままとする。

---

# 1. Completion Definition

各表・図は、以下をすべて満たした場合のみ **COMPLETE** とする。

- [ ] 生成コードが実装されている
- [ ] 入力 artifact / schema が固定されている
- [ ] synthetic / fixture test がある
- [ ] 実験 artifact から生成できる
- [ ] 同一 config + seed で再現できる
- [ ] 欠損・unevaluable 条件を安全に処理できる
- [ ] 数値集計と図表の表示内容が一致する
- [ ] 出力ファイルが artifact registry / manifest に記録される
- [ ] Primary / Sensitivity の境界を破らない
- [ ] 論文・報告書用の caption / interpretation contract が定義されている

---

# 2. Master Status Table

| ID | Type | Deliverable | Main Question | Priority | Intended Placement | Code | Test | Artifact Generated | Validated | Final Status |
|---|---|---|---|---|---|---|---|---|---|---|
| T01 | Table | Dataset / outer-fold summary | データ量・fold偏り・評価可能性 | Required | Main/Supplement | [ ] | [ ] | [ ] | [ ] | TODO |
| T02 | Table | Primary frozen configuration | 何を事前固定したか | Required | Main | [ ] | [ ] | [ ] | [ ] | TODO |
| T03 | Table | Primary 4-condition performance | Persistence / Self / Full / PCMCI | Critical | Main | [ ] | [ ] | [ ] | [ ] | TODO |
| T04 | Table | PCMCI vs Self paired effect | RQ1 / H1 incremental forecastability | Critical | Main | [ ] | [ ] | [ ] | [ ] | TODO |
| T05 | Table | PCMCI vs Full + sparsity | RQ2 / H2 performance-efficiency | Critical | Main | [ ] | [ ] | [ ] | [ ] | TODO |
| T06 | Table | Region-wise effect | target region 別効果 | Required | Main/Supplement | [ ] | [ ] | [ ] | [ ] | TODO |
| T07 | Table | Null / falsification summary | lag / region / sparsity / time specificity | Critical | Main | [ ] | [ ] | [ ] | [ ] | TODO |
| T08 | Table | Edge / lag stability | RQ4 discovery stability | Required | Main/Supplement | [ ] | [ ] | [ ] | [ ] | TODO |
| T09 | Table | Sensitivity summary | Primary結論の頑健性 | Required | Supplement | [ ] | [ ] | [ ] | [ ] | TODO |
| F01 | Figure | Primary condition comparison | 4条件の予測性能差 | Critical | Main | [ ] | [ ] | [ ] | [ ] | TODO |
| F02 | Figure | Subject-level paired PCMCI vs Self | 被験者単位の改善・異質性 | Critical | Main | [ ] | [ ] | [ ] | [ ] | TODO |
| F03 | Figure | Incremental Gain by target region | どの部位で追加情報が有効か | Critical | Main | [ ] | [ ] | [ ] | [ ] | TODO |
| F04 | Figure | Lag-response curve | RQ3 / H3 lag specificity | Critical | Main | [ ] | [ ] | [ ] | [ ] | TODO |
| F05 | Figure | Null / falsification effect comparison | PCMCI優位性の破壊検証 | Critical | Main | [ ] | [ ] | [ ] | [ ] | TODO |
| F06 | Figure | Performance–sparsity plot | Fullに対する疎性効率 | Important | Main | [ ] | [ ] | [ ] | [ ] | TODO |
| F07 | Figure | Region→region edge stability heatmap | 安定した部位間リンク | Important | Main | [ ] | [ ] | [ ] | [ ] | TODO |
| F08 | Figure | Region × lag stability heatmap | 安定した region–lag | Important | Main/Supplement | [ ] | [ ] | [ ] | [ ] | TODO |
| F09 | Figure | Outer-fold effect distribution | fold間再現性 | Important | Supplement | [ ] | [ ] | [ ] | [ ] | TODO |
| F10 | Figure | Sensitivity forest plot | Sensitivity across methods/settings | Important | Main/Supplement | [ ] | [ ] | [ ] | [ ] | TODO |
| F11 | Figure | Forecast trajectory example | SelfとPCMCI差の直感的例示 | Supporting | Main | [ ] | [ ] | [ ] | [ ] | TODO |
| F12 | Figure | Data-quality / motion diagnostics | 静止・欠損・運動量等の診断 | Diagnostic | Supplement | [ ] | [ ] | [ ] | [ ] | TODO |
| F13 | Figure | Prediction-error distribution | 外れ値依存の確認 | Diagnostic | Supplement | [ ] | [ ] | [ ] | [ ] | TODO |
| F14 | Figure | Metric concordance plot | 指標間で結論が一致するか | Supporting | Supplement | [ ] | [ ] | [ ] | [ ] | TODO |

---

# 3. Primary Tables

## T01 — Dataset / Outer-Fold Summary

**Purpose**
- 被験者数
- frame 数
- valid frame 数
- missingness
- motion magnitude
- outer fold 構成
- evaluable / unevaluable target-fold

**Required inputs**
- SplitManifest
- FaceTimeSeries metadata
- valid_mask
- evaluability artifact

**Implementation**
- [ ] 集計関数
- [ ] fold単位集計
- [ ] region単位集計
- [ ] missing / invalid frame 集計
- [ ] evaluable / unevaluable count
- [ ] Markdown / CSV 出力
- [ ] test

**Output**
- [ ] `artifacts/analysis/tables/T01_dataset_outer_fold_summary.csv`
- [ ] `artifacts/analysis/tables/T01_dataset_outer_fold_summary.md`

---

## T02 — Primary Frozen Configuration

**Purpose**  
Primary 実験前に固定した科学仕様を機械的に出力する。

**Include**
- PCMCI+
- ParCorr
- `tau_max = 10`
- `pc_alpha`
- Ridge
- `h = 1`
- region definition
- feature representation
- outer split
- bootstrap settings
- Null settings
- lag-response grid
- seed / schema version

**Implementation**
- [ ] Scientific Freeze artifact reader
- [ ] schema validation
- [ ] Markdown table generator
- [ ] immutable / hash information
- [ ] test

**Output**
- [ ] `artifacts/analysis/tables/T02_primary_frozen_configuration.md`

---

## T03 — Primary 4-Condition Performance

Conditions:

- Persistence
- Self
- Full
- PCMCI

**Recommended columns**
- condition
- metric
- median
- mean
- 95% bootstrap CI
- subject count
- evaluable unit count

**Implementation**
- [ ] PredictionArtifact reader
- [ ] condition aggregation
- [ ] subject-level metric aggregation
- [ ] median
- [ ] bootstrap CI
- [ ] evaluability reporting
- [ ] test

**Output**
- [ ] `artifacts/analysis/tables/T03_primary_condition_performance.csv`
- [ ] `artifacts/analysis/tables/T03_primary_condition_performance.md`

---

## T04 — PCMCI vs Self Paired Effect

Primary effect:

\[
\Delta E_s = E_{\mathrm{Self},s} - E_{\mathrm{PCMCI},s}
\]

Positive value = PCMCI improvement.

**Required statistics**
- median paired difference
- mean paired difference
- 95% bootstrap CI
- N subjects
- evaluable N
- region breakdown if applicable

**Implementation**
- [ ] exact subject matching
- [ ] same-unit support enforcement
- [ ] paired difference calculation
- [ ] bootstrap CI
- [ ] sign convention test
- [ ] missing-pair rejection
- [ ] test

**Output**
- [ ] `artifacts/analysis/tables/T04_pcmci_vs_self_paired_effect.csv`
- [ ] `artifacts/analysis/tables/T04_pcmci_vs_self_paired_effect.md`

---

## T05 — PCMCI vs Full + Sparsity

**Purpose**

Evaluate simultaneously:

\[
E_{\mathrm{PCMCI}} \;\text{vs}\; E_{\mathrm{Full}}
\]

and

\[
|P_{\mathrm{PCMCI}}| \ll |P_{\mathrm{Full}}|
\]

**Recommended columns**
- target region
- Full error
- PCMCI error
- paired difference
- Full feature count
- PCMCI feature count
- feature ratio
- performance conclusion

**Implementation**
- [ ] performance join
- [ ] feature provenance reader
- [ ] feature count
- [ ] feature ratio
- [ ] paired performance statistic
- [ ] test

**Output**
- [ ] `artifacts/analysis/tables/T05_pcmci_vs_full_sparsity.csv`
- [ ] `artifacts/analysis/tables/T05_pcmci_vs_full_sparsity.md`

---

## T06 — Region-Wise Effect

**Primary quantity**

\[
\Delta E_j =
\operatorname{median}_s
(E_{\mathrm{Self},s,j}-E_{\mathrm{PCMCI},s,j})
\]

**Implementation**
- [ ] region aggregation
- [ ] paired subject support
- [ ] 95% bootstrap CI
- [ ] evaluable count
- [ ] multiple region handling
- [ ] test

**Output**
- [ ] `artifacts/analysis/tables/T06_region_wise_effect.csv`
- [ ] `artifacts/analysis/tables/T06_region_wise_effect.md`

---

## T07 — Null / Falsification Summary

**Primary controls**
- PCMCI
- lag-shift
- random-region
- matched sparsity
- time-shuffle

**Sensitivity**
- autocorrelation-preserving surrogate

**Implementation**
- [ ] condition mapping
- [ ] matched evaluation support
- [ ] PCMCI vs each Null paired difference
- [ ] repeated matched-sparsity aggregation
- [ ] random-region aggregation
- [ ] lag-shift aggregation
- [ ] time-shuffle aggregation
- [ ] test

**Output**
- [ ] `artifacts/analysis/tables/T07_falsification_summary.csv`
- [ ] `artifacts/analysis/tables/T07_falsification_summary.md`

---

## T08 — Edge / Lag Stability

**Required summaries**
- outer-fold selection frequency
- bootstrap selection frequency
- source region
- target region
- lag
- selected count
- total opportunities
- frequency

**Implementation**
- [ ] ParentSet aggregation
- [ ] fold frequency
- [ ] bootstrap frequency
- [ ] region→region aggregation
- [ ] region×lag aggregation
- [ ] test

**Output**
- [ ] `artifacts/analysis/tables/T08_edge_lag_stability.csv`
- [ ] `artifacts/analysis/tables/T08_edge_lag_stability.md`

---

## T09 — Sensitivity Summary

**Sensitivity candidates**
- GPDC
- LPCMCI
- phase-shuffled surrogate
- circular-shift surrogate
- `h > 1`
- GRU if executed

**Implementation**
- [ ] sensitivity registry
- [ ] Primary reference join
- [ ] effect extraction
- [ ] CI extraction
- [ ] conclusion consistency field
- [ ] test

**Output**
- [ ] `artifacts/analysis/tables/T09_sensitivity_summary.csv`
- [ ] `artifacts/analysis/tables/T09_sensitivity_summary.md`

---

# 4. Primary Figures

## F01 — Primary Condition Comparison

**Data**
- Persistence
- Self
- Full
- PCMCI
- subject-level values
- median
- 95% CI

**Implementation**
- [ ] plotting function
- [ ] subject point rendering
- [ ] median rendering
- [ ] CI rendering
- [ ] fixed metric labels
- [ ] evaluability annotation
- [ ] test

**Output**
- [ ] `artifacts/analysis/figures/F01_primary_condition_comparison.png`
- [ ] vector format if required

---

## F02 — Subject-Level Paired PCMCI vs Self

**Primary quantity**

\[
\Delta E_s = E_{\mathrm{Self},s}-E_{\mathrm{PCMCI},s}
\]

**Interpretation**
- `> 0`: PCMCI improvement
- `= 0`: no difference
- `< 0`: Self better

**Implementation**
- [ ] paired subject join
- [ ] zero reference line
- [ ] subject-level points
- [ ] median
- [ ] 95% bootstrap CI
- [ ] stable ordering
- [ ] test

**Output**
- [ ] `artifacts/analysis/figures/F02_subject_paired_pcmci_vs_self.png`

---

## F03 — Incremental Gain by Target Region

**Implementation**
- [ ] region-wise paired effects
- [ ] median
- [ ] CI
- [ ] zero reference
- [ ] evaluable N
- [ ] test

**Output**
- [ ] `artifacts/analysis/figures/F03_incremental_gain_by_region.png`

---

## F04 — Lag-Response Curve

Primary grid:

```yaml
delta_frames: [-2, -1, 0, 1, 2]
reference_delta: 0
```

**Primary requirements**
- same selected region(s)
- same feature count
- common shift of selected parents
- complete symmetric grid
- no clipping
- no wrapping
- no feature dropping
- target-fold unevaluable if grid cannot be completed
- same support across all deltas

**Quantity**

\[
\Delta E(\Delta)
=
E(\tau^\*+\Delta)-E(\tau^\*)
\]

**Implementation**
- [ ] grid reader
- [ ] complete-grid filter
- [ ] same-unit support
- [ ] median curve
- [ ] 95% CI at each delta
- [ ] frames x-axis
- [ ] milliseconds secondary/report field
- [ ] unevaluable count
- [ ] zero/reference marker
- [ ] test

**Output**
- [ ] `artifacts/analysis/figures/F04_lag_response_curve.png`
- [ ] `artifacts/analysis/tables/F04_lag_response_source.csv`

---

## F05 — Null / Falsification Effect Comparison

**Conditions**
- PCMCI
- lag-shift
- random-region
- matched sparsity
- time-shuffle

**Implementation**
- [ ] common effect scale
- [ ] subject-level aggregation
- [ ] median
- [ ] 95% CI
- [ ] repeated Null summary
- [ ] test

**Output**
- [ ] `artifacts/analysis/figures/F05_falsification_effect_comparison.png`

---

## F06 — Performance–Sparsity Plot

**X-axis**
- feature count
- or `|P_PCMCI| / |P_Full|`

**Y-axis**
- prediction error

**Implementation**
- [ ] feature count provenance
- [ ] Full / PCMCI mapping
- [ ] point aggregation
- [ ] region/fold optional grouping
- [ ] test

**Output**
- [ ] `artifacts/analysis/figures/F06_performance_sparsity.png`

---

## F07 — Region→Region Edge Stability Heatmap

**Cell**

\[
P(i \rightarrow j \text{ selected})
\]

**Implementation**
- [ ] source-target matrix
- [ ] fold aggregation
- [ ] bootstrap aggregation
- [ ] stable region order
- [ ] missing cell semantics
- [ ] test

**Output**
- [ ] `artifacts/analysis/figures/F07_region_edge_stability_heatmap.png`

---

## F08 — Region × Lag Stability Heatmap

**Axes**
- rows: source→target relation
- columns: lag `1..10`
- values: selection frequency

**Implementation**
- [ ] relation×lag matrix
- [ ] `tau_max=10` validation
- [ ] fold/boot frequency
- [ ] missing vs zero distinction
- [ ] test

**Output**
- [ ] `artifacts/analysis/figures/F08_region_lag_stability_heatmap.png`

---

# 5. Secondary / Diagnostic Figures

## F09 — Outer-Fold Effect Distribution

- [ ] fold-level effect extraction
- [ ] fold identifiers
- [ ] distribution plot
- [ ] pooled median reference
- [ ] test
- [ ] output generated

Output:
`artifacts/analysis/figures/F09_outer_fold_effect_distribution.png`

---

## F10 — Sensitivity Forest Plot

**Rows**
- Primary ParCorr + Ridge
- GPDC
- LPCMCI
- phase surrogate
- circular shift
- h > 1
- GRU if executed

**X-axis**
- `Self − PCMCI` paired effect

- [ ] effect reader
- [ ] CI reader
- [ ] Primary reference marker
- [ ] zero reference
- [ ] Sensitivity ordering
- [ ] test
- [ ] output generated

Output:
`artifacts/analysis/figures/F10_sensitivity_forest_plot.png`

---

## F11 — Forecast Trajectory Example

- [ ] example selection rule frozen
- [ ] no cherry-picking from outcome
- [ ] y_true
- [ ] Self prediction
- [ ] PCMCI prediction
- [ ] time axis
- [ ] region label
- [ ] test
- [ ] output generated

Output:
`artifacts/analysis/figures/F11_forecast_trajectory_example.png`

---

## F12 — Data-Quality / Motion Diagnostics

Suggested panels/data products:
- subject frame counts
- valid frame ratio
- missing ratio
- region motion magnitude
- static vs active interval proportion
- sequence length
- evaluability counts

- [ ] frame count plot
- [ ] missingness plot
- [ ] motion magnitude plot
- [ ] static/activity diagnostic
- [ ] evaluability diagnostic
- [ ] test
- [ ] output generated

Output directory:
`artifacts/analysis/figures/F12_data_quality/`

---

## F13 — Prediction-Error Distribution

- [ ] subject-level distribution
- [ ] region-level optional split
- [ ] Self / PCMCI comparison
- [ ] outlier-safe display
- [ ] test
- [ ] output generated

Output:
`artifacts/analysis/figures/F13_prediction_error_distribution.png`

---

## F14 — Metric Concordance

Candidate metrics:
- Position RMSE
- Velocity RMSE
- Acceleration RMSE
- Temporal correlation
- Peak timing error
- Onset timing error
- Lag preservation

- [ ] metric registry
- [ ] sign normalization
- [ ] metric-wise effect calculation
- [ ] cross-metric comparison
- [ ] test
- [ ] output generated

Output:
`artifacts/analysis/figures/F14_metric_concordance.png`

---

# 6. Recommended Implementation Order

## Phase A — Primary Statistical Backbone

- [ ] T03 Primary 4-condition performance
- [ ] T04 PCMCI vs Self paired effect
- [ ] F01 Primary condition comparison
- [ ] F02 Subject-level paired effect

**Gate A**
- [ ] `Self − PCMCI` の符号・集計単位・CIが全artifactで一致する

---

## Phase B — Lag Specificity

- [ ] F04 Lag-response curve
- [ ] lag-response source table
- [ ] complete symmetric grid enforcement
- [ ] unevaluable count report

**Gate B**
- [ ] Δごとに同一 support を使用
- [ ] clipping / wrapping / dropping が発生しない

---

## Phase C — Falsification

- [ ] T07 Null summary
- [ ] F05 Null comparison
- [ ] random-region
- [ ] matched sparsity
- [ ] time-shuffle
- [ ] lag-shift

**Gate C**
- [ ] Null mapping が outer-train で freeze されている

---

## Phase D — Region-Level Effect

- [ ] T06 Region-wise table
- [ ] F03 Region-wise figure

---

## Phase E — Discovery Stability

- [ ] T08 Stability table
- [ ] F07 Region→region heatmap
- [ ] F08 Region×lag heatmap

---

## Phase F — Sparsity

- [ ] T05 PCMCI vs Full + sparsity
- [ ] F06 Performance–sparsity plot

---

## Phase G — Sensitivity

Primary Freeze 後のみ実行。

- [ ] T09 Sensitivity summary
- [ ] F10 Sensitivity forest plot

---

## Phase H — Diagnostics / Supporting

- [ ] T01 Dataset summary
- [ ] F09 Fold distribution
- [ ] F11 Forecast example
- [ ] F12 Data-quality diagnostics
- [ ] F13 Error distribution
- [ ] F14 Metric concordance

---

# 7. Publication-Level Completion Gate

## Main Tables
- [ ] T02
- [ ] T03
- [ ] T04
- [ ] T05
- [ ] T07

## Main Figures
- [ ] F01
- [ ] F02
- [ ] F03
- [ ] F04
- [ ] F05
- [ ] F06
- [ ] F07 or F08

## Supplement / Audit
- [ ] T01
- [ ] T06
- [ ] T08
- [ ] T09
- [ ] F09
- [ ] F10
- [ ] F11
- [ ] F12
- [ ] F13
- [ ] F14

---

# 8. Final Completion Summary

Update this block when implementation progresses.

```yaml
analysis_outputs:
  total_tables: 9
  total_figures: 14
  total_outputs: 23

  tables:
    implemented: 0
    remaining: 9

  figures:
    implemented: 0
    remaining: 14

  critical_primary:
    total:
      - T03
      - T04
      - T05
      - T07
      - F01
      - F02
      - F03
      - F04
      - F05
    implemented: []
    remaining:
      - T03
      - T04
      - T05
      - T07
      - F01
      - F02
      - F03
      - F04
      - F05

  publication_ready: false
```

---

# 9. Update Rule

実装時は単に master table の `Code = [x]` にするだけでなく、以下を更新する。

1. 該当項目の `Implementation` checklist
2. Output artifact path
3. test の有無
4. artifact generation status
5. validation status
6. Master Status Table
7. Final Completion Summary

**禁止事項**

- コードファイルが存在するだけで COMPLETE にしない
- plot 関数が存在するだけで COMPLETE にしない
- synthetic output だけで publication-ready にしない
- Primary と Sensitivity の結果を同じ status として混同しない
- 実データ未実行を「解析完了」と扱わない
