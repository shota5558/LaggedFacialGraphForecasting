# Lagged Facial Graph Forecasting

PCMCI+ と Ridge を用いて、自然な表情・発話中の顔部位間遅延予測構造と、その被験者間の共通性・一般化可能性を評価する研究実装です。

## Authoritative specification

2026-09-12 以降、本研究の正式仕様は次です。

- [Authoritative Primary Experiment Specification（2026-09-12）](docs/authoritative_primary_experiment_spec_2026-09-12.md)
- [Adopted Decisions（2026-09-12）](docs/adopted_decisions_2026-09-12.md)
- [本実験に必要な定義の最終推薦案（2026-09-11）](docs/primary_experiment_recommendation_2026-09-11.md) — 2026-09-12 に**全面採用済み**
- [実験設計書：顔部位間の遅延予測構造と被験者間の共通性](docs/experiment_design.md)

`primary_experiment_recommendation_2026-09-11.md` 内の設計上の推薦値・規則は、本日以降参考案ではなく採用済み Primary specification として扱います。実データからしか得られない配布版、file hash、実 fps、model hash、実 fold、実 support 等は、未確認の事実を創作せず preflight で materialize します。

旧 experimental plan、detailed design、Scientific Freeze v6、旧 Issue が上記仕様と矛盾する場合は、上記 authoritative specification を優先し、旧記述を `SUPERSEDED` とします。

## Primary の中心設計

- 主表現: 2D 正規化変位
- Primary discovery: PCMCI+ + ParCorr
- Primary forecaster: Ridge
- horizon: h=1
- 全候補 Predictive Gain Landscape
- Persistence / Self / Full / PCMCI-block の4条件比較
- Selection Enrichment: 1,000 random sets
- 選択 lag 中心 response: 約 ±100 ms、complete symmetric grid
- Discovery stability: outer-train group bootstrap 100回
- 主集団要約: 被験者単位、median を中心に効果量と分布を推定
- nominal 95% bootstrap interval: 10,000 group resamples
- held-out significance test / FDR は Primary の研究完了条件にしない

Random-region、Time-shuffle、joint-set matched-sparsity再学習、circular/phase surrogate、GPDC、LPCMCI、GRU、h>1 等は Primary completion blocker ではありません。

## 実装・移行管理

- [Protocol authority registry / migration plan](docs/protocol_authority_and_migration.md)
- [新指標に基づく解析図表の再構成案（2026-09-11）](docs/analysis_tables_figures_redesign_2026-09-11.md)
- [研究計画の検討資料](docs/experimental_plan.md)
- [実装移行の詳細設計](docs/detailed_design.md)
- [要求トレーサビリティ](docs/requirement-matrix.md)
- [Issue 実装記録](docs/issue-implementation-2026-09-10.md)

文書の採用だけでは executable migration の完了を意味しません。`configs/scientific_freeze.yaml`、schema、config loader、実装、tests、dry run、leakage audit、hash付き freeze を authoritative specification と一致させた後にのみ、新 Primary 実験を開始します。
