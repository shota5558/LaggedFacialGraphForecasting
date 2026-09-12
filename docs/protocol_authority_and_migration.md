# Protocol authority registry and migration plan

更新日: 2026-09-11

対象: `shota5558/LaggedFacialGraphForecasting`  
基準実装: `dev@9bbc3d946c0a64935b8786d9413abe53e6cefe29`

この文書は Issue #25 / R-01 の独立レビュー用索引である。科学値を新たに決定せず、原典、旧仕様との差分、責任 Issue、受入証拠を一か所に固定する。

## Authority registry

| 優先 | 役割 | 版 / SHA-256 | repository representation |
|---|---|---|---|
| 1 | 後続のユーザー判断（矛盾する旧記述より優先） | 2026-09-10、2026-09-11追記 | `docs/adopted_decisions_2026-09-10.md` |
| 2 | 科学上の正本・後続判断の反映元 | `PCMCI_facial_motion_research_plan_generalized_predictive_structure_2026-09-09.docx` / `7d77f30febaf4856131f2bb9930a2da93ec8252b60e5137e46c96f58550bc249` | `docs/experimental_plan.md` |
| 3 | 実装契約 | 2026-09-09 Markdown design | `docs/detailed_design.md` |
| 4 | 継承資料 | `PCMCI_facial_motion_implementation_plan_2026-09-08.docx` / `9ba024806749268f5a456267a616149f507f366a9dc3d31ff683205220697ab3` | 矛盾しない実装原則だけを継承 |
| 5 | 移行元 | Scientific Freeze schema v6 | `configs/scientific_freeze.yaml`, `schemas/scientific_freeze.schema.json` |

原 DOCX の記録上の所在は `C:/Users/yukit/OneDrive/デスクトップ/研究/`。別 repository の合成 DDL 回収研究、旧 DOCX の package 名 `facial_pcmci`、mock 出力は本研究の科学仕様・実成果へ混入しない。

## Approved dispositions

2026-09-11の[本実験に必要な定義の最終推薦案](primary_experiment_recommendation_2026-09-11.md)は、今回の必須範囲に限定したR-02/R-03の判断材料である。具体値の推薦と実データでの確認を分け、必要な実物確認を同書第14節へ集約する。推薦文書の作成だけでは実行用freezeを変更しない。旧計画・Issue本文の速度主指標や任意解析の必須化を、最新ユーザー判断に優先させない。

| 差分 | disposition | migration owner |
|---|---|---|
| 主表現 / 主指標 | 正規化変位と点ごとの平均 Euclidean error を新計画候補として採用。実データ topology・単位・QC の確定までは実行 freeze を更新しない | #26 R-02, #27 R-03 |
| matched sparsity | 新 enrichment は 1,000 反復。旧 v6 の joint Null 100 反復とは別 estimand・別 protocol version とする | #27 R-03, #30 R-06, #34 R-18 |
| lag response | 新 Primary は edge-centered response。旧 common-shift は別 ID の補助解析として保持する | #27 R-03, #35 R-19 |
| landscape cell | source-region × lag block を採用候補とし、component topology と同数層化は実データ extractor 確定後に freeze する | #26 R-02, #27 R-03, #33 R-17 |
| enrichment | cell-G 集合平均を主定義、joint-set Ridge gain は別名の補助比較とする | #27 R-03, #34 R-18 |
| GRU / GPDC / LPCMCI / phase surrogate | 必須ではない。Primary freeze 後、必要性を記録した独立 Sensitivity としてのみ実行する | #21, #46–#48 |
| circular shift | 主要 Sensitivity 候補。shift 幅など未決値の確定と Primary freeze 前には実行しない | #27 R-03, #49 |

「候補」「暫定」の値は freeze 済みの値ではない。実装者は性能結果を根拠に補完しない。

## Migration gates

1. R-02 が dataset、extractor/topology、split、normalization、QC、sampling、usable-time の式と値を確定する。
2. R-03 が aggregation、support、candidate/history range、bootstrap、multiple-comparison、enrichment、population-response の契約を確定する。
3. R-04/R-05/R-06 が承認 protocol、raw support、解析統計を fail-closed に接続する。
4. R-17/R-18/R-19 を synthetic contract test まで実装する。
5. schema/protocol version を上げ、旧 v6 artifact を新版成果として受理しない migration test を追加する。
6. R-08/R-09/R-10 の producer、runner、DRY audit が通った後だけ実データ Primary を開始する。
7. Primary result manifest の完全性と hash を R-13 で freeze した後だけ Sensitivity を実行する。

## Traceability and acceptance evidence

- F-ID → Q-ID →責任 Issue / Task →既存証拠・回帰条件: `docs/audit-report.md` と `docs/requirement-matrix.md`
- Issue 一覧と依存関係: `docs/issue-drafts/README.md`、投稿結果は `docs/issue-drafts/POSTED.md`
- Primary hypothesis / endpoint / scope: `docs/experimental_plan.md` §§2–6, 9
- 実装順序と gate: `docs/detailed_design.md` §§3, 13–15
- 採用事項と残る未決事項: `docs/adopted_decisions_2026-09-10.md` §§1–9

R-01 の文書成果は上記で充足する。これは R-02/R-03 の科学決定、実行用 freeze の更新、実データ実験、または下流 Issue の完了を意味しない。
