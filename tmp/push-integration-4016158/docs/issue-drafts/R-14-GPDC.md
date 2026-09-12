# [R-14-GPDC] PCMCI+ + GPDCの本番接続・独立実行・受入

## 適用基準・着手条件

基準コード: [dev / 9bbc3d9](https://github.com/shota5558/LaggedFacialGraphForecasting/tree/9bbc3d946c0a64935b8786d9413abe53e6cefe29)。対象はLaggedFacialGraphForecasting。以下のファイル名のみの記載は原則 `src/lagged_facial_graph_forecasting/` 配下を指す。

新研究計画は2026-09-09版。Primaryはh=1、PCMCI+/ParCorr/Ridge、matched sparsity=1000反復。旧v6の100回は旧版の説明であり新版の選択肢ではない。旧全親同時shiftと新edge-centered解析は別estimand。本文の `D-xx` は監査書の決定ID、`Dxx` は新研究計画の決定IDであり同番号とは限らない。後掲の新設計の仕様を優先する。

未決の科学値は依存する決定Issueで式・具体値・根拠・決定日を確定し、本Issue本文にも同期してから依存実装/実runに進む。候補値は採用値ではない。予測結果を見て決定しない。

## 依存関係

- R-07a
- R-13
- [#20](https://github.com/shota5558/LaggedFacialGraphForecasting/issues/20)

## タスク仕様

**Purpose / Source:** Q63、#21、#22の実装済み部品。  
**Inputs:** R-13 manifest、各methodの事前承認config、同じsplitを含むprovenance。  
**Tasks:** R-14-GPDC、R-14-LPCMCI、R-14-phase、R-14-circular、R-14-horizon、R-14-GRUを各々独立受入。GRUは新計画でoptional、Issue #21では必須なのでR-01で採否を整合。surrogateは新enrichment/population構造の検証にも接続。  
**Implementation:** 新手法を再実装せず既存adapterをorchestrate。各設定（h、shift規則、GRU訓練等）を実結果前に承認。  
**Execution:** method別のproduction入口は現状未確認。既存公開APIから作るthin CLIを個別PRで確定し、その --help とmanifest guardを確認してからrun。架空コマンドで実行済みとしない。  
**Tests / Checks:** guard、独立seed/version、Primary hash不変、h>1 availability、null session boundaries。  
**Outputs:** artifacts/sensitivity/ 配下のmethod別run manifest・predictions/metrics・paired effect。  
**Acceptance:** 採用された各行に実run ID・config/seed/SHA・source Primary freeze hash・失敗状況を記録。1成功で6件完了にしない。  
**Failure:** Primary改変、manifest不一致、未承認設定は停止。  
**Dependencies / Blocks:** R-13/R-07a / R-16-T09/F10。  
**Parallelizable / Complexity:** 承認後は異なるoutput root間で可 / 各M〜L。

## 自己完結のための仕様補足

## 個別スコープ

本IssueはPCMCI+ + GPDCのみ。親は #21。既存 `src/lagged_facial_graph_forecasting/sensitivity_gpdc.py` と対応testsを使用する。

事前決定・検証項目：CI test変更、Ridge/split/h=1保持、GPDC設定とseed。

新計画D09の設定・適用partition・反復数・seedを記録後、Primaryとは別run rootで実行。landscape/enrichment/populationへの頑健性も接続する。GRUは任意で本Issueの必須条件ではない。

## 完了記録

- [ ] 上記実施範囲・出力・回帰条件を満たした。
- [ ] 実装PRと検証コマンド/結果、残る科学判断を記録した。
- [ ] 実行・成果受入タスクの場合、実run ID・git SHA・config/manifest/source hash・artifact・失敗/評価不能数・検算結果を記録した。ソフトウェア成功だけで実成果完了にしない。

