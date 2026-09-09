# [21][S] Sensitivity

## 2026-09-09改訂計画に対する追加実施仕様

既存本文を保持して本節を追記し、新旧の矛盾は本節に同期する。新しい実行Issueを重複作成しない。

## 適用基準・着手条件

基準コード: [dev / 9bbc3d9](https://github.com/shota5558/LaggedFacialGraphForecasting/tree/9bbc3d946c0a64935b8786d9413abe53e6cefe29)。対象はLaggedFacialGraphForecasting。以下のファイル名のみの記載は原則 `src/lagged_facial_graph_forecasting/` 配下を指す。

新研究計画は2026-09-09版。Primaryはh=1、PCMCI+/ParCorr/Ridge、matched sparsity=1000反復。旧v6の100回は旧版の説明であり新版の選択肢ではない。旧全親同時shiftと新edge-centered解析は別estimand。本文の `D-xx` は監査書の決定ID、`Dxx` は新研究計画の決定IDであり同番号とは限らない。後掲の新設計の仕様を優先する。

未決の科学値は依存する決定Issueで式・具体値・根拠・決定日を確定し、本Issue本文にも同期してから依存実装/実runに進む。候補値は採用値ではない。予測結果を見て決定しない。


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

## 改訂後の必須確認

全候補landscape・cell-G enrichment・edge-centered population解析を含める。matchedは1000反復。科学的未決事項が残る工程は開始しない。GRUは任意とし、既存S-06を必須完了条件から外す。GPDC/LPCMCI/phase/circular/h>1は各子Issueで独立受入する。

依存Issueの具体的なGitHub番号は投稿後に管理Issueから転記する。実受入証拠はrun ID、git SHA、config/manifest hash、artifact、failure/unevaluable件数、監査判定。
