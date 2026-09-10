# [18][PR] Primary Full Run

## 2026-09-09改訂計画に対する追加実施仕様

既存本文を保持して本節を追記し、新旧の矛盾は本節に同期する。新しい実行Issueを重複作成しない。

## 適用基準・着手条件

基準コード: [dev / 9bbc3d9](https://github.com/shota5558/LaggedFacialGraphForecasting/tree/9bbc3d946c0a64935b8786d9413abe53e6cefe29)。対象はLaggedFacialGraphForecasting。以下のファイル名のみの記載は原則 `src/lagged_facial_graph_forecasting/` 配下を指す。

新研究計画は2026-09-09版。Primaryはh=1、PCMCI+/ParCorr/Ridge、matched sparsity=1000反復。旧v6の100回は旧版の説明であり新版の選択肢ではない。旧全親同時shiftと新edge-centered解析は別estimand。本文の `D-xx` は監査書の決定ID、`Dxx` は新研究計画の決定IDであり同番号とは限らない。後掲の新設計の仕様を優先する。

未決の科学値は依存する決定Issueで式・具体値・根拠・決定日を確定し、本Issue本文にも同期してから依存実装/実runに進む。候補値は採用値ではない。予測結果を見て決定しない。


**Purpose / Source:** F-15、Q57、#18 PR-01〜10。  
**Input / Configuration:** immutable data manifest、承認済みprotocol、全split、実行SHA・software lock。  
**Execution command:** R-09で確定する本番CLI `python scripts/run_primary.py --config configs/primary_run.yaml`（現在は未存在、preflightを通った版のみ）。  
**Implementation:** 実験実行と記録のみ。  
**Tests / Scientific checks:** 起動前R-10、foldごとleakage/freeze/support、全8条件＋新3解析、matched N_matched、lag全grid。  
**Expected output:** 全foldの選択状態・予測・metric・failure/unevaluable ledger・manifest。  
**Validation criterion:** 予定したfold/target/condition/repeat集合に対し成功・承認unevaluable・失敗が全件accounted。未完了を成功扱いしない。  
**Failure condition:** hash drift、未固定値、漏洩、欠落条件、科学状態を変えるresumeは停止。性能低下そのものは実装FAILにしない。  
**Acceptance:** #18の全受入条件、run ID/config hash/git SHA/result artifact/audit dispositionを記録。  
**Dependencies / Blocks:** R-10 / R-12。  
**Parallelizable / Complexity:** NO / XL（被験者数・計算量未定）。

## 改訂後の必須確認

全候補landscape・cell-G enrichment・edge-centered population解析を含める。matchedは1000反復。科学的未決事項が残る工程は開始しない。

依存Issueの具体的なGitHub番号は投稿後に管理Issueから転記する。実受入証拠はrun ID、git SHA、config/manifest hash、artifact、failure/unevaluable件数、監査判定。
