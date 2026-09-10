# [17][I4] Scientific Audit Gate

## 2026-09-09改訂計画に対する追加実施仕様

既存本文を保持して本節を追記し、新旧の矛盾は本節に同期する。新しい実行Issueを重複作成しない。

## 適用基準・着手条件

基準コード: [dev / 9bbc3d9](https://github.com/shota5558/LaggedFacialGraphForecasting/tree/9bbc3d946c0a64935b8786d9413abe53e6cefe29)。対象はLaggedFacialGraphForecasting。以下のファイル名のみの記載は原則 `src/lagged_facial_graph_forecasting/` 配下を指す。

新研究計画は2026-09-09版。Primaryはh=1、PCMCI+/ParCorr/Ridge、matched sparsity=1000反復。旧v6の100回は旧版の説明であり新版の選択肢ではない。旧全親同時shiftと新edge-centered解析は別estimand。本文の `D-xx` は監査書の決定ID、`Dxx` は新研究計画の決定IDであり同番号とは限らない。後掲の新設計の仕様を優先する。

未決の科学値は依存する決定Issueで式・具体値・根拠・決定日を確定し、本Issue本文にも同期してから依存実装/実runに進む。候補値は採用値ではない。予測結果を見て決定しない。


**Purpose / Source:** F-14、Q55/56、#16/#17。  
**Inputs:** R-09の実入口、R-08のexport、承認protocol、隔離validation fixture。  
**Implementation:** 新アルゴリズムなし。test compositionを本番compositionへ置換して監査を配線。  
**Tests:** 同config/seedのsynthetic runを2回、承認N_matched全反復、全許可条件、hash/ParentSet/alpha/predictions/metrics一致。failure injection/resumeとsupport一致を確認。  
**Scientific checks:** 本番データDRYが必要なら対象と再利用方針を事前指定し、同じouter-testを見て再調整しない。syntheticからREAL DATA VERIFIEDへ昇格させない。  
**Execution:** 既存DRY/I4 tests＋production integration。declared environmentで全suiteとCI。  
**Artifacts:** run ID・SHA・config/manifest hash付きDRY/I4 report、検証scopeラベル。  
**Acceptance:** 全機械監査PASS、未固定fieldゼロ、独立review、#16/#17に正しいscopeで証拠。  
**Dependencies / Blocks:** R-08/09 / R-11。  
**Parallelizable / Complexity:** NO / M。

## 改訂後の必須確認

全候補landscape・cell-G enrichment・edge-centered population解析を含める。matchedは1000反復。科学的未決事項が残る工程は開始しない。

依存Issueの具体的なGitHub番号は投稿後に管理Issueから転記する。実受入証拠はrun ID、git SHA、config/manifest hash、artifact、failure/unevaluable件数、監査判定。
