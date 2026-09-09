# [20][FREEZE] PRIMARY FREEZE

## 2026-09-09改訂計画に対する追加実施仕様

既存本文を保持して本節を追記し、新旧の矛盾は本節に同期する。新しい実行Issueを重複作成しない。

## 適用基準・着手条件

基準コード: [dev / 9bbc3d9](https://github.com/shota5558/LaggedFacialGraphForecasting/tree/9bbc3d946c0a64935b8786d9413abe53e6cefe29)。対象はLaggedFacialGraphForecasting。以下のファイル名のみの記載は原則 `src/lagged_facial_graph_forecasting/` 配下を指す。

新研究計画は2026-09-09版。Primaryはh=1、PCMCI+/ParCorr/Ridge、matched sparsity=1000反復。旧v6の100回は旧版の説明であり新版の選択肢ではない。旧全親同時shiftと新edge-centered解析は別estimand。本文の `D-xx` は監査書の決定ID、`Dxx` は新研究計画の決定IDであり同番号とは限らない。後掲の新設計の仕様を優先する。

未決の科学値は依存する決定Issueで式・具体値・根拠・決定日を確定し、本Issue本文にも同期してから依存実装/実runに進む。候補値は採用値ではない。予測結果を見て決定しない。


**Purpose / Source:** Q59/60、#20、F-11。  
**Inputs:** 完了したR-11/R-12、既存13種と新landscape/enrichment/populationのrequired artifact。  
**Files:** 最小freeze writer、既存artifact_registry/failure_handling/sensitivity_execution。  
**Implementation:** required types全て・全実行unit・新3解析の完全性を検査し、承認migration後schemaでmanifestをatomicに生成。現manifest schema v1のrequired setは旧13種なので、新成果のrequired type/構成とconsumerの検証を明示version migrationする。旧manifestを新研究完了として受理しない。hashがあるだけの空dummy fileを受入証拠にしない。  
**Execution:** 提案 `python scripts/freeze_primary.py --manifest <validated-primary-manifest>`（未実装）。  
**Tests:** 欠落type、file欠落、hash改変、重複path、別run、未完了fold、overwriteを拒否。consumer round-trip。  
**Artifacts:** artifacts/primary/freeze_manifest.json、integrity audit。  
**Acceptance:** 既存13種＋新3解析＋全予定単位を参照、same inputで同digest、freeze後上書き不可、独立review、#20 PASS。  
**Failure:** incomplete scienceをPASSにしない。  
**Dependencies / Blocks:** R-12 / R-14。  
**Parallelizable / Complexity:** Primary読取図表と可 / M。

## 改訂後の必須確認

全候補landscape・cell-G enrichment・edge-centered population解析を含める。matchedは1000反復。科学的未決事項が残る工程は開始しない。

依存Issueの具体的なGitHub番号は投稿後に管理Issueから転記する。実受入証拠はrun ID、git SHA、config/manifest hash、artifact、failure/unevaluable件数、監査判定。
