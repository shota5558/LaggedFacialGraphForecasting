# [R-16-LANDSCAPE] landscapeの新図表生成と実成果受入

## 適用基準・着手条件

基準コード: [dev / 9bbc3d9](https://github.com/shota5558/LaggedFacialGraphForecasting/tree/9bbc3d946c0a64935b8786d9413abe53e6cefe29)。対象はLaggedFacialGraphForecasting。以下のファイル名のみの記載は原則 `src/lagged_facial_graph_forecasting/` 配下を指す。

新研究計画は2026-09-09版。Primaryはh=1、PCMCI+/ParCorr/Ridge、matched sparsity=1000反復。旧v6の100回は旧版の説明であり新版の選択肢ではない。旧全親同時shiftと新edge-centered解析は別estimand。本文の `D-xx` は監査書の決定ID、`Dxx` は新研究計画の決定IDであり同番号とは限らない。後掲の新設計の仕様を優先する。

未決の科学値は依存する決定Issueで式・具体値・根拠・決定日を確定し、本Issue本文にも同期してから依存実装/実runに進む。候補値は採用値ではない。予測結果を見て決定しない。

## 依存関係

- R-17
- R-08
- [#19](https://github.com/shota5558/LaggedFacialGraphForecasting/issues/19)

## タスク仕様

## 実装・成果

全cell表・source→target/lag-band集約・landscape図。全Ω、G=E_self−E_cell、単位、subject分母、探索的cell順位を検算。

analysis_pipeline.pyのsource-table→plot→registryを再利用し、独立した出力ID `LANDSCAPE` を追加。入力は対応解析のraw recordと確定統計。canonical CSVを先に保存し、PNG/SVG、caption、analysis_artifact_registry.csv、analysis_manifest.jsonを生成する。既存T/Fを上書きしない。

## 受入条件

- [ ] rawから再計算しCSVと図の値・単位・CI・分母が一致。
- [ ] 欠落record、別run/config/support、未定義metricを拒否する回帰と手計算fixtureを追加。
- [ ] 実artifactのrun ID、SHA、config/source/support hash、seed、failure件数を保存。
- [ ] 目視・数値検算・review後にtrackerの実成果完了を記録。
- [ ] mockはMOCK DATA / NOT A SCIENTIFIC RESULTと表示。

ソフトウェアとfixture検証は先行可能。実成果受入は #19 完了後。

## 自己完結のための仕様補足

## 11. 統計・出力の再生成

### 11.1 統計入力の検証

Pairing keyはrun/protocol、fold、subject、target、h、metric、estimand、supportを含む。subject IDだけのmergeや、行数だけの一致を許可しない。重複キー、missing pair、異なる単位・指標方向を拒否する。

主指標はconfigで明示し、任意のlower-is-better指標へfallbackしない。matched反復のmean/medianは凍結済み解析規則に従い、table生成側で切り替えない。点推定とbootstrapで同じestimandを使う。

CI bootstrapはsubjectを再標本化し、被験者に付随するedge/regionを保持する。discovery bootstrapの引数とは別namespaceにする。seed、反復数、対象subject ID、CI方式を保存する。

### 11.2 必須出力

|出力|必要な内容|
|---|---|
|データ・QC表|被験者、sequence、frame、欠損・除外理由、sampling|
|4条件表|subject/region別誤差、paired効果、平均/中央値/CI、特徴数|
|Landscape表・図|全cell G、候補集合、support、部位対・帯域集約|
|Enrichment表・図|selected値、1,000反復分布、効果・CI、estimand|
|Population表・図|edge別差、Δ別中央値・95% CI、分母、frame/ms|
|Null比較|操作別結果、mapping、保持量、失敗率|
|安定性|train bootstrap/fold頻度、分母、部位・成分投影|
|実行監査|config/code/data hash、freeze履歴、失敗・再開、完全性|
|Sensitivity|Primary参照hash、変更点、独立結果、制約|

Core prediction/metricから解析tableへのexportは単一の検証経路を用いる。CSVを手編集して報告値を整えない。各集計値から元recordへ遡れるキーを残す。

## 完了記録

- [ ] 上記実施範囲・出力・回帰条件を満たした。
- [ ] 実装PRと検証コマンド/結果、残る科学判断を記録した。
- [ ] 実行・成果受入タスクの場合、実run ID・git SHA・config/manifest/source hash・artifact・失敗/評価不能数・検算結果を記録した。ソフトウェア成功だけで実成果完了にしない。

