# Lagged Facial Graph Forecasting

PCMCI+ と Ridge を用いて、顔部位間の遅延予測構造を held-out 被験者で評価する研究実装です。

## 実験設計

- [実験設計書：顔部位間の遅延予測構造と被験者間の共通性](docs/experiment_design.md)

実験内容はこの設計書を入口とします。目的、データ、測定、分割、学習、評価、統計、図表、実行手順、完了条件を単独で読める形で記載しています。実データ確認前の設計値と、本実験開始前に登録する実値を区別しています。

## 検討資料・実装管理

- [新指標に基づく解析図表の再構成案（2026-09-11）](docs/analysis_tables_figures_redesign_2026-09-11.md)
- [本実験に必要な定義の最終推薦案（2026-09-11）](docs/primary_experiment_recommendation_2026-09-11.md)
- [研究計画の検討資料](docs/experimental_plan.md)
- [実装移行の詳細設計](docs/detailed_design.md)
- [原典レジストリと移行計画](docs/protocol_authority_and_migration.md)
- [要求トレーサビリティ](docs/requirement-matrix.md)
- [Issue 実装記録](docs/issue-implementation-2026-09-10.md)

上記は検討経緯と実装管理のための資料です。実験仕様の確認には実験設計書を用います。

実行用Scientific Freezeはschema v6です。本実験の開始には、設計値の採用、実データ確認、config/schemaと実行経路の整合、検証と凍結が必要です。設計書の完成やmockの成功は、実データ実験の実施・受容を意味しません。
