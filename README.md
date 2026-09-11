# Lagged Facial Graph Forecasting

PCMCI+ と Ridge を用いて、顔部位間の遅延予測構造を held-out 被験者で評価する研究実装です。

## 現在の基準

- [新指標に基づく解析図表の再構成案（2026-09-11）](docs/analysis_tables_figures_redesign_2026-09-11.md)
- [本実験に必要な定義の最終推薦案（2026-09-11）](docs/primary_experiment_recommendation_2026-09-11.md)
- [研究計画](docs/experimental_plan.md)
- [詳細設計](docs/detailed_design.md)
- [原典レジストリと移行計画](docs/protocol_authority_and_migration.md)
- [要求トレーサビリティ](docs/requirement-matrix.md)
- [Issue 実装記録](docs/issue-implementation-2026-09-10.md)

実行用 Scientific Freeze は旧 protocol v6 のままです。新計画の科学判断が未確定な項目は候補値を採用値として扱わず、R-02/R-03 の完了後に versioned migration します。mock の成功は実データ成果や publication-ready の証拠ではありません。

2026-09-11の推薦案は今回の必須解析に範囲を限定し、NoXiを主データの第一候補とします。ユーザー判断により集団の共通性を主目的として維持し、利用可能なCAVIARES（1話者）は集団実験の代用にしません。推薦案の具体値は実行用freezeへの反映・実データでの確認前であり、実験開始可能を意味しません。
