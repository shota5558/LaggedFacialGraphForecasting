# 残存課題 Issue 投稿案

送信先: [shota5558/LaggedFacialGraphForecasting](https://github.com/shota5558/LaggedFacialGraphForecasting/issues)

**状態: ユーザー承認後、GitHubへ投稿済み。新規53件、既存6件更新。[投稿結果とIssueリンク](POSTED.md)を参照。issues.jsonと各本文ファイルは承認時の原稿として保持。**

新規は管理1件＋作業52件。既存 #16〜#21 は本文追記6件。研究仕様の未決値は確定していない。

- [管理Issue本文](00-governing.md)
- [投稿用JSON](issues.json)

## 新規作業Issue

| ID | タイトル | 依存 |
|---|---|---|
| R-01 | [[R-01] 新計画とv6の差分をIssue・migration planへ確定](R-01.md) | なし |
| R-02 | [[R-02] データ・split・前処理の科学決定](R-02.md) | R-01 |
| R-03 | [[R-03] 推論・安定性の科学契約を確定](R-03.md) | R-01 |
| R-04 | [[R-04] data manifestと承認protocolのenforcement](R-04.md) | R-02, R-03 |
| R-05 | [[R-05] 同一評価supportをraw predictionまで強制](R-05.md) | R-02, R-03 |
| R-06 | [[R-06] 解析とScientific Freezeの統計契約を統一](R-06.md) | R-03, R-05 |
| R-07a | [[R-07a] real Sensitivity解析のmanifest barrier](R-07a.md) | R-01 |
| R-07b | [[R-07b] Primary-only解析の独立入力](R-07b.md) | R-07a |
| R-17 | [[R-17] 固定候補全体のpredictive gain landscape](R-17.md) | R-03, R-05 |
| R-18 | [[R-18] PCMCI selection enrichment](R-18.md) | R-17, R-06 |
| R-19 | [[R-19] Edge-centered population lag response](R-19.md) | R-17, R-05, R-03 |
| R-08 | [[R-08] raw scientific artifactからanalysis入力を生成](R-08.md) | R-04, R-06, R-07b, R-17, R-18, R-19 |
| R-09 | [[R-09] 全Primary条件のproduction runner](R-09.md) | R-04, R-06, R-17, R-18, R-19 |
| R-10 | [[R-10] 実行protocolのDRY / I4受入](R-10.md) | R-08, R-09 |
| R-13 | [[R-13] PRIMARY FREEZE作成・検証](R-13.md) | R-03 |
| R-15-G10 | [[R-15-G10] Position RMSEの仕様確定と指標実装](R-15-G10.md) | R-02, R-03 |
| R-15-G11 | [[R-15-G11] Acceleration RMSEの仕様確定と指標実装](R-15-G11.md) | R-02, R-03 |
| R-15-G12 | [[R-15-G12] Temporal correlationの仕様確定と指標実装](R-15-G12.md) | R-02, R-03 |
| R-15-G13 | [[R-15-G13] Peak timingの仕様確定と指標実装](R-15-G13.md) | R-02, R-03 |
| R-15-G14 | [[R-15-G14] Onset timingの仕様確定と指標実装](R-15-G14.md) | R-02, R-03 |
| R-15-G15 | [[R-15-G15] Lag preservationの仕様確定と指標実装](R-15-G15.md) | R-02, R-03 |
| R-14-GPDC | [[R-14-GPDC] PCMCI+ + GPDCの本番接続・独立実行・受入](R-14-GPDC.md) | R-07a, R-13, #20 |
| R-14-LPCMCI | [[R-14-LPCMCI] LPCMCIの本番接続・独立実行・受入](R-14-LPCMCI.md) | R-07a, R-13, #20 |
| R-14-phase | [[R-14-phase] Phase-shuffled surrogateの本番接続・独立実行・受入](R-14-phase.md) | R-07a, R-13, #20 |
| R-14-circular | [[R-14-circular] Circular-shift surrogateの本番接続・独立実行・受入](R-14-circular.md) | R-07a, R-13, #20 |
| R-14-horizon | [[R-14-horizon] h > 1の本番接続・独立実行・受入](R-14-horizon.md) | R-07a, R-13, #20 |
| R-16-T01 | [[R-16-T01] T01：dataset summaryの実成果生成・検算](R-16-T01.md) | R-08, #19 |
| R-16-T02 | [[R-16-T02] T02：config tableの実成果生成・検算](R-16-T02.md) | R-08, #19 |
| R-16-T03 | [[R-16-T03] T03：4条件表の実成果生成・検算](R-16-T03.md) | R-08, #19 |
| R-16-T04 | [[R-16-T04] T04：Self−PCMCIの実成果生成・検算](R-16-T04.md) | R-08, #19 |
| R-16-T05 | [[R-16-T05] T05：sparsityの実成果生成・検算](R-16-T05.md) | R-08, #19 |
| R-16-T06 | [[R-16-T06] T06：region表の実成果生成・検算](R-16-T06.md) | R-08, #19 |
| R-16-T07 | [[R-16-T07] T07：Null表/分布の実成果生成・検算](R-16-T07.md) | R-08, #19 |
| R-16-T08 | [[R-16-T08] T08：stabilityの実成果生成・検算](R-16-T08.md) | R-08, #19 |
| R-16-T09 | [[R-16-T09] T09：summaryの実成果生成・検算](R-16-T09.md) | R-08, #19, #20, #21 |
| R-16-F01 | [[R-16-F01] F01：condition plotの実成果生成・検算](R-16-F01.md) | R-08, #19 |
| R-16-F02 | [[R-16-F02] F02：paired effectの実成果生成・検算](R-16-F02.md) | R-08, #19 |
| R-16-F03 | [[R-16-F03] F03：region effectsの実成果生成・検算](R-16-F03.md) | R-08, #19 |
| R-16-F04 | [[R-16-F04] F04：curveの実成果生成・検算](R-16-F04.md) | R-08, #19 |
| R-16-F05 | [[R-16-F05] F05：Null効果の実成果生成・検算](R-16-F05.md) | R-08, #19 |
| R-16-F06 | [[R-16-F06] F06：sparsity plotの実成果生成・検算](R-16-F06.md) | R-08, #19 |
| R-16-F07 | [[R-16-F07] F07：region heatmapの実成果生成・検算](R-16-F07.md) | R-08, #19 |
| R-16-F08 | [[R-16-F08] F08：heatmapの実成果生成・検算](R-16-F08.md) | R-08, #19 |
| R-16-F09 | [[R-16-F09] F09：fold分布の実成果生成・検算](R-16-F09.md) | R-08, #19 |
| R-16-F10 | [[R-16-F10] F10：forestの実成果生成・検算](R-16-F10.md) | R-08, #19, #20, #21 |
| R-16-F11 | [[R-16-F11] F11：exampleの実成果生成・検算](R-16-F11.md) | R-08, #19 |
| R-16-F12 | [[R-16-F12] F12：diagnosticsの実成果生成・検算](R-16-F12.md) | R-08, #19 |
| R-16-F13 | [[R-16-F13] F13：distributionの実成果生成・検算](R-16-F13.md) | R-08, #19 |
| R-16-F14 | [[R-16-F14] F14：concordanceの実成果生成・検算](R-16-F14.md) | R-08, #19 |
| R-16-LANDSCAPE | [[R-16-LANDSCAPE] landscapeの新図表生成と実成果受入](R-16-LANDSCAPE.md) | R-17, R-08, #19 |
| R-16-ENRICHMENT | [[R-16-ENRICHMENT] enrichmentの新図表生成と実成果受入](R-16-ENRICHMENT.md) | R-18, R-08, #19 |
| R-16-POPULATION | [[R-16-POPULATION] populationの新図表生成と実成果受入](R-16-POPULATION.md) | R-19, R-08, #19 |

## 既存Issueへの追記

- [#16 [16][DRY] One Outer-Fold Scientific Dry Run](existing-16.md)
- [#17 [17][I4] Scientific Audit Gate](existing-17.md)
- [#18 [18][PR] Primary Full Run](existing-18.md)
- [#19 [19][STAT] Primary Statistics & Final Analysis](existing-19.md)
- [#20 [20][FREEZE] PRIMARY FREEZE](existing-20.md)
- [#21 [21][S] Sensitivity](existing-21.md)

## 投稿時の扱い

管理Issueを作成し、作業Issueを依存順で投稿。R-IDを返却されたGitHub番号へ置換し本文・管理一覧を更新する。既存Issueは最新本文を再取得して追記し、他の編集を消さない。実験の実行は今回の作業範囲に含めない。
