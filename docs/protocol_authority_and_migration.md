# Protocol Authority Registry and Migration Plan

更新日: 2026-09-12  
対象: `shota5558/LaggedFacialGraphForecasting`

## 1. Authority registry

2026-09-12 のユーザー決定により、2026-09-11 の最終推薦案を全面採用した。以後の authority hierarchy は次とする。

| 優先 | 役割 | repository representation |
|---|---|---|
| 1 | 正式な Primary specification declaration | `docs/authoritative_primary_experiment_spec_2026-09-12.md` |
| 2 | 採用対象となった詳細設計本文 | `docs/primary_experiment_recommendation_2026-09-11.md` |
| 3 | 読みやすい統合実験設計 | `docs/experiment_design.md` |
| 4 | 採用判断・履歴 | `docs/adopted_decisions_2026-09-10.md` および 2026-09-12 の adoption record |
| 5 | 移行対象の旧研究計画・詳細設計 | `docs/experimental_plan.md`, `docs/detailed_design.md` |
| 6 | Legacy executable protocol | `configs/scientific_freeze.yaml`, `schemas/scientific_freeze.schema.json` |

下位文書、config、schema、Issue が上位仕様と矛盾する場合、上位仕様を優先し、下位記述は `SUPERSEDED` とする。

## 2. 2026-09-12 に正式採用された主要決定

以下は「推薦候補」ではなく Primary の設計決定となった。

- 主研究対象: 自然な表情・発話中の顔部位間遅延予測構造と被験者間共通性
- 主表現: 点ごとの2D正規化変位
- データ第一選択: NoXi の単一配布版
- 解析条件: 本人のみ発話 / 本人非発話
- 8 region / 29 landmark / 58 scalar component mapping
- MediaPipe Face Landmarker IMAGE mode
- calibration / QC / missingness / cadence 規則
- Outer: dependency-group 5-fold、成立不能時のみ3-fold、LOSO fallback禁止
- Inner: group 3-fold
- `h=1`
- inter-region lag: `L=floor(0.5*fps)`、1 frame刻み
- lag bands: `(0,100]`, `(100,250]`, `(250,500] ms`
- Self history の予備データによる決定アルゴリズム
- PCMCI+ + ParCorr, `pc_alpha=0.01`
- scalar discovery → region-block OR projection
- Ridge の被験者等重み学習、alpha tuning contract
- condition×fold×subject 単位の common support
- Predictive Gain Landscape `G = E_Self - E_cell`
- Selection Enrichment: cell-G 集合平均、1000 random sets
- centered lag response: 約±100 ms、complete symmetric grid
- OOF group bootstrap 10000回、nominal 95% percentile interval
- held-out significance test / FDR を Primary completion criterion にしない
- discovery stability: outer-train group resampling 100回

## 3. Primary mandatory boundary

本実験の必須解析は以下。

1. Predictive Gain Landscape
2. Persistence / Self / Full / PCMCI-block 4条件比較
3. Selection Enrichment
4. selected-lag centered response
5. discovery stability
6. data / execution audit

以下は Primary completion blocker ではない。

- Random-region
- Time-shuffle
- joint-set matched-sparsity model comparison
- phase-shuffled / circular-shift surrogate
- GPDC
- LPCMCI
- GRU
- h>1
- GNN
- secondary motion metrics

必要な場合のみ PRIMARY FREEZE 後の Sensitivity / Optional namespace で実行する。

## 4. 実物 preflight と科学仕様を分離する

以下は科学的未決事項ではなく、実データから materialize する実物確認である。

- NoXi の利用可能な配布版・利用条件・file hash
- subject/group/session/sequence inventory
- actual fps / cadence
- extractor exact version / model hash
- landmark overlay / 左右 / pose axis / tracking quality
- actual `L` / lag-band membership
- adopted Self history `H`
- final alpha grid（必要な場合のみ予備段階で拡張）
- actual split manifests
- common support counts / duration
- dry run / leakage / reproducibility evidence

これらを未取得のまま値で埋めない。

## 5. Superseded legacy decisions

本仕様により少なくとも以下は旧 Primary 仕様として superseded となる。

- velocity を Primary target / metric とする規則
- `tau_max=10` 固定
- centered lag grid `[-2,-1,0,1,2]` 固定
- Random-region / Time-shuffle / joint-set matched sparsity を Primary 必須 Null とする規則
- held-out significance / FDR を Primary の必須完了判定とする規則
- LOSO への自動 fallback
- scalar parent の exact component のみを forecasting feature とする旧 contract
- sensitivity の未決値が Primary 実行を block する設計

## 6. Executable migration gates

設計採用は完了したが、実行環境の移行は別に完了させる。

1. `configs/scientific_freeze.yaml` を新 protocol version へ移行
2. `configs/primary_run.yaml` を新 Primary mandatory scope へ移行
3. `schemas/scientific_freeze.schema.json` を同期
4. config loader / preflight validator を同期
5. region block / support / landscape / enrichment / centered-response contracts を同期
6. old-v6 artifact を新 Primary evidence として拒否する regression test を追加
7. synthetic integration を通す
8. one-fold dry run を通す
9. leakage / reproducibility audit を通す
10. preflight 実物値と hash を固定
11. 新 Primary Full Run を開始
12. PRIMARY FREEZE 後に Sensitivity を許可

## 7. Issue authority

- #24: active backlog / dependency index
- #25: protocol migration / scientific adoption record
- #28: data / support contract enforcement
- #31: Primary / Sensitivity boundary
- #33: Predictive Gain Landscape
- #34: Selection Enrichment
- #35: centered Population Lag Response
- #36: raw artifact → analysis export
- #37: production runner
- #38: production integration / dry-run audit
- #16 / #17 / #18 / #19 / #20: formal execution gates

旧 Issue は履歴として残し、上記 authoritative specification と競合する場合は新 work の authority にしない。

## 8. Completion rule

README や文書へリンクしただけでは migration complete としない。

`specification → config/schema → implementation → tests → dry run → leakage audit → frozen real preflight values → hash` が一致して初めて executable migration complete とする。
