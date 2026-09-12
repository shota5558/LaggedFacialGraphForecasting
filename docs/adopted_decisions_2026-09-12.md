# Adopted Decisions — 2026-09-12

## Status

2026-09-12、ユーザー指示により `docs/primary_experiment_recommendation_2026-09-11.md` を本研究の正式な Primary specification として全面採用した。

この決定により、同文書内の「推薦」「候補」「本案」とされていた科学設計値・規則は、実データ依存の事実を除き `ADOPTED` とする。

正式な authority は `docs/authoritative_primary_experiment_spec_2026-09-12.md` を参照する。

## 採用事項

- 主目的: 自然な表情・発話中の顔部位間遅延予測構造と被験者間共通性の解明
- 主表現: 2D正規化変位
- 主データ第一選択: NoXi単一配布版
- 発話条件: 本人のみ発話 / 本人非発話
- dependency-group outer 5-fold、必要時のみ3-fold、LOSO自動fallback禁止
- 8 region / 29 landmark / 58 scalar components
- MediaPipe Face Landmarker IMAGE mode
- QC / calibration / missingness / cadence規則
- `h=1`
- `L=floor(0.5*fps)`
- lag bands `(0,100]`, `(100,250]`, `(250,500] ms`
- Self history決定手順
- PCMCI+ + ParCorr, `pc_alpha=0.01`
- scalar discovery → region-block OR projection
- Ridge被験者等重み学習
- common evaluation support
- full candidate Predictive Gain Landscape
- Selection Enrichment 1000 random sets
- centered lag response 約±100 ms
- held-out significance/FDRをPrimary completion criterionにしない
- OOF group bootstrap 10000回、nominal 95% interval
- discovery stability group resampling 100回

## Superseded

- velocity主target / velocity RMSE主指標
- fixed `tau_max=10`
- fixed `[-2,-1,0,1,2]` centered grid
- Random-region / Time-shuffle / joint-set matched sparsityをPrimary必須Nullとする規則
- held-out significance / FDRを必須完了判定とする規則
- LOSO自動fallback
- exact selected scalar componentだけをforecastへ渡す旧契約
- Sensitivity未決値でPrimaryをblockする運用

## 未決ではなく preflight で materialize する事項

- NoXi配布版・利用条件・file hash
- extractor/model exact version/hash
- actual fps/cadence
- actual subject/group/session inventory
- actual split manifests
- actual Self history H
- 必要なら予備段階で拡張されたalpha grid
- actual support counts/durations
- dry run / leakage / reproducibility evidence
- executable freeze hash

これらはouter-testの結果で決めない。
