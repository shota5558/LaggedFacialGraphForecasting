# Adopted Decisions — 2026-09-12

Status: **ADOPTED**. The 2026-09-11 Primary recommendation was adopted in
full on 2026-09-12. This record promotes its scientific design rules, but does
not invent facts that require the real data preflight.

## Adopted Primary design

- Primary representation: pointwise 2D normalized displacement.
- Primary data: one NoXi distribution; CAVIARES is not an automatic fallback.
- Conditions: self-speaking and self-non-speaking, excluding overlap,
  non-speech events, unknown/transition intervals, and ±200 ms around
  boundaries.
- Split: dependency-group outer 5-fold once, with a single 3-fold fallback only
  when the stated group-count requirements cannot be met; inner 3-fold;
  no automatic LOSO or frame split.
- Measurement: MediaPipe Face Landmarker IMAGE mode, one face, detection and
  presence thresholds 0.5, BlendShapes off, pose matrix on; 8 regions, 29
  landmarks, and 58 x/y scalar components using the adopted mapping in the
  authoritative specification.
- Forecast lag: `h=1`, `L=floor(0.5*fps)`, integer lags `1..L`, and bands
  `(0,100]`, `(100,250]`, `(250,500]` ms.
- Discovery: PCMCI+ with ParCorr, `pc_alpha=0.01`; scalar links are preserved
  and OR-projected to source-region × target-region × lag blocks. Forecasting
  adds every usable scalar in the selected source block.
- Forecasting: Ridge with input-only standardization, subject-equal weights,
  common multi-output alpha, and the adopted nine-value base grid.
- Primary analyses: full predictive-gain landscape, four-condition comparison,
  1,000-repeat cell-G enrichment, selected-lag response with
  `M=floor(0.1*fps)`, discovery stability with 100 group resamples, and audit.
- Inference: subject-level median-centered summaries and nominal OOF
  dependency-group bootstrap, 10,000 resamples, linear percentile interval;
  held-out significance and FDR are not completion criteria.

## Superseded

Scientific Freeze v6, its fixed `tau_max=10`, fixed `[-2,-1,0,1,2]` response,
velocity primary metric, exact-selected-scalar forecast rule, and mandatory
Random-region/Time-shuffle/joint-set Null lanes are not the adopted Primary
protocol. Existing v6 artifacts require explicit migration and cannot be
presented as adopted Primary results.

## Preflight-only facts

NoXi distribution/file hashes, exact extractor/model versions, actual fps,
subject/group inventory, realized split manifests, selected Self history, and
support counts are materialized only from preflight evidence. They must not be
selected using formal outer-test outcomes.
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
