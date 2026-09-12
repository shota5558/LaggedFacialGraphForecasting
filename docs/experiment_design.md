# 顔部位間の遅延予測構造と被験者間の共通性：実験設計書

作成日：2026年9月12日  
文書の状態：**AUTHORITATIVE DESIGN / 実データ preflight 前**

> **Authority notice — 2026-09-12**  
> `docs/primary_experiment_recommendation_2026-09-11.md` の推薦案は、2026-09-12 に本仕様として全面採用された。本書はその採用内容を読みやすく統合した実験設計書であり、科学設計上の「候補」「推薦値」として再審議するものではない。  
> 実データからしか取得できない配布版、file hash、実 fps、extractor/model hash、実 fold、実 support 等は未取得の事実として preflight で materialize し、未確認の値を創作しない。  
> 詳細な authority と supersession 規則は [Authoritative Primary Experiment Specification](authoritative_primary_experiment_spec_2026-09-12.md) を正とする。

## 1. 研究概要と設計の状態

本研究は、自然な対話における顔運動を対象に、ある部位の過去が別の部位の次時点の運動を予測する情報を持つか、その情報が学習に含まれない被験者にもどの程度共通するかを調べる。対象部位自身の履歴による予測を基準に、他部位の履歴を加えたときの誤差減少を、すべての部位対と時間差について評価する。PCMCI+による関係の選択は、その予測上の価値と安定性を併せて調べる。

本書で定める科学設計は採用済みである。今後の未完了事項は、科学設計の再選択ではなく、実データ・実 extractor・実 fps に依存する preflight と executable migration である。

| 状態 | 内容 |
|---|---|
| 採用済み科学仕様 | 集団の共通性、正規化変位、点ごとの平均ユークリッド誤差、被験者・対話groupを分離した評価、全候補の予測情報、PCMCI+・ParCorr・Ridge、1フレーム先予測、Landscape、Enrichment、centered lag response、stability |
| 採用済み具体設計 | NoXi第一選択、29点mapping、QC運用値、校正、Self履歴決定手順、`L=floor(0.5f)`、3 lag bands、OR projection、Ridge重み、共通support、±約100 ms centered response、効果量推定中心 |
| preflightで確定する実値 | データ配布版と利用条件、file/model hash、実fps、適格人数、実split、採用H、必要時の最終alpha grid、実support、dry-run結果 |

以下の設計値を outer-test の結果で選び直してはならない。preflight で物理的・測定的に成立しない場合のみ、outer-test を見る前に破綻証拠と改訂理由を記録し protocol version を上げる。

## 2. 研究の問いと実施範囲

主たる問いは、①どの部位対に追加予測情報があるか、②どの時間帯に分布するか、③被験者間で符号と大きさがどの程度共通するか、の三つとする。

| 必須解析 | 評価する内容 | 主な出力 |
|---|---|---|
| 全候補Landscape | Self履歴を超える追加予測情報の分布 | 全cellのG、部位対×lag帯域の集団要約 |
| 被験者間の共通性 | 集団の典型値、改善方向の共通性、個人差 | 被験者別G、G>0割合、人数・group数 |
| 4モデル比較 | 持続予測、Self、Full、PCMCI-blockの誤差と入力規模 | 被験者別誤差、対応差、入力数 |
| Selection enrichment | 選択集合が同じ入力規模の集合より高いcell-Gを含むか | 1,000ランダム集合、集合平均、Enrichment |
| 選択lag中心の応答 | 選択lagから時間位置をずらしたときの誤差変化 | 絶対lag曲線、中心化曲線、対象集合 |
| 発見安定性 | 学習する集団の構成による選択の変動 | 100回の再探索、選択頻度と分母 |
| データ・実行監査 | 測定品質、評価可能範囲、計算と再生成の正しさ | 品質記録、support、失敗台帳、結果manifest |

本実験の完了範囲は上表とする。効果が正であること、有意差、予測性能の最高値、Fullとの同等性、一定の疎性は完了条件にしない。

Random-region、Time-shuffle、joint-set matched-sparsity再学習、phase/circular surrogate、GPDC、LPCMCI、GRU、h>1、GNN、速度・加速度・peak/onset等のsecondary metricsはPrimary completion blockerとしない。

対象とする関係は、観測された顔運動の遅延予測依存および条件付き依存である。PCMCI+の因果解釈には仮定が必要であり、矢印から直接的な筋肉間因果、lagから生理的伝達時間を結論しない。

## 3. 記号と評価単位

| 記号・用語 | 定義 |
|---|---|
| s、c、k | 被験者、発話条件、outer-fold |
| i、j、p | source領域、target領域、顔の測定点 |
| f | 採用データのフレームレート（frame/秒） |
| t、u | 予測起点tと予測対象時点u=t+1 |
| τ | target時点から遡るlag。入力時点はu−τ。τ=1は予測起点tの現在観測 |
| H、L | Selfの最大履歴長と他部位の最大lag。単位はframe |
| q、q_ref、d | 幾何学的正規化座標、個人の固定基準形状、変位d=q−q_ref |
| ブロック | 一つのsource領域の、あるlagにおける全採用点のx・y成分 |
| cell e=(i,j,τ) | Selfへ一つのsource領域×lagブロックを追加する比較単位。i≠j |
| Ω | すべての領域間cellからなる候補集合 |
| support U | モデル間で共通に用いる予測対象時点の集合。行ID・点集合・hashを持つ |
| 独立group | 同一人物、対話ペア、共有セッションでつながる記録の連結成分 |
| OOF結果 | 各被験者を学習から外したouter-foldで得た評価結果 |

被験者を集団要約の基本単位とする。記録、frame、cell、ランダム反復、foldを独立した被験者として数えない。groupは分割と再標本化で依存関係を保持する単位として用いる。

## 4. データと発話条件

### 4.1 対象データ

主データは NoXi の原映像・音声・注釈を含む**単一配布版を第一選択**とする。実際の利用許可、取得、配布版、hash は preflight で固定する。異なるコーパスや配布版を混合しない。

同一人物、対話ペア、共有sessionでつながる記録は同一 dependency group とする。CAVIARES は主集団実験の自動代替にはしない。

### 4.2 解析条件

| 条件 | 含める状態 |
|---|---|
| 本人のみ発話 | 本人が言語的に発話し、相手は発話していない区間 |
| 本人非発話 | 相手のみ発話する聞き手状態、および双方非発話。両状態のflagと時間は個別保存 |

重複発話、本人の笑い・咳等の非言語音、不明区間・移行区間は学習と評価から除く。発話境界の前後200 msを除外し、最も古い入力からtargetまで同じ解析条件が続く行だけを使う。条件は本人音声に基づく注釈で定め、口運動から選ばない。

## 5. 予備解析、人数、データ分割

共通seedは `20260911` とし、用途・fold・条件・target・反復を名前空間で分離する。

予測結果を見る前に独立groupをseedで並べ、合計5人以上に初めて達する最小group集合を予備解析へ確保し、そのgroupの関連記録を本実験から除く。

被験者・条件の適格性は、校正・QC・履歴・境界の除外後の共通supportが合計60秒以上。outer-testではfold固有mask適用後30秒以上を評価下限とする。

### 5.1 Outer / Inner

- outer: dependency group単位の5-fold × 1
- inner: outer-train内group単位の3-fold
- 各被験者はouter-testへ1回だけ寄与
- 条件間で同じassignment

5-foldの成立条件を満たさない場合のみouter 3-foldを1回試す。3-foldでも成立しない場合はその条件の集団実験を開始しない。LOSO・frame-wise splitへの自動変更は禁止する。

## 6. 測定点、座標、校正

主表現は各点の2D正規化変位 `d(p,t)=q(p,t)-q_ref(p)`。region centroid、PCA、velocity targetへ置換しない。

Primary extractorは MediaPipe Face Landmarker IMAGE mode とし、`num_faces=1`、detection/presence threshold 0.5、BlendShapes off、pose確認用matrix onとする。exact package versionとmodel hashはpreflightで固定する。

採用region / point mapping:

| 領域 | index |
|---|---|
| 左眉 | 336, 296, 300 |
| 右眉 | 107, 66, 70 |
| 左眼瞼 | 385, 386, 380, 374 |
| 右眼瞼 | 158, 159, 153, 145 |
| 左頬 | 425, 280 |
| 右頬 | 205, 50 |
| 口 | 61, 291, 13, 14, 37, 267, 84, 314 |
| 顎 | 176, 152, 400 |

計29点・58成分。実modelへのoverlayでmapping成立を確認する。

眼角 33/133 と 362/263 を基準に translation / scale / roll を補正する。参照距離20 pixel未満は無効。品質適格な連続10秒を個人校正区間とし、qの時間中央値をq_refとする。校正区間は学習・評価へ含めない。

## 7. 時刻・QC

- 原fps維持
- resampling / interpolation禁止
- expected interval から5%超のgap、missing、duplicate、cut、条件変更はsequence境界
- 非有限値、画像外、顔未検出、明瞭な遮蔽は無効
- |yaw| または |pitch| > 20° は無効
- 低運動や大振幅だけで除外しない
- outer-train被験者の80%以上で利用可能かつx/y分散がともに1e-12超の点を使用可能とする
- region内に使用不能点があればregion block全体を使用不能
- unavailableをG=0にしない

## 8. Lag と Self history

`u=t+1`, `h=1`, source=`u-τ`, `τ>=1`。

inter-region最大lagは

`L = floor(0.5 f)`

とし、1..Lの全整数を候補とする。旧 `tau_max=10` 固定は superseded。

lag bands:

- B1=(0,100] ms
- B2=(100,250] ms
- B3=(250,500] ms

Self history H は予備被験者で0.5 / 1 / 2秒を比較し、最良誤差の1.01倍以内となる最短履歴を採用する。2秒が1秒より1%以上良い場合のみ4秒を追加し、4秒でも改善が継続する場合は自動延長せず設計レビューへ戻す。採用Hは全fold・条件で固定する。

## 9. PCMCI+ と ParentSet projection

Primary discovery:

- PCMCI+ + ParCorr
- `pc_alpha=0.01`
- `significance=analytic`
- `mask_type=xyz`
- `recycle_residuals=False`
- `tau_min=0`, `tau_max=L`
- majority collider rule
- `conflict_resolution=True`
- `reset_lagged_links=False`
- `max_combinations=1`
- `fdr_method=none`

Discovery nodeはlandmark-coordinate scalar。

region block selectionは **OR projection** とする。source regionのいずれかのscalarからtarget regionのいずれかのscalarへlagged directed linkが1本以上あれば `(i,j,τ)` blockをselectedとし、forecastingではsource region blockの全使用可能scalarを追加する。

## 10. Ridge と common support

Ridgeはinterceptあり、targetは標準化せず、inputのみtrainで標準化する。被験者ごとの総学習重みを1、各rowを1/n_sとする。

基本alpha gridは `1e-4 ... 1e4` の9候補。inner-validationの被験者別主誤差等重み平均で選択し、数値同点のみ大きいalphaを選ぶ。予備でgrid端の改善が継続する場合のみ事前規則で拡張し、outer-test後の変更は禁止。

supportはcondition×fold×subjectごとに一つに統一し、Full / 4条件 / 全cell / Enrichment / centered responseで同一行・点集合を使う。support hashを保存する。

## 11. 主誤差と Landscape

`E_m(s,c,j)=mean_u mean_p ||d_hat_m(p,u)-d(p,u)||_2`

`G_s,c(i,j,τ)=E_Self(s,c,j)-E_cell(s,c,i,j,τ)`

別supportのEを引かない。帯域内Gは被験者内でlag等重み平均した後、被験者間中央値を取る。

## 12. Selection Enrichment

selected block集合をscalar成分数で層別し、同じ各層個数のrandom setを候補空間から非復元抽出する。

- 1000 repeats
- set内重複禁止
- selected setとの重複は許可
- repeat間同一setは許可
- lag分布はmatchingしない
- outer-trainで生成・freeze

`Enrichment = mean(G_selected) - mean_r mean(G_random_r)`

これはcell-G集合平均のestimandであり、joint-set Ridge gainとは別。empty selectionはunevaluable。

## 13. Centered lag response

`E(e,τ)` は Self + 当該source-region×lag block。

中心化幅は約±100 ms:

`M=floor(0.1f)`, `Δ=-M,...,0,...,+M`

complete symmetric gridを必須とし、clip / wrap / one-sided gridは禁止する。

`D_s,e(Δ)=E_s(e,τ*+Δ)-E_s(e,τ*)`

同一被験者・同一source→targetの複数lagを平均 → 被験者内で部位対平均 → 被験者間中央値、の順で集約する。

## 14. 統計と不確かさ

Primaryは効果量と分布の推定を目的とし、held-out significance test / FDR / non-inferiority marginをPrimary完了条件にしない。

OOF結果についてouter-fold内で独立groupを復元抽出し、10000 resamples、2.5/97.5 percentile、linear interpolationの名目95% intervalを計算する。groupが10未満ならCIはunevaluableとする。

## 15. Discovery stability

各outer-train・条件で独立groupを復元抽出し100回再探索する。series内時間順序を保持し、group copyを連結しない。scalar link / region×lag block / region-pair×bandの選択頻度と分母を保存する。stability頻度でPrimary selectionを再選択しない。

## 16. 失敗・受容

空selectionではPCMCI-blockはSelfと同一、Enrichment / centered responseはunevaluable。短区間・QC不適格・数値退化等は0埋めせず理由付きunevaluable/failedとする。一時I/O失敗は同一設定・seedで最大2回再試行し、3回目も失敗ならfailedを残す。

結果が仮説を支持しないことと実験失敗を区別する。

## 17. 必須保存物

データ利用条件、subject/group/session/sequence、timestamp/fps、extractor/model hash、point mapping、QC、calibration、q_ref、mask、split、Ω、H、scaler/weights/alpha、PCMCI raw outputs、block mapping、predictions、ground truth、support hash、G、random sets、centered curves、bootstrap indices、seed registry、failure ledger、environment、git SHA、config hashを保存する。

## 18. 本実験開始前の preflight

残るのは次の実物確認である。

- NoXi配布版・利用条件・file hash
- subject/group/session inventory
- extractor exact version / model hash
- landmark overlay / topology / pose axis / tracking確認
- actual fps / cadence
- actual L / lag-band membership
- adopted H
- final alpha grid
- deterministic split
- condition別common support
- dry run / leakage / reproducibility audit
- hash付きexecutable freeze

これらが成立しない場合はouter-testを見る前に改訂理由を登録する。

## 19. Executable migration

本仕様を `configs/scientific_freeze.yaml`、`configs/primary_run.yaml`、schema、config loader、support/landscape/enrichment/centered-response implementation、testsへ移行する。

旧v6 artifactを新Primaryの完了証拠として受理しない。文書更新だけでmigration completeとしない。
