# Authoritative Primary Experiment Specification

更新日: 2026-09-12  
状態: **AUTHORITATIVE / ADOPTED**  
対象 branch: `dev` への移行基準

## 0. 本文書の効力

2026-09-12 のユーザー決定により、`docs/primary_experiment_recommendation_2026-09-11.md` に記載された本実験の推薦設計を、**参考案ではなく本研究の正式仕様として全面採用する**。

同推薦文書中の「推薦」「候補」「本案」「今回新たに推薦する」という表現は、科学設計上は本日以降 **ADOPTED** と読み替える。ただし、実データを見なければ取得できない事実（配布版、file hash、実 fps、model hash、実際の fold 割当て、実 support 数等）を未確認のまま創作してはならない。

競合時の優先順位は次の通り。

1. 本文書
2. `docs/primary_experiment_recommendation_2026-09-11.md` の全設計規則
3. 本仕様へ同期済みの `docs/experiment_design.md` / config / schema / Issue
4. 2026-09-10 までの採用記録のうち本仕様と矛盾しない部分
5. 旧 experimental plan / detailed design / Scientific Freeze v6 / 旧 Issue

旧文書・旧 Issue・旧 config が本仕様と矛盾する場合、旧記述は `SUPERSEDED` とする。

---

# 1. 研究目的と Primary の範囲

主目的は、**自然な表情・発話中の顔運動における部位間の遅延予測構造と、その被験者間の共通性・一般化可能性を定量化すること**である。

主たる研究単位は、全候補 `source region × target region × lag` に対する追加予測価値であり、PCMCI+ は全候補空間の中から選択された構造の価値・安定性を評価するために用いる。

必須 Primary 解析は以下とする。

1. 全候補 Predictive Gain Landscape
2. Persistence / Self / Full / PCMCI-block の4条件比較
3. Selection Enrichment
4. 選択 lag 中心の response
5. Discovery stability
6. データ・実行監査

研究完了条件を、統計的有意差、Full との同等性、SOTA、一定の疎性、正の効果の存在には置かない。否定的結果も有効な科学的結果として扱う。

## Primary から外すもの

以下は本実験完了の blocker としない。

- Random-region
- Time-shuffle
- joint-set matched-sparsity Ridge の 1,000 再学習
- circular shift
- phase-shuffled surrogate
- GPDC
- LPCMCI
- GRU
- GNN / Temporal GNN
- h > 1
- 別データセットでの再現
- 本人専用モデル
- 動的グラフ
- AU / BlendShapes 比較
- velocity / acceleration / peak / onset を Primary 必須指標とすること

これらを行う場合は Primary Freeze 後の Sensitivity / Optional analysis とする。

---

# 2. データと評価条件

## 2.1 主データ

主データの第一選択は **NoXi の原映像・音声・注釈を含む単一配布版** とする。

- NoXi-J 等との自動統合は禁止
- 異なるコーパスの混合は禁止
- 被験者数は実配布版の実 ID から数える
- 利用不能の場合、CAVIARES へ自動代替しない
- CAVIARES は必要に応じて I/O / 計算経路の予備確認には利用できるが、被験者間一般化の主データにはしない

## 2.2 条件

解析条件は次の2条件を登録し、条件内で独立に discovery / forecasting を実行する。

- 本人のみ発話
- 本人非発話

重複発話、本人の笑い・咳等の非言語音、不明・移行区間は学習・評価から除外する。条件境界前後 200 ms は除外し、入力履歴から target まで同一条件である行のみ使用する。

---

# 3. 分割・予備解析・seed

共通 seed は `20260911` とし、用途・fold・条件・target・反復を名前空間で分離し、既存 SeedRegistry の SHA-256 方式で派生する。

## 3.1 予備被験者

予測結果を見る前に独立 group を seed 順に並べ、合計5人以上に初めて到達する最小 group 集合を予備解析用に確保し、その関連記録を本実験から除外する。

予備解析は次のみに使用する。

- QC
- landmark / topology 確認
- 校正
- Self history 決定
- Ridge alpha grid 確認
- 計算量確認
- dry run

G の大きさ、有意性、都合のよい部位・lag の選別には使用しない。

## 3.2 Outer / Inner

- Outer: dependency group 単位の 5-fold × 1
- Inner: outer-train 内 group 単位の 3-fold
- 各被験者は outer-test に1回のみ寄与
- 条件間で同じ group assignment を使用

5-fold が、各 test に2以上の適格 group、かつ各 inner-validation に2以上の適格 group を配置できない場合のみ、同じ要件で outer 3-fold を1回試す。

3-fold でも成立しない場合、その条件の集団実験は開始不可とする。

**LOSO・frame-wise split へ自動変更してはならない。**

---

# 4. 顔表現・測定

## 4.1 主表現

主表現は点ごとの 2D 正規化変位

`d(p,t) = q(p,t) - q_ref(p)`

とする。

- x / y をそのまま入力・予測する
- region centroid へ圧縮しない
- PCA を Primary に用いない
- velocity を主 target にしない
- z を実測 3D とみなして主入力へ追加しない

## 4.2 抽出器

MediaPipe Face Landmarker の IMAGE mode を Primary 測定器として採用する。

- `num_faces = 1`
- detection threshold = 0.5
- presence threshold = 0.5
- BlendShapes output = off
- pose 確認用 transformation matrix = on
- temporal tracking / 追加時間平滑化を使わない

実行前に exact package version と model asset hash を固定する。

## 4.3 8領域・29点・58成分

| 領域 | MediaPipe index |
|---|---|
| 左眉 | 336, 296, 300 |
| 右眉 | 107, 66, 70 |
| 左眼瞼 | 385, 386, 380, 374 |
| 右眼瞼 | 158, 159, 153, 145 |
| 左頬 | 425, 280 |
| 右頬 | 205, 50 |
| 口 | 61, 291, 13, 14, 37, 267, 84, 314 |
| 顎 | 176, 152, 400 |

計29点・58 scalar 成分とする。

この mapping は設計として採用するが、採用 model の実出力に対する overlay で topology / 左右 / tracking 妥当性を preflight 確認する。破綻した場合は **outer-testを見る前に**改訂理由を登録して新 protocol version とする。

## 4.4 正規化・校正

参照眼角を右側 `33, 133`、左側 `362, 263` とする。眼角中点間の移動・スケール・roll を補正し、

`q(p,t)=R_t[pixel(p,t)-c_t]/s_t`

を用いる。

- 参照距離 20 pixel 未満は無効
- 品質適格な連続10秒を個人校正区間とする
- 各採用点の時間中央値を `q_ref` とする
- 校正区間は学習・評価へ使用しない
- q_ref を test 全体から再推定しない

---

# 5. QC・時間規則

- 原 fps を維持する
- resampling / interpolation は行わない
- expected interval `1/f` から 5% を超える gap、missing、duplicate timestamp、cut、condition change は連続区間境界
- 非有限値、画像外、顔未検出、明瞭な遮蔽は無効
- yaw / pitch の絶対値が20度を超える frame は除外
- 大振幅・速度のみを理由に外れ値除去しない
- 低運動のみを理由に被験者除外しない
- outer-train で80%以上の被験者に有効データがあり、被験者等重み平均分散が x/y とも `> 1e-12` の点を使用可能とする
- region 内の採用点が1点でも使用不能なら、その fold では region block 全体を使用不能とする
- unavailable を `G=0` に置換しない

---

# 6. Lag・Self history

## 6.1 Forecast semantics

- horizon: `h = 1`
- target: `u = t + 1`
- input: `u - tau`
- `tau = 1` は forecast origin `t` の現在観測

## 6.2 Inter-region lag range

最大 lag は

`L = floor(0.5 * f)`

とし、`tau = 1, ..., L` の全整数を候補とする。

旧固定値 `tau_max = 10` は本仕様では superseded とする。

lag band は次の3帯域。

- B1 = `(0, 100] ms`
- B2 = `(100, 250] ms`
- B3 = `(250, 500] ms`

実 fps から各 frame lag を ms へ変換し、帯域へ割り当てる。

## 6.3 Self history

候補を 0.5 / 1 / 2 秒とし、予備被験者のみで 60% train / 20% alpha selection / 20% history comparison を行う。

最良誤差の 1.01 倍以内となる最短履歴を採用する。

2秒が1秒より1%以上良い場合のみ4秒を追加する。4秒が2秒よりさらに1%以上良い場合、自動拡張せず設計レビューへ戻す。

採用 H は全 fold / 全条件 / 全追加モデルで共通とする。

---

# 7. PCMCI+ discovery と region-block projection

Primary discovery は以下で固定する。

- Tigramite PCMCI+
- ParCorr
- `pc_alpha = 0.01`
- `significance = analytic`
- `mask_type = xyz`
- `recycle_residuals = False`
- `tau_min = 0`
- `tau_max = L`
- contemporaneous relation は raw discovery には保存するが forecasting input には使わない
- collider rule = majority
- `conflict_resolution = True`
- `reset_lagged_links = False`
- `max_combinations = 1`
- max condition counts = None
- `fdr_method = none`

Discovery node は landmark-coordinate scalar とする。

**Scalar → region block の Primary projection は OR projection を正式採用する。**

source region `i` のいずれかの scalar から target region `j` のいずれかの scalar へ、lag `tau` の directed link が1本以上ある場合、`(i,j,tau)` block を selected とする。

Forecasting では、その source region block の使用可能な全 scalar 成分を明示的に追加する。

raw scalar links と region-block projection の両方を保存する。

---

# 8. Ridge forecasting

共通予測器は Ridge とする。

- intercept あり
- target は標準化しない
- input のみ train で標準化
- multi-output で同じ alpha を使用
- 被験者ごとの総学習重みを1にする
- 被験者 s の各行の重みを `1 / n_s` とする
- scaler の平均・分散と Ridge loss の双方に被験者等重みを反映する

alpha grid は

`1e-4, 1e-3, ..., 1e4`

の9候補を基本とする。

inner-validation の被験者別主誤差の等重み平均で選択する。数値差 `<= 1e-12` のみ同点とし、大きい alpha を選ぶ。

予備段階で grid 端が隣接候補より1%以上良い場合のみ、その方向へ10倍刻みで全モデル共通拡張し、各側最大3段階とする。outer-test を見て grid を拡張してはならない。

---

# 9. 比較条件と共通 support

比較モデルは以下とする。

- Persistence
- Self
- Full
- PCMCI-block
- cell model = Self + 単一 source-region × lag block

support は **condition × fold × subject ごとに一つの共通 support** とし、Full / 4条件 / 全cell / Enrichment / centered lag response で同じ target 行と点集合を使用する。

count の一致だけでなく exact support ID / hash を保存して照合する。

---

# 10. 主誤差・Predictive Gain Landscape

主誤差は、target region の各点について x/y の Euclidean error を先に計算し、点と時刻を平均する。

`E_m(s,c,j) = mean_u mean_p ||d_hat_m(p,u)-d(p,u)||_2`

cell gain は

`G_s,c(i,j,tau) = E_Self(s,c,j) - E_cell(s,c,i,j,tau)`

とする。

- `G > 0` は Self より改善
- 直接因果効果とは解釈しない
- 別 support 間の error を引かない

帯域集約は、被験者内で帯域に含まれる全 lag の G を等重み平均し、その後に被験者間中央値を取る。

主報告には中央値、平均、`G>0` 被験者割合、評価人数、独立 group 数、名目95%区間、評価可能点集合を含める。

---

# 11. Selection Enrichment

PCMCI selected block 集合を `S_j` とする。

Random set は source block の実 scalar 成分数で層別し、selected set と同じ各層個数を候補空間から非復元抽出する。

- repeat count = `1000`
- 同一 set 内重複禁止
- selected set との重なりは許可
- repeat 間の同一集合は許可
- lag 分布は matching しない
- random set は outer-train で生成・freeze
- outer-test の G で再抽出しない

定義:

`A_selected(s,j) = mean_{e in S_j} G_s(e)`

`A_random(s,j,r) = mean_{e in A_j,r} G_s(e)`

`Enrichment(s,j) = A_selected(s,j) - mean_r A_random(s,j,r)`

これは **cell-G 集合平均の enrichment** であり、joint-set Ridge gain とは別 estimand とする。

empty selected set は `unevaluable` とし0にしない。

---

# 12. 選択 lag 中心 response

`E(e,tau)` は `Self + 当該 source-region × lag block` の誤差とし、他 selected block は同時投入しない。

中心化幅は正式に **約 ±100 ms** とする。

`M = floor(0.1 * f)`

`Delta = -M, ..., 0, ..., +M`

- 1 frame 刻み
- complete symmetric grid 必須
- clip 禁止
- wrap 禁止
- one-sided grid 禁止
- 全 Delta で同じ edge / subject / support を使用

`D_s,e(Delta) = E_s(e,tau*+Delta) - E_s(e,tau*)`

集約順序は

1. 同一被験者・同一 source→target の複数 selected lag を算術平均
2. 被験者内で評価可能な部位対を算術平均
3. 被験者間中央値

とする。

`Delta=0` は定義上0であり、効果の証拠とはしない。

---

# 13. 統計的立場・区間

本実験は **効果量と分布の推定を主目的** とする。

- held-out significance test を Primary にしない
- FDR 判定を行わない
- non-inferiority margin を置かない
- p-value を研究完了条件にしない

OOF の frozen subject results に対し、outer-fold 内で独立 group を復元抽出し、同一 group に属する被験者・cell・curve をまとめて resample する。

- resamples = `10000`
- nominal interval = 95%
- percentile = 2.5 / 97.5%
- quantile interpolation = linear
- 全図表で同じ bootstrap index を共有

この区間は「固定 OOF 結果に対する被験者構成の変動を示す名目95% bootstrap区間」と表現する。

評価可能 group が10未満の要約では CI を評価不能とし、点推定と個別値のみ報告する。

---

# 14. Discovery stability

各 outer-train・条件で独立 group を復元抽出し、**100回** PCMCI+ を再探索する。

- series 内時間順序保持
- group が複数回選ばれた場合は独立 series copy として渡す
- copy を連結しない
- point mask / 等時間採用区間は original outer-train のものを固定
- scalar link / region×lag block / region-pair×band の頻度を保存
- stability 頻度で Primary ParentSet を再選択しない

成功・失敗・選択可能回数を別々に保存し、失敗を未選択0として扱わない。

---

# 15. 失敗・評価不能

- empty PCMCI selection: PCMCI-block prediction は Self と同一。Enrichment と centered response は unevaluable
- 短区間 / QC 不適格 / unusable region: unevaluable。0埋め禁止
- ParCorr の自由度不足・数値退化: failed / unevaluable を明示
- temporary I/O failure: 同一設定・seedで最大2回再試行。3回目も失敗なら failed を残す
- code / config / data を変更した場合、version/hash を更新し影響範囲を再計算
- 原因不明の失敗を残した解析は complete としない

---

# 16. 必須保存物

少なくとも以下を provenance 付きで保存する。

- dataset / usage terms reference
- subject / dependency group / session / sequence
- raw timestamps / fps
- extractor version / model hash
- point mapping
- QC / calibration interval / q_ref / mask
- split manifests
- candidate Omega
- Self history H
- scaler / sample weights / alpha
- PCMCI raw graph / p_matrix / val_matrix / scalar node mapping
- block projection mapping
- predictions / ground truth / timestamps / support hash
- G
- random sets / repeat values
- centered response
- bootstrap indices
- seed registry
- failure ledger
- environment / git SHA / config hash

---

# 17. 本実験開始前に残るもの

本仕様の科学設計は採用済みであり、以下は「未決の科学仕様」ではない。実物確認により値を materialize する preflight である。

1. NoXi の取得可能な配布版、利用条件、file hash
2. subject / group / session / sequence inventory
3. extractor exact version / model hash
4. overlay による 29-point mapping / 左右 / pose axis / tracking 品質確認
5. 実 fps / cadence 層
6. 実 fps から算出した `L` と band 内 lag
7. 予備解析で確定した Self history `H`
8. 必要なら予備解析で拡張した最終 alpha grid
9. deterministic outer / inner split realization
10. 条件別 common support 数・時間
11. dry run / leakage audit / reproduction evidence
12. hash 付き executable freeze

これらの preflight で本仕様の運用閾値が成立しない場合、**outer-test の結果を見る前に**破綻証拠と改訂理由を記録し、protocol version を上げて再登録する。

---

# 18. Repository migration rule

以下は本仕様への移行対象である。

- `README.md`
- `docs/experiment_design.md`
- `docs/protocol_authority_and_migration.md`
- `docs/adopted_decisions_2026-09-10.md` の後続追記
- `docs/experimental_plan.md`
- `docs/detailed_design.md`
- `configs/scientific_freeze.yaml`
- `configs/primary_run.yaml`
- `schemas/scientific_freeze.schema.json`
- scientific config loader / preflight validation
- GitHub Issue #24 / #25 / #28 / #31 / #33 / #34 / #35 / #36 / #37 / #38 および実行 gate Issues

旧 v6 artifact や旧仕様で生成された result は、新 protocol の Primary 完了証拠として受理しない。

**文書リンクの追加だけで migration 完了としない。** Config / schema / implementation / tests / dry run / freeze hash まで一致して初めて executable migration 完了とする。
