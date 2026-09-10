> 履歴資料：2026-09-10に独立サブシステムの導入を撤回。以下の導入判断・必須要件は現行ではありません。[価値再評価](methodological_validation_value_decision.md)と[正本](experimental_plan.md)を参照してください。

# Methodological validation：第2次検討

2026-09-09。[第1次監査](methodological_validation_audit.md)と[導入仕様](methodological_validation_subsystem.md)の再検討。今回は数学的反例、現行コードの追加照合、一次論文の確認を行った。数値例は式をJavaScriptで評価したもので、PCMCIや実研究pipelineの実行結果ではない。

## 1. 判断の更新

導入価値は高い。ただし、研究の中心を「3D合成顔からの因果グラフ完全回収」へ移さない。**新研究の評価計算を既知例で検証すること**と、**raw landmarkから研究変数までの観測過程を検証すること**を別の責務として完成させる。

前回仕様の修正点は次のとおり。

|前回の扱い|今回の修正|
|---|---|
|速度の積分をgeometry生成の第一候補とする|固定参照・可観測性・有限時間での成立を証明できるときの候補に限定|
|cleanの小さい誤差を根拠に構造GTを使用|構造を保存する写像の条件と数値一致の両方を要求。近似一致だけなら近似観測の性能評価|
|全nodeを独立に表現できなければbasisを改善|実データ表現そのものが正規化で低rankなら、basisでは解消できないと判定|
|各robustness条件にも全研究解析を繰り返す余地|完全な新研究E2Eは事前指定の基準条件で実施。noise/head全水準への高価な展開は別に必要性を判断|

これは原案の価値を下げる変更ではなく、検証できる主張を明確にする変更である。

## 2. 相関が高くてもグラフは保存されない

独立な2系列について、`g_t=A g_(t-1)+ε_t`、`A=diag(0.8,0.2)`、`Cov(ε)=I`とする。元のcross-variable edgeは空である。

観測に10%の混合 `y=H g`、`H=[[1,0.1],[0,1]]` を加えると、Hは可逆であるにもかかわらず、

```text
A_observed = H A H^(-1) = [[0.8, -0.06], [0, 0.2]]
Cov(observed innovation) = H H^T = [[1.01, 0.1], [0.1, 1]]
```

となる。観測上の条件付き平均にcross-lag係数が現れ、innovationも同時相関する。これはPCMCIが必ずそのedgeを出すという予測ではなく、「元のゼロパターンが保存される」という主張への代数的反例である。

一般の可逆混合でも `A_l→H A_l H^(-1)` となり、元のedge集合は保たれない。固定した非ゼロ対角スケールと変数の並べ替えなら、この線形独立innovation設定ではゼロパターンを対応付けられる。ただしnode identityと単位への逆対応を明記する必要がある。

**設計上の帰結**：cross-talkの小ささ、相関0.99、低RMSEは信号品質の指標であり、グラフ保存の証明ではない。近似混合の下では「latentに対する回収性能」を測れるが、その誤差をdiscovery実装の誤りと断定しない。

## 3. 正規化が独立nodeを作れなくする場合

現行`normalize_translation`は指定参照点の重心を全点から引く。

例えば、解析regionが参照集合全体を重複なく分割し、region点数をn_r、正規化後の重心をz_rとすると、`Σ_r n_r z_r=0` が各frame・軸で厳密に成り立つ。この条件下では速度にも同じ線形制約がある。K regionの同一軸は最大K−1自由度となり、K個の独立innovationを持つ任意VARをそのまま埋め込むことはできない。

これは現行の参照点が実際にこの集合であるという断定ではない。D02が未確定なため、条件付きの設計警告である。固定参照を分析対象外に確保できれば、この特定の制約を回避できる可能性がある。

**必要な先行確認**：領域index、参照index、集約重みから解析的な制約を調べる。低rankをsyntheticだけのPCAやvariable削除で解消しない。実データの変数定義に問題があれば研究仕様として判断する。顔の自由度とPCMCI変数数を一致させるためにsemantic basisを増やすだけでは不十分である。

## 4. 安定な速度と有界な顔位置は別条件

定常AR(1)速度gの係数をa、定常分散をσ_g²とし、一定dtでNステップ積分すると、

```math
\operatorname{Var}(p_N-p_0)
=dt^2\sigma_g^2\left[N+2\sum_{k=1}^{N-1}(N-k)a^k\right].
```

|例示用条件|値|
|---|---|
|a|0.8|
|dt|1/30秒|
|N|1,000|
|速度の定常SD|0.1 座標単位/秒|
|積分位置変化のSD|約0.3155 座標単位|

これらは採用パラメータではない。安定速度の長時間積分で位置分散が増えることの例示である。非退化Gaussian自体にも無限の裾があり、全生成列について有限の顔形状上限を絶対保証できない。

位置をclip、終点へ戻す、列全体の平均速度を引く、上限内に入るseedだけ採ると生成分布が変わる。安定化springを入れる方法も速度VARと別のモデルになる。系列を短く分割し全区間を採用する設計は可能だが、任意長の連続顔運動を再現したことにはならない。リセットの前後に差分・lagを作らない。

したがって、積分生成は**有限時間の実装試験**として限定する。位置の有界性と定常速度を同時に重視する自然運動生成は別モデルにし、そのlatent graphを速度の正解に転用しない。

## 5. 位置の白色noiseが速度では色付きになる

位置noise η_tが独立、分散σ_η²であっても、差分後のnoiseは `ν_t=(η_t−η_(t−1))/dt` となる。

```text
Var(ν_t) = 2 σ_η² / dt²
Cov(ν_t,ν_(t−1)) = −σ_η² / dt²
Corr(ν_t,ν_(t−1)) = −1/2
```

dt=1/30秒、位置noise SD=0.001座標単位なら速度noise SDは約0.04243座標単位/秒である。独立jitterを追加したつもりでも、discovery入力にはlag 1のnoise相関が入る。集約でnoiseが低下する程度は、landmark間の独立性に依存する。共通参照のnoiseは部位間にも伝播する。

**導入する検査**：noiseの入力SDだけでなく、前処理後の分散と自己・相互相関を保存する。noiseがある条件で独立innovation VARを完全回収できないことを、直ちにバグにしない。測定誤差を明示的に扱う必要性は[測定誤差下の因果探索研究](https://proceedings.mlr.press/v124/saeed20a.html)とも整合するが、同論文の補正法を本研究に自動導入するものではない。

## 6. timestampを保存するだけでは物理lagが一致しない

現行`validate_timestamps`は正の不等間隔を許可する。`FaceTimeSeries`も単調増加を確認するが、`sampling_rate`との間隔一致を要求しない。adapterはtimestampを渡す一方、design matrixは配列indexのlagで入力を取る。

例として時刻0、1、2、4の列を連続4行にすれば、最後のlag 1は2秒であり、他のlag 1は1秒になる。metadataのsampling_rateを1と書いても直らない。

**修正**：基準benchmarkは等間隔・dt明示とし、欠落frameは所定gridのinvalid行として保持する。現実の不等間隔は別に検出し、許容範囲・除外・変換方針をD02/D03で固定する。一般raw validatorが不等間隔を受け付けること自体をバグとはしないが、discovery直前の等間隔契約は要検証。frame→msを名目fpsで換算できる条件を明記する。

高頻度の生成系列を間引く場合にも元のgraphを自動継承しない。VAR(1)のk間引きでは遷移がA^kになり、noise共分散も変わる。サブサンプリングで因果構造の同定が難しくなる点は[原著研究](https://proceedings.mlr.press/v37/gongb15.html)でも扱われている。

## 7. 本研究に必要なのは予測利得のoracleも持つこと

構造GTだけでは、landscape/enrichmentの計算が正しいかを検証しきれない。線形Gaussian基準例では、Self履歴S、追加scalar Z、target Yを使い、最適線形予測のMSE改善を独立に計算できる。

Sで線形残差化したY、ZをY⊥、Z⊥とすると、Var(Z⊥)>0のもとで、

```math
\Delta\operatorname{MSE}_{\mathrm{oracle}}
=\frac{\operatorname{Cov}(Y^\perp,Z^\perp)^2}
{\operatorname{Var}(Z^\perp)}.
```

複数特徴では対応する条件付き共分散行列を使う。定常VARの共分散から計算するか、独立した十分大きな参照生成標本を使いMonte Carlo誤差を示す。後者を厳密な真値と呼ばない。

これは母集団・非正則化最適線形予測の診断である。有限trainでtuneしたRidgeのheld-out Gと一致する義務はない。RMSE差なら `sqrt(MSE_Self)−sqrt(MSE_augmented)` であり、上式のMSE差をそのままRMSE差として採点しない。GTを使うoracleは評価専用に隔離する。

簡単な反例：独立な白色Xに対して、`M_t=bX_(t−1)+ε_M,t`、`Y_t=cM_(t−1)+ε_Y,t`なら、X→Yの直接edgeはないがX_(t−2)はY_tの予測情報を持つ。したがって非直接edgeのGをすべて0にするテストは誤りである。

## 8. 全候補ΩをParentSetから作らない

新たな具体的再利用上の注意がある。`null_matched_sparsity.py::build_matched_sparsity_candidate_space`は、ParentSetに現れたtarget dimensionだけを使い、ParentSetが空なら空候補を返す。旧matched controlの実装としては意味があるが、新研究の全候補landscape生成器にはならない。

**必須の回帰条件**：同じprotocolでParentSetを空にしても、landscape Ωのcell数と順序が変わらないこと。変わるのはselected flag、matched抽出数、enrichmentの評価可能性であり、全候補Gの計算範囲ではない。discoveryが全く選ばなかった場合こそ、予測情報が本当に乏しいのか、選択が取り逃したのかを区別する価値がある。

## 9. 1,000反復は有意性も計算量も自動決定しない

matched集合の反復分布は、定義したrandom selection基準との比較である。PCMCI集合がそのrandom抽出則と交換可能であるという根拠がなければ、順位をそのまま校正済み帰無検定p値とは呼べない。被験者の不確実性とrandom集合のMonte Carlo変動も別である。D06/D08で順位・差・CI・検定の役割を決める。

一方、enrichmentが固定済みcell Gの集合内集約なら、全cellを一度評価した後に1,000集合をindexで集約できる。集合ごとにRidgeを再fitする必要はない。各集合を一括入力するNullモデル比較には別途fitが必要で、両者は別estimandである。

したがって、**1,000反復を減らさず計算を抑える余地がある**。同一protocol・fold・subject・target・support・metric・tuningのhashが一致するcellだけ再利用する。supportが違えばSelf誤差を使い回さない。

## 10. どこまで共通化し、どこを独立に検証するか

実データとsyntheticの処理経路を共通化するのは適切だが、生成器と評価器の両方が同じ誤ったlag変換を呼べば、共通バグのままPASSする。

以下の少数例には、共通helperから独立に手計算した期待値を持たせる。

- h=1、target index=10、τ=3ならsource index=7、origin index=9。
- 非対称な単一edgeでsource/targetの転置が検出される。
- 既知の等速位置から、単位を含む期待速度と端点maskが得られる。
- GTあり推定空、GT空推定あり、両者空でmetric定義が区別される。
- ParentSetが空でもΩが保持される。

逆写像に合わせて作ったgeometryは変換の実装確認に向くが、実際の表情の妥当性を独立に証明しない。独立形状・参照点近傍変形・観測noiseを用いたstress試験を併記し、検証の射程を明示する。

## 11. 必須範囲を絞った実行案

|区分|最初に実施する内容|目的|
|---|---|---|
|決定論的検査|上記の少数手計算例、時刻・mask・境界・freeze・Ω|実装の正しさ|
|直接node基準|cross-edgeなし、明瞭な単一edge、媒介chain、自己相関を持つ複数lag|構造と予測情報の違いを含む基準性能|
|Geometry単位検査|固定参照条件での有限時間信号一致、rank検査|raw→変数の契約|
|新研究E2E|事前指定した基準条件で4条件、全G、1,000反復、population応答|本研究に必要な全出力の接続|
|観測stress|headのみ、jitter、参照点欠損、面外回転をまず別々に付加|偽構造・信号減衰・適用限界|

基準条件と反復数は実行前に固定し、結果を見て簡単な条件へ置換しない。全stress水準への1,000回集合一括Ridgeと全統計の反復を初期必須にしない。合成データの臨床的・生理学的リアリズムは別研究課題である。

形態差はbaselineから分離する。同じlatent graphでもsubjectごとの観測写像H_sにより、観測係数や分散が異なり得る。sequenceを分けてTigramiteへ渡すことは、pooled CIの分布同質性の保証ではない。複数datasetのcontext問題は[J-PCMCI+の原著](https://proceedings.mlr.press/v216/gunther23a.html)で扱われているが、ここでPrimary手法を変更する必要はない。まず同質baselineと形態stressを分けて限界を測る。

## 12. 次に確定する順序

最優先はD02〜D04のうち、region/anchor/重心集約/速度・変位/単位/等間隔/成分対応を確定することである。これが定まらなければ、写像のrankも速度積分の成立も判定できない。次に全候補Ωとenrichment/populationの演算を確定し、最後に必要なbenchmark反復数と性能受容規則を選ぶ。

方法検証の導入は継続する。ただし現段階で約束する成果は「現実の顔に近い合成系でグラフを回収する」ではなく、「定義された入力表現・観測条件・評価計算について、正しさと失敗する範囲を説明できる」である。

