> 履歴資料：2026-09-10に独立サブシステムの導入を撤回。以下の導入判断・必須要件は現行ではありません。[価値再評価](methodological_validation_value_decision.md)と[正本](experimental_plan.md)を参照してください。

# Methodological Validation Subsystem：導入監査

監査日：2026-09-09。判定：**目的は採用。原案のままの実装・Stage 1 PASS宣言は不可。修正版を研究計画に導入する。**

## 対象と証拠の範囲

対象原案：`C:/Users/yukit/OneDrive/デスクトップ/研究/methodological validation subsystem.md`。SHA-256：`0457296e68f41fe20a7dd00d6481305c8b3cd11a1b2af3200948cb693412f248`。

照合対象は[現行研究計画](experimental_plan.md)、[詳細設計](detailed_design.md)、ローカル実装commit `9bbc3d946c0a64935b8786d9413abe53e6cefe29`。docsは監査開始時点でgit未追跡であり、上記commitの内容とは区別する。文書に記載された過去のユーザー指示は、今回の新たな実行指示として扱っていない。現行計画との整合性を評価対象にした。

今回は文書・ソース・既存テストの静的監査である。既存テストの存在を合格証拠にはしていない。ベンチマーク、実データrun、テスト再実行は未実施。Python/pyはPATH上で確認できず、uvの起動もアクセス拒否だった。コード・実行用freezeは変更していない。

## 継承すべき点

既知生成過程、同一前処理、subject分離、outer-test隔離、GTの評価専用利用、未来情報の監査、段階別artifact、安定VAR、事前固定benchmark、写実性を必須にしない方針は適切。合成検証を実際の顔の因果構造の証明としない主張範囲も維持する。

## 重要指摘

### MV-01／最優先：継承する科学仕様の版が違う

原案§2、31、36、50は既存freezeをそのまま継承し、Self対PCMCIとcommon-shiftを中心にしている。一方、現行研究計画§6、12、13は全候補landscape、1,000反復のenrichment、edge-centered population応答を要求する。`scientific_config.py`のv6は100反復・common-shiftである。

**修正**：新計画に整合する解決済みprotocolを継承する。旧v6はlegacy検証と明記する。原案§31の数値を新計画へ再固定しない。D01〜D10の未決値はsynthetic側の既定値で補完しない。4条件比較だけでは新研究のmethodological validation完了にならない。

### MV-02／最優先：位置に投影したVARを速度の正解グラフとして使えない

原案§4.3の `c_t=M g_t`、`V_t=V0+B c_t` はgを位置変形へ写す。現行`motion_features.py::compute_velocity`は位置差分を時間差で割る。正規化・集約を固定線形写像Hで近似しても、観測速度は `y_t=H(g_t-g_(t-1))/dt` でありgそのものではない。

H=I、dt一定の場合でも、`g_t=Σ A_l g_(t-l)+ε_t`から得られる差分系列は `y_t=Σ A_l y_(t-l)+(ε_t-ε_(t-1))/dt`。新しい残差は時間相関を持ち、元の独立innovation VARと同じ条件付き独立構造を仮定できない。行列・変数名の一致だけでは解消しない。

**修正**：直接node-space検証とgeometry経由検証を分ける。速度を正解にする場合は、生成側で位置を積分し、共通前処理後に所定の速度・単位・時点へ戻ることを確認する。積分位置の長期漂流、正規化の非線形性、変形許容範囲は別途検証する。位置VARを残す場合はlatent graphを構造回収の正解とせず、観測変換のrobustness条件として報告する。

### MV-03／重大：semantic basisが現行観測で消える・独立nodeを表現できない

`region_aggregation.py::aggregate_regions`はregion内の単純平均を取る。対称なmouth-widthやeye-open変形は、実際に動いていても重心応答が0になり得る。原案§9の応答比は分母0の定義もなく、basis番号aと観測variable番号aの対応も一般には一対一でない。cross-talkが小さいことだけでは観測可能性は保証されない。

**修正**：basisごとの符号付き応答、応答の単位、intended mapping、ゼロ応答、rank、特異値またはcondition numberを記録する。必要variable数より低rankならS1/S2を不合格とする。小振幅の局所Jacobianだけで全振幅の同一性を証明しない。semantic名の全種類実装は必須から外し、固定した実データ変数を表現できる最小basisを使う。除外閾値は開発calibration段階で定め、凍結評価の失敗例を事後削除しない。

### MV-04／重大：任意3D回転の除去能力は現行コードにない

`spatial_normalization.py::normalize_rotation`は指定2軸平面の参照点対を整列する。任意のyaw/pitch/rollを推定する3D剛体姿勢回復ではない。原案§13〜14のH3を常に信号回復すべき条件にすると、既存手法の適用範囲外を実装defectと誤分類する。

**修正**：現行変換で除去可能な運動の確認と、面外運動へのstress testを分ける。3D入力形状の受付と3D姿勢補正能力を区別する。参照点の表情変形・欠損による全regionへの影響も測る。共通実データ前処理を3D化する場合は、別の科学仕様変更として記録する。

### MV-05／重大：Selfの単位が本研究とずれている

原案§25は「対象変数自身」と「inter-variable」を使う。現行研究計画§5.3および`design_matrix.py`は、対象**regionの全成分**の固定履歴に、選択された**別region**成分を追加する。例えばmouth::vx予測でmouth::vyはSelf側である。

**修正**：本研究ではSelf-region全成分を固定し、追加集合をcross-regionに限定する。scalar self-link、同region別成分、別regionリンクをグラフ評価で分離する。`ParentSet`への変換で失われるsame-region linkまで評価したい場合は、PCMCI raw出力を用いる。

### MV-06／重大：グラフ回収と予測利得を同じ正解にしない

直接edgeがない変数も媒介経路や自己相関を通じて予測情報を持ち得る。直接edgeがあってもSelfや他の特徴との冗長性、有限標本、Ridge縮小により追加利得が小さくなる。原案§30のeffect増大→利得増大、§32〜33の正しいlag/Nullの優劣は、普遍的な合格条件ではない。

**修正**：構造GT、数値的なsignal identity、予測estimandを分ける。開発用に空cross-region graph、明瞭な単一edge、chain/fork、複数lagを含める。GTは予測モデルやNull選択へ渡さない。既知edgeを使ったoracle予測は別の診断結果であり、Primary PCMCI成績に混ぜない。enrichmentには全候補cellのGを使い、集合一括Ridge利得で置換しない。

### MV-07／重大：統計的受容条件と反復設計が未定義

原案ACの多くは「計算できる」であり、信号保持や偽陽性の許容基準を定めていない。逆に全seedでF1=1やNull劣化を要求すれば確率的検定を誤評価する。複数subjectが同じgraphを共有しても、独立なgraph replicateにはならない。

**修正**：契約PASS、数値一致PASS、確率的性能評価、robustness限界を別欄にする。grid、系列長、burn-in、graph数、innovation反復数、seed、SNR、数値許容誤差、性能受容規則をbenchmark前に固定する。性能CIは独立dataset/graph replicateを単位にし、実研究のheld-out効果CIのsubject単位と区別する。outer foldやframeを独立反復として水増ししない。pc_alphaをそのままグラフ全体FPRの合格値としない。

### MV-08／中：generatorの向き・独立性・ノイズ軸が曖昧

原案§4.3の行列積ではA[target,source]だが、§10.2の成分式はA[source,target]になっている。Gaussianの指定だけでは時間・変数間独立性や共分散を定めない。またinnovation全体を同率で増やしても、線形系の標準化された相関/SNRは変わらない場合がある。

**修正**：`A[lag-1,target,source]`に統一し、単一有向edgeテストで転置を検出する。mandatory生成は時間独立・変数間独立、正の対角innovation分散を明記する。target innovationと観測noiseを別軸にし、実効SNRを保存する。安定性は最終係数に対して検証し、spectral radiusだけで十分なburn-inと判断しない。subject間係数も基準条件では共通にし、形態差が観測スケールやpooled discoveryへ与える影響を別条件で測る。

### MV-09／中：評価・raw契約・未来情報の具体化が必要

原案§28のempty semanticsは良いが、F1はprecision未定義時も集合式 `2TP/(2TP+FP+FN)` で定義できるケースがある。多lag assignmentの未対応edge数・tie-break、region投影時の重複排除も要固定。lag-response中心は推定lagでありGT lagではない（§32は混同の余地）。

`RawLandmarkSequence`はlandmarks/timestamps/source_pathを持つが、座標系やvalidityの完全契約ではない。`RegionDefinition`も任意index mappingを受け取る型であり、実データ468点対応表が確定済みとは言えない。現行Primaryの`sequence_assembly.py`は補間なしを要求するので、新たな補間機構の導入は不要。

**修正**：§28に候補空間・対象レベル・空集合・未対応lagの数を追記する。情報隔離は引数境界とprovenanceで検証し、GTをsubject入力の同一コンテナに混ぜない。未来を改変しても起点以前の予測特徴が変わらない検査を行う。欠損は行削除で時刻を詰めず、差分両端とlag/target maskに伝播させる。通常予測とtest系列を並べ替える破壊実験の情報集合を区別する。

### MV-10／中：実装済み部品と完成した共通経路を混同しない

`synthetic.py`は独立AR(1)をFaceTimeSeriesへ直接生成するV0 fixture。`test_pcmci_known_lag_validation.py`は直接入力で既知単一lagを確認するテスト、`test_synthetic_motion_validation.py`は少数点の既知運動テストである。いずれも468点raw→新研究の全解析を通すStage 1完成証拠ではない。

**修正**：既存loader、normalization、aggregation、motion、split、Tigramite、design matrixを再利用し、共通のraw→FaceTimeSeries経路を先に接続する。写真生成・FLAME・Blender・再検出は原案どおり将来拡張。Stage 1はランドマーク以降の検証であり検出器の検証ではない。

## 外部一次資料の確認

- [Runge (2020), PCMCI+原論文](https://proceedings.mlr.press/v124/runge20a.html)：因果充足性を前提とした時系列CI探索。geometry mixingや未除去頭部運動のある条件で、理想生成グラフの回収保証へ拡張しない。
- [Google MediaPipe Face Mesh公式資料](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md)：468点とmetric 3D空間の説明。canonical geometryとscreen座標の契約を同一視しない。採用assetはcommit・hash・license・軸・単位を固定する。

MV-02の差分式とMV-03の可観測性判断は、原案の式と現行コードからの本監査の導出であり、上記論文の直接の主張ではない。

## 導入判断

追記：前回修正案自体を再検討した[第2次検討](methodological_validation_second_review.md)を参照。積分生成の条件、観測混合とグラフ保存、正規化のrank、等間隔、空ParentSetでも維持するΩについて追加の制約を導入仕様版2へ反映した。

[修正版導入仕様](methodological_validation_subsystem.md)を参照。監査と計画への組込みは完了。実行可能なStage 1は未完成であり、新protocol、実データregion/raw契約、signal identityの証明、共通E2E接続、性能受容基準の確定後に検証する。

