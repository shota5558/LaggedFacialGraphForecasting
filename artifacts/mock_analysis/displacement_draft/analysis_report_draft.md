# 顔部位間の遅延予測構造 解析レポート草案

作成日 2026-09-11  
版 displacement-report-draft-v1  
**MOCK DATA / NOT A SCIENTIFIC RESULT**

本書は、正規化変位の平均ユークリッド誤差を主指標とする解析の、図表と文章の草案である。30人・15組の架空groupから作った模擬予測を集約し、全候補G、被験者間の共通性、時間構造、4条件の予測、選択集合の濃縮を同じ指標で表示した。数値と形状は生成規則の結果であり、顔運動に関する実験上の発見ではない。

**今回実際に行った処理**は、模擬座標・予測からの主誤差計算、全候補Gの集約、1,000ランダム集合との比較、選択lag中心の集約、group単位10,000回のbootstrap、図表・本文の生成である。PCMCI探索、Ridge学習・inner調整、映像からの点抽出、実データのQCは実行していない。選択履歴100回とQC値も模擬値である。

## 1 解析対象と主指標

主誤差は、各点のx・yの予測差のユークリッド距離を取り、領域内の点と被験者内の評価時点を等重みで平均したEとする。点をまとめた重心の誤差や速度RMSEではない。全顔は先に8領域を等重み平均する。

`E = mean_time mean_point sqrt((pred_x − true_x)^2 + (pred_y − true_y)^2)`

`G(source→target, lag) = E_Self − E_(Self＋単一source領域lagブロック)`

G>0は追加ブロックにより誤差が小さくなることを示す。cell-Gは加算可能な効果ではなく、選択集合の同時入力による改善量とも異なる。表のE・G・Z・Rは眼角中点間距離比の **×10⁻³** 表示であり、改善率%やmmではない。

模擬設定は30 fps、h=1、他部位lagは1〜15 frame（33.3〜500 ms）、Self履歴30 frame。帯域は0〜100、100超〜250、250超〜500 msで、それぞれ3・4・8 lagを含む。点数は8領域29点58成分。このfps・履歴・点mappingが実データで確定・検証済みという意味ではない。

### N-T1 データと評価support

| 条件     |   人数 |   group数 |    元時間 秒 |    校正後 秒 |    QC後 秒 |   境界履歴後 秒 |   共通support 秒 |
|:-------|-----:|---------:|---------:|---------:|---------:|----------:|--------------:|
| 本人非発話  |   30 |       15 | 2700.000 | 2400.000 | 2160.000 |  1800.000 |       900.000 |
| 本人のみ発話 |   30 |       15 | 2700.000 | 2400.000 | 2160.000 |  1800.000 |       900.000 |

各被験者の各条件は900評価frameで、各図表は同じsupport hashを使う。上表の工程時間は表示確認用の模擬QC台帳。最後のfoldでは右頬を学習側使用不能とする例を置いたため、右頬関連は24人、他の部位対は原則30人で評価する。全顔4条件は24人・12groupとなる。1個のcell計算失敗も注入し、対応する帯域の部分平均を禁止している。

## 2 全候補の追加予測情報

![N-F1 全候補の帯域G  横source 縦target](C:/Users/yukit/OneDrive/ドキュメント/ChatGPT/PCMCI＋/artifacts/mock_analysis/displacement_draft/figures/N_F1_landscape.png)

被験者内の全lag平均後に被験者間中央値。斜線は自己対、灰色は評価不能。


各条件168個の部位対×帯域セルをすべて描いた。模擬値は正・負・ほぼ0が混在するよう生成した。本文を実データで書き換える際は、部位対と帯域、Gの大きさ、正の割合と分母を併記する。結果から最大lagや良かったcellだけを選び直さない。

![N-F2 改善の符号と分母  横source 縦target](C:/Users/yukit/OneDrive/ドキュメント/ChatGPT/PCMCI＋/artifacts/mock_analysis/displacement_draft/figures/N_F2_commonality.png)

セル内は正の人数/評価人数。0は正に含めない。真の効果がある人口の割合ではない。


同じ中央値でも被験者ごとの符号は異なり得るため、G>0人数と全被験者の値を併せて報告する。分母が異なるセルの大小を直接比較する場合は、共通被験者で対応差を再計算する。[全被験者のGとfold別の補足図](C:/Users/yukit/OneDrive/ドキュメント/ChatGPT/PCMCI＋/artifacts/mock_analysis/displacement_draft/supplement.md)に個別分布を示した。

## 3 絶対lagと中心化応答

![N-F3A 全56部位対の絶対lag  本人のみ発話](C:/Users/yukit/OneDrive/ドキュメント/ChatGPT/PCMCI＋/artifacts/mock_analysis/displacement_draft/figures/N_F3_absolute_speaking.png)

横source・縦target。全候補を固定順に表示。実上限500 ms、h=1。帯は点ごとの名目区間。


![N-F3A 全56部位対の絶対lag  本人非発話](C:/Users/yukit/OneDrive/ドキュメント/ChatGPT/PCMCI＋/artifacts/mock_analysis/displacement_draft/figures/N_F3_absolute_non_speaking.png)

横source・縦target。全候補を固定順に表示。実上限500 ms、h=1。帯は点ごとの名目区間。


絶対lag曲線は全56部位対を保持した。生成規則には広いピーク、上限付近まで続く形、口→顎の平坦な形を含む。これは図の読み分けを確認するための形状である。絶対lagの上限に改善が続く場合は終端未確認と記載する。

![N-F3B 選択lag中心の応答](C:/Users/yukit/OneDrive/ドキュメント/ChatGPT/PCMCI＋/artifacts/mock_analysis/displacement_draft/figures/N_F3_centered.png)

Δ=0は定義上0。正はずらした時の悪化。全Δで同じedge・部位対・被験者。


中心化はSelf＋単一ブロックの誤差差を、同じ部位対の選択lag平均、被験者内の部位対平均、被験者間中央値の順で集約した。±3 frame（±100 ms）の全gridで対象を固定した。選択lagからずらして誤差が下がる負値も残す。Δ=0の0は定義上の値で、時間特異性の証拠にはしない。

### N-T3B 中心化対象と境界除外

| 条件     |   選択edge数 |   対称grid適格 |   境界除外 |   失敗edge被験者単位 |   境界除外率 % |
|:-------|----------:|-----------:|-------:|--------------:|----------:|
| 本人非発話  |       111 |         74 |     37 |             0 |    33.333 |
| 本人のみ発話 |       114 |         76 |     38 |             0 |    33.333 |

edge数はfold・条件・部位対・lagの一意な選択数であり、被験者数ではない。失敗edge被験者単位は境界除外後の異なる分母なので別列にした。foldごとの分子・分母と各Δの被験者・部位対・edge数は数値表に保存した。

## 4 四条件の予測と入力数

![N-F4 4条件の誤差と対応差](C:/Users/yukit/OneDrive/ドキュメント/ChatGPT/PCMCI＋/artifacts/mock_analysis/displacement_draft/figures/N_F4_models.png)

全顔は8領域が揃う被験者のみ。点は被験者、太線は名目区間。差は被験者内で計算。


### N-T2A 全顔の主誤差

| 条件     | モデル         |   E中央値 | 名目95%区間       |   人数 |   group数 |
|:-------|:------------|-------:|:--------------|-----:|---------:|
| 本人のみ発話 | Persistence |  6.579 | 6.562 ～ 6.592 |   24 |       12 |
| 本人のみ発話 | Self        |  7.071 | 6.888 ～ 7.245 |   24 |       12 |
| 本人のみ発話 | Full        |  6.528 | 6.345 ～ 6.690 |   24 |       12 |
| 本人のみ発話 | PCMCI-block |  6.632 | 6.463 ～ 6.774 |   24 |       12 |
| 本人非発話  | Persistence |  6.588 | 6.572 ～ 6.603 |   24 |       12 |
| 本人非発話  | Self        |  7.471 | 7.288 ～ 7.645 |   24 |       12 |
| 本人非発話  | Full        |  6.928 | 6.745 ～ 7.090 |   24 |       12 |
| 本人非発話  | PCMCI-block |  7.029 | 6.894 ～ 7.179 |   24 |       12 |

### N-T2B 被験者内の対応差

| 条件     | 対応差              |    中央値 | 名目95%区間         |   人数 |
|:-------|:-----------------|-------:|:----------------|-----:|
| 本人非発話  | full-pcmci       | -0.083 | -0.134 ～ -0.047 |   24 |
| 本人非発話  | persistence-self | -0.880 | -1.034 ～ -0.706 |   24 |
| 本人非発話  | self-full        |  0.512 | 0.421 ～ 0.640   |   24 |
| 本人非発話  | self-pcmci       |  0.433 | 0.370 ～ 0.540   |   24 |
| 本人のみ発話 | full-pcmci       | -0.074 | -0.109 ～ -0.058 |   24 |
| 本人のみ発話 | persistence-self | -0.500 | -0.670 ～ -0.288 |   24 |
| 本人のみ発話 | self-full        |  0.512 | 0.421 ～ 0.640   |   24 |
| 本人のみ発話 | self-pcmci       |  0.462 | 0.349 ～ 0.569   |   24 |

正の対応差は左側のモデルから右側のモデルへの改善を示す。対応差の中央値は、モデル別中央値の差とは限らない。空選択の場合はPCMCI-blockをSelfと同じ予測にして0差を残した。Fullとの同等性・非劣性や疎性の合格判定は置いていない。

### N-T2C 入力規模

| 条件     | モデル         | Self列数範囲   | 追加block数範囲   | 追加scalar列数範囲   |
|:-------|:------------|:-----------|:-------------|:---------------|
| 本人非発話  | Full        | 120–480    | 90–105       | 570–810        |
| 本人非発話  | PCMCI-block | 120–480    | 0–3          | 0–38           |
| 本人非発話  | Self        | 120–480    | 0–0          | 0–0            |
| 本人のみ発話 | Full        | 120–480    | 90–105       | 570–810        |
| 本人のみ発話 | PCMCI-block | 120–480    | 1–3          | 6–38           |
| 本人のみ発話 | Self        | 120–480    | 0–0          | 0–0            |

範囲の単位は評価可能なfold×targetである。Persistenceは現在のtarget変位をそのまま使い、Ridge用の履歴列数表からは分けた。図の模型予測器は実際にRidgeで学習していないため、入力数は選択mappingから計算した設計上の列数である。

## 5 選択集合の濃縮

![N-F5 選択cellの濃縮](C:/Users/yukit/OneDrive/ドキュメント/ChatGPT/PCMCI＋/artifacts/mock_analysis/displacement_draft/figures/N_F5_enrichment.png)

選択cell-G平均から1,000ランダム集合平均の算術平均を引く。空選択は不能。


### N-T3A 評価可能な選択targetの要約

| 条件     |   selected平均Gの中央値 |   random期待値の中央値 |   Z中央値 | 名目95%区間         |   人数 |   group数 |
|:-------|------------------:|----------------:|-------:|:----------------|-----:|---------:|
| 本人非発話  |            -0.267 |          -0.187 | -0.077 | -0.097 ～ -0.066 |   30 |       15 |
| 本人のみ発話 |             0.178 |          -0.028 |  0.210 | 0.184 ～ 0.225   |   30 |       15 |

targetごとの選択cell-G平均から、ブロック数と実scalar成分数を揃えた1,000ランダム集合の平均Gの算術平均を引いた。各被験者でtargetを等重み平均してから集団中央値を取る。表の3列はそれぞれ個別値から要約しているため、表示された中央値同士の差がZ中央値に一致するとは限らない。

ランダム集合は模擬test値の生成前に固定し、同じfoldの全被験者で共有した。集合内重複なし、選択集合との重なりと反復間重複は許した。空選択はZ=0とせず評価不能。予定targetの計算失敗は成功targetだけの要約へ置き換えず、集約失敗として残した。1,000回を独立人数や厳密な置換p値には用いない。

[全被験者とtargetの1,000集合分布](C:/Users/yukit/OneDrive/ドキュメント/ChatGPT/PCMCI＋/artifacts/mock_analysis/displacement_draft/supplement.md)も保存した。正のZでもselected平均Gが負なら、Selfを改善したという結論にはならない。

## 6 不確かさと解釈の範囲

全図表はouter-fold内で独立groupを復元抽出する同一の10,000抽出indexを使う。選ばれたgroupの全被験者をまとめて残し、各集約順序を保った。区間は固定OOF値における被験者構成の変動を示す名目95%区間である。学習集合の重複や探索・学習をやり直す不確かさを解決するものではない。10group未満では区間を出さない。曲線は点ごとの区間で、多重比較補正済み・同時区間ではない。

本草案では有意性検定、星印、p/q値、結果に基づく表示の足切りを使わない。本人発話と本人非発話は別条件の記述であり、差を発話の因果効果とは解釈しない。source→targetは遅延予測関係として述べ、生理学的な直接因果や伝達時間とはしない。

## 7 実データ版への差し替え箇所

データ利用と独立group、採用点と測定品質、fps・Self履歴・共通support、実際の探索・Ridge調整・再学習を確認した後、承認protocolの実artifactから全指標を再計算する。旧速度RMSE集計を名称変更して読み込むことはできない。本草案生成は本実験freezeや科学的受容の完了ではない。

[補足資料 図表 設定 全数値への索引](C:/Users/yukit/OneDrive/ドキュメント/ChatGPT/PCMCI＋/artifacts/mock_analysis/displacement_draft/supplement.md)  
[生成設定](C:/Users/yukit/OneDrive/ドキュメント/ChatGPT/PCMCI＋/artifacts/mock_analysis/displacement_draft/input/config.json)  
[出力registry](C:/Users/yukit/OneDrive/ドキュメント/ChatGPT/PCMCI＋/artifacts/mock_analysis/displacement_draft/analysis_artifact_registry.csv)  
[出力manifest](C:/Users/yukit/OneDrive/ドキュメント/ChatGPT/PCMCI＋/artifacts/mock_analysis/displacement_draft/analysis_manifest.json)

再生成コマンド（repository root、analysis依存関係を導入したPython）:

```powershell
& .venv/Scripts/python.exe scripts/generate_displacement_mock_report.py
```
