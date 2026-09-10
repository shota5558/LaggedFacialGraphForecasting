# Design Document Residual-Issue Analysis & Execution Plan

監査日: 2026-09-09 JST  
対象: **shota5558/LaggedFacialGraphForecasting / dev**  
対象SHA: **9bbc3d946c0a64935b8786d9413abe53e6cefe29**  
担当: Codex / GPT-6。新規実装・科学値の採用・Primary/Sensitivity実験・GitHubへの投稿は行っていない。

## 1. Executive Summary

**Current phase:** 2026-09-09改訂研究計画に対し、devは旧protocolの基盤実装・synthetic integration・mock解析段階。新主解析の設計移行と実データpreflightが必要。  
**Overall readiness:** **NOT READY FOR PRIMARY FULL RUN**。Sensitivity実行も未許可。

| 項目 | 判定 |
|---|---|
| Complete | 124要求中31は、既存の狭いソフトウェア契約としてCOMPLETE。研究全体の完了を表さない |
| Partial | 19。実runへの接続、同一support、統計・解析間の整合性など |
| Remaining | SPEC_UNDECIDED 26、SPEC_CONFLICT 5、BLOCKED 1、NOT_IMPLEMENTED 8、NOT_EXECUTED 28、NOT_TESTED 2、NOT_DOCUMENTED 4 |
| Blockers | 最新計画とv6の競合、landscape/enrichment/edge-centered population解析未実装、data manifest・split・topology・表現未決、本番runner |
| Scientific decisions required | D-01〜D-13。候補は本書に記すが、採用やfreezeはしていない |
| Real result | 追跡対象の実Primary予測、最終統計、Primary freeze manifest、実Sensitivity成果は未確認。リポジトリ内には存在せず、PR-01も未実行と明記 |
| Analysis outputs | T01〜T09 / F01〜F14の23出力はmock実装あり。実データ生成・検証は0とtrackerに明記 |

**進行を止めているものは、単なるテスト不足だけではない。** 実験仕様が欠けているうえ、解析側には現在のテストで検出されない不一致がある。特に、matched-sparsityの中央値規則が図表側で平均に変わる点は、凍結仕様に対する確認済みの実装不一致である。

監査で確認した最重要事項:

1. **対象の取り違えを回避した。** 初期cwdのremoteは facial-coordination-graph、SHA a505cc7。これは依頼先と別研究である。指定リポジトリを隔離取得した。初期cwdの合成DDL研究計画やP0-1〜P0-10仕様は、依頼先の科学的規範として採用していない。
2. **ユーザー提供の2026-09-09改訂研究計画を原典として追加した。** 主研究はregion–lag predictive information landscape、selection enrichment、population-level lag structure。devには原典登録と新3解析の実装がない。
3. **Scientific Freeze v6とPrimary結果のfreezeは別物。** 前者は既存の固定値、後者はIssue #20の全成果物固定。v6が存在しても #20完了とはならない。
4. **CI成功は実験完了を示さない。** 対象SHAのGitHub CIは766 passed / 2 skipped / 3 warnings。ローカル重点検証は132 passed。それでも下記F-04〜F-09を確認した。
5. **次はR-01。** 計画1000回/v6 100回、edge別curve/全親同時shift、新3解析の差分をgoverning Issueとmigration planへ整理する。承認前にPrimaryを開始しない。

### 監査範囲と証拠の限界

- ユーザー提供DOCXの全278非空段落（表を含む）、devのtracked tree、Scientific Freeze、schema、全23 Issue本文、Issue #23コメント、関連Git履歴、関連実装・テスト、setup artifact、mock inputを読んだ。
- すべての関数の形式証明や全テストの独立妥当性証明はしていない。要件の細分化にはユーザー提供研究計画を含む。独立した詳細実装設計書は未入手。計画のAggregateやcomponentの曖昧さは推測しない。
- 外部ストレージや別リポジトリに実データ・成果がないとは断定しない。所在が不明なものを完了証拠に使っていない。
- 既存の他作業・未追跡ファイルは保全した。監査は隔離cloneで実施し、tracked source/config/testは変更していない。
- 上位ワークスペースから与えられたGit運用・Ponytail gateは運用上の指示として尊重したが、その合成DDL科学文書は別研究なので継承しない。

### 実行した検証

| 検証 | 結果・解釈 |
|---|---|
| GitHub Actions run 34243176049 | 対象SHA一致、766 passed / 2 skipped / 3 warnings、75.22秒。これは今回APIとログで再確認した既存CI |
| ローカル重点suite | 132 passed / 50.30秒。ローカルpytest 9.1.1は宣言範囲<9外であり正式環境の全suite代替ではない。config/core schema/leakage/statistics/DRY/audit/null/lag/analysis/mock/freeze-barrier |
| 最初のローカル探索 | dcor不足でcollection失敗。重点suiteの初回はtabulate不足で4 failed / 128 passed。環境不足として記録 |
| ローカル全suite再試行 | dcorの推移依存array_api_compat不足でcollection 2 errors。**ローカル全suite成功とは主張しない** |
| 環境対応 | 共有venvは変更せず、監査専用 .audit-deps に宣言済み依存dcor/tabulateを置いた。最終重点suiteではtabulateを使用 |
| 追加診断 | 不一致n_valid、指標差替え、manifestなしT09、反復mean、3反復受理、Sensitivity CSV必須を再現 |
| CLI | scripts/generate_analysis_outputs.py --help を確認 |
| 実データ | 読取り・実験・Sensitivity実行なし |

CI: [対象SHAのテスト結果](https://github.com/shota5558/LaggedFacialGraphForecasting/actions/runs/34243176049)。診断入力は **MOCK DATA / NOT A SCIENTIFIC RESULT**。

## 2. Authoritative Documents

版の判定はmtimeではなく、tracked版・schema・migration記述・Git履歴・Issue対応を優先した。

| Priority | Document | Version | Role / 判定 |
|---|---|---|---|
| 1 | ユーザー提供研究計画書 | 2026-09-09研究目的一般化改訂、本文P0007 | 最新の主目的・RQ1–5/H1–4・新解析要求。実行freezeとの競合を保持 |
| 2 | configs/scientific_freeze.yaml | schema 6、117331f、2026-09-08 07:44 JST | 現devの実行固定値。新計画との差分は未移行 |
| 2 | schemas/scientific_freeze.schema.json / scientific_config.py | v6相当 | machine-readable schemaと実行validator。規範に対する実装であり新たな科学根拠ではない |
| 3 | schemas/core_contracts_v3.yaml | schema 3、9439cac、2026-09-08 05:18 JST | canonical contract・component/target dimension provenance・migration |
| 4 | Issues #1〜#21 | 監査時点の本文を保存 | 実装・実験の順序、gate、acceptance。科学値はfreezeと照合 |
| 5 | Issue #22 | Sensitivity pre-implementation | synthetic先行実装の許可。実データ実行は #20後 |
| 5 | Issue #23 | analysis output契約 | T01〜T09/F01〜F14、統計・provenance・出力要求 |
| 6 | PCMCI_analysis_tables_figures_implementation_tracker.md | 2026-09-08 audit、837315b | 出力別実装状態。Issue #23とFreezeを規範と自ら明記 |
| 7 | configs/primary_run.yaml | schema 3 | seed 20260908、artifact root。実験のdata/split/topology値は含まない |
| 8 | artifacts/setup/*/task_result.yaml | 各過去SHA | 実装証跡・過去の失敗。最新の科学仕様に優先しない |
| 9 | schemas/core_contracts_v1.yaml / v2.yaml | v1/v2 | **OBSOLETE for new runs**。履歴・明示migration専用 |
| 10 | README.md | # TESTのみ | 導線不備。科学規範ではない |
| — | 対象repoのAGENTS.md | 不在 | 隣接別研究の科学規範を持ち込まない |

参照先: [Freeze v6](https://github.com/shota5558/LaggedFacialGraphForecasting/blob/9bbc3d946c0a64935b8786d9413abe53e6cefe29/configs/scientific_freeze.yaml)、[Core v3](https://github.com/shota5558/LaggedFacialGraphForecasting/blob/9bbc3d946c0a64935b8786d9413abe53e6cefe29/schemas/core_contracts_v3.yaml)、[Primary preflight記録](https://github.com/shota5558/LaggedFacialGraphForecasting/blob/9bbc3d946c0a64935b8786d9413abe53e6cefe29/artifacts/setup/PR-01/task_result.yaml)。

### 受領研究計画の版と読取証拠

正式入力はユーザー提供の [2026-09-09改訂研究計画書](C:/Users/yukit/OneDrive/デスクトップ/研究/PCMCI_facial_motion_research_plan_generalized_predictive_structure_2026-09-09.docx)。
本文P0007に「研究目的一般化改訂 2026年9月9日」と明記。SHA256は
7d77f30febaf4856131f2bb9930a2da93ec8252b60e5137e46c96f58550bc249。

全278非空段落・11表をOOXML順に読み取った。画像、native math、tracked insertion/deletion、commentsは0。
docPropsの日付は2013年のテンプレート値なので版判定に使わず、本文改訂日と内容を採用した。
P番号は evidence/research-plan-extracted.txt の段落IDでありページ番号ではない。
パッケージのDOCX rendererはLibreOffice/soffice不在で失敗したため、ページレイアウト検証やページ番号の主張はしていない。原文は変更していない。
本監査は計画の要件と実装の照合であり、計画中の文献新規性そのものを再検証したものではない。

最新計画は**研究目的・要求の規範**。一方、現devの**実行固定値はv6**のままなので、競合を黙って解消せず明示migrationを計画する。

### 既に解消可能な版の食い違い

- DRY/task_result.yaml末尾のbootstrap・matched repeat未固定という記載は、後続v6で解決済み。新しい未決定事項として数え直さない。
- Issue #4のschema v1記述はCore v3のmigration記録により履歴と判定する。新runにv1を採用しない。
- #15はOPENだがG-00〜04の進行gateはPASSと本文で明記。G-13/G-14はPrimary開始のblockerではない。
- #16/#17はOPEN、setupはPASS。synthetic software gateの証拠は認めるが、本番のdata/topology仕様を経た実行gateとは別に再確認する。
- #23のCLOSEDは、ownerの受入コメントでsoftware完了と実結果未完了を区別している。closure自体を誤りとは断定しない。

## 3. Requirement Coverage Matrix

完全な124行の台帳: **[requirement-matrix.md](requirement-matrix.md)**  
機械可読版: **[requirement-matrix.csv](requirement-matrix.csv)**

全行にID、要求、Source/Section、Type、Mandatory、Frozen、Status、Evidence、Remainingを記録した。Q01〜Q101が契約・実装・実行要求、T01〜T09/F01〜F14が個別出力要求。

| ID | Requirement | Status | Evidence | Remaining |
|---|---|---|---|---|
| Q01 | 新計画のrepository登録・追跡 | NOT_DOCUMENTED | DOCX受領、hash確定。repo未登録 | R-01 |
| Q02〜10 | 既存Primary固定定義 | COMPLETE（個別ソフトウェア契約） | freeze/validator/関連tests/CI | 本番runは別要求 |
| Q17〜18 | split方式の確定 | SPEC_UNDECIDED | PR-01 | R-02 |
| Q22〜30 | data/topology/feature/time | 各行参照 | 汎用APIとPR-01 | R-02/R-04 |
| Q41〜42 | matched反復数・median | SPEC_CONFLICT | 計画1000/v6 100、解析meanの別不一致 | R-01/D-10/R-06 |
| Q45 | 完全に同じ評価support | PARTIAL | _exact_pairがn_validを無視 | R-05 |
| Q48/Q51/Q67 | 集約・安定性bootstrap・多重比較 | SPEC_UNDECIDED | freeze未記載、code引数/平均 | R-03 |
| Q54 | 全条件本番runner | NOT_IMPLEMENTED | production runnerはV0 Self、compositionはtest内 | R-09 |
| Q55〜56 | 本番protocolのDRY/I4 | PARTIAL | synthetic限定の過去・現行テスト | R-10 |
| Q57〜59/Q63 | Primary→STAT→FREEZE→Sensitivity | NOT_EXECUTED | 実成果物なし、既存Issue OPEN | R-11〜14 |
| Q61〜62 | 図表のSensitivity境界 | PARTIAL | boolのみ、CSV常時必須 | R-07a/R-07b |
| T01〜F14 | 旧契約の23出力 | NOT_EXECUTED | tracker real 0/23 | 各R-16-output |
| Q79〜Q101 | 新landscape/enrichment/population解析 | 個別行参照 | production/schemaなし | R-17〜19 |

COMPLETEは一つの検証可能な条件だけに付けた。例えば「tau_max=10のconfigを強制する」が完了していても、「凍結した実データ全foldを実行した」はNOT_EXECUTEDのままである。

## 4. Remaining Issues

以下のIDは監査所見ID。GitHubに新規投稿していない。新規実装前にR-01で既存Issueまたは新規governing Issueに紐付ける。共通audit sourceは本書・対象SHA・evidence内ログ。

### Scientific / Specification

| Finding | Severity / Blocking | Problem・Evidence | Expected behavior / Task |
|---|---|---|---|
| F-01 | HIGH / YES | 最新原典は受領したがrepo/Issueへ未移行。詳細実装設計は未入手 | 原典、版、主仮説、主要対比を追跡。R-01 |
| F-02 | CRITICAL / YES | dataset manifest・split・mapping・正規化参照点・motion feature未固定。PR-01既知所見 | 原典・データ仕様から承認してfreeze。R-02/R-04 |
| F-03 | HIGH / YES（統計確定まで） | subject内region集約、discovery bootstrap、multiple comparison、exclusion/constant-unit処理が一意でない | D-05〜08を解決。R-03。統計bootstrap10000とは別 |
| F-12 | MEDIUM / 部分的NO | G-10〜15のoperational definition未決。G-13/14は明示non-blocking | 指標ごとに採否と仕様を固定。R-15。全指標をPrimary blockerにしない |
| F-13 | HIGH / YES（本番起動） | Ridge候補はcode constantだがAPIは別grid許可、DRYは3候補 | 本番protocolで承認gridを強制。generic APIの自由度だけをbugと呼ばず、run境界不足として扱う |

### 2026-09-09改訂による追加所見

| Finding | Severity / Blocking | Source / evidence | 影響・必要な解消 |
|---|---|---|---|
| F-20 | CRITICAL / YES | 計画§5.6/7.4 P0081/P0124は1000、freeze v6は100 | 反復数競合。最新計画への明示migrationを推奨、D-10。既存v6を黙って変更しない |
| F-21 | CRITICAL / YES | 計画§5.5/7.3 A,D。全ΩのSelf+cell学習・G・source→target/帯域集約が必要 | production cell-grid経路なし。Full-history joint fitでは代替できない。R-17 |
| F-22 | CRITICAL / YES | 計画§5.6のAggregate(G_selected)−Aggregate(G_matched) | cell gainの集合集約かjoint-set再学習gainかが曖昧。T07誤差差と同値とは限らない。D-12/R-18 |
| F-23 | CRITICAL / YES | 計画§5.7/7.4はedge別τ*整列、v6は全selected parentsのcommon shift | 異なるestimand。旧F04で新population要求を完了にできない。D-13/R-19 |
| F-24 | HIGH / YES | 計画§5.1は2D候補、§5.5はregion単位の単一feature。Core v3はexact scalar component | cell定義とfeature-count単位を固定。D-11。部位vectorを一scalarと数えない |
| F-25 | MEDIUM / NO（GRUのみ） | 計画§9/13はGRU任意、Issue #21は6種必須。新主解析は旧出力tracker未掲載 | GRU採否と新出力契約を明示。R-01/R-16-new。他Sensitivityを免除しない |

共通回帰要求: 新解析のcandidate digestをouter-test前後で一致させる。変更されたrepeat、cell/component、edge単位、集約はschema versionとmigration記録に保存する。
実装Issueはsynthetic integrationで閉じられる責任と、実科学runを必要とする #18〜21 の責任を分離する。新研究scopeを旧PASSへ遡及適用しない。

### Implementation

| Finding | Severity / Blocking | 確認済みの動作 | 最小修正・回帰要求 |
|---|---|---|---|
| F-04 | HIGH / YES（推論・報告） | analysis_pipeline.py:282 _exact_pairはn_validをmergeに含めない。100 vs 1件でも4ペアを受理 | R-05。frame/target/component identityのdigestを原predictionsから導出し照合。件数だけ同じ・時刻違いも拒否 |
| F-05 | HIGH / YES | table_t07:554前後はrepeatをmean。v6のmedianと不一致。診断[0,0,9]で期待0、実際3。現v6で100固定なのに3反復も受理 | R-06。subject-region内median→承認region集約→subject間median。移行後N_matchedのunique repeat完全性・欠落・重複・中断を回帰 |
| F-06 | HIGH / YES | _primary_metric:209が任意lower-is-better指標を許可。position_rmse差替えを受理。seed欠落で0 | R-06。freeze由来のprimary_metric/seedを必須化。CSVの利用可能性で選択しない |
| F-07 | HIGH / YES（Sensitivity出力） | table_t09:669はprimary_frozen=Trueだけでreal扱いを許可。manifestがなくても5行受理 | R-07a。既存validate_primary_freeze_manifestを再利用し、参照元run/hashとの結合も検査 |
| F-08 | MEDIUM / YES（Primary出力） | AnalysisInputs.from_directory:54とgenerate_analysis_outputs:889がSensitivity CSVを無条件要求/読取り | R-07b。include_sensitivity=Falseでは入力不要・読取りなし・hash対象外 |
| F-09 | HIGH / YES（実run→図表） | 解析は事前集計CSV/JSONが入口。core v3 predictionsから当該schemaへの本番producerがない | R-08。lossless exportと元manifest/行supportを接続 |
| F-10 | HIGH / YES | runner.py:71はV0 Self専用。全条件pipelineはtests::_run_one_foldにある | R-09。既存部品を構成するthin production runner。新推定器は不要 |
| F-11 | HIGH / YES（#20） | #20のmanifest consumerはあるが、全Primary成果を集める受入済みwriter/runなし | R-13。既存registry/atomic writeを使用し完全性を検査 |

F-07の診断は、図表関数に対するソフトウェア入力の検査である。実Sensitivityアルゴリズムを実データで動かした証拠ではない。F-04も実結果が汚染されたと断定するものではなく、不正supportを防げないという再現済みの欠陥である。

### Validation / Experiment

- **F-14 HIGH / YES:** DRYはsynthetic、単一outer fold、3 alpha、単一matched sample。real topology・100反復・全target・全foldの本番経路は未検証。R-10。
- **F-15 HIGH / YES:** Primary full run、比較統計、lag response、bootstrap stability、failure/unevaluable解析の実証拠がない。R-11/R-12。
- **F-16 HIGH / YES（Sensitivity）:** Primary result freeze、実Sensitivityは未実行。R-13/R-14。
- local full suiteは環境依存不足で未完走。対象SHA CI成功を別の証拠として残す。依存不足を製品不具合や科学FAILと混同しない。

### Documentation / Traceability

- **F-17 MEDIUM / NO:** DRYの残課題欄はv6以前。#16/#17とsetupの状態、#4のschema記述も同期が必要。R-01/R-10。
- **F-18 MEDIUM / YES（公開）:** mockには既存のFAKE表記はあるが、今回要求された `MOCK DATA` / `NOT A SCIENTIFIC RESULT` の文字列ではない。意味は明示済みだが、要求への完全一致は未完。R-08。
- **F-19 HIGH / YES（公開）:** analysis manifestのcode_versionは固定文字列で、raw scientific runのgit SHA・freeze digestへの導線を完成させる必要がある。input CSVのhashだけではraw→表の科学的同一性を証明しない。R-08/R-16。
- README導線、原典Version、Issueの科学実行受入証拠をR-01で整える。

### 差分タイプ分類

| Type | 実例 |
|---|---|
| Design-only | #18全fold orchestration、#20実結果freeze writer、実data manifest |
| Code-only / authority unclear | T04のregion平均、任意metric fallback、seed=0。新計画もAggregate詳細は未定義のためoperational definitionを要確定 |
| Test-only composition | tests/test_one_fold_scientific_dry_run.py::_run_one_fold。本番composition APIではない |
| Implemented-but-unvalidated | generic anatomy/normalizationを実データに適用する経路 |
| Documented-but-not-enforced | matched median/count、analysisでのfreeze manifest、同一frame support |
| Executed-but-not-reproducible | 実科学run自体が未確認なので、この類型の実runは認定しない。setup PASSだけで実結果の再現性を主張しない |

各修正IssueにはF-ID、該当Q-ID、上記severity、証拠、規範、期待動作、回帰条件、freeze impactを記載する。F-04〜11は原則としてソフトウェア回帰＋raw-to-output integrationで閉じられる実装Issueとし、実科学runの受入は #18〜21 に残す。F-02/03/12は決定未了のまま実装着手しない。モデル追加・新しい研究仮説・別データ探索はscope外。

## 5. Critical Blockers

### B1: 研究原典・実データprotocolが接続されていない

**Problem:** 計画1000/v6 100、edge別/全親同時shiftの競合、新主解析の欠落。dataset・split・解剖学的mapping・運動表現も未確定。  
**Why blocking:** 学習対象・評価母集団・誤差の単位が変わる。コードの例示値では決められない。  
**Required resolution:** R-01で原典確定、R-02でD-02〜05を承認、R-04でmanifestとschemaに接続。  
**Affected downstream:** R-09〜16。データファイル未入手ならR-04以降はBLOCKEDを維持。

### B1a: 新主解析の仕様と実行経路がない

**Problem:** landscape全Ω、selection enrichment、edge-centered population responseの原要求が新たに明確になったがdevにはない。  
**Why blocking:** 主推論が旧Self−PCMCI/全親lag-shiftから広がっており、旧8条件のみのrunではRQ1〜5とH2/H3を満たさない。  
**Required resolution:** R-01のmigration、D-10〜13、R-17〜19。  
**Affected downstream:** R-08/09/10/11/12/13、surrogateによる新指標検証、最終図表。

### B2: 推論値が実装経路によって変わる

**Problem:** matched repeatのmedian/mean不一致、support同一性不足、Primary metric fallback。  
**Why blocking:** 同じraw結果から異なる効果量・CI・結論が出る。  
**Required resolution:** R-03の統計決定とR-05/R-06の実装修正。  
**Affected downstream:** R-10/R-12/R-16。Primaryを実行してから分析方針を選ばない。

### B3: 本番実行とfreeze境界の接続不足

**Problem:** production全条件runner、raw-to-analysis export、結果freezeの完成証拠がない。  
**Why blocking:** synthetic test関数のPASSは本番CLIの全条件・provenance・resumeを保証しない。  
**Required resolution:** R-08/R-09、R-10の本番経路検証、R-13。  
**Affected downstream:** 全fold run、分析、Sensitivity、公開。

### B4: Primary-only/Sensitivity出力の境界が不十分

**Problem:** Primary-onlyでもSensitivity CSVを要求。real T09/F10許可はboolに依存。  
**Why blocking:** 正しい順序で利用できず、freeze検証が実行APIと図表APIで異なる。  
**Required resolution:** R-07a/R-07b。  
**Affected downstream:** Primary解析、#20後のSensitivity公開。フラグで迂回しない。

## 6. Unresolved Scientific Decisions

以下は**提案**であり、いずれも未採用。`[承認値]` を含むfreeze文案はレビュー用テンプレートで、実行可能なconfigへ入れてはいけない。研究対象・estimandを変えるD-01〜13は **REQUIRES_SCIENTIFIC_DECISION** として扱う。既存freezeの数値を再決定する提案ではない。

### 科学仕様inventory

| Item | State | 判定 |
|---|---|---|
| Primary hypothesis | PARTIALLY_FROZEN | 計画RQ1–5/H1–4、主推論enrichment/populationを明示。repo移行と確認的階層は未確定 |
| Primary endpoint | PARTIALLY_FROZEN | velocity_rmse名称とsubject median/CIはある。表現・region集約が未固定 |
| Primary model | FROZEN | PCMCI+ / ParCorr / Ridge |
| Baseline | FROZEN | Persistence / Self / Full。forecasting semanticsあり |
| Null | FROZEN | lag-shift / random-region / matched-sparsity / time-shuffle |
| Sensitivity | PARTIALLY_FROZEN | 6種と実行順は固定。実study-level各設定は別途必要 |
| outer split | PARTIALLY_FROZEN | subject-disjointのみ固定。LOSO/K-fold未選択 |
| inner tuning | PARTIALLY_FROZEN | inner-train scaler・outer-trainのみ。fold設計・本番grid強制は未完 |
| feature construction | PARTIALLY_FROZEN | mean centroidとexact componentはあり。motion表現・topology未決 |
| lag definition | PARTIALLY_FROZEN | frame-based target-relative lagとtau>=hは固定。物理時間・不規則標本方針未決 |
| lag boundary policy | PARTIALLY_FROZEN | 旧set-shiftは固定。新edge-centeredへのgrid/除外単位の継承はD-13 |
| tau_max / pc_alpha / CI test | FROZEN | 10 / 0.01 / ParCorr |
| prediction horizon | FROZEN | h=1 |
| region dimensionality | PARTIALLY_FROZEN | component identity保持、実2D/3D未選択 |
| normalization | UNDECIDED | 解剖学的reference points/pairs/axes未固定 |
| missingness | PARTIALLY_FROZEN | validity mask・補間none。quality cutoff/除外未固定 |
| evaluability | PARTIALLY_FROZEN | lag gridは固定。全studyの不足support処理未固定 |
| aggregation | PARTIALLY_FROZEN | median(subject)・median(matched repeats)固定。region/session内集約未決 |
| statistical unit | FROZEN | subject。framesは独立反復にしない |
| bootstrap unit | PARTIALLY_FROZEN | paired CIはsubject、discovery stabilityのblock設計は別 |
| bootstrap method / repetitions | PARTIALLY_FROZEN | paired CIはpercentile/10000。discovery側未固定 |
| confidence interval | FROZEN | paired CI 95% |
| random seed policy | PARTIALLY_FROZEN | paired experiment seed・matched split seed固定。全本番namespace接続待ち |
| matched sparsity | PARTIALLY_FROZEN | 計画1000/v6 100。region/component feature単位も要決定。解析median不一致は別件 |
| multiple comparison handling | PARTIALLY_FROZEN | 個別cellは探索的、主推論enrichment/population。H1–H4階層・family補正は未定 |
| exclusion rule | PARTIALLY_FROZEN | grid規則あり。subject/session/qualityは未固定 |
| failure handling | PARTIALLY_FROZEN | atomic/resume primitivesあり。本番停止・再試行・分母の規則未完成 |

### Decision options一覧

| Decision | Options | Recommendation | Reason | User decision required |
|---|---|---|---|---|
| D-01 新計画migration | 明示移行 / 旧版を別研究として保持 / 新旧混在 | 明示移行 | 最新研究要求を実行freezeへ接続 | YES |
| D-02 data/split | LOSO / fixed grouped K-fold / その他事前指定 | manifestと被験者数・計算量確認後に選択 | nやsession構造が不明な今は数値を決められない | YES |
| D-03 anatomy/normalization | 使用extractorの解剖学的定義 / 別承認mapping / generic例示 | 使用データの定義をversion/hashで固定 | 無関係なlandmark indexを流用しない | YES |
| D-04 motion/metric | velocity直接予測 / displacement予測＋明示変換 / 別endpoint | 原典が許せばvelocity直接予測 | velocity_rmseとの単位整合が最短 | YES |
| D-05 timing/missing/evaluability | uniform sampling限定 / 事前定義再標本化 / irregular-frame estimand | 等間隔データに限定できるか先に確認 | 補間noneを保つ案が既存契約に近い | YES |
| D-06 subject内集約 | 等重みregion平均 / pooled RMSE / region別のみ | 原典が許せば等重みregion平均→subject median | 現T04との整合、長系列偏重を避ける。ただし採用しない | YES |
| D-07 stability bootstrap | 事前数値固定 / train-only校正規則固定 / 実結果後選択 | train-only/calibrationでblock設計を決め事前freeze | time dependenceと計算量の両立 | YES |
| D-08 multiple comparisons | enrichment/populationのjoint family / 階層H1–4 / 記述的区分 | 新主推論2系統を保ってfamily承認 | Self−PCMCI単独へ縮小しない | YES |
| D-09 G-10〜15 | 定義して追加 / 明示secondary延期 / 暗黙計算 | 原典照合後、各指標別に採否 | peak/onsetを全体blockerにしない | YES |
| D-10 matched repeats | 1000へ明示移行 / 計画を100へ再改訂 / 混在 | 1000への明示migration | 新計画の明文を保存 | YES |
| D-11 landscape cell | scalar component / region vector bundle / 事後選択 | exact component案をレビュー | 既存v3と計画の意味を整合 | YES |
| D-12 enrichment | cell G aggregate / joint-set gain / 別名併記 | 両者を区別しPrimaryを承認 | 同値ではない | YES |
| D-13 population curve | Self+edge / set内1edge shift / 全親同時shift | Self+edge案をレビュー | edge-centered式と整合 | YES |

### D-01 — 最新計画の明示migration

**Decision:** 2026-09-09改訂の原典登録、旧v6との差分、移行先protocol版。  
**Why it matters:** 旧8条件だけでは新主研究を満たさない。  
**Options:** A 新計画へ明示migration、B 旧計画と別runで保持、C 新旧混在。  
**Recommended option:** A。Cは禁止。  
**Rationale:** 最新の研究目的を保存しつつfreeze provenanceを維持する。  
**Risk:** 計算量増加、曖昧なestimand、新旧testの誤った完了判定。  
**Freeze text（案）:** 「研究規範は2026-09-09改訂計画 hash [記録済みdigest]。旧v6からの変更は[承認migration ID]に列挙し、landscape/enrichment/edge-centered population responseを事前仕様に含める。」  
**Implementation consequence:** 原典・要求・governing Issue・新schema版を対応付ける。  
**Validation:** C-08〜12とQ79〜101にdispositionがあり、新旧混在を拒否。

### D-02 — dataset、outer/inner split

**Decision:** immutable data manifest、被験者/session対応、split方式と必要なfold数。  
**Why it matters:** 被験者独立性・学習規模・計算量・再現性が変わる。  
**Options:** A LOSO、B 固定grouped K-fold、C 原典所定の別subject split。  
**Recommended option:** manifestに基づきA/Bを比較し承認。被験者数を知らずにKを埋めない。  
**Rationale:** 両実装が存在することは採用根拠ではない。  
**Risk:** 小標本の内側fold不足、sessionの跨ぎ、重複subjectのbootstrap。  
**Freeze text（案）:** 「Primaryはdataset manifest [digest] に含まれる被験者を用い、outer=[承認方式/数]、inner=[承認方式/数]とする。各被験者はouter-testへ一度だけ寄与し、sessionはsubjectとともに分割する。分割は結果に依存しない。」  
**Implementation consequence:** manifest schema、splitの単一入口、run preflight。  
**Validation:** 漏洩・欠落・重複・manifest変更を拒否し、same seedで同じsplit digest。

### D-03 — 解剖学的mappingと正規化

**Decision:** extractor/version、landmark番号、region、座標次元、translation/scale/rotation参照。  
**Why it matters:** region時系列の意味と予測の難しさを変える。  
**Options:** A データの正式mapping、B 科学的レビュー済み代替、C synthetic例示の流用。  
**Recommended option:** A。Cは不採用。  
**Rationale:** 実topologyに対する参照点の妥当性を検証できる。  
**Risk:** 参照点欠測・scaleゼロ・軸退化・左右対応誤り。  
**Freeze text（案）:** 「extractor [版/hash]、region mapping [artifact/hash]、dimension [labels]、translation [indices]、scale [pair]、rotation [pair/axes]を使用する。退化時は[承認failure policy]とし、別参照点へ暗黙fallbackしない。」  
**Implementation consequence:** generic APIへ承認値を渡すadapter。  
**Validation:** bounds、label uniqueness、左右/軸、退化・欠測のnegative test、実sampleのmetadata QA。

### D-04 — 運動表現とVelocity RMSE

**Decision:** 予測yをvelocityにするか、別表現を変換して評価するか。  
**Why it matters:** 現metricsは配列RMSEをvelocity_rmseと名付けるだけで単位を検証しない。  
**Options:** A 正規化座標のvelocity直接予測、B displacement予測＋Δt変換、C position予測＋変換。  
**Recommended option:** 原典が許せばA。B/Cは追加provenanceが必要。  
**Risk:** 不規則Δt、微分境界、座標scale、component欠測。  
**Freeze text（案）:** 「予測対象は[承認表現]、時刻単位は[承認単位]、差分端点は[承認規則]とする。Velocity RMSEは同一supportの全承認componentに対する sqrt(mean((y_pred-y_true)^2)) とし、表現・単位をartifactに保持する。」  
**Implementation consequence:** feature construction、PredictionArtifact metadataまたは参照manifest、metricの検証。  
**Validation:** analytic motion fixtureで単位・境界を検証。displacementをvelocityとして入力したら拒否。

### D-05 — sampling、quality、欠測、evaluability

**Decision:** 不規則timestamp、nominal sampling、quality threshold、最低有効support、除外・failureの規則。  
**Why it matters:** index lag 10が同じ物理時間を意味するか、手法間でどの行を比べるかが変わる。  
**Options:** A uniform-cadenceデータのみ、B 承認resampling、C frame-indexのみをestimandにする。  
**Recommended option:** Aの成立可否をmetadataで評価。Bは補間noneからの明示protocol変更。  
**Risk:** 多数除外による母集団変化、不規則gapを飛び越えるlag、未評価の隠蔽。  
**Freeze text（案）:** 「時刻整合許容差は[承認値]、quality基準は[承認規則]。無効行は保持・flagし、比較supportは事前定義の[intersection規則]から構成する。対象外subject/session/target-foldと理由・分母を全件報告し、成績による除外をしない。」  
**Implementation consequence:** data adapter、shared support builder、support hash、unevaluable ledger。  
**Validation:** 同数別時刻・欠測・全欠測・short series・irregular gap・不足subjectを検査。

### D-06 — subject内集約

**Decision:** region/session/component→subjectの演算順序・重み。  
**Why it matters:** mean of region RMSE、pooled RMSE、median of region effectsは一致しない。  
**Options:** A region等重み平均→subject median、B frame pooling→subject median、C region別推定のみ。  
**Recommended option:** 原典が単一全顔endpointを求めるならAを検討。CならT04の全体推定をPrimaryと呼ばない。  
**Risk:** region missingnessによりsubjectごとに異なるestimandになる。  
**Freeze text（案）:** 「各subject-regionの誤差を同一評価supportで算出し、[承認region集合/重み]でsubject効果を定義する。被験者間は中央値。既存Null誤差比較のmatched-sparsityは各subject-regionで承認N_matched反復の中央値を先に取り、その後同じ集約を適用する。新enrichmentのAggregateはD-12の承認定義と区別する。」  
**Implementation consequence:** 統計APIとanalysisで共通のreducerを使う。  
**Validation:** 不均衡region/frame数・極端値で各演算の違いが出るfixtureを使用しT04/F02/STATを一致させる。

### D-07 — discovery stability bootstrap / region projection

**Decision:** boot_samples、boot_blocklength、seed規則、component→region selected/opportunityの数え方。  
**Why it matters:** CIのsubject bootstrap 10000とは別であり、依存保存・分母・edge頻度が変わる。  
**Options:** A 事前数値固定、B train-only規則で校正しfreeze、C Primary結果を見て調整。  
**Recommended option:** Bまたは原典指定A。Cは禁止。  
**Risk:** block長不足、セッション横断、component数が違うregionの機会数bias。  
**Freeze text（案）:** 「discovery stabilityはouter-train内で[承認block方式/長さ/回数]、seed [namespace]を用いる。region-lag selectedは[承認OR/別定義]、denominatorは[承認機会単位]。欠測機会と観測ゼロを区別し、頻度からPrimary ParentSetを再選択しない。」  
**Implementation consequence:** configに引数を記録しproducerで集約。  
**Validation:** bootstrap known-count、欠けた機会、D>1、subject/session境界、Primary ParentSet不変。

### D-08 — 新計画の主推論と複数対比

**Decision:** 主推論enrichment/populationとH1–H4の関係、family、pointwise/simultaneous CI。  
**Why it matters:** §0/7.3は個別cell探索的、主推論enrichment/populationを明示。Self−PCMCIのみへ縮小できない。  
**Options:** A 2主推論を共同family・H1/H4補助、B 階層的H1–H4、C 推定的区分を明示承認。  
**Recommended option:** Aを科学レビューの候補とする。具体補正・成功閾値は創作しない。  
**Rationale:** 新研究の中心を保ちながら複数対比を管理できる。  
**Risk:** 帯域探索の事後転用、H4非劣性marginの事後設定。  
**Freeze text（案）:** 「主推論は[承認enrichment]と[承認population estimand]。H1/H4の役割、family、補正は[承認規則]。cell順位は探索的。帯域・成功判定をouter-testから変更しない。」  
**Implementation consequence:** comparison registryとcaptionを新主推論に対応。  
**Validation:** 全主張と事前registry一致。探索的最大cellをPrimaryへ昇格しない。

### D-09 — 追加metric群 G-10〜15

共通: **単に関数名があるだけでmetricのoperational definition済みとはしない**。各項目を独立決定・独立Taskとする。

| Metric | Options / 推奨候補 | Risk / Freeze text案 / Validation |
|---|---|---|
| Position RMSE | A forecast-origin positionから積分再構成、B absolute position直接予測、C secondary延期。原典確認までC | 起点・Δt・scale不足。文案:「position=[承認再構成式]、起点=[artifact field]」。既知軌跡と端点検証 |
| Acceleration RMSE | A予測velocityから同じ差分演算、B加速度直接予測、C延期。Aは表現決定後 | h=1独立予測の連結が必要。文案:「差分端点/Δt=[承認規則]」。不等間隔analytic fixture |
| Temporal correlation | A component別Pearson＋承認集約、B rank correlation、C延期。Aを候補にするだけ | constant series・vector集約。文案:「family=[承認]、constant=[unevaluable等承認規則]」。定数・NaN・逆相関test |
| Peak timing | A事前定義signalのpeak、B各component、Coptional延期。Cを推奨 | tie/no peak/閾値が結果依存化。文案:「signal/peak/tie/no-peak=[承認]」。複数同高峰・無峰fixture |
| Onset timing | A固定閾値/基線/hysteresis、B既存検証済みdetector、Coptional延期。Cを推奨 | baseline leakage・任意閾値。文案:「baseline/threshold/hysteresis=[承認]」。無onset・再crossingfixture |
| Lag preservation | A指定pairのlag estimator、Bselected-link一致指標、Csecondary延期。原典確認までC | GTなしの意味づけ。文案:「pairs/estimator/score=[承認]」。known-lag synthetic testと実データでの限定解釈 |

G-13/G-14を延期しても既存のG-core gateを取り消さない。G-10/11/12/15の必須性は原典との対応で決め、F14に未定義metricを埋めない。

### D-10 — Matched sparsity 1000回とv6 100回

**Decision:** 最新計画の1000回をどのprotocol版で受け入れるか。  
**Why it matters:** null分布の精度、計算量、再現性、既存証拠の意味が変わる。  
**Options:** A 新計画に合わせ1000へ明示migration、B 計画を科学レビューで100へ再改訂、C 新旧を混在。  
**Recommended option:** A。Cは禁止。  
**Rationale:** 計画§7.4は「1000に固定」と繰り返し明記する。最新目的に合わせる最小の移行である。  
**Risk:** matched実行量が現freeze比10倍。既存単発DRYで許容コストを判断できない。  
**Freeze text（案）:** 「承認migration [ID] によりmatched-sparsity repeat_countを100から1000に変更する。全outer fold・対象unitで1000個の一意なreplicateを生成し、候補集合・seed規則・集約を事前固定する。旧100反復を新1000反復の完了証拠としない。」  
**Implementation consequence:** config/schema/validator、反復実行、count完全性、artifact schema、計算budget。  
**Validation:** 1000 unique、999/重複/欠落を拒否、repeat seed再現、同設定で同digest。実1000反復は承認後のみ。  
**Requires user/scientific decision:** YES。新値を本監査で採用していない。

### D-11 — Landscapeのcellと集約

**Decision:** Ωのsource/targetはregionかscalar componentか、Self+(i,τ)のfeature単位、K/L/h/次元、source→target/lag帯域の集約。  
**Why it matters:** D=2のregion vector追加は単一scalar追加と違う。候補数とmatched sparsity数が変わる。  
**Options:** A exact scalar-component cellを基本に承認region projection、B region vector bundleをcellとしscalar countも併記、C outcomeで良いcomponentを選択。  
**Recommended option:** 既存Core v3を活かすAを候補とする。ただし研究計画がregion vectorを意図するならBへ明示変更。Cは禁止。  
**Rationale:** component provenanceを失わず曖昧な「1 feature」を解消できる。  
**Risk:** Aを自動採用するとregion-level estimandを狭める。Bは入力数増加・matched数の再定義を伴う。  
**Freeze text（案）:** 「Ωは[承認region/component仕様]、i≠j、1≤τ≤L、τ≥hを満たす全候補。各cellの追加featureは[承認定義]、全cellのλはouter-train内で決定する。Gを[承認重み/演算]でsource→targetおよび事前帯域[承認境界]に集約する。個別cell順位は探索的で、候補変更には使わない。」  
**Implementation consequence:** cell ID/schema、generic design matrixの再利用、grid digest、cell単位予測・support・alpha記録、新出力。  
**Validation:** Ωを手で列挙できる小fixture、D>1、i=j除外、tau>=h、欠測cell、unit-count、held-out値改変でもgrid不変。  
**Requires user/scientific decision:** YES。帯域cutpoint・K・componentを推測しない。

### D-12 — Selection enrichmentのestimand

**Decision:** Aggregate(G_selected)−Aggregate(G_matched)が個別cellのG集約か、集合をjointに再学習した予測gain差か。  
**Why it matters:** 冗長/相互抑制featureではcell gain平均とjoint-set gainは一致しない。  
**Options:** A 計画式どおりcell Gの集合集約、B selected/matched各joint modelのgain比較、C 両者を別名で事前指定。  
**Recommended option:** Cを検討し、計画式に対応するAを主推論として保持、Bは既存Null比較として明示。最終Primary選択は科学レビューで決める。  
**Rationale:** 現T07を新定義と誤認せず既存比較も再利用できる。  
**Risk:** 主推論を増やすためfamilyのD-08と依存する。計算量・set countの単位が変わる。  
**Freeze text（案）:** 「enrichmentは[承認A/B]と定義し、Aggregateは[承認演算/重み]。matched集合はouter-train由来の同一候補空間・同じ承認feature数でN_matched反復。被験者内の反復集約と被験者間推論は[承認順序]。joint-set gainを報告する場合は別metric IDとする。」  
**Implementation consequence:** set ID/replicate ID、selected membership、cell→set join、subject効果分布、必要なら既存joint Null再利用。  
**Validation:** cell/jointが異なるsynthetic fixtureで別値・別名、空selected、D>1、matched count・candidate space・1000/承認数、seed・supportを検証。  
**Requires user/scientific decision:** YES。既存v6 repeat medianと新Aggregateの関係を明示する。

### D-13 — Edge-centered population responseとbootstrap

**Decision:** E(e,τ)で変える入力、他parentの扱い、edge identity、Δgrid、境界除外、median_e,sの重み、CIのresampling unit。  
**Why it matters:** 全親同時shiftはedge別curveと異なる。edge×subjectを独立反復にするとsubject独立性を壊す。  
**Options:** A Self+当該edgeのみを変化、B selected set内で当該edgeのみ変化し他を固定、C 旧全親同時shiftを新curveと呼ぶ。  
**Recommended option:** 計画§5.5のcell Gと整合するAを候補、Bは条件付きedge寄与として別estimand。Cは不採用。  
**Rationale:** edge-centered実体を保持し、R-17のcell予測を再利用できる。  
**Risk:** A/Bは異なる仮説。median_e,sはedge数の多いsubjectに重みがかかる。subject-first medianへの変更は式変更。  
**Freeze text（案）:** 「各e=(i,j,τ*)のEは[承認A/B]、Δは[承認対称grid]、無効edgeは[承認完全grid規則]。点推定は計画式median_e,sを[維持/承認変更]し、CIはsubjectをclusterとして全edge曲線をまとめてresampleする[承認方式/回数/seed]。旧set-shift曲線は別IDで保持する。」  
**Implementation consequence:** edge/fold/subject/Δ artifact、support ledger、cluster bootstrap、old/new output IDs。既存paired bootstrap10000を自動流用せず適用可否を記録。  
**Validation:** 2親が異なるlagのfixtureで全親shiftとの差を検出。境界edge、empty set、same support、同一subjectのedge追加による疑似n増加防止、計画式の手計算一致。  
**Requires user/scientific decision:** YES。新curveに旧[-2,-1,0,1,2]を引き継ぐかも明示承認。

### Conflict records

| Conflict ID | Document A | Document B / implementation | Conflict | Impact | Resolution / User decision |
|---|---|---|---|---|---|
| C-01 | Freeze v6 repeat_aggregation=median | analysis table_t07=mean | 演算が違う | HIGH | v6に合わせ修正。NO（新科学値なし） |
| C-02 | #23 exact same-unit support | _exact_pair ignores n_valid | 同一unit名でも異なるframe集合を許可 | HIGH | support証拠で照合。support選択規則自体はD-05 YES |
| C-03 | #22 real execution requires valid #20 manifest | T09 checks bool | 図表入口の証拠が弱い | HIGH | 共通validator再利用。NO |
| C-04 | #23 Primary-only outputs | sensitivity.csv always required | 禁止された時系列依存 | MEDIUM | 入力を条件付きにする。NO |
| C-05 | DRY old remaining_issue | Freeze v6 | 既に固定済み数値を未固定と記載 | LOW | v6優先、記録同期。NO |
| C-06 | 既存G統計laneのregion別結果 | T04 region mean→subject median | 原典にも全体集約のoperational definitionがない | HIGH | D-06。YES |
| C-07 | Issue #4 v1 | Core v3 explicit migration | 版違い | LOW | v3を新規runに採用。NO |
| C-08 | 新計画§7.4 repeat_count=1000 | Freeze v6=100 | 明示固定値の競合 | CRITICAL | D-10、明示migration。YES |
| C-09 | 新計画§5.7 edge-centered median_e,s | v6 common_shift_all_selected_parents | estimand/集約単位が違う | CRITICAL | D-13、新旧別ID。YES |
| C-10 | 新計画§5.1 vector region / §5.5 single feature | Core v3 exact scalar component | cellと疎性カウントが曖昧 | HIGH | D-11。YES |
| C-11 | 新計画§5.6 Aggregate(G) | T07 joint-null error difference | 同値と限らない | CRITICAL | D-12。YES |
| C-12 | 新計画§13 GRU必要時のみ | Issue #21全6実験必須 | optional範囲の競合 | MEDIUM | R-01で採否。YES |

## 7. Dependency Graph

~~~text
R-01 新計画/v6差分・Issue・migration
 ├─ R-02 data/split/表現 → R-04 manifest/enforcement
 ├─ R-03 統計/主推論/追加freeze
 ├─ R-05 support → R-06 repeat/metric/seed
 └─ R-07a manifest guard → R-07b Primary-only入力
R-02/R-03/R-05 → R-17 全Ω landscape
                     ├─ R-18 selection enrichment
                     └─ R-19 edge-centered population response
R-04/R-06/R-17〜19 → R-09 本番runner
R-08 新旧raw→analysis export（schema確定後R-09と並行可）
                         ↓
                  R-10 DRY/I4
                         ↓
                  R-11 Primary full run
                         ↓
                  R-12 全Primary STAT
                         ↓
                  R-13 Primary result freeze
                         ↓
                  R-14 承認Sensitivity
                         ↓
                  R-16 個別出力受入
~~~

実行前Scientific Freezeと実行後Primary result freezeは別工程。新3解析は実行前に仕様固定。
R-15追加metricは採否を承認した分のみ依存へ追加する。

| Task | depends_on | blocks | can_run_parallel_with / Effort |
|---|---|---|---|
| R-01 | 原典・証拠（READY） | 全migration | なし / M |
| R-02 | R-01 | R-04/05/17 | R-03 / M |
| R-03 | R-01,D-08/10–13承認 | R-06/17/18/19 | R-02 / M |
| R-04 | R-02 | R-08/09 | R-07 / L |
| R-05 | R-02,D-05/13 | R-06/17/19 | metadata整理 / M |
| R-06 | R-03/05 | R-08/09/18 | R-04 / M |
| R-07a | R-01 | R-14/16-Sensitivity | R-04 / S |
| R-07b | R-07a | R-08/16 | R-04 / S |
| R-17 | R-02/03/05,D-11 | R-18/19/09 | R-07別編集面 / L |
| R-18 | R-17/06,D-12 | R-09/12 | R-19、schema確定後 / M–L |
| R-19 | R-17/05,D-13 | R-09/12 | R-18、schema確定後 / L |
| R-08 | R-04/05/06/07b/17–19 schema | R-10/12/16 | R-09 / L |
| R-09 | R-04/05/06/17–19 | R-10 | R-08 / L |
| R-10 | R-08/09 | R-11 | optional設計 / M |
| R-11 | R-10 | R-12 | 結果アクセス集中管理 / XL |
| R-12 | R-11 | R-13/16-Primary | 図表準備 / L |
| R-13 | R-12 | R-14 | Primary読取図表 / M |
| R-14 | R-13/07a | R-16-Sensitivity | 別run root / 各M–L |
| R-15-metric | 個別承認 | 対応出力 | schema競合なければ / 各S–M |
| R-16-output | R-12またはR-14 | 最終報告 | 別出力 / 各S–M |

同じanalysis_pipeline.pyや未確定契約を並列編集しない。Critical pathは新計画migration→新解析→integration→本実験。旧8条件を先に走らせてから新gridを決める順序は不可。

## 8. Recommended Execution Order

1. **Scientific migration:** R-01、R-02/R-03。1000/100・新estimand・component・集約・主推論を確定。
2. **Core contract:** R-04/05/06/07a/07b。現v6不一致の修正と承認migrationを区別。
3. **新主解析:** R-17 landscape → R-18 enrichment / R-19 population curve。
4. **Integration:** R-08/R-09 → R-10。全候補・承認反復数・全解析まで検証。
5. **Primary:** R-11 → R-12。外側testでgrid・帯域・Null・集約を変えない。
6. **Result freeze / Sensitivity:** R-13 → R-14。surrogateで新主解析も検証。
7. **Reporting:** 各R-16-output。旧23出力に新3解析の成果契約を追加。旧F04を別estimandへ黙って置換しない。

## 9. Detailed Task Specifications

### 共通実装・受入ルール

以下の将来Taskでの **N_matched** はD-10で承認する反復数を指す。現行v6=100、新計画=1000の両方を証拠として保持する。旧仕様の説明に100とあっても、新本番Taskで100を自動採用する意味ではない。R-01で新計画への明示1000回migrationを推奨し、採用記録後に実装する。

- 実装前にgoverning Issueを確定。既存 #18/#19/#20/#21 を再利用できる実行は新規重複Issueを作らない。新しい不具合はF-IDごとに責任を分ける。
- current devからtask branch、PR to dev、CI、独立review、修正、merge、post-merge validation。scientific-semantic changeをPRに明記する。
- commit/pushする場合は指定Ponytail review gateを守る。対象repoにgate scriptがないことを理由に存在しないコマンドを捏造しない。適用する運用をR-01で明文化する。
- 以下の `python ...` は対象repoの宣言依存を導入した環境で実行する標準形。今回確認した既存CLI以外は「提案CLI」と明記した。
- テストのPASSだけで実行Taskを閉じない。実行Taskはrun ID、resolved config hash、git SHA、seed、manifest hash、artifact、failure/unevaluable counts、audit dispositionが必要。
- frozen値・閾値を変更してテストを通さない。承認済み仕様と実装を合わせる。
- 実runの失敗は保存し、対象fold/subjectを勝手に落とさない。real outputをmock outputに置き換えない。

### R-01 — 新計画とv6の差分をIssue・migration planへ確定

**Purpose / Source:** Q01、F-01/F-17、全Issueの前提を一意にする。  
**Current problem:** 最新計画を受領したがrepo未登録。repeat・lag単位・新主解析に差分。新所見は未Issue化。  
**Required decision:** D-01/D-10〜13。  
**Inputs:** 本書、124要求、現Issue snapshot、正式原典のURL/path/版。  
**Files likely affected:** README.md、提案 docs/research_protocol.md、監査台帳。  
**Implementation:** 文書整理のみ。F-ID→Q-ID→Issue→Task→受入証拠を表にする。既存Issue再利用/新規Issue案を区別。  
**Tests / Scientific checks:** link/版/節対応、v6/v3との矛盾、Primary hypothesis/endpoint/scope確認。  
**Execution:** read-only Git/Issue照合、承認済み原典の取り込み案作成。  
**Expected artifacts:** authority registry、decision log、Issue-ready findings、実装Plan。  
**Acceptance criteria:** 受領原典のhash/版/差分を明記、別研究を混入しない、F-IDに責任Issueがある、未採用科学値を明示、独立review可能。  
**Dependencies / Blocks:** なし / 全Taskのauthority。  
**Parallelizable / Complexity:** NO / M。  
**Stop:** 新旧仕様のdisposition未承認なら依存実装をdecision-blockedにする。既知のsoftware所見整理は継続可能。

### R-02 — データ・split・前処理の科学決定

**Purpose / Source:** Q17/18/22〜30/53/68、F-02、#18。  
**Required decision:** D-02〜05。  
**Inputs:** dataset inventory（結果を含まないmetadata）、extractor spec、原典。  
**Files:** 提案protocol文書、configs/scientific_freeze.yamlのmigration案、データmanifest schema案。  
**Implementation:** split/region/dimension/reference points/motion/time/quality/不足support規則を一つのprotocol版として確定。コードは決定後R-04。  
**Tests:** schema例の正負ケース設計、subject/session独立性、値の根拠対応。  
**Scientific checks:** outer-test outcomeに触れず選定。例示30Hz/landmark番号を採用しない。  
**Execution / Outputs:** 承認付きdecision record・freeze差分・受入条件。  
**Acceptance:** 各必須fieldの具体値またはアルゴリズム・version/hash・承認根拠がある。未決fieldでは実行不可。  
**Dependencies / Blocks:** R-01 / R-04/05/09。  
**Parallelizable / Complexity:** R-03と可 / M。データ提供が必要ならBLOCKED。

### R-03 — 推論・安定性の科学契約を確定

**Purpose / Source:** Q48/51/52/67/Q79〜101、F-03/F-20〜24、計画§5/7/8.1、#15/#19/#23。  
**Decision:** D-06/07/08/D-10〜13。  
**Inputs:** 原典、既存subject/region統計、stability raw graph契約。  
**Files:** protocol、scientific_freeze.yaml/schema、承認comparison registry案。  
**Implementation:** region/session集約、stability resampling/opportunity、主対比と多重比較に加え、新Ω、cell/component、enrichment、edge-centered応答の式とfreeze migrationを記述。  
**Tests:** 極端値/不均衡support fixtureの期待値、component→region count fixtureの期待分母。  
**Scientific checks:** paired bootstrap 10000/percentile/95%をdiscovery bootstrapへ誤転用しない。  
**Execution / Artifacts:** decision recordと実装可能な数式・演算順序。  
**Acceptance:** 同一入力の期待値を手計算でき、STATとT/Fが同じestimandを計算する。  
**Dependencies / Blocks:** R-01 / R-06/08/12/17/18/19。  
**Parallelizable / Complexity:** R-02と可 / M。

### R-04 — data manifestと承認protocolのenforcement

**Purpose / Source:** F-02、Q53、#18 PR-01。  
**Current problem:** 汎用APIの必須引数と実study値が接続されていない。  
**Decision:** R-02で完了していること。  
**Inputs:** 承認protocol、実landmark .npy/.npzの安全な所在、subject/session metadata。  
**Files:** configs/、schemas/、landmark_io.py、regions.py、spatial_normalization.py、splits.py、提案preflight入口。  
**Implementation:** immutable manifestを読み、hash・shape・timestamps・topologyを検査して既存関数へ渡す。結果を見てfallbackしない。  
**Tests:** file/hash mismatch、subject重複、split漏洩、範囲外landmark、axis退化、未知feature、time単位不足で拒否。  
**Execution:** 提案CLI `python scripts/run_primary.py --config configs/primary_run.yaml --preflight`（R-09で入口確定）。現時点でこのCLIは存在しない。  
**Artifacts:** data_manifest、resolved protocol、preflight report、split manifests。  
**Acceptance:** 全必須値が記録され、drift/欠落/CLI overrideを拒否。本番outer-test predictionを作らず検査可能。  
**Dependencies / Blocks:** R-02 / R-08/09。  
**Parallelizable / Complexity:** R-07と別編集面ならYES / L。

### R-05 — 同一評価supportをraw predictionまで強制

**Purpose / Source:** F-04、Q38/45、#23。  
**Decision:** D-05のsupport規則。  
**Inputs:** PredictionArtifact v3、Null/lag grid、承認support規則。  
**Files:** metrics.py、statistics.py、primary_statistics.py、analysis_pipeline.py、必要最小限のschema。  
**Implementation:** subject/region/target_dimensions/forecast_origin/target_time/valid rowsのcanonical support digestを作り、比較前に一致検証。countsだけで代用しない。  
**Tests:** 同数異時刻、不一致mask、target dimension入替、duplicate、片側欠落、全欠測を拒否。shared-intersection適用後は意図通り一致。  
**Scientific checks:** no silent dropping。bootstrap unitはsubject。  
**Execution:** `python -m pytest tests/test_subject_level_paired_difference.py tests/test_primary_condition_alignment.py tests/test_analysis_pipeline.py` と新回帰。  
**Artifacts:** support ledger、paired output、regression evidence。  
**Acceptance:** 診断100 vs 1を拒否。同数別supportも拒否。正しいpairは既存効果量を保存。  
**Dependencies / Blocks:** R-01/02 / R-06/08/09。  
**Parallelizable / Complexity:** 統計面の他Taskとは逐次 / M。

### R-06 — 解析とScientific Freezeの統計契約を統一

**Purpose / Source:** F-05/F-06/F-13、Q41/42/50/66。  
**Decision:** D-06/D-10/D-12。現v6のmedian違反と新enrichment定義を区別し、承認migration後のN_matchedを使う。  
**Inputs:** 承認migration後freeze、承認集約、N_matched反復raw metric。  
**Files:** analysis_pipeline.py、primary_statistics.py、統計reducer、analysis export config。  
**Implementation:** repeat中央値を先に計算。承認N_matched unique repeatsと全unit・condition完全性を検証。primary_metric/seedをfreeze由来必須値にしfallbackを除去。既存計算を再利用。  
**Tests:** [0,0,9]相当の承認N_matched反復skew fixtureでmeanとの差を検出。N_matched−1/重複/不足unit/指標差替え/seed欠落を拒否。  
**Scientific checks:** 数値規範を変更しない。region集約はR-03承認後。  
**Execution:** 既存統計/analysis/null tests＋新回帰、raw→T07/F05一致検証。  
**Artifacts:** 正規のpaired/null統計、同一値照合表。  
**Acceptance:** T07/F05/STATでmedian一致、承認反復数完全性、固定metric/seed、同じsupport。  
**Dependencies / Blocks:** R-03/05 / R-08/10/12。  
**Parallelizable / Complexity:** R-04と可 / M。

### R-07a — real Sensitivity解析のmanifest barrier

**Purpose / Source:** F-07、Q61、#20/#22/#23。  
**Required decision:** なし（既存境界の強制）。  
**Files:** analysis_pipeline.py、scripts/generate_analysis_outputs.py、sensitivity_execution.pyを再利用。  
**Implementation:** boolean自己申告を実行証拠にせず既存manifest validatorを呼ぶ。元Sensitivity artifactのPrimary reference/hashと照合。  
**Tests:** missing/invalid/incomplete/hash改変manifest、別run referenceを拒否。syntheticは明示mock例外＋非科学表示を保持。  
**Execution:** `python -m pytest tests/test_sensitivity_freeze_barrier.py tests/test_analysis_pipeline.py`。  
**Artifacts / Acceptance:** manifestなし診断が拒否される。正しいpost-freeze fixtureだけ通る。Primary出力は上書きしない。  
**Dependencies / Blocks:** R-01 / R-14/16-T09/F10。  
**Parallelizable / Complexity:** R-07bとは逐次 / S。

### R-07b — Primary-only解析の独立入力

**Purpose / Source:** F-08、Q62、#23。  
**Required decision:** なし。  
**Files:** AnalysisInputs、generate_analysis_outputs、CLI、tests。  
**Implementation:** include_sensitivity=Falseならpath解決・read・provenance検査・input hash対象からSensitivityを外す。空の偽Sensitivity CSVを作る回避策は禁止。  
**Tests:** sensitivity.csvを持たないfixtureでPrimary21出力成功。include=Trueなら入力とbarrierを必須化。既存fileを読み取っていないこともspyで検証。  
**Execution:** `python -m pytest tests/test_analysis_pipeline.py`。  
**Artifacts / Acceptance:** T01〜T08、F01〜F09/F11〜F14が独立生成され、T09/F10なし。manifest sensitivity_included=false。  
**Dependencies / Blocks:** R-01/07a / R-08/16。  
**Parallelizable / Complexity:** NO（同一surface）/ S。

### R-08 — raw scientific artifactからanalysis入力を生成

**Purpose / Source:** F-09/F-18/F-19、Q65/75/78、#23。  
**Inputs:** core v3 PredictionArtifact、ParentSet、NullMapping、split、quality、software/seed provenance。  
**Decision:** R-02/R-03でmetric/region/stability schemaの意味が確定。  
**Files:** 最小のexport module/script、analysis input schema、analysis_pipeline.py。  
**Implementation:** 各CSVをrawから生成し、run ID・git SHA・resolved config/freeze hash・source hash・support digestを結び付ける。canonical configからanalysis configを生成し手編集しない。  
**Tests:** test helperではなくproduction export→analysisのsynthetic integration。CSV改変・source hash欠落・dimension消失を拒否。mockへ指定2文字列を併記。  
**Execution:** 提案CLI `python scripts/export_analysis_inputs.py --manifest <validated-run-manifest>`（未実装）。既存解析CLIへ接続。  
**Artifacts:** dataset_summary、primary_config、metrics、feature_counts、null_metrics、lag_response、edge_stability、prediction_trajectory、必要時sensitivity。  
**Acceptance:** raw→reducer→CSV→figureの全経路を追跡、hashと数値を再計算できる。mockとrealを混在させない。  
**Dependencies / Blocks:** R-03/04/05/06/07bとR-17〜19の出力schema / R-10/12/16。  
**Parallelizable / Complexity:** R-09とはcontract固定後 / L。

### R-09 — 全Primary条件のproduction runner

**Purpose / Source:** F-10/F-13、Q54/64、#18。  
**Inputs:** R-04 protocol/data、split、既存discovery/design-matrix/Ridge/null/registry/lock。  
**Files:** runner.pyまたは最小composition module、提案scripts/run_primary.py、runner config、integration tests。  
**Implementation:** existing partsを構成。全target・fold・condition・新解析、承認N_matched repeats、lag-grid evaluability、承認alpha grid、frozen state、resumeを扱う。test内_run_one_foldをproductionからimportしない。  
**Tests:** 小規模syntheticの全経路、承認反復count、empty parents/boundary、failure injection、resume identity。  
**Scientific checks:** outer-train discovery/tuning/null、fold freeze前outer-test不可、exact component、h=1、同一forecaster、no sensitivity。  
**Execution:** 提案 `python scripts/run_primary.py --config configs/primary_run.yaml --preflight` とsynthetic integration入口。CLI名はPRで最終確定。  
**Outputs:** per-fold preprocessing/ParentSet/alpha/Null/freeze/predictions/metrics、run manifest、failure ledger。  
**Acceptance:** 本番入口をtestsから呼べる。例示gridでなく承認protocolを使う。失敗・retryで選択状態が変わらない。  
**Dependencies / Blocks:** R-04/05/06/R-17〜19 / R-10。  
**Parallelizable / Complexity:** R-08とcontract固定後 / L。

### R-10 — 実行protocolのDRY / I4受入

**Purpose / Source:** F-14、Q55/56、#16/#17。  
**Inputs:** R-09の実入口、R-08のexport、承認protocol、隔離validation fixture。  
**Implementation:** 新アルゴリズムなし。test compositionを本番compositionへ置換して監査を配線。  
**Tests:** 同config/seedのsynthetic runを2回、承認N_matched全反復、全許可条件、hash/ParentSet/alpha/predictions/metrics一致。failure injection/resumeとsupport一致を確認。  
**Scientific checks:** 本番データDRYが必要なら対象と再利用方針を事前指定し、同じouter-testを見て再調整しない。syntheticからREAL DATA VERIFIEDへ昇格させない。  
**Execution:** 既存DRY/I4 tests＋production integration。declared environmentで全suiteとCI。  
**Artifacts:** run ID・SHA・config/manifest hash付きDRY/I4 report、検証scopeラベル。  
**Acceptance:** 全機械監査PASS、未固定fieldゼロ、独立review、#16/#17に正しいscopeで証拠。  
**Dependencies / Blocks:** R-08/09 / R-11。  
**Parallelizable / Complexity:** NO / M。

### R-11 — Primary Full Run

**Purpose / Source:** F-15、Q57、#18 PR-01〜10。  
**Input / Configuration:** immutable data manifest、承認済みprotocol、全split、実行SHA・software lock。  
**Execution command:** R-09で確定する本番CLI `python scripts/run_primary.py --config configs/primary_run.yaml`（現在は未存在、preflightを通った版のみ）。  
**Implementation:** 実験実行と記録のみ。  
**Tests / Scientific checks:** 起動前R-10、foldごとleakage/freeze/support、全8条件＋新3解析、matched N_matched、lag全grid。  
**Expected output:** 全foldの選択状態・予測・metric・failure/unevaluable ledger・manifest。  
**Validation criterion:** 予定したfold/target/condition/repeat集合に対し成功・承認unevaluable・失敗が全件accounted。未完了を成功扱いしない。  
**Failure condition:** hash drift、未固定値、漏洩、欠落条件、科学状態を変えるresumeは停止。性能低下そのものは実装FAILにしない。  
**Acceptance:** #18の全受入条件、run ID/config hash/git SHA/result artifact/audit dispositionを記録。  
**Dependencies / Blocks:** R-10 / R-12。  
**Parallelizable / Complexity:** NO / XL（被験者数・計算量未定）。

### R-12 — Primary Statistics & Final Analysis

**Purpose / Source:** Q58、#19、Q46〜52。  
**Inputs:** R-11 frozen predictions、null repeats、ParentSet、stability、新landscape/enrichment/population raw artifacts。  
**Configuration:** v6のsubject median・percentile10000・95%、R-03の追加承認規則。  
**Implementation:** 既存統計APIを再利用し、本番orchestrationとexportを実行。  
**Execution:** 提案 `python scripts/run_primary_statistics.py --manifest <primary-run-manifest>`（未実装入口）。  
**Tests / Scientific checks:** Self/Full/matched/random/lag/time-shuffle全比較、effect sign、same support、counts、bootstrap seed、stability分母、事後選択なし。  
**Outputs:** subject effects、region effects、median/CI、null distribution、lag response、selection stability、failure分析、新G landscape/source→target/lag-band/enrichment/edge-centered population curve。  
**Acceptance:** 手計算fixture・raw再集計・T/F source値との一致。#19の全比較とprovenanceを記録。  
**Failure:** 欠落・unmatched・別config/run混在なら停止し、結果を見て設定を変更しない。  
**Dependencies / Blocks:** R-11/R-08 / R-13/16-Primary。  
**Parallelizable / Complexity:** NO（統計状態確定）/ L。

### R-13 — PRIMARY FREEZE作成・検証

**Purpose / Source:** Q59/60、#20、F-11。  
**Inputs:** 完了したR-11/R-12、既存13種と新landscape/enrichment/populationのrequired artifact。  
**Files:** 最小freeze writer、既存artifact_registry/failure_handling/sensitivity_execution。  
**Implementation:** required types全て・全実行unit・新3解析の完全性を検査し、承認migration後schemaでmanifestをatomicに生成。現manifest schema v1のrequired setは旧13種なので、新成果のrequired type/構成とconsumerの検証を明示version migrationする。旧manifestを新研究完了として受理しない。hashがあるだけの空dummy fileを受入証拠にしない。  
**Execution:** 提案 `python scripts/freeze_primary.py --manifest <validated-primary-manifest>`（未実装）。  
**Tests:** 欠落type、file欠落、hash改変、重複path、別run、未完了fold、overwriteを拒否。consumer round-trip。  
**Artifacts:** artifacts/primary/freeze_manifest.json、integrity audit。  
**Acceptance:** 既存13種＋新3解析＋全予定単位を参照、same inputで同digest、freeze後上書き不可、独立review、#20 PASS。  
**Failure:** incomplete scienceをPASSにしない。  
**Dependencies / Blocks:** R-12 / R-14。  
**Parallelizable / Complexity:** Primary読取図表と可 / M。

### R-14 — Sensitivity実行（承認した独立run群）

**Purpose / Source:** Q63、#21、#22の実装済み部品。  
**Inputs:** R-13 manifest、各methodの事前承認config、同じsplitを含むprovenance。  
**Tasks:** R-14-GPDC、R-14-LPCMCI、R-14-phase、R-14-circular、R-14-horizon、R-14-GRUを各々独立受入。GRUは新計画でoptional、Issue #21では必須なのでR-01で採否を整合。surrogateは新enrichment/population構造の検証にも接続。  
**Implementation:** 新手法を再実装せず既存adapterをorchestrate。各設定（h、shift規則、GRU訓練等）を実結果前に承認。  
**Execution:** method別のproduction入口は現状未確認。既存公開APIから作るthin CLIを個別PRで確定し、その --help とmanifest guardを確認してからrun。架空コマンドで実行済みとしない。  
**Tests / Checks:** guard、独立seed/version、Primary hash不変、h>1 availability、null session boundaries。  
**Outputs:** artifacts/sensitivity/ 配下のmethod別run manifest・predictions/metrics・paired effect。  
**Acceptance:** 採用された各行に実run ID・config/seed/SHA・source Primary freeze hash・失敗状況を記録。1成功で6件完了にしない。  
**Failure:** Primary改変、manifest不一致、未承認設定は停止。  
**Dependencies / Blocks:** R-13/R-07a / R-16-T09/F10。  
**Parallelizable / Complexity:** 承認後は異なるoutput root間で可 / 各M〜L。

### R-15 — 追加metricの独立Task群

**Purpose / Source:** Q69〜74、#15 G-10〜15、D-09。  
**Task IDs:** R-15-G10、R-15-G11、R-15-G12、R-15-G13、R-15-G14、R-15-G15。  
**Inputs:** 個別operational definitionと必要provenance。  
**Files:** metric module、必要最小schema、個別tests、metric registry。  
**Implementation:** D-09表の承認式のみ。基底motion_featuresの存在をmetric実装済みと扱わない。  
**Tests:** D-09のmetric別analytic/negative fixtures、units・missingness・support。  
**Execution:** 個別 `python -m pytest tests/test_<approved_metric>.py`（新test名は実装PRで確定）。承認済みreal predictionsに適用。  
**Outputs:** 各metricのsubject-region結果、provenance、NA理由。  
**Acceptance:** 文書・実装・test・実runが一致、結果を見た閾値調整なし。  
**Dependencies / Blocks:** R-01/02/03＋個別承認 / 当該出力のみ。  
**Parallelizable / Complexity:** schema競合がなければ可 / 各S〜M。peak/onsetの延期を明示可能。

### R-16 — 出力を1件ずつ科学的受入

**Task IDs:** R-16-T01〜T09、R-16-F01〜F14。これは23個の独立Taskの共通仕様で、単一の「23件完了」Taskではない。  
**Purpose / Source:** #23 completion contract、matrix各T/F行。  
**Inputs:** R-08で生成したreal analysis input、R-12確定統計。T09/F10はR-14も必須。  
**Files:** 個別output、registry/manifest、tracker該当1行。  
**Implementation:** 既存生成関数を用いる。不具合なら当該所見として別Issue化し、数値を手編集しない。  
**Execution:** 既存 `python scripts/generate_analysis_outputs.py --input-dir <validated-analysis-input-dir>`。Primaryはinclude-sensitivityを指定しない。SensitivityはR-07aで確定したmanifest付き引数を使う。現在のboolのみの入口で公開しない。  
**Tests:** 該当出力の契約tests、raw数値との照合、source CSVとPNG/SVGの値・ラベル・単位・凡例・CIを確認。  
**Scientific checks:** effect sign、固定metric、同一support、未評価分母、subject unit、no post-hoc selection。  
**Expected output / per-output criterion:**

| Output task | Input → output | 独立した受入条件 |
|---|---|---|
| T01 | data/split/quality → dataset summary | 予定/使用subject・frame・fold・欠測数がmanifestと一致 |
| T02 | resolved protocol → config table | freeze hashと全承認設定が一致 |
| T03 | predictions→metrics → 4条件表 | 四条件のunit・metric・median/CI一致 |
| T04 | paired effects → Self−PCMCI | 承認region集約・subject median/CI・sign一致 |
| T05 | Full/PCMCI+feature counts → sparsity | exact pairとfeature ratio一致 |
| T06 | region paired effects → region表 | region別subject分母・CI一致 |
| T07 | 全Null repeats → Null表/分布 | 承認N_matched反復median、全Null条件・seed・分母一致 |
| T08 | raw discovery boot/folds → stability | component→region規則、selected/opportunity、zero/NA一致 |
| T09 | post-freeze Sensitivity results → summary | 承認run群のscope・Primary参照hash・metric/horizon区別 |
| F01 | T03 source → condition plot | subject points/median/CIがsource値と一致 |
| F02 | T04 source → paired effect | zero基準・sign・subject点・CI一致 |
| F03 | T06 source → region effects | region順序・CI・解釈一致 |
| F04 | lag response → curve | 5delta同じsupport、frame/ms、unevaluable件数 |
| F05 | T07 source → Null効果 | repeat集約とeffect signがT07一致 |
| F06 | T05 source → sparsity plot | x/y単位、ratio、同じsubject support |
| F07 | T08 projection → region heatmap | opportunity規則、NAとzero、軸label |
| F08 | T08 lag projection → heatmap | tau=1..10、region/lag順、NAとzero |
| F09 | paired source → fold分布 | foldを独立subject反復と呼ばない |
| F10 | T09 source → forest | manifest verified、h>1別estimand、方向・CI |
| F11 | raw trajectory → example | 結果に依存しないlexicographic選択、時間・componentを明示 |
| F12 | quality source → diagnostics | threshold/欠測/活動量の意味と分母一致 |
| F13 | primary errors → distribution | subject単位・Self/PCMCI同じ対象 |
| F14 | 承認metric registry → concordance | 未定義G指標を入れず、direction normalizationとCIを照合 |

**Acceptance（各行）:** 実artifact生成、source hash、数値検算、目視確認、独立reviewを記録した後だけtrackerをCOMPLETEにする。  
**Dependencies / Blocks:** PrimaryはR-12、SensitivityはR-14 / 最終報告。  
**Parallelizable / Complexity:** 別出力なら可 / 各S〜M。  
**Failure condition:** source mismatch、欠落証拠、誤ラベル、未定義metricはpublication_ready=false。mockをrealに昇格しない。

### R-17 — 固定候補全体のpredictive gain landscape

**Purpose / Source requirement:** Q79〜87/Q96/Q99、計画§3/5.5/7.3 A,D/8.1。  
**Current problem:** Full-historyの一括入力モデルは存在するが、全候補Self+cellを別々に学習する経路、cell G、source→target/帯域集約がない。  
**Required decision:** D-11、D-04/05、D-08の記述/主推論区分。  
**Inputs:** 承認Ω、subject split、Self-history、resolved feature/component定義、Ridge調整規則、support。  
**Files likely affected:** design_matrix.pyの再利用、提案landscape composition module、cell schema、R-08 export、tests。  
**Implementation:** Ωを結果と独立に列挙しdigest固定。各cellのSelf+feature modelをouter-train/inner CVのみでfitし、全cell選択状態をfreeze後にouter-testを評価。G=E_Self−E_Self+cellをsubject別に保存。承認規則でsource→target/lag-band/sign consistencyを集約する。新回帰器を作らない。  
**Tests:** 小さな手動Ωの完全性、同じSelf reference、exact component、feature ordering、tau>=h、subject分割、欠落/重複cell、外側値改変時にΩ/α決定が不変、同一seed再現。  
**Scientific checks:** 個別cell最大値をPrimary設定に使わない。cell gainをPCMCI conditional link strengthや物理因果と同一視しない。NaN/unevaluableの分母を保持。  
**Execution:** 本番CLIはR-09へ接続する提案。実装PRでsynthetic fixture入口を確定し、全候補テスト後にのみR-11で実データ実行。既存run_primary.pyがあるとは主張しない。  
**Expected artifacts:** candidate_grid.json、cell fit/alpha provenance、cell predictions、cell_gain table、source_target_gain、lag_band_gain、sign_consistency、support/unevaluable ledger。名称は新Issueで確定。  
**Acceptance criteria:** 全Ω候補が成功/承認unevaluable/失敗としてaccounted、raw predictionからGを再計算できる、同一support、result-driven選択なし、core v3 component provenanceを保存。  
**Dependencies / Blocks:** R-02/03/05、D-11 / R-18/19/09/12。  
**Parallelizable / Estimated complexity:** R-07の別編集面ならYES / L。  
**Real-run closure:** 実装Issueはsynthetic integrationまで。研究受入はR-11/R-12/R-16-LANDSCAPEでrun ID・hash・SHA・結果を記録。  
**Out of scope:** 新discovery手法、事後grid最適化、文献noveltyの拡張。

### R-18 — PCMCI selection enrichment

**Purpose / Source requirement:** Q88〜90、計画§5.6/7.3 B/8.1、F-22。  
**Current problem:** T07のNull error differenceだけではAggregate(G)を実装したことにならない。  
**Required decision:** D-10/D-11/D-12、主推論family D-08。  
**Inputs:** R-17 cell G、outer-train選択membership、matched candidate space、承認N_matched集合、support、Self reference。  
**Files likely affected:** 最小enrichment reducer、null_matched_sparsity.pyの再利用、set/membership schema、export/analysis、tests。  
**Implementation:** selectedとmatchedを同じcell/feature単位で構成。承認Aggregateでsubject単位のenrichmentを算出し、replicate全分布を残す。joint-set gainを併記する場合は別metric IDで既存Null modelを再利用。  
**Tests:** redundant/interacting featureによりcell aggregateとjoint gainが異なるfixture、同数matched、候補範囲、空selected、重複replicate、N_matched−1、別support、同seed/同集合。  
**Scientific checks:** outer-testでmatched spaceやrepeat数を変更しない。repeatを独立subjectとしてbootstrapしない。採用Primary estimandが事前registryと一致。  
**Execution:** synthetic composition→R-09→R-11/R-12。新CLI名は実装PRで決める。  
**Expected artifacts:** selected_membership、matched_membership+seed、per_subject_per_repeat_gain、enrichment_distribution、subject_enrichment、median/CI、candidate/support hash。  
**Acceptance criteria:** 承認数のunique repeat完全性、式からの手計算一致、cell/joint別名、全subjectの分母、T07との関係を説明、Primaryの再選択なし。  
**Dependencies / Blocks:** R-17/R-06、D-12 / R-09/R-12/R-16-ENRICHMENT。  
**Parallelizable / Complexity:** R-19とschema確定後のみYES / M–L。  
**Real-run closure:** #18/#19へ実enrichmentのrun ID・config hash・SHA・結果・disposition。  
**Out of scope:** 同じouter-testで最良selectorを選ぶこと。

### R-19 — Edge-centered population lag response

**Purpose / Source requirement:** Q91〜95、計画§5.7/7.4/8.1、F-23。  
**Current problem:** 旧v6の全親同時shiftはselected edge eごとの応答を分離しない。median_e,sの式・cluster CIも未実装。  
**Required decision:** D-13、D-05、D-08。  
**Inputs:** frozen selected edgesとτ*、R-17 cell predictions（D-13 Aなら再利用）、承認Δgrid/edge eligibility、subject support。  
**Files likely affected:** 新edge-response composition/reducer、null_lag_response.pyの既存validity処理の再利用、population schema、cluster bootstrap、analysis、tests。  
**Implementation:** edgeごとτ*+Δを評価し同じedge/subject referenceから差を計算。承認したmedian_e,s点推定を算出。CIは承認cluster単位で作る。旧set-responseを別IDとして残す。  
**Tests:** 2親・異なるτ*のfixture、全親同時shiftとの数値差、境界で完全grid欠落、空ParentSet、frame/ms、同一support、median手計算、subject内edge依存を保ったbootstrap再現。  
**Scientific checks:** edge数を独立subject nにしない。subject cluster CIは凍結discovery/学習状態に条件付けた不確実性か、学習変動も含めるかを明示。discovery bootstrapとouter-test effect CIを混同しない。  
**Execution:** synthetic known-lag/flat-band fixtures、R-09へ接続後R-11/R-12。  
**Expected artifacts:** edge_subject_delta_response、eligibility/exclusion ledger、population_response、cluster_CI、old_set_curve（実施時）、frozen grid/reference/provenance。  
**Acceptance criteria:** D-13承認式と一致、reference Δ0は同定義で0、全Δ共通support、zero/NA区別、旧F04を新結果と誤表示しない、lag-bandの解釈をcaptionに保持。  
**Dependencies / Blocks:** R-17/R-05、D-13 / R-09/R-12/R-16-POPULATION。  
**Parallelizable / Complexity:** R-18と共通schema確定後のみYES / L。  
**Real-run closure:** 実曲線・CI・件数・run ID/hash/SHAを #19/結果freezeへ含める。  
**Out of scope:** 成績を見てΔ範囲を拡大、都合の悪いedgeを除外、帯域の事後Primary化。

### 新研究目的の追加出力Task

旧23出力の一覧は最新計画全体の完了基準ではない。次をR-16の独立成果Taskとして追加し、正式table/figure番号はgoverning Issueで確定する。

| Task | Input → artifact | Acceptance |
|---|---|---|
| R-16-LANDSCAPE | R-17 G → 全cell表、source→target/lag-band集約、landscape可視化 | Ω完全性、G/sign/units、subject分母、探索的cell順位の明示、raw一致 |
| R-16-ENRICHMENT | R-18 → selected/matched分布、subject enrichment、CI図表 | cell/joint定義、N_matched、membership/seed、effect方向、family、Primary参照一致 |
| R-16-POPULATION | R-19 → edge-centered curve、CI、edge/subject/unevaluable数 | median/cluster単位、Δ0 reference、lag frame/ms、exact/帯域解釈、旧set curveと区別 |

各TaskはR-12の確定raw/統計がInput、既存source-table→plot/registry machineryを再利用、個別negative/source-reconciliation testsを実施する。
実行はR-08/R-16に接続する既存解析CLIの拡張案であり、現CLIに新3出力機能があるとはしない。
各出力にrun ID・SHA・freeze/config/source hash・件数・source値照合・目視review・dispositionを要求し、一つの成功で3つともCOMPLETEにしない。
Effortは各M、相互に異なる成果物なら並行可。研究上の新しい成功閾値を描画実装で追加しない。

## 10. Immediate Next Task

### NEXT TASK

**Task ID:** R-01  
**Task name:** 2026-09-09研究計画とFreeze v6の差分をgoverning Issueとmigration planへ確定

**Why this is next:** 原典は受領済み。1000/100反復、edge-centered/全親同時shift、新3解析欠落、component/region表現に実質差分がある。旧protocolの先行実行では最新主研究を満たさない。

**Prerequisites:** **READY**。原典・SHA・124要求・再現証拠が揃い、差分計画を開始できる。科学値の採用・migration実装・実データ実行は未許可。

**Exact work:**

1. DOCX本文改訂日とhashをregistryに登録し、§3–5/7.3–7.4/8.1/13をQ79〜101へ対応付ける。
2. C-08（1000/100）、C-09（edge/set shift）、C-10（region/component）、C-11（cell/joint gain）、C-12（GRU任意性）をdisposition付きで整理する。
3. R-17/R-18/R-19を別論理責任のgoverning Issue案にし、既存 #18/#19/#20/#23へ新主解析を接続するPlanを作る。
4. D-10〜13の選択肢・freeze文案を科学レビュー可能にする。最新計画1000回への移行を推奨するがv6を黙って変更しない。
5. F-04〜08の既存software不一致を、新仕様の未決事項と分けて回帰条件付きIssue案にする。

**Acceptance criteria:**

- [ ] 原典hash/改訂日/節をrepository planから追跡できる。
- [ ] 1000/100競合とmigration先が明示され、未承認値を採用していない。
- [ ] landscape/enrichment/population解析を旧T07/F04で実装済みと扱っていない。
- [ ] regionとscalar componentのcell/feature数を区別している。
- [ ] 各F-ID/Q-IDにgoverning Issue・Task・回帰条件・freeze impactがある。
- [ ] cell探索的、主推論enrichment/population、subject依存性、outer-test非選択を保存。
- [ ] 次の一件の実装Taskが独立review可能な粒度で確定。

**Stop condition:** 科学的disposition未承認の差分では依存実装を停止。未承認の1000回化・grid追加・point estimator変更・Primary/real Sensitivity実行はしない。既存違反の証拠・Issue案・Plan整理は進められる。
