## 目的

2026-09-09改訂研究計画に対する残存課題を、単独で着手・検証できるIssueへ分割する。基準コードはdev / 9bbc3d946c0a64935b8786d9413abe53e6cefe29。ローカルdocs/audit-report.md・experimental_plan.md・detailed_design.md・requirement-matrix.mdを根拠に、必要仕様を各Issue本文へ転記する。

## 確定している研究要件

Primary h=1、PCMCI+ / ParCorr / Ridge、matched sparsityは1000反復。4条件比較に加えて、全候補G landscape・cell-G selection enrichment・edge-centered population lag responseを必須とする。旧v6の100反復と全親同時shiftは移行元であり、新解析完了の証拠にはしない。

科学仕様の未決値は決定Issueに残す。データ・cell単位・Aggregate・E(e)文脈・統計規則を実装者が既定値で補わない。Issue化は科学値の採用や本実験開始を意味しない。

## 作業単位

基盤・仕様・新解析15件、追加指標6件、Sensitivity5件、旧図表23件、新図表3件を作成する。実験実行は既存 #18、最終統計は #19、実結果固定は #20、Sensitivity全体は #21 を再利用。#16/#17のdry-run・auditも維持する。GRUは必要時の任意拡張。

## 実行順

原典登録 → 科学仕様決定・config移行 → データ/support/統計/境界修正 → landscape → enrichment・population → export・runner → #16/#17 → #18 → #19 → #20 → #21。

各Issueには変更箇所、入力/出力、実装契約、回帰・完了条件、依存Issueを記載。決定結果は依存Issue本文にも具体値・根拠として同期する。実装Issueはソフトウェア検証で受入可能だが、実行・図表受入はrun ID・git SHA・config/source hash・実artifact・検算を必要とする。

## 一覧

作成後にIssueリンクを追記する。
