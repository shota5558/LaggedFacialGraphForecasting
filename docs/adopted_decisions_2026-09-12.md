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
