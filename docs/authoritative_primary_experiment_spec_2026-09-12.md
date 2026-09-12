# Authoritative Primary Experiment Specification

Updated: 2026-09-12  
Status: **AUTHORITATIVE / ADOPTED**  
Protocol ID: `primary-2026-09-12-adopted`  
Schema: `7`

This document is the scientific authority for the executable Primary protocol.
The detailed design is inherited from
`docs/primary_experiment_recommendation_2026-09-11.md`; the adopted decision
record summarizes the promotion. Where an older plan, Issue, config, or
Scientific Freeze v6 conflicts with this document, the older rule is
`SUPERSEDED`.

## Fixed rules

The Primary target is cross-region lagged predictive structure and its
cross-subject generalization in natural facial motion. The required analyses
are the full Predictive Gain Landscape, Persistence/Self/Full/PCMCI-block
comparison, 1,000-repeat Selection Enrichment, selected-lag-centered response,
100-replicate dependency-group discovery stability, and data/execution audit.

The input is pointwise 2D normalized displacement. The adopted measurement is
MediaPipe Face Landmarker IMAGE mode with one face, detection/presence 0.5,
BlendShapes off, and pose-matrix output on. The mapping is 8 regions, 29
landmarks, and 58 x/y scalars. It is checked by preflight overlay; an actual
model/topology failure requires a new protocol version before formal evaluation.

Discovery is PCMCI+ and ParCorr with `pc_alpha=0.01`, analytic significance,
`mask_type=xyz`, `recycle_residuals=False`, `tau_min=0`, `tau_max=L`, majority
collider handling, conflict resolution, `max_combinations=1`, no FDR, and no
contemporaneous forecast input. Scalar links are retained and OR-projected to
region blocks; a selected block contributes all usable source scalars.

The forecast horizon is `h=1`; inter-region candidates are every integer lag
from 1 through `L=floor(0.5*fps)`. Lag bands are `(0,100]`, `(100,250]`, and
`(250,500]` ms. The selected-lag response uses a single source-region × lag
block with `Delta=-M..M`, `M=floor(0.1*fps)`, complete symmetric support, and
no clipping, wrapping, or one-sided grid.

Ridge has an intercept, input-only train-fitted standardization, subject-equal
training weight, one common multi-output alpha, and base grid
`1e-4..1e4` by powers of ten. Selection uses inner subject-equal main error;
numeric ties within `1e-12` choose the larger alpha. Outer-test outcomes cannot
expand the grid.

All conditions and landscape cells share exact evaluation support. The primary
cell gain is `G=E_Self-E_cell`; unavailable cells remain unevaluable rather
than zero. Enrichment is the cell-G set mean against 1,000 matched scalar-count
random sets, not joint-set Ridge gain. Summaries are subject-level medians.
Nominal OOF intervals resample dependency groups 10,000 times with shared
indices and linear percentile quantiles; fewer than ten evaluable groups makes
the interval unevaluable. Discovery stability resamples independent groups 100
times while preserving series order and without redefining the Primary
ParentSet.

## Preflight boundary

The specification fixes the rules; preflight materializes only the real-data
facts: NoXi distribution and hashes, extractor/model hashes, fps/cadence,
subject/group/session inventory, pilot exclusion, H, realized splits, common
support, and the executable-freeze hash. No formal outer-test result may be
used to choose those facts. Until those files exist and are hash-verified, the
repository protocol remains `preflight_required` and a real runner must fail
closed.
