# Migration Status — 2026-09-12

## Scientific design

- [x] Final recommendation fully adopted as authoritative specification.
- [x] Authority hierarchy documented.
- [x] README updated.
- [x] Integrated experiment design updated.
- [x] Superseded legacy decisions enumerated.

## Executable migration

The following remain implementation work and are not falsely marked complete:

- [ ] `configs/scientific_freeze.yaml` protocol migration
- [ ] `configs/primary_run.yaml` migration
- [ ] schema migration
- [ ] config loader / preflight migration
- [ ] scalar→region-block OR projection implementation audit
- [ ] dynamic `L=floor(0.5*fps)` implementation
- [ ] common-support enforcement audit
- [ ] Landscape / Enrichment / centered response migration
- [ ] old-v6 artifact rejection regression tests
- [ ] one-fold real-data dry run
- [ ] leakage/reproducibility audit
- [ ] hash-backed executable freeze

Scientific adoption and executable migration are deliberately separated.
