# Migration Status — 2026-09-12

Scientific adoption is complete. Executable migration is represented by
protocol `primary-2026-09-12-adopted`, schema 7, and an explicit
`preflight_required` state. No real Primary run is authorized until the
preflight facts, support manifests, and executable-freeze record are present
and hash-verified.

The old v6 config and Primary freeze manifest are incompatible with this
protocol. Preflight and synthetic validation may inspect the new contract;
real execution fails closed until materialization is complete.
