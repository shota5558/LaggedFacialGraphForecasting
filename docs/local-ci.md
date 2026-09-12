# Local CI — GitHub Actions parity

This repository provides a local CI runner so routine validation does not need to consume GitHub Actions minutes.

## Scope

The local runner mirrors the commands currently executed by the two workflows on `dev`.

| GitHub Actions workflow | Local suite | Environment | Test command |
|---|---|---|---|
| `.github/workflows/test.yml` | `core` | Python 3.11, `pip install -e .[test]` | `python -m pytest` |
| `.github/workflows/sensitivity-torch-test.yml` | `sensitivity` | Python 3.11, CPU PyTorch, `pip install -e .[test]` | `python -m pytest tests/test_sensitivity_gru.py tests/test_sensitivity_integration_torch.py -q` |

The runner intentionally does **not** change scientific configuration, split rules, leakage guards, Primary/Sensitivity separation, or experiment artifacts. It only reproduces software test execution locally.

## Windows / PowerShell

From the repository root:

```powershell
# Standard test workflow equivalent
.\scripts\run_local_ci.ps1 -Suite core

# CPU PyTorch sensitivity workflow equivalent
.\scripts\run_local_ci.ps1 -Suite sensitivity

# Run both sequentially
.\scripts\run_local_ci.ps1 -Suite all
```

For a clean environment recreation:

```powershell
.\scripts\run_local_ci.ps1 -Suite all -RecreateVenv
```

For repeated development runs after dependencies are already installed:

```powershell
.\scripts\run_local_ci.ps1 -Suite core -SkipInstall
```

## Direct Python invocation

The runner requires Python 3.11 because GitHub Actions is pinned to Python 3.11.

Windows:

```powershell
py -3.11 scripts/run_local_ci.py --suite core
```

Linux/macOS with Python 3.11 active:

```bash
python3.11 scripts/run_local_ci.py --suite core
```

## Isolation

The runner uses:

```text
.venv-local-ci/
```

This is separate from the developer's normal virtual environment. It reduces false passes caused by packages that happen to be installed globally or in another project environment.

## Artifacts

Every run writes evidence under:

```text
artifacts/local-ci/YYYYMMDD-HHMMSS/
```

Typical contents:

```text
environment.json
pip-freeze.txt
install-core.log
core.log
pytest-core.xml
install-sensitivity.log
sensitivity.log
pytest-sensitivity.xml
summary.json
```

`environment.json` records the commit, branch, dirty working-tree state, Python version, and pip version. `pip-freeze.txt` records the resolved package set. JUnit XML can be consumed by editors or later CI tooling.

These local runtime artifacts are ignored by Git and should not be committed as research results.

## Recommended development policy

Use local CI as the default pre-push gate:

```text
edit
  ↓
local targeted pytest
  ↓
local CI core
  ↓
local CI sensitivity only when relevant
  ↓
commit / push
```

When GitHub Actions quota is constrained, avoid opening a PR solely to obtain a test run. Run Local CI first and reserve Actions for integration checkpoints, release/freeze evidence, or manually requested remote verification.

## Parity boundary

The local runner reproduces the workflow **commands and Python version**, but a Windows workstation is not identical to GitHub's `ubuntu-latest` host. Therefore:

- `core` PASS means the same Python-level test suite passed locally.
- `sensitivity` PASS means the same CPU-PyTorch test files passed locally.
- OS-specific behavior can still differ from GitHub-hosted Ubuntu.

For high-confidence freeze/release points, a remote Linux run remains useful when quota is available.

## Failure handling

The runner exits non-zero when installation or tests fail. For `--suite all`, the default is fail-fast after the core suite. To execute sensitivity even after a core failure:

```powershell
.\scripts\run_local_ci.ps1 -Suite all -KeepGoing
```

Inspect `summary.json` plus the corresponding log and JUnit XML for the exact failure.
