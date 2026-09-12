from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENV = REPO_ROOT / ".venv-local-ci"
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "local-ci"


def _run(
    command: Sequence[str],
    *,
    cwd: Path = REPO_ROOT,
    log_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    printable = " ".join(command)
    print(f"\n>>> {printable}")
    if log_path is None:
        return subprocess.run(command, cwd=cwd, env=env, check=False).returncode

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write(f"\n>>> {printable}\n")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()


def _capture(command: Sequence[str], *, cwd: Path = REPO_ROOT) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip()


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _require_python_311() -> None:
    if sys.version_info[:2] != (3, 11):
        raise SystemExit(
            "Local CI requires Python 3.11 to match GitHub Actions. "
            "On Windows run: py -3.11 scripts/run_local_ci.py --suite core"
        )


def _create_venv(venv_dir: Path, recreate: bool) -> Path:
    if recreate and venv_dir.exists():
        shutil.rmtree(venv_dir)
    python = _venv_python(venv_dir)
    if not python.exists():
        print(f"Creating isolated CI environment: {venv_dir}")
        rc = _run([sys.executable, "-m", "venv", str(venv_dir)])
        if rc != 0:
            raise SystemExit(rc)
    return python


def _install_core(python: Path, log_path: Path) -> None:
    commands = [
        [str(python), "-m", "pip", "install", "--upgrade", "pip"],
        [str(python), "-m", "pip", "install", "-e", ".[test]"],
    ]
    for command in commands:
        rc = _run(command, log_path=log_path)
        if rc != 0:
            raise RuntimeError(f"Install step failed with exit code {rc}: {' '.join(command)}")


def _install_sensitivity(python: Path, log_path: Path) -> None:
    commands = [
        [str(python), "-m", "pip", "install", "--upgrade", "pip"],
        [
            str(python),
            "-m",
            "pip",
            "install",
            "torch",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ],
        [str(python), "-m", "pip", "install", "-e", ".[test]"],
    ]
    for command in commands:
        rc = _run(command, log_path=log_path)
        if rc != 0:
            raise RuntimeError(f"Install step failed with exit code {rc}: {' '.join(command)}")


def _run_core(python: Path, run_dir: Path) -> int:
    return _run(
        [
            str(python),
            "-m",
            "pytest",
            "--junitxml",
            str(run_dir / "pytest-core.xml"),
        ],
        log_path=run_dir / "core.log",
    )


def _run_sensitivity(python: Path, run_dir: Path) -> int:
    return _run(
        [
            str(python),
            "-m",
            "pytest",
            "tests/test_sensitivity_gru.py",
            "tests/test_sensitivity_integration_torch.py",
            "-q",
            "--junitxml",
            str(run_dir / "pytest-sensitivity.xml"),
        ],
        log_path=run_dir / "sensitivity.log",
    )


def _write_environment_snapshot(python: Path, run_dir: Path) -> None:
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "git_head": _capture(["git", "rev-parse", "HEAD"]),
        "git_branch": _capture(["git", "branch", "--show-current"]),
        "git_status": _capture(["git", "status", "--short"]),
        "host_python": sys.version,
        "ci_python": _capture([str(python), "--version"]),
        "pip": _capture([str(python), "-m", "pip", "--version"]),
    }
    (run_dir / "environment.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "pip-freeze.txt").write_text(
        _capture([str(python), "-m", "pip", "freeze"]) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run local tests with the same commands used by GitHub Actions."
    )
    parser.add_argument(
        "--suite",
        choices=("core", "sensitivity", "all"),
        default="core",
        help="core=test.yml, sensitivity=sensitivity-torch-test.yml, all=both",
    )
    parser.add_argument(
        "--venv",
        type=Path,
        default=DEFAULT_VENV,
        help="isolated virtual environment path",
    )
    parser.add_argument(
        "--recreate-venv",
        action="store_true",
        help="delete and recreate the local CI virtual environment",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="reuse the existing environment and skip pip install steps",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="when --suite all is used, run sensitivity even if core fails",
    )
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    _require_python_311()

    venv_dir = args.venv.resolve()
    python = _create_venv(venv_dir, args.recreate_venv)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ARTIFACT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    summary: dict[str, object] = {
        "suite": args.suite,
        "run_id": run_id,
        "results": {},
        "overall": "PASS",
    }
    results: dict[str, object] = summary["results"]  # type: ignore[assignment]

    try:
        if args.suite in ("core", "all"):
            if not args.skip_install:
                _install_core(python, run_dir / "install-core.log")
            _write_environment_snapshot(python, run_dir)
            rc = _run_core(python, run_dir)
            results["core"] = {"exit_code": rc, "status": "PASS" if rc == 0 else "FAIL"}
            if rc != 0:
                summary["overall"] = "FAIL"
                if not args.keep_going or args.suite != "all":
                    return rc

        if args.suite in ("sensitivity", "all"):
            if not args.skip_install:
                _install_sensitivity(python, run_dir / "install-sensitivity.log")
            _write_environment_snapshot(python, run_dir)
            rc = _run_sensitivity(python, run_dir)
            results["sensitivity"] = {
                "exit_code": rc,
                "status": "PASS" if rc == 0 else "FAIL",
            }
            if rc != 0:
                summary["overall"] = "FAIL"
                return rc

        return 0 if summary["overall"] == "PASS" else 1
    except RuntimeError as exc:
        summary["overall"] = "ERROR"
        summary["error"] = str(exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nLocal CI artifacts: {run_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
