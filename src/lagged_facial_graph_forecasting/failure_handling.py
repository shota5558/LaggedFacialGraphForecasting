"""Atomic artifact publication for Runner Core failure handling.

A scientific result is visible at its final artifact path only after its writer
returns successfully.  Failed writes are confined to ``*.partial`` staging files
and cleaned before the original exception is re-raised.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from typing import Iterator

from .core_contracts import ExperimentConfig


class FailureHandlingError(RuntimeError):
    """Raised when an artifact cannot be published safely."""


def _resolve_inside(path: Path, parent: Path, field_name: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(parent)
    except ValueError as exc:
        raise FailureHandlingError(f"{field_name} must resolve inside {parent}") from exc
    return resolved


@contextmanager
def atomic_artifact_output(
    final_path: str | Path,
    *,
    config: ExperimentConfig,
    repository_root: str | Path = Path("."),
) -> Iterator[Path]:
    """Yield a staging path and publish it atomically only after success.

    The final path must be beneath the current Primary ``artifact_root`` and must
    not already exist.  The caller writes exclusively to the yielded staging path.
    On normal exit the staging file is atomically promoted with ``os.replace``;
    on any exception it is deleted and the exception is propagated unchanged.
    """

    root = Path(repository_root).resolve()
    artifact_root = _resolve_inside(
        root / config.artifact_root,
        root,
        "artifact_root",
    )

    target = Path(final_path)
    if not target.is_absolute():
        target = root / target
    target = _resolve_inside(target, artifact_root, "final_path")
    if target.exists():
        raise FailureHandlingError(f"final artifact already exists: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".partial",
        dir=target.parent,
    )
    os.close(fd)
    staging = Path(temporary_name)

    try:
        yield staging
        if not staging.is_file():
            raise FailureHandlingError("artifact writer did not leave a staging file")
        if target.exists():
            raise FailureHandlingError(
                f"final artifact appeared during staged write: {target}"
            )
        os.replace(staging, target)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
