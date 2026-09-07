"""Content-addressed artifact registry for Runner Core provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path

from .core_contracts import ExperimentArtifact, ExperimentConfig


class ArtifactRegistryError(ValueError):
    """Raised when an artifact cannot be safely registered."""


def _resolve_inside(path: Path, parent: Path, field_name: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(parent)
    except ValueError as exc:
        raise ArtifactRegistryError(f"{field_name} must resolve inside {parent}") from exc
    return resolved


@dataclass(slots=True)
class ArtifactRegistry:
    """Register immutable file provenance under one ExperimentConfig artifact root."""

    config: ExperimentConfig
    repository_root: Path | str = Path(".")
    _entries: dict[str, ExperimentArtifact] = field(init=False, default_factory=dict)
    _artifact_root: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.repository_root = Path(self.repository_root).resolve()
        artifact_root = self.repository_root / self.config.artifact_root
        self._artifact_root = _resolve_inside(
            artifact_root, self.repository_root, "artifact_root"
        )

    @property
    def entries(self) -> tuple[ExperimentArtifact, ...]:
        """Return registry entries sorted by repository-relative path."""

        return tuple(self._entries[path] for path in sorted(self._entries))

    def register_file(
        self,
        path: str | Path,
        *,
        artifact_type: str,
        outer_fold: int | None = None,
        condition: str | None = None,
    ) -> ExperimentArtifact:
        """Hash and register an existing completed file under the artifact root."""

        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.repository_root / candidate
        candidate = _resolve_inside(candidate, self._artifact_root, "artifact path")
        if candidate.name.endswith(".partial"):
            raise ArtifactRegistryError(
                "incomplete staging artifacts must not be registered"
            )
        if not candidate.is_file():
            raise ArtifactRegistryError(f"artifact file not found: {candidate}")

        relative_path = candidate.relative_to(self.repository_root).as_posix()
        sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
        entry = ExperimentArtifact(
            experiment_id=self.config.experiment_id,
            artifact_type=artifact_type,
            relative_path=relative_path,
            sha256=sha256,
            outer_fold=outer_fold,
            condition=condition,
        )

        previous = self._entries.get(relative_path)
        if previous is not None and previous != entry:
            raise ArtifactRegistryError(
                "artifact path is already registered with different provenance"
            )
        self._entries[relative_path] = entry
        return entry
