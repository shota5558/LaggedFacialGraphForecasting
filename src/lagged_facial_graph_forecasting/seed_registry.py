"""Deterministic named seed derivation for reproducible experiment components."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path


class SeedRegistryError(ValueError):
    """Raised when a seed namespace is invalid."""


@dataclass(slots=True)
class SeedRegistry:
    """Derive stable uint32 seeds from one frozen experiment seed and named keys."""

    root_seed: int
    _seeds: dict[str, int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.root_seed, int) or isinstance(self.root_seed, bool) or self.root_seed < 0:
            raise SeedRegistryError("root_seed must be a non-negative integer")

    def seed_for(self, key: str) -> int:
        if not isinstance(key, str) or not key.strip():
            raise SeedRegistryError("seed key must be a non-empty string")
        key = key.strip()
        existing = self._seeds.get(key)
        if existing is not None:
            return existing
        digest = hashlib.sha256(f"{self.root_seed}:{key}".encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], byteorder="big", signed=False)
        self._seeds[key] = seed
        return seed

    @property
    def entries(self) -> tuple[tuple[str, int], ...]:
        return tuple((key, self._seeds[key]) for key in sorted(self._seeds))

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "root_seed": self.root_seed,
            "seeds": {key: value for key, value in self.entries},
        }

    def write_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.payload(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output_path
