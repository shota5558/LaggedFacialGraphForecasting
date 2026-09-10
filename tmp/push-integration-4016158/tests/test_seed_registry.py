from __future__ import annotations

from pathlib import Path

import pytest

from lagged_facial_graph_forecasting.seed_registry import SeedRegistry, SeedRegistryError


def test_seed_derivation_is_stable_and_order_independent(tmp_path: Path) -> None:
    first = SeedRegistry(20260908)
    a_first = first.seed_for("fold/0/null/random_region")
    b_first = first.seed_for("fold/0/inner_cv")
    path_a = first.write_json(tmp_path / "a.json")

    second = SeedRegistry(20260908)
    b_second = second.seed_for("fold/0/inner_cv")
    a_second = second.seed_for("fold/0/null/random_region")
    path_b = second.write_json(tmp_path / "b.json")

    assert a_first == a_second
    assert b_first == b_second
    assert a_first != b_first
    assert path_a.read_bytes() == path_b.read_bytes()


def test_repeated_key_returns_same_seed() -> None:
    registry = SeedRegistry(17)

    assert registry.seed_for("fold/2") == registry.seed_for("fold/2")
    assert len(registry.entries) == 1


def test_seed_is_uint32_compatible() -> None:
    seed = SeedRegistry(17).seed_for("pcmci/bootstrap/0")

    assert 0 <= seed <= 2**32 - 1


@pytest.mark.parametrize("value", [-1, True])
def test_invalid_root_seed_is_rejected(value: object) -> None:
    with pytest.raises(SeedRegistryError, match="root_seed"):
        SeedRegistry(value)  # type: ignore[arg-type]


def test_empty_seed_key_is_rejected() -> None:
    with pytest.raises(SeedRegistryError, match="seed key"):
        SeedRegistry(17).seed_for("   ")
