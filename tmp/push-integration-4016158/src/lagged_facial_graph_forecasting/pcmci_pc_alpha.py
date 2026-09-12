"""Frozen Primary PCMCI+ ``pc_alpha``.

The study-level decision is final: Primary PCMCI+ uses ``pc_alpha = 0.01`` for
every outer fold. This is a scientific freeze parameter, not a runtime tuning
knob. It must not be selected from outer-test outcomes, Sensitivity results, or
Tigramite model-selection defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Mapping


PRIMARY_PC_ALPHA_SCHEMA_VERSION = 1
PRIMARY_PC_ALPHA = 0.01


@dataclass(frozen=True, slots=True)
class PrimaryPCAlpha:
    """Immutable Primary ``pc_alpha`` fixed to 0.01."""

    pc_alpha: float = PRIMARY_PC_ALPHA
    schema_version: int = PRIMARY_PC_ALPHA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PRIMARY_PC_ALPHA_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {PRIMARY_PC_ALPHA_SCHEMA_VERSION}"
            )
        if not isinstance(self.pc_alpha, Real) or isinstance(self.pc_alpha, bool):
            raise TypeError("Primary PCMCI+ pc_alpha must be a real scalar")

        pc_alpha = float(self.pc_alpha)
        if pc_alpha != PRIMARY_PC_ALPHA:
            raise ValueError(
                "Primary PCMCI+ pc_alpha is scientifically frozen to "
                f"{PRIMARY_PC_ALPHA}; got {pc_alpha}"
            )
        object.__setattr__(self, "pc_alpha", pc_alpha)

    def tigramite_kwargs(self) -> Mapping[str, object]:
        """Return the immutable D-05 contribution to ``run_pcmciplus`` kwargs."""

        return MappingProxyType({"pc_alpha": self.pc_alpha})


def freeze_primary_pc_alpha(
    pc_alpha: float = PRIMARY_PC_ALPHA,
) -> PrimaryPCAlpha:
    """Return the final Primary pc_alpha freeze; reject any value other than 0.01."""

    return PrimaryPCAlpha(pc_alpha=pc_alpha)
