"""Frozen Primary PCMCI+ maximum lag.

The study-level decision is final: Primary PCMCI+ uses ``tau_max = 10`` frames
for every outer fold.  This value is a scientific freeze parameter, not a runtime
tuning knob, and must not be changed after observing outer-test or Sensitivity
results.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Mapping


PRIMARY_TAU_MAX_SCHEMA_VERSION = 2
PRIMARY_FORECAST_HORIZON = 1
PRIMARY_TAU_MAX = 10


@dataclass(frozen=True, slots=True)
class PrimaryTauMax:
    """Immutable Primary ``tau_max`` fixed to 10 frames."""

    tau_max: int = PRIMARY_TAU_MAX
    schema_version: int = PRIMARY_TAU_MAX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PRIMARY_TAU_MAX_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {PRIMARY_TAU_MAX_SCHEMA_VERSION}"
            )
        if not isinstance(self.tau_max, Integral) or isinstance(self.tau_max, bool):
            raise TypeError("Primary PCMCI+ tau_max must be an integer")

        tau_max = int(self.tau_max)
        if tau_max != PRIMARY_TAU_MAX:
            raise ValueError(
                "Primary PCMCI+ tau_max is scientifically frozen to "
                f"{PRIMARY_TAU_MAX}; got {tau_max}"
            )
        if tau_max < PRIMARY_FORECAST_HORIZON:
            raise ValueError(
                "Primary PCMCI+ tau_max must be >= the frozen forecast horizon "
                f"({PRIMARY_FORECAST_HORIZON})"
            )
        object.__setattr__(self, "tau_max", tau_max)

    def tigramite_kwargs(self) -> Mapping[str, object]:
        """Return the immutable D-04 contribution to ``run_pcmciplus`` kwargs."""

        return MappingProxyType({"tau_max": self.tau_max})


def freeze_primary_tau_max(tau_max: int = PRIMARY_TAU_MAX) -> PrimaryTauMax:
    """Return the final Primary maximum-lag freeze; reject any value other than 10."""

    return PrimaryTauMax(tau_max=tau_max)
