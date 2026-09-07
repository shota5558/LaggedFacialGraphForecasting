"""Fail-closed handling for the Primary PCMCI+ maximum lag.

D-04 intentionally does not choose a study-specific numeric maximum lag.  The
research specification requires that value to be fixed before discovery, but no
numeric value is currently frozen.  This module therefore accepts only an
explicitly supplied integer and records it as an immutable configuration value.
No data-dependent or outer-test-dependent inference path exists here.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Mapping


PRIMARY_TAU_MAX_SCHEMA_VERSION = 1
PRIMARY_FORECAST_HORIZON = 1


@dataclass(frozen=True, slots=True)
class PrimaryTauMax:
    """Explicitly frozen ``tau_max`` for Primary PCMCI+ discovery.

    ``tau_max`` has no default by design.  A caller must supply the value chosen
    by the pre-registered scientific configuration before discovery is run.
    Primary forecasting uses ``h=1``, so ``tau_max`` must include at least lag 1.
    """

    tau_max: int
    schema_version: int = PRIMARY_TAU_MAX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PRIMARY_TAU_MAX_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {PRIMARY_TAU_MAX_SCHEMA_VERSION}"
            )
        if not isinstance(self.tau_max, Integral) or isinstance(self.tau_max, bool):
            raise TypeError("Primary PCMCI+ tau_max must be an integer")

        tau_max = int(self.tau_max)
        if tau_max < PRIMARY_FORECAST_HORIZON:
            raise ValueError(
                "Primary PCMCI+ tau_max must be >= the frozen forecast horizon "
                f"({PRIMARY_FORECAST_HORIZON})"
            )
        object.__setattr__(self, "tau_max", tau_max)

    def tigramite_kwargs(self) -> Mapping[str, object]:
        """Return the immutable D-04 contribution to ``run_pcmciplus`` kwargs."""

        return MappingProxyType({"tau_max": self.tau_max})


def freeze_primary_tau_max(tau_max: int) -> PrimaryTauMax:
    """Freeze one explicit Primary maximum lag without consulting any data."""

    return PrimaryTauMax(tau_max=tau_max)
