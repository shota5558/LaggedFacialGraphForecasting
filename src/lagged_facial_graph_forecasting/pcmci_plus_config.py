"""Static Primary PCMCI+ configuration excluding D-04/D-05 free parameters."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, object as _typing_object


PCMCI_PLUS_CONFIG_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PrimaryPCMCIPlusConfig:
    """Frozen static options for Tigramite ``run_pcmciplus``.

    ``tau_max`` and ``pc_alpha`` are intentionally absent: they are separate
    scientifically consequential tasks (D-04 and D-05).  The options here match
    Tigramite's stable PCMCI+ defaults, but are recorded explicitly so an upstream
    default change cannot silently alter a Primary run.
    """

    tau_min: int = 0
    contemp_collider_rule: str = "majority"
    conflict_resolution: bool = True
    reset_lagged_links: bool = False
    max_combinations: int = 1
    fdr_method: str = "none"
    schema_version: int = PCMCI_PLUS_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PCMCI_PLUS_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {PCMCI_PLUS_CONFIG_SCHEMA_VERSION}"
            )
        if self.tau_min != 0:
            raise ValueError("Primary PCMCI+ tau_min is frozen to 0")
        if self.contemp_collider_rule != "majority":
            raise ValueError("Primary PCMCI+ collider rule is frozen to 'majority'")
        if self.conflict_resolution is not True:
            raise ValueError("Primary PCMCI+ conflict_resolution is frozen to True")
        if self.reset_lagged_links is not False:
            raise ValueError("Primary PCMCI+ reset_lagged_links is frozen to False")
        if self.max_combinations != 1:
            raise ValueError("Primary PCMCI+ max_combinations is frozen to 1")
        if self.fdr_method != "none":
            raise ValueError("Primary PCMCI+ fdr_method is frozen to 'none'")

    def tigramite_kwargs(self) -> Mapping[str, object]:
        """Return immutable kwargs for D-06, excluding tau_max and pc_alpha."""

        return MappingProxyType(
            {
                "tau_min": self.tau_min,
                "contemp_collider_rule": self.contemp_collider_rule,
                "conflict_resolution": self.conflict_resolution,
                "reset_lagged_links": self.reset_lagged_links,
                "max_combinations": self.max_combinations,
                "fdr_method": self.fdr_method,
            }
        )


def make_primary_pcmci_plus_config() -> PrimaryPCMCIPlusConfig:
    """Return the frozen static Primary PCMCI+ configuration."""

    return PrimaryPCMCIPlusConfig()
