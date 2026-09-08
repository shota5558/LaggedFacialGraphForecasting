"""Lag/horizon alignment utilities for the V0 vertical slice."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AlignedIndices:
    """Aligned source, forecast-origin, and target indices for one lag."""

    source_index: np.ndarray
    forecast_origin: np.ndarray
    target_index: np.ndarray
    lag: int
    horizon: int


def aligned_indices(
    n_steps: int,
    *,
    lag: int,
    horizon: int = 1,
) -> AlignedIndices:
    """Return all valid direct-observation indices for a target-relative lag.

    For forecast origin ``t`` and horizon ``h``, the target is ``t + h``. A parent
    at target-relative lag ``tau`` is observed at ``t + h - tau``. Direct forecasting
    therefore requires ``tau >= h`` so the source is no later than forecast origin.

    V0 implements only the frozen Primary horizon ``h = 1``.
    """

    if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 2:
        raise ValueError("n_steps must be an integer >= 2")
    if not isinstance(horizon, int) or isinstance(horizon, bool):
        raise ValueError("horizon must be an integer")
    if horizon != 1:
        raise ValueError("V0 Primary horizon is frozen to h=1")
    if not isinstance(lag, (int, np.integer)) or isinstance(lag, (bool, np.bool_)):
        raise ValueError("lag must be an integer")
    lag = int(lag)
    if lag < horizon:
        raise ValueError("direct-observation forecasting requires lag >= horizon")

    first_target = lag
    if first_target >= n_steps:
        raise ValueError(
            f"lag={lag} leaves no aligned samples for n_steps={n_steps}"
        )

    target_index = np.arange(first_target, n_steps, dtype=np.int64)
    forecast_origin = target_index - horizon
    source_index = target_index - lag

    if np.any(source_index > forecast_origin):
        raise RuntimeError("alignment invariant violated: source is after forecast origin")

    return AlignedIndices(
        source_index=source_index,
        forecast_origin=forecast_origin,
        target_index=target_index,
        lag=lag,
        horizon=horizon,
    )
