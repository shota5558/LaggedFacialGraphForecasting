"""Synthetic fixtures for the V0 vertical slice.

V0 intentionally avoids real facial data and causal discovery. The generator below
creates deterministic, stable self-history signals that are sufficient to validate
the experiment backbone before any PCMCI+ integration.
"""

from __future__ import annotations

import numpy as np

from .contracts import FaceTimeSeries


def generate_synthetic_face_time_series(
    *,
    subject_id: str,
    n_steps: int = 120,
    region_ids: tuple[str, ...] = ("left_eye", "mouth", "jaw"),
    dimensions: tuple[str, ...] = ("vx", "vy"),
    sampling_rate: float = 30.0,
    seed: int = 0,
    autoregressive_coefficient: float = 0.8,
    innovation_scale: float = 0.1,
) -> FaceTimeSeries:
    """Create one deterministic synthetic subject-level facial-motion series.

    The process is independent across region/dimension channels and follows a stable
    AR(1) recursion. Cross-region structure is deliberately absent in V0-01 because
    the vertical slice validates plumbing, not causal-discovery performance.
    """

    if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 2:
        raise ValueError("n_steps must be an integer >= 2")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not np.isfinite(autoregressive_coefficient) or abs(autoregressive_coefficient) >= 1:
        raise ValueError("autoregressive_coefficient must be finite with abs(value) < 1")
    if not np.isfinite(innovation_scale) or innovation_scale <= 0:
        raise ValueError("innovation_scale must be finite and > 0")

    region_ids = tuple(region_ids)
    dimensions = tuple(dimensions)
    if not region_ids:
        raise ValueError("region_ids must not be empty")
    if not dimensions:
        raise ValueError("dimensions must not be empty")

    rng = np.random.default_rng(seed)
    K = len(region_ids)
    D = len(dimensions)

    X = np.empty((n_steps, K, D), dtype=float)
    stationary_scale = innovation_scale / np.sqrt(1.0 - autoregressive_coefficient**2)
    X[0] = rng.normal(loc=0.0, scale=stationary_scale, size=(K, D))
    innovations = rng.normal(
        loc=0.0,
        scale=innovation_scale,
        size=(n_steps - 1, K, D),
    )
    for t in range(1, n_steps):
        X[t] = autoregressive_coefficient * X[t - 1] + innovations[t - 1]

    return FaceTimeSeries(
        X=X,
        subject_id=subject_id,
        time_index=np.arange(n_steps, dtype=float),
        region_id=region_ids,
        dimension=dimensions,
        valid_mask=np.ones((n_steps, K, D), dtype=bool),
        sampling_rate=sampling_rate,
    )
