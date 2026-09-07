"""Primary metric primitives for the V0 vertical slice."""

from __future__ import annotations

import numpy as np

from .contracts import PredictionArtifact


def velocity_rmse(artifact: PredictionArtifact) -> float:
    """Compute RMSE over all valid prediction rows and velocity dimensions."""

    valid_rows = np.asarray(artifact.valid_mask, dtype=bool)
    if not np.any(valid_rows):
        raise ValueError("Velocity RMSE requires at least one valid prediction row")

    residual = artifact.y_pred[valid_rows] - artifact.y_true[valid_rows]
    return float(np.sqrt(np.mean(np.square(residual))))
