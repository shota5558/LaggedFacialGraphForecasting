from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting import PredictionArtifact, velocity_rmse


def _artifact(*, valid_mask: np.ndarray) -> PredictionArtifact:
    return PredictionArtifact(
        outer_fold=0,
        subject_id=("s01", "s01", "s01"),
        region_id=("mouth", "mouth", "mouth"),
        condition="self",
        forecast_origin=np.array([0.0, 1.0, 2.0]),
        target_time=np.array([1.0, 2.0, 3.0]),
        y_true=np.array([[0.0, 0.0], [10.0, 10.0], [2.0, 4.0]]),
        y_pred=np.array([[1.0, -1.0], [20.0, 20.0], [4.0, 2.0]]),
        valid_mask=valid_mask,
    )


def test_velocity_rmse_uses_all_valid_velocity_components() -> None:
    artifact = _artifact(valid_mask=np.array([True, False, True]))

    # Squared errors on valid rows: [1, 1, 4, 4], mean=2.5.
    assert velocity_rmse(artifact) == pytest.approx(np.sqrt(2.5))


def test_velocity_rmse_ignores_invalid_rows() -> None:
    artifact = _artifact(valid_mask=np.array([True, False, False]))

    assert velocity_rmse(artifact) == pytest.approx(1.0)


def test_velocity_rmse_rejects_no_valid_rows() -> None:
    artifact = _artifact(valid_mask=np.array([False, False, False]))

    with pytest.raises(ValueError, match="at least one valid prediction row"):
        velocity_rmse(artifact)
