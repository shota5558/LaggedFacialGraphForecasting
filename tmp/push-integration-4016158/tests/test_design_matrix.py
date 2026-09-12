from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting import ContractError, DesignMatrix


def _valid_kwargs() -> dict:
    return {
        "X": np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [1.0, 1.1, 1.2],
            ],
            dtype=float,
        ),
        "y": np.array(
            [[0.2, 0.1], [0.3, 0.2], [0.4, 0.3], [0.5, 0.4]], dtype=float
        ),
        "subject_id": ("s01", "s01", "s02", "s02"),
        "region_id": ("mouth", "mouth", "jaw", "jaw"),
        "target_dimensions": ("vx", "vy"),
        "forecast_origin": np.array([10.0, 11.0, 10.0, 11.0]),
        "target_time": np.array([11.0, 12.0, 11.0, 12.0]),
        "feature_names": ("mouth.vx", "mouth.vx", "jaw.vy"),
        "feature_lags": (1, 2, 1),
        "valid_mask": np.array([True, True, True, True]),
    }


def test_design_matrix_accepts_row_feature_and_target_provenance() -> None:
    matrix = DesignMatrix(**_valid_kwargs())

    assert matrix.X.shape == (4, 3)
    assert matrix.y.shape == (4, 2)
    assert len(matrix.subject_id) == 4
    assert len(matrix.region_id) == 4
    assert matrix.target_dimensions == ("vx", "vy")
    assert matrix.feature_names == ("mouth.vx", "mouth.vx", "jaw.vy")
    assert matrix.feature_lags == (1, 2, 1)
    assert matrix.valid_mask.dtype == np.bool_


def test_rejects_non_2d_X() -> None:
    kwargs = _valid_kwargs()
    kwargs["X"] = np.zeros((4, 3, 1), dtype=float)

    with pytest.raises(ContractError, match="DesignMatrix.X must be 2D"):
        DesignMatrix(**kwargs)


def test_rejects_y_row_mismatch() -> None:
    kwargs = _valid_kwargs()
    kwargs["y"] = np.zeros((3, 2), dtype=float)

    with pytest.raises(ContractError, match="y must have shape"):
        DesignMatrix(**kwargs)


def test_rejects_target_dimension_count_mismatch() -> None:
    kwargs = _valid_kwargs()
    kwargs["target_dimensions"] = ("vx",)

    with pytest.raises(ContractError, match="target_dimensions must contain 2 entries"):
        DesignMatrix(**kwargs)


def test_rejects_row_provenance_length_mismatch() -> None:
    kwargs = _valid_kwargs()
    kwargs["subject_id"] = ("s01", "s01", "s02")

    with pytest.raises(ContractError, match="subject_id must contain 4 entries"):
        DesignMatrix(**kwargs)


def test_rejects_target_not_after_forecast_origin() -> None:
    kwargs = _valid_kwargs()
    kwargs["target_time"] = np.array([11.0, 11.0, 11.0, 12.0])

    with pytest.raises(ContractError, match="strictly after forecast_origin"):
        DesignMatrix(**kwargs)


def test_rejects_feature_metadata_length_mismatch() -> None:
    kwargs = _valid_kwargs()
    kwargs["feature_lags"] = (1, 2)

    with pytest.raises(ContractError, match="feature_lags must contain 3 entries"):
        DesignMatrix(**kwargs)


@pytest.mark.parametrize("bad_lags", [(0, 2, 1), (-1, 2, 1), (1.5, 2, 1)])
def test_rejects_non_positive_or_non_integer_feature_lags(bad_lags: tuple) -> None:
    kwargs = _valid_kwargs()
    kwargs["feature_lags"] = bad_lags

    with pytest.raises(ContractError, match="positive integers"):
        DesignMatrix(**kwargs)


def test_rejects_duplicate_feature_name_lag_pair() -> None:
    kwargs = _valid_kwargs()
    kwargs["feature_names"] = ("mouth.vx", "mouth.vx", "mouth.vx")
    kwargs["feature_lags"] = (1, 2, 1)

    with pytest.raises(ContractError, match="pairs must be unique"):
        DesignMatrix(**kwargs)


def test_rejects_non_boolean_valid_mask() -> None:
    kwargs = _valid_kwargs()
    kwargs["valid_mask"] = np.ones(4, dtype=np.int8)

    with pytest.raises(ContractError, match="boolean dtype"):
        DesignMatrix(**kwargs)


def test_valid_rows_must_be_finite_but_invalid_rows_may_be_missing() -> None:
    kwargs = _valid_kwargs()
    kwargs["X"] = kwargs["X"].copy()
    kwargs["X"][3, 0] = np.nan
    kwargs["valid_mask"] = np.array([True, True, True, False])

    matrix = DesignMatrix(**kwargs)
    assert np.isnan(matrix.X[3, 0])
    assert not matrix.valid_mask[3]

    kwargs["valid_mask"] = np.array([True, True, True, True])
    with pytest.raises(ContractError, match="valid rows of X must be finite"):
        DesignMatrix(**kwargs)
