from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy.stats import bootstrap as scipy_bootstrap

from lagged_facial_graph_forecasting.bootstrap_statistics import (
    PRIMARY_BOOTSTRAP_CONFIDENCE_LEVEL,
    BootstrapMedianCI,
    bootstrap_median_paired_difference_ci,
)
from lagged_facial_graph_forecasting.statistics import PairedMetricDifference


def _paired(
    *,
    fold: int,
    subject: str,
    difference: float,
    region: str = "mouth",
) -> PairedMetricDifference:
    comparison_value = 2.0
    return PairedMetricDifference(
        outer_fold=fold,
        subject_id=subject,
        region_id=region,
        metric_name="velocity_rmse",
        reference_condition="self",
        comparison_condition="pcmci",
        reference_value=comparison_value + difference,
        comparison_value=comparison_value,
        difference=difference,
        n_valid=20,
    )


def _sample() -> tuple[PairedMetricDifference, ...]:
    differences = (-0.3, -0.1, 0.0, 0.2, 0.4, 0.7, 1.0, 1.4)
    return tuple(
        _paired(fold=index, subject=f"s{index}", difference=difference)
        for index, difference in enumerate(differences)
    )


def test_bootstrap_ci_uses_scipy_bootstrap_for_subject_level_median() -> None:
    sample = _sample()
    result = bootstrap_median_paired_difference_ci(
        sample,
        n_resamples=1000,
        seed=1234,
        method="percentile",
    )[0]

    differences = np.asarray([row.difference for row in sample], dtype=float)
    expected = scipy_bootstrap(
        (differences,),
        np.median,
        confidence_level=0.95,
        n_resamples=1000,
        method="percentile",
        vectorized=False,
        random_state=np.random.default_rng(1234),
    )

    assert isinstance(result, BootstrapMedianCI)
    assert result.median_difference == pytest.approx(float(np.median(differences)))
    assert result.ci_low == pytest.approx(float(expected.confidence_interval.low))
    assert result.ci_high == pytest.approx(float(expected.confidence_interval.high))
    assert result.confidence_level == PRIMARY_BOOTSTRAP_CONFIDENCE_LEVEL == 0.95
    assert result.n_subjects == 8
    assert result.n_resamples == 1000
    assert result.seed == 1234
    assert result.method == "percentile"
    assert result.subject_ids == tuple(f"s{index}" for index in range(8))
    assert result.outer_folds == tuple(range(8))


def test_bootstrap_ci_is_reproducible_for_same_explicit_seed() -> None:
    first = bootstrap_median_paired_difference_ci(
        _sample(),
        n_resamples=500,
        seed=20260908,
        method="basic",
    )
    second = bootstrap_median_paired_difference_ci(
        _sample(),
        n_resamples=500,
        seed=20260908,
        method="basic",
    )

    assert first == second


def test_bootstrap_ci_keeps_region_groups_separate() -> None:
    sample = _sample() + tuple(
        _paired(
            fold=index,
            subject=f"s{index}",
            region="eye",
            difference=difference,
        )
        for index, difference in enumerate((0.1, 0.2, 0.4, 0.8))
    )
    results = bootstrap_median_paired_difference_ci(
        sample,
        n_resamples=500,
        seed=7,
        method="percentile",
    )

    assert [result.region_id for result in results] == ["eye", "mouth"]
    assert [result.n_subjects for result in results] == [4, 8]
    assert all(result.metric_name == "velocity_rmse" for result in results)
    assert all(result.reference_condition == "self" for result in results)
    assert all(result.comparison_condition == "pcmci" for result in results)


def test_bootstrap_ci_requires_at_least_two_subjects() -> None:
    with pytest.raises(ValueError, match="at least two held-out subjects"):
        bootstrap_median_paired_difference_ci(
            (_paired(fold=0, subject="s0", difference=0.2),),
            n_resamples=100,
            seed=1,
            method="percentile",
        )


def test_unfrozen_bootstrap_settings_have_no_implicit_defaults() -> None:
    parameters = inspect.signature(bootstrap_median_paired_difference_ci).parameters
    assert parameters["n_resamples"].default is inspect.Parameter.empty
    assert parameters["seed"].default is inspect.Parameter.empty
    assert parameters["method"].default is inspect.Parameter.empty

    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_median_paired_difference_ci(
            _sample(), n_resamples=0, seed=1, method="percentile"
        )
    with pytest.raises(ValueError, match="seed"):
        bootstrap_median_paired_difference_ci(
            _sample(), n_resamples=100, seed=-1, method="percentile"
        )
    with pytest.raises(ValueError, match="method"):
        bootstrap_median_paired_difference_ci(
            _sample(), n_resamples=100, seed=1, method="studentized"
        )
