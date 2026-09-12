import json
from pathlib import Path

import pandas as pd
import pytest

from lagged_facial_graph_forecasting.analysis_pipeline import (
    AnalysisOutputError,
    landscape_aggregate_sources,
)


def _source():
    return pd.DataFrame([
        {"subject_id": subject, "source_region": "eye", "target_region": "mouth",
         "lag": lag, "gain": gain, "status": "evaluable"}
        for subject, gains in (("s1", (1.0, 3.0)), ("s2", (5.0, 7.0)))
        for lag, gain in enumerate(gains, 1)
    ])


def _config(bands):
    config = json.loads(Path("data/mock_analysis/mock_primary_config.json").read_text())
    config["landscape"] = {"lag_bands": bands,
                           "aggregation_id": "cell_mean_then_subject_median",
                           "ci_method": "subject_percentile"}
    return config


@pytest.mark.parametrize("bands", [
    None, {}, {"early": [1.5, 2]}, {"early": [True, 2]},
    {"early": ["1", 2]}, {"early": [1, float("inf")]},
    {"early": [1, None]}, {"early": [0, 2]}, {"early": [2, 1]},
    {"": [1, 2]}, {1: [1, 2]}, {"early": [1, 2, 3]},
    {"early": [1, 2], "late": [2, 3]},
    # Overlap is invalid even if none of the evaluable cells use it.
    {"early": [1, 3], "late": [3, 4]},
])
def test_landscape_rejects_invalid_band_configuration(bands):
    with pytest.raises(AnalysisOutputError, match="lag.band"):
        landscape_aggregate_sources(_source(), _config(bands))


@pytest.mark.parametrize("status", ["evaluable", "unevaluable", "failed"])
def test_landscape_bands_must_cover_all_candidates(status):
    source = _source()
    source.loc[source.lag == 2, "status"] = status
    with pytest.raises(AnalysisOutputError, match="every candidate cell"):
        landscape_aggregate_sources(source, _config({"early": [1, 1]}))


def test_landscape_band_aggregates_match_hand_calculation():
    source = _source()
    pair, bands = landscape_aggregate_sources(
        source, _config({"early": [1, 1], "late": [2, 2]})
    )
    assert pair.iloc[0].median_gain == 4.0  # median of subject means 2 and 6
    assert pair.iloc[0].n_subjects == 2
    assert pair.iloc[0].n_cells == 4
    assert bands.median_gain.tolist() == [3.0, 5.0]
    assert bands.n_cells.tolist() == [2, 2]
    assert bands.n_subjects.tolist() == [2, 2]
    source.loc[(source.subject_id == "s2") & (source.lag == 2), "status"] = "failed"
    _, bands = landscape_aggregate_sources(source, _config({"all": [1, 2]}))
    assert bands.iloc[0].median_gain == 3.5  # median of subject means 2 and 5
    assert bands.iloc[0].n_cells == 3
