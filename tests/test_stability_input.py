import pandas as pd
import pytest

from lagged_facial_graph_forecasting.analysis_pipeline import AnalysisOutputError, table_t08


def _edges():
    return pd.read_csv("data/mock_analysis/mock_edge_stability.csv").iloc[:1].copy()


@pytest.mark.parametrize("prefix,freq", [
    ("fold", "outer_fold_selection_frequency"),
    ("bootstrap", "bootstrap_selection_frequency"),
])
def test_stability_rejects_selected_count_above_opportunities(prefix, freq):
    frame = _edges().astype(object)
    frame.loc[frame.index[0], f"{prefix}_selected_count"] = 5
    frame.loc[frame.index[0], f"{prefix}_opportunities"] = 4
    frame.loc[frame.index[0], freq] = 1.25
    with pytest.raises(AnalysisOutputError):
        table_t08(frame)


def test_stability_rejects_non_integer_lag():
    frame = _edges().astype(object)
    frame.loc[frame.index[0], "lag"] = 1.5
    with pytest.raises(AnalysisOutputError, match="lag must be an integer"):
        table_t08(frame)


@pytest.mark.parametrize("selected", [0, 4])
def test_stability_preserves_zero_and_full_selection(selected):
    frame = _edges()
    frame["fold_selected_count"] = selected
    frame["outer_fold_selection_frequency"] = selected / 4
    result = table_t08(frame)
    assert result.iloc[0].outer_fold_selection_frequency == selected / 4
