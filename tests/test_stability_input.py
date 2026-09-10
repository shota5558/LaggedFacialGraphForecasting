import pandas as pd
import pytest

from lagged_facial_graph_forecasting.analysis_pipeline import AnalysisOutputError, table_t08


def _edges():
    return pd.read_csv("data/mock_analysis/mock_edge_stability.csv").iloc[:1].copy()


@pytest.mark.parametrize("prefix,freq", [
    ("fold", "outer_fold_selection_frequency"),
    ("bootstrap", "bootstrap_selection_frequency"),
])
@pytest.mark.parametrize("selected,opportunities", [(-1, 4), (5, 4), (0.5, 4), (1, 4.5), (1, float("inf")), (True, 4)])
def test_stability_rejects_invalid_counts_even_when_ratio_matches(prefix, freq, selected, opportunities):
    frame = _edges().astype(object)
    frame.loc[frame.index[0], f"{prefix}_selected_count"] = selected
    frame.loc[frame.index[0], f"{prefix}_opportunities"] = opportunities
    frame.loc[frame.index[0], freq] = selected / opportunities
    with pytest.raises(AnalysisOutputError):
        table_t08(frame)


@pytest.mark.parametrize("lag", [1.5, True, None, float("inf")])
def test_stability_rejects_non_integer_lag(lag):
    frame = _edges().astype(object)
    frame.loc[frame.index[0], "lag"] = lag
    with pytest.raises(AnalysisOutputError, match="lag must be an integer"):
        table_t08(frame)


@pytest.mark.parametrize("selected", [0, 4])
def test_stability_preserves_zero_and_full_selection(selected):
    frame = _edges()
    frame["fold_selected_count"] = selected
    frame["outer_fold_selection_frequency"] = selected / 4
    result = table_t08(frame)
    assert result.iloc[0].outer_fold_selection_frequency == selected / 4
