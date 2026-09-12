"""Report-link checks for the displacement analysis draft."""
from __future__ import annotations

from pathlib import Path

from lagged_facial_graph_forecasting.displacement_report import _markdown_path


def test_markdown_paths_are_relative_to_the_report_root():
    root = Path("report")
    figure = root / "figures" / "N_F1_landscape.png"
    assert _markdown_path(root, figure) == "figures/N_F1_landscape.png"
    assert not _markdown_path(root, figure).startswith(("C:/", "/"))
