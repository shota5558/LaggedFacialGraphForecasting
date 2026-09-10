from __future__ import annotations

import numpy as np
import pytest

from lagged_facial_graph_forecasting.contracts import FaceTimeSeries, InnerFold, SplitManifest
from lagged_facial_graph_forecasting.pcmci_link_extraction import (
    PCMCIPlusLinkExtractionError,
    extract_significant_pcmciplus_links,
)
from lagged_facial_graph_forecasting.tigramite_adapter import (
    face_time_series_to_tigramite_dataframe,
)


def _bundle():
    manifest = SplitManifest(
        outer_fold=0,
        train_subject_ids=("s1", "s2"),
        test_subject_ids=("outer_test",),
        inner_folds=(
            InnerFold(
                inner_train_subject_ids=("s1",),
                inner_val_subject_ids=("s2",),
            ),
        ),
        seed=11,
    )
    series = []
    for subject_id in manifest.train_subject_ids:
        values = np.zeros((20, 1, 2), dtype=float)
        series.append(
            FaceTimeSeries(
                X=values,
                subject_id=subject_id,
                time_index=np.arange(20, dtype=float) / 25.0,
                region_id=("mouth",),
                dimension=("vx", "vy"),
                valid_mask=np.ones_like(values, dtype=bool),
                sampling_rate=25.0,
            )
        )
    return face_time_series_to_tigramite_dataframe(manifest, tuple(series))


def _empty_result(node_count: int = 2) -> dict[str, np.ndarray]:
    shape = (node_count, node_count, 11)
    return {
        "graph": np.full(shape, "", dtype="<U3"),
        "p_matrix": np.ones(shape, dtype=float),
        "val_matrix": np.zeros(shape, dtype=float),
    }


def test_extracts_nonempty_graph_entries_with_exact_component_provenance() -> None:
    bundle = _bundle()
    result = _empty_result()
    result["graph"][0, 1, 2] = "-->"
    result["p_matrix"][0, 1, 2] = 0.004
    result["val_matrix"][0, 1, 2] = 0.42

    links = extract_significant_pcmciplus_links(result, bundle)

    assert len(links) == 1
    link = links[0]
    assert link.source_node_index == 0
    assert link.target_node_index == 1
    assert link.lag == 2
    assert link.link_mark == "-->"
    assert link.p_value == pytest.approx(0.004)
    assert link.test_statistic == pytest.approx(0.42)
    assert link.source_variable_name == "mouth::vx"
    assert link.target_variable_name == "mouth::vy"
    assert (link.source_region, link.source_dimension) == ("mouth", "vx")
    assert (link.target_region, link.target_dimension) == ("mouth", "vy")


def test_graph_is_authoritative_and_small_p_value_does_not_create_link() -> None:
    result = _empty_result()
    result["p_matrix"][0, 1, 3] = 1e-12
    result["val_matrix"][0, 1, 3] = 0.99

    assert extract_significant_pcmciplus_links(result, _bundle()) == ()


def test_d07_preserves_contemporaneous_entries_for_d08() -> None:
    result = _empty_result()
    result["graph"][0, 1, 0] = "-->"
    result["graph"][1, 0, 0] = "<--"
    result["p_matrix"][0, 1, 0] = 0.003
    result["p_matrix"][1, 0, 0] = 0.003
    result["val_matrix"][0, 1, 0] = 0.31
    result["val_matrix"][1, 0, 0] = 0.31

    links = extract_significant_pcmciplus_links(result, _bundle())

    assert len(links) == 2
    assert {link.lag for link in links} == {0}
    assert {link.link_mark for link in links} == {"-->", "<--"}


def test_extraction_order_is_deterministic_target_then_lag_then_source() -> None:
    result = _empty_result()
    for source, target, lag in ((1, 1, 3), (0, 1, 1), (1, 0, 2), (0, 0, 4)):
        result["graph"][source, target, lag] = "-->"
        result["p_matrix"][source, target, lag] = 0.005
        result["val_matrix"][source, target, lag] = 0.2

    links = extract_significant_pcmciplus_links(result, _bundle())

    assert [(x.target_node_index, x.lag, x.source_node_index) for x in links] == [
        (0, 2, 1),
        (0, 4, 0),
        (1, 1, 0),
        (1, 3, 1),
    ]


@pytest.mark.parametrize("missing_key", ["graph", "p_matrix", "val_matrix"])
def test_rejects_missing_tigramite_result_matrix(missing_key: str) -> None:
    result = _empty_result()
    del result[missing_key]

    with pytest.raises(PCMCIPlusLinkExtractionError, match=missing_key):
        extract_significant_pcmciplus_links(result, _bundle())


def test_rejects_primary_shape_drift() -> None:
    result = _empty_result()
    result["graph"] = np.full((2, 2, 10), "", dtype="<U3")

    with pytest.raises(PCMCIPlusLinkExtractionError, match="graph shape"):
        extract_significant_pcmciplus_links(result, _bundle())


def test_rejects_nonfinite_diagnostics_for_retained_link() -> None:
    result = _empty_result()
    result["graph"][0, 1, 1] = "-->"
    result["p_matrix"][0, 1, 1] = np.nan
    result["val_matrix"][0, 1, 1] = 0.2

    with pytest.raises(PCMCIPlusLinkExtractionError, match="finite"):
        extract_significant_pcmciplus_links(result, _bundle())


def test_rejects_noncanonical_inputs() -> None:
    with pytest.raises(PCMCIPlusLinkExtractionError, match="result must"):
        extract_significant_pcmciplus_links([], _bundle())  # type: ignore[arg-type]
    with pytest.raises(PCMCIPlusLinkExtractionError, match="TigramiteDataFrameBundle"):
        extract_significant_pcmciplus_links(_empty_result(), object())  # type: ignore[arg-type]
