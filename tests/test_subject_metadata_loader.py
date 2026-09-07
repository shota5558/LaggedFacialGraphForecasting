from __future__ import annotations

import pytest

from lagged_facial_graph_forecasting.subject_metadata import (
    SubjectMetadataLoadError,
    load_subject_metadata_csv,
)


def test_subject_metadata_loader_reads_rows_and_preserves_values(tmp_path) -> None:
    path = tmp_path / "subjects.csv"
    path.write_text(
        "subject_id,session,cohort\n"
        "s02,visit_b,control\n"
        "s01,visit_a,case\n",
        encoding="utf-8",
    )

    rows = load_subject_metadata_csv(path)

    assert rows == (
        {"subject_id": "s02", "session": "visit_b", "cohort": "control"},
        {"subject_id": "s01", "session": "visit_a", "cohort": "case"},
    )


def test_subject_metadata_loader_does_not_preempt_b01_id_validation(tmp_path) -> None:
    path = tmp_path / "subjects.csv"
    path.write_text(
        "subject_id,session\n"
        ",visit_a\n"
        "s01,visit_b\n"
        "s01,visit_c\n",
        encoding="utf-8",
    )

    rows = load_subject_metadata_csv(path)

    assert [row["subject_id"] for row in rows] == ["", "s01", "s01"]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("session,cohort\nvisit_a,case\n", "subject_id"),
        ("subject_id,subject_id\ns01,s01\n", "unique"),
        ("subject_id,session\ns01\n", "too few"),
        ("subject_id,session\ns01,visit_a,extra\n", "too many"),
    ],
)
def test_subject_metadata_loader_rejects_malformed_tables(
    tmp_path, content: str, message: str
) -> None:
    path = tmp_path / "subjects.csv"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(SubjectMetadataLoadError, match=message):
        load_subject_metadata_csv(path)


def test_subject_metadata_loader_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(SubjectMetadataLoadError, match="does not exist"):
        load_subject_metadata_csv(tmp_path / "missing.csv")
