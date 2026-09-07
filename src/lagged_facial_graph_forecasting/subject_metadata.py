"""Subject metadata loading and validation for the split/leakage lane."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path


class SubjectMetadataLoadError(ValueError):
    """Raised when a subject metadata file cannot be parsed safely."""


class SubjectMetadataValidationError(ValueError):
    """Raised when loaded subject identifiers violate the metadata contract."""


def load_subject_metadata_csv(path: str | Path) -> tuple[dict[str, str], ...]:
    """Load a UTF-8 CSV metadata table without performing subject-ID validation.

    B-00 owns file parsing only. B-01 separately validates subject identifiers,
    including blank and duplicate IDs, so this loader intentionally preserves row
    values as supplied by the metadata source.
    """

    path = Path(path)
    if not path.is_file():
        raise SubjectMetadataLoadError(f"subject metadata file does not exist: {path}")

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            if not fieldnames:
                raise SubjectMetadataLoadError("subject metadata CSV must contain a header")
            if any(name is None or not name.strip() for name in fieldnames):
                raise SubjectMetadataLoadError("subject metadata CSV header names must be non-empty")
            if len(set(fieldnames)) != len(fieldnames):
                raise SubjectMetadataLoadError("subject metadata CSV header names must be unique")
            if "subject_id" not in fieldnames:
                raise SubjectMetadataLoadError(
                    "subject metadata CSV must contain a 'subject_id' column"
                )

            rows: list[dict[str, str]] = []
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise SubjectMetadataLoadError(
                        f"subject metadata CSV row {row_number} has too many columns"
                    )
                if any(value is None for value in row.values()):
                    raise SubjectMetadataLoadError(
                        f"subject metadata CSV row {row_number} has too few columns"
                    )
                rows.append({name: row[name] for name in fieldnames})
    except UnicodeDecodeError as exc:
        raise SubjectMetadataLoadError("subject metadata CSV must be UTF-8 encoded") from exc
    except csv.Error as exc:
        raise SubjectMetadataLoadError(f"invalid subject metadata CSV: {exc}") from exc

    return tuple(rows)


def validate_subject_ids(rows: tuple[Mapping[str, str], ...]) -> tuple[str, ...]:
    """Validate subject identifiers without constructing or selecting a split."""

    rows = tuple(rows)
    if not rows:
        raise SubjectMetadataValidationError("subject metadata must contain at least one row")

    subject_ids: list[str] = []
    seen: set[str] = set()
    for row_index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise SubjectMetadataValidationError(
                f"subject metadata row {row_index} must be a mapping"
            )
        if "subject_id" not in row:
            raise SubjectMetadataValidationError(
                f"subject metadata row {row_index} is missing subject_id"
            )
        subject_id = row["subject_id"]
        if not isinstance(subject_id, str):
            raise SubjectMetadataValidationError(
                f"subject_id at row {row_index} must be a string"
            )
        if not subject_id.strip():
            raise SubjectMetadataValidationError(
                f"subject_id at row {row_index} must be non-empty"
            )
        if subject_id != subject_id.strip():
            raise SubjectMetadataValidationError(
                f"subject_id at row {row_index} must not contain leading/trailing whitespace"
            )
        if subject_id in seen:
            raise SubjectMetadataValidationError(f"duplicate subject_id is forbidden: {subject_id!r}")
        seen.add(subject_id)
        subject_ids.append(subject_id)

    return tuple(subject_ids)
