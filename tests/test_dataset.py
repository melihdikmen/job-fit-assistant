"""Tests for loading, validating and extending the labeled test set."""

import json
from pathlib import Path
from typing import Any

import pytest

from job_fit.dataset import (
    DatasetError,
    Posting,
    add_posting,
    check_coverage,
    label_counts,
    load_postings,
    main,
)


def _row(**overrides: Any) -> dict[str, Any]:
    """Return a valid test set row as a dict, with overrides."""
    row: dict[str, Any] = {
        "id": "acme-1",
        "company": "Acme",
        "role": "Senior Mobile Engineer",
        "text": "We are hiring a senior engineer.",
        "label": "APPLY",
        "note": "main stack matches",
    }
    row.update(overrides)
    return row


def _write(path: Path, *rows: dict[str, Any]) -> Path:
    """Write rows to a JSONL file and return its path."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _postings(apply: int, prepare: int, skip: int) -> list[Posting]:
    """Build a set with the given number of postings per label."""
    labels = ["APPLY"] * apply + ["PREPARE"] * prepare + ["SKIP"] * skip
    return [Posting(**_row(id=f"p-{index}", label=label)) for index, label in enumerate(labels)]


def test_valid_file_loads_in_order(tmp_path: Path) -> None:
    """Rows come back as postings, in file order."""
    path = _write(tmp_path / "set.jsonl", _row(), _row(id="acme-2", label="SKIP"))

    postings = load_postings(path)

    assert [posting.id for posting in postings] == ["acme-1", "acme-2"]
    assert postings[1].label == "SKIP"


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    """An empty line between rows is not an error."""
    path = tmp_path / "set.jsonl"
    path.write_text(json.dumps(_row()) + "\n\n", encoding="utf-8")

    assert len(load_postings(path)) == 1


def test_note_is_optional(tmp_path: Path) -> None:
    """A row without a note loads with an empty note."""
    row = _row()
    del row["note"]

    assert load_postings(_write(tmp_path / "set.jsonl", row))[0].note == ""


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    """A path that does not exist gives a clear error."""
    with pytest.raises(DatasetError, match="not found"):
        load_postings(tmp_path / "missing.jsonl")


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    """A file without postings is an error."""
    path = tmp_path / "set.jsonl"
    path.write_text("\n", encoding="utf-8")

    with pytest.raises(DatasetError, match="empty"):
        load_postings(path)


def test_invalid_json_is_rejected_with_its_line_number(tmp_path: Path) -> None:
    """A broken line is reported by number."""
    path = tmp_path / "set.jsonl"
    path.write_text(json.dumps(_row()) + "\n{not json\n", encoding="utf-8")

    with pytest.raises(DatasetError, match=r"set\.jsonl:2"):
        load_postings(path)


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"label": "MAYBE"}, "label"),
        ({"text": "   "}, "text"),
        ({"id": ""}, "id"),
        ({"surprise": 1}, "surprise"),
    ],
)
def test_invalid_field_is_rejected_by_name(
    tmp_path: Path, overrides: dict[str, Any], field: str
) -> None:
    """An unknown label, an empty text or id, and an extra field are all rejected."""
    path = _write(tmp_path / "set.jsonl", _row(**overrides))

    with pytest.raises(DatasetError, match=field):
        load_postings(path)


def test_missing_field_is_rejected(tmp_path: Path) -> None:
    """A row without a label cannot be scored."""
    row = _row()
    del row["label"]

    with pytest.raises(DatasetError, match="label"):
        load_postings(_write(tmp_path / "set.jsonl", row))


def test_error_message_does_not_echo_the_posting_text(tmp_path: Path) -> None:
    """A validation error names the field but not the row's content."""
    path = _write(tmp_path / "set.jsonl", _row(text="SECRET-MARKER", label="MAYBE"))

    with pytest.raises(DatasetError) as caught:
        load_postings(path)

    assert "SECRET-MARKER" not in str(caught.value)


def test_duplicate_id_is_rejected_with_both_lines(tmp_path: Path) -> None:
    """Two rows with the same id are reported with their line numbers."""
    path = _write(tmp_path / "set.jsonl", _row(), _row())

    with pytest.raises(DatasetError, match=r"set\.jsonl:2: duplicate id 'acme-1'.*line 1"):
        load_postings(path)


def test_label_counts_include_labels_with_no_postings() -> None:
    """Every label is counted, zero included."""
    assert label_counts(_postings(2, 0, 1)) == {"APPLY": 2, "PREPARE": 0, "SKIP": 1}


def test_complete_set_has_no_coverage_problems() -> None:
    """Twenty postings with at least five per label pass."""
    assert check_coverage(_postings(8, 5, 7)) == []


def test_small_set_is_reported() -> None:
    """Fewer than twenty postings is a shortfall."""
    problems = check_coverage(_postings(5, 5, 5))

    assert problems == ["15 postings, need at least 20"]


def test_unbalanced_set_is_reported_per_label() -> None:
    """A label with fewer than five postings is named."""
    problems = check_coverage(_postings(14, 4, 6))

    assert problems == ["4 postings labeled PREPARE, need at least 5"]


def test_add_posting_writes_a_row_that_loads_back(tmp_path: Path) -> None:
    """Multi-line text with quotes and Turkish letters survives the round trip."""
    path = tmp_path / "data" / "set.jsonl"
    text = 'İş ilanı\n"Senior" engineer\twanted'

    add_posting(path, Posting(**_row(text=text)))
    add_posting(path, Posting(**_row(id="acme-2")))

    postings = load_postings(path)
    assert [posting.id for posting in postings] == ["acme-1", "acme-2"]
    assert postings[0].text == text
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_add_posting_refuses_a_duplicate_id(tmp_path: Path) -> None:
    """An id that is already in the file is not added again."""
    path = _write(tmp_path / "set.jsonl", _row())

    with pytest.raises(DatasetError, match="already exists"):
        add_posting(path, Posting(**_row()))

    assert len(load_postings(path)) == 1


def test_add_posting_repairs_a_missing_final_newline(tmp_path: Path) -> None:
    """A file that does not end in a newline still gets one row per line."""
    path = tmp_path / "set.jsonl"
    path.write_text(json.dumps(_row()), encoding="utf-8")

    add_posting(path, Posting(**_row(id="acme-2")))

    assert len(load_postings(path)) == 2


def test_cli_add_appends_from_a_text_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The add command reads the posting from a file and reports the new balance."""
    posting_file = tmp_path / "posting.txt"
    posting_file.write_text("Line one\nLine two", encoding="utf-8")
    data = tmp_path / "set.jsonl"
    argv = ["add", "--id", "acme-1", "--company", "Acme", "--role", "Engineer"]
    argv += ["--label", "PREPARE", "--file", str(posting_file), "--data", str(data)]

    exit_code = main(argv)

    assert exit_code == 0
    assert load_postings(data)[0].text == "Line one\nLine two"
    assert "postings: 1" in capsys.readouterr().out


def test_cli_add_rejects_an_empty_posting_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty text file is refused and nothing is written."""
    posting_file = tmp_path / "posting.txt"
    posting_file.write_text("", encoding="utf-8")
    data = tmp_path / "set.jsonl"
    argv = ["add", "--id", "acme-1", "--company", "Acme", "--role", "Engineer"]
    argv += ["--label", "SKIP", "--file", str(posting_file), "--data", str(data)]

    exit_code = main(argv)

    assert exit_code == 1
    assert not data.exists()
    assert "text" in capsys.readouterr().err


def test_cli_check_fails_on_an_incomplete_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The check command exits with 1 and names the shortfall."""
    data = _write(tmp_path / "set.jsonl", _row())

    exit_code = main(["check", "--data", str(data)])

    assert exit_code == 1
    assert "incomplete: 1 postings, need at least 20" in capsys.readouterr().out


def test_cli_check_passes_on_a_complete_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The check command exits with 0 when size and balance are met."""
    rows = [posting.model_dump() for posting in _postings(7, 6, 7)]
    data = _write(tmp_path / "set.jsonl", *rows)

    exit_code = main(["check", "--data", str(data)])

    assert exit_code == 0
    assert "coverage: ok" in capsys.readouterr().out
