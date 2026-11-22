"""Tests for loading, validating and extending the labeled test set."""

import json
from pathlib import Path
from typing import Any

import pytest

from job_fit.dataset import (
    HELDOUT_SHARE,
    MIN_DEV_PER_LABEL,
    MIN_DEV_POSTINGS,
    MIN_PER_LABEL,
    MIN_POSTINGS,
    DatasetError,
    LabelRecord,
    Posting,
    add_posting,
    apply_labels,
    check_coverage,
    format_agreement,
    heldout_size,
    label_counts,
    label_session,
    labeling_order,
    load_label_log,
    load_postings,
    main,
    parse_label,
    split_postings,
    write_postings,
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
    """A set with the required size and the required number per label passes."""
    assert check_coverage(_postings(MIN_PER_LABEL, MIN_PER_LABEL, MIN_POSTINGS)) == []


def test_small_set_is_reported() -> None:
    """Fewer postings than required is a shortfall."""
    size = MIN_PER_LABEL * 3
    problems = check_coverage(_postings(MIN_PER_LABEL, MIN_PER_LABEL, MIN_PER_LABEL))

    assert problems == [f"{size} postings, need at least {MIN_POSTINGS}"]


def test_unbalanced_set_is_reported_per_label() -> None:
    """A label with fewer postings than required is named."""
    short = MIN_PER_LABEL - 1
    problems = check_coverage(_postings(MIN_POSTINGS, short, MIN_PER_LABEL))

    assert problems == [f"{short} postings labeled PREPARE, need at least {MIN_PER_LABEL}"]


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
    assert f"incomplete: 1 postings, need at least {MIN_POSTINGS}" in capsys.readouterr().out


def test_cli_check_passes_on_a_complete_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The check command exits with 0 when size and balance are met."""
    complete = _postings(MIN_PER_LABEL, MIN_PER_LABEL, MIN_POSTINGS)
    rows = [posting.model_dump() for posting in complete]
    data = _write(tmp_path / "set.jsonl", *rows)

    exit_code = main(["check", "--data", str(data)])

    assert exit_code == 0
    assert "coverage: ok" in capsys.readouterr().out


def test_dev_minimums_are_what_a_complete_set_leaves_after_the_split() -> None:
    """The dev thresholds follow from the full-set thresholds and the held-out share."""
    dev, _ = split_postings(_postings(MIN_PER_LABEL, MIN_PER_LABEL, MIN_PER_LABEL))

    assert set(label_counts(dev).values()) == {MIN_DEV_PER_LABEL}
    assert MIN_DEV_POSTINGS == MIN_POSTINGS - heldout_size(MIN_POSTINGS) - 1


@pytest.mark.parametrize(
    "counts",
    [(MIN_PER_LABEL, MIN_PER_LABEL, MIN_POSTINGS - 2 * MIN_PER_LABEL), (9, 6, 25), (13, 13, 14)],
)
def test_dev_part_of_a_complete_set_meets_the_dev_minimums(counts: tuple[int, int, int]) -> None:
    """However a complete set is balanced, its dev part passes the check eval makes."""
    dev, _ = split_postings(_postings(*counts))

    assert check_coverage(dev, MIN_DEV_POSTINGS, MIN_DEV_PER_LABEL) == []


@pytest.mark.parametrize(("count", "expected"), [(0, 0), (1, 0), (2, 1), (3, 1), (10, 3)])
def test_heldout_size_keeps_every_label_on_both_sides(count: int, expected: int) -> None:
    """A label with two or more postings gives at least one to each side."""
    assert heldout_size(count, HELDOUT_SHARE) == expected


def test_split_is_stratified_and_complete() -> None:
    """Every posting lands on exactly one side and each label is split by the share."""
    postings = _postings(10, 10, 20)

    dev, heldout = split_postings(postings)

    assert sorted(p.id for p in dev + heldout) == sorted(p.id for p in postings)
    assert label_counts(heldout) == {"APPLY": 3, "PREPARE": 3, "SKIP": 6}
    assert label_counts(dev) == {"APPLY": 7, "PREPARE": 7, "SKIP": 14}


def test_split_does_not_depend_on_file_order() -> None:
    """The same postings split the same way however the file is sorted."""
    postings = _postings(10, 10, 20)

    _, heldout = split_postings(postings)
    _, heldout_reversed = split_postings(list(reversed(postings)))

    assert {p.id for p in heldout} == {p.id for p in heldout_reversed}


def test_split_changes_with_the_seed() -> None:
    """A different seed picks different held-out postings."""
    postings = _postings(10, 10, 20)

    _, first = split_postings(postings, seed=1)
    _, second = split_postings(postings, seed=2)

    assert {p.id for p in first} != {p.id for p in second}


def test_write_postings_round_trips(tmp_path: Path) -> None:
    """Postings written to a file load back unchanged."""
    path = tmp_path / "data" / "out.jsonl"
    postings = _postings(1, 1, 1)

    write_postings(path, postings)

    assert load_postings(path) == postings


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("a", "APPLY"), (" Apply ", "APPLY"), ("P", "PREPARE"), ("skip", "SKIP"), ("x", None)],
)
def test_parse_label_accepts_letters_and_words(answer: str, expected: str | None) -> None:
    """A label can be typed as its first letter or in full, in any case."""
    assert parse_label(answer) == expected


def test_labeling_order_is_fixed_and_not_the_file_order() -> None:
    """Postings are shown in a repeatable shuffled order."""
    postings = _postings(5, 5, 5)

    first = [p.id for p in labeling_order(postings)]
    second = [p.id for p in labeling_order(list(reversed(postings)))]

    assert first == second
    assert first != [p.id for p in postings]


class FakeTerminal:
    """Stands in for the candidate at the keyboard: scripted answers, recorded output."""

    def __init__(self, *answers: str) -> None:
        """Store the answers to give, in order.

        Args:
            *answers: What the candidate types at each prompt.
        """
        self.answers = list(answers)
        self.shown: list[str] = []
        self.saved: list[LabelRecord] = []

    def ask(self, prompt: str) -> str:
        """Return the next scripted answer.

        Args:
            prompt: Prompt shown to the candidate.

        Returns:
            The scripted answer.
        """
        self.shown.append(prompt)
        return self.answers.pop(0)

    def show(self, text: str) -> None:
        """Record text printed to the candidate.

        Args:
            text: Printed text.
        """
        self.shown.append(text)


def _session(terminal: FakeTerminal, postings: list[Posting], done: frozenset[str]) -> bool:
    """Run a labeling session against a fake terminal.

    Args:
        terminal: Scripted candidate.
        postings: Rows to label.
        done: Ids labeled earlier.

    Returns:
        What ``label_session`` returned.
    """
    return label_session(
        postings, done, ask=terminal.ask, show=terminal.show, save=terminal.saved.append
    )


def test_label_session_never_shows_the_suggested_label_or_note() -> None:
    """Only company, role and text are shown; the row's label and note stay hidden."""
    posting = Posting(**_row(label="PREPARE", note="HIDDEN-NOTE", text="Posting body."))
    terminal = FakeTerminal("s", "wrong stack")

    finished = _session(terminal, [posting], frozenset())

    printed = "\n".join(terminal.shown)
    assert finished is True
    assert "Posting body." in printed
    assert "Acme / Senior Mobile Engineer" in printed
    assert "HIDDEN-NOTE" not in printed
    assert "PREPARE" not in printed.replace("[p]repare", "")
    assert terminal.saved == [
        LabelRecord(id="acme-1", suggested="PREPARE", own="SKIP", note="wrong stack")
    ]


def test_label_session_asks_again_after_an_invalid_answer() -> None:
    """An answer that is not a label is not saved; the prompt repeats."""
    terminal = FakeTerminal("maybe", "a", "")

    _session(terminal, [Posting(**_row())], frozenset())

    assert [record.own for record in terminal.saved] == ["APPLY"]


def test_label_session_skips_postings_labeled_earlier() -> None:
    """A second session continues where the first one stopped."""
    postings = _postings(1, 1, 1)
    terminal = FakeTerminal("a", "")

    finished = _session(terminal, postings, frozenset({"p-0", "p-1"}))

    assert finished is True
    assert [record.id for record in terminal.saved] == ["p-2"]


def test_label_session_can_be_quit_and_keeps_what_was_saved() -> None:
    """Quitting ends the session as unfinished without losing earlier answers."""
    postings = _postings(1, 1, 1)
    terminal = FakeTerminal("a", "", "q")

    finished = _session(terminal, postings, frozenset())

    assert finished is False
    assert len(terminal.saved) == 1


def test_apply_labels_updates_existing_rows_and_appends_new_ones() -> None:
    """Own labels replace suggested ones; rows new to the set are added."""
    existing = [Posting(**_row(id="old", label="APPLY", note="suggested"))]
    source = [existing[0], Posting(**_row(id="new", label="SKIP", note="suggested"))]
    records = [
        LabelRecord(id="old", suggested="APPLY", own="PREPARE", note="mine"),
        LabelRecord(id="new", suggested="SKIP", own="SKIP", note="also mine"),
    ]

    merged = apply_labels(existing, source, records)

    assert [(p.id, p.label, p.note) for p in merged] == [
        ("old", "PREPARE", "mine"),
        ("new", "SKIP", "also mine"),
    ]
    assert existing[0].label == "APPLY"


def test_format_agreement_reports_differences_and_kappa() -> None:
    """Full agreement on a mixed set is kappa 1; each difference is listed."""
    same = [
        LabelRecord(id="a", suggested="APPLY", own="APPLY"),
        LabelRecord(id="b", suggested="SKIP", own="SKIP"),
    ]
    differing = [*same, LabelRecord(id="c", suggested="PREPARE", own="SKIP")]

    assert "kappa 1.00" in format_agreement(same)
    assert "0 differ" in format_agreement(same)
    assert "c: suggested PREPARE, yours SKIP" in format_agreement(differing)


def test_cli_label_writes_own_labels_and_the_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Labeling a file of new rows adds them to the test set with the typed labels."""
    data = _write(tmp_path / "set.jsonl", _row(id="old"))
    source = _write(tmp_path / "new.jsonl", _row(id="new", label="APPLY", note="suggested"))
    log = tmp_path / "log.jsonl"
    terminal = FakeTerminal("s", "not for me")

    exit_code = main(
        ["label", "--source", str(source), "--data", str(data), "--log", str(log)],
        ask=terminal.ask,
    )

    assert exit_code == 0
    assert [(p.id, p.label, p.note) for p in load_postings(data)] == [
        ("old", "APPLY", "main stack matches"),
        ("new", "SKIP", "not for me"),
    ]
    assert load_label_log(log) == [
        LabelRecord(id="new", suggested="APPLY", own="SKIP", note="not for me")
    ]
    assert "1 differ from the suggested label" in capsys.readouterr().out


def test_cli_label_changes_nothing_until_every_row_is_labeled(tmp_path: Path) -> None:
    """Quitting early keeps the log but leaves the test set as it was."""
    data = _write(tmp_path / "set.jsonl", _row(id="one"), _row(id="two"))
    before = data.read_text(encoding="utf-8")
    log = tmp_path / "log.jsonl"
    terminal = FakeTerminal("s", "", "q")

    main(["label", "--source", str(data), "--data", str(data), "--log", str(log)], ask=terminal.ask)

    assert data.read_text(encoding="utf-8") == before
    assert len(load_label_log(log)) == 1


def test_cli_split_writes_both_files_and_refuses_to_split_twice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A second split would move postings across the held-out line, so it is refused."""
    complete = _postings(MIN_PER_LABEL, MIN_PER_LABEL, MIN_POSTINGS)
    data = _write(tmp_path / "set.jsonl", *[posting.model_dump() for posting in complete])
    dev, heldout = tmp_path / "dev.jsonl", tmp_path / "heldout.jsonl"
    argv = ["split", "--data", str(data), "--dev", str(dev), "--heldout", str(heldout)]

    first = main(argv)
    second = main(argv)

    assert first == 0
    assert len(load_postings(dev)) + len(load_postings(heldout)) == len(complete)
    assert second == 1
    assert "already exists" in capsys.readouterr().err


def test_cli_split_refuses_an_incomplete_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A set below the required size is not split without the explicit flag."""
    data = _write(tmp_path / "set.jsonl", _row())
    argv = [
        "split",
        "--data",
        str(data),
        "--dev",
        str(tmp_path / "d"),
        "--heldout",
        str(tmp_path / "h"),
    ]

    assert main(argv) == 1
    assert "incomplete" in capsys.readouterr().err


def test_confirm_mode_shows_the_suggestion_and_enter_accepts_it() -> None:
    """With the suggestion on screen, an empty answer keeps its label and its note."""
    posting = Posting(**_row(label="PREPARE", note="one gap"))
    terminal = FakeTerminal("", "")

    finished = label_session(
        [posting],
        frozenset(),
        ask=terminal.ask,
        show=terminal.show,
        save=terminal.saved.append,
        show_suggestion=True,
    )

    assert finished is True
    assert "Suggested: PREPARE (one gap)" in terminal.shown
    assert terminal.saved == [
        LabelRecord(
            id="acme-1", suggested="PREPARE", own="PREPARE", note="one gap", saw_suggestion=True
        )
    ]


def test_confirm_mode_lets_the_candidate_change_label_and_note() -> None:
    """A typed label and reason replace the suggested ones."""
    posting = Posting(**_row(label="PREPARE", note="one gap"))
    terminal = FakeTerminal("s", "wrong city")

    label_session(
        [posting],
        frozenset(),
        ask=terminal.ask,
        show=terminal.show,
        save=terminal.saved.append,
        show_suggestion=True,
    )

    record = terminal.saved[0]
    assert (record.suggested, record.own, record.note) == ("PREPARE", "SKIP", "wrong city")
    assert record.saw_suggestion is True


def test_blind_mode_records_that_the_suggestion_was_hidden() -> None:
    """A label given without the suggestion on screen is marked as such."""
    terminal = FakeTerminal("a", "")

    _session(terminal, [Posting(**_row())], frozenset())

    assert terminal.saved[0].saw_suggestion is False


def test_blind_mode_does_not_accept_an_empty_answer() -> None:
    """Without a suggestion on screen, Enter is not a label; the prompt repeats."""
    terminal = FakeTerminal("", "p", "")

    _session(terminal, [Posting(**_row(label="SKIP"))], frozenset())

    assert [record.own for record in terminal.saved] == ["PREPARE"]


def test_agreement_report_counts_labels_given_with_the_suggestion_on_screen() -> None:
    """The report says how many labels were confirmed rather than given blind."""
    records = [
        LabelRecord(id="a", suggested="APPLY", own="APPLY", saw_suggestion=True),
        LabelRecord(id="b", suggested="SKIP", own="SKIP"),
    ]

    assert "suggestion on screen while labeling: 1 of 2" in format_agreement(records)


def test_cli_label_confirm_mode_accepts_suggestions_with_enter(tmp_path: Path) -> None:
    """``--show-suggestion`` lets a whole file be confirmed with the Enter key."""
    data = tmp_path / "set.jsonl"
    source = _write(tmp_path / "new.jsonl", _row(id="new", label="SKIP", note="suggested"))
    log = tmp_path / "log.jsonl"
    terminal = FakeTerminal("", "")

    exit_code = main(
        [
            "label",
            "--source",
            str(source),
            "--data",
            str(data),
            "--log",
            str(log),
            "--show-suggestion",
        ],
        ask=terminal.ask,
    )

    assert exit_code == 0
    assert [(p.id, p.label, p.note) for p in load_postings(data)] == [("new", "SKIP", "suggested")]
    assert load_label_log(log)[0].saw_suggestion is True
