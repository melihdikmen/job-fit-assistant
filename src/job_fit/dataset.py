"""Labeled test set: load, validate, label and split ``data/postings.jsonl``.

One posting per line. Labels are set by the candidate, never by a model.

Add a posting from a text file (one command, wrapped here for readability)::

    uv run python -m job_fit.dataset add --id acme-2026-11 --company Acme
        --role "Senior Mobile Engineer" --label APPLY --file posting.txt --note "..."

Label postings yourself, one at a time, without seeing any suggested label::

    uv run python -m job_fit.dataset label --source data/postings.jsonl

Check the size and balance of the set, then split it into a dev and a held-out file::

    uv run python -m job_fit.dataset check
    uv run python -m job_fit.dataset split
"""

import argparse
import random
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

from job_fit.metrics import LABELS, cohen_kappa
from job_fit.schema import DecisionLabel

AskFn = Callable[[str], str]
ShowFn = Callable[[str], None]
SaveFn = Callable[["LabelRecord"], None]

DEFAULT_DATA_PATH = Path("data/postings.jsonl")
DEV_DATA_PATH = Path("data/postings_dev.jsonl")
HELDOUT_DATA_PATH = Path("data/postings_heldout.jsonl")
LABEL_LOG_PATH = Path("data/label_log.jsonl")

MIN_POSTINGS = 40
MIN_PER_LABEL = 8
# Share of every label that is set aside and looked at once, at the very end.
HELDOUT_SHARE = 0.3
SPLIT_SEED = 2026
# Postings are shown for labeling in a shuffled order: the files are roughly sorted by
# label, so file order would give the answer away.
LABEL_ORDER_SEED = 2026

QUIT_ANSWER = "q"
LABEL_ANSWERS: dict[str, DecisionLabel] = {
    "a": "APPLY",
    "apply": "APPLY",
    "p": "PREPARE",
    "prepare": "PREPARE",
    "s": "SKIP",
    "skip": "SKIP",
}
LABEL_PROMPT = "Your label: [a]pply / [p]repare / [s]kip, or [q]uit and continue later > "
NOTE_PROMPT = "Your reason in a few words (Enter for none) > "
CONFIRM_LABEL_PROMPT = (
    "Enter to accept the suggestion, or [a]pply / [p]repare / [s]kip to change it, "
    "or [q]uit and continue later > "
)
CONFIRM_NOTE_PROMPT = "Your reason in a few words (Enter keeps the suggested note) > "
SEPARATOR = "=" * 78

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class DatasetError(ValueError):
    """The test set file is missing, malformed or inconsistent."""


class Posting(BaseModel):
    """One labeled job posting.

    Only ``text`` is ever shown to a model. The other fields stay on the evaluation
    side; sending ``label`` or ``note`` would leak the answer.

    Attributes:
        id: Unique identifier, e.g. "acme-2026-11".
        company: Hiring company, for the error report.
        role: Job title, for the error report.
        text: Full posting text as published.
        label: The candidate's own decision, hard rules included.
        note: The candidate's short reason for the label.
    """

    model_config = ConfigDict(extra="forbid")

    id: NonEmptyText
    company: NonEmptyText
    role: NonEmptyText
    text: NonEmptyText
    label: DecisionLabel
    note: str = ""


class LabelRecord(BaseModel):
    """One labeling decision by the candidate, next to the label that was suggested.

    Attributes:
        id: Posting id.
        suggested: Label the row carried before the candidate looked at it.
        own: Label the candidate chose.
        note: The candidate's reason.
        saw_suggestion: True if the suggested label was on screen when the candidate
            chose. Such a label is confirmed by the candidate, not independent of the
            suggestion.
    """

    model_config = ConfigDict(extra="forbid")

    id: NonEmptyText
    suggested: DecisionLabel
    own: DecisionLabel
    note: str = ""
    saw_suggestion: bool = False


def _describe_problems(error: ValidationError) -> str:
    """Summarize a validation error without echoing the posting text.

    Args:
        error: Error raised while validating one row.

    Returns:
        The problems on one line, e.g. "label: Input should be 'APPLY', ...".
    """
    parts = []
    for problem in error.errors(include_input=False, include_url=False):
        field = ".".join(str(part) for part in problem["loc"]) or "row"
        parts.append(f"{field}: {problem['msg']}")
    return "; ".join(parts)


def load_postings(path: Path) -> list[Posting]:
    """Read and validate the test set.

    Args:
        path: JSONL file with one posting per line. Blank lines are ignored.

    Returns:
        The postings in file order.

    Raises:
        DatasetError: If the file is missing or empty, a line is not valid JSON, a
            field is missing or invalid, or an id occurs twice. The message names the
            line.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as err:
        raise DatasetError(f"Test set not found: {path}") from err

    postings: list[Posting] = []
    line_of_id: dict[str, int] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            posting = Posting.model_validate_json(line)
        except ValidationError as err:
            raise DatasetError(f"{path}:{line_number}: {_describe_problems(err)}") from err
        if posting.id in line_of_id:
            raise DatasetError(
                f"{path}:{line_number}: duplicate id '{posting.id}' "
                f"(first used on line {line_of_id[posting.id]})"
            )
        line_of_id[posting.id] = line_number
        postings.append(posting)

    if not postings:
        raise DatasetError(f"Test set is empty: {path}")
    return postings


def write_postings(path: Path, postings: Sequence[Posting]) -> None:
    """Write postings as one correctly escaped JSON line each, replacing the file.

    Args:
        path: JSONL file; its folder is created if missing.
        postings: Postings to write, in this order.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for posting in postings:
            file.write(f"{posting.model_dump_json()}\n")


def label_counts(postings: Sequence[Posting]) -> dict[DecisionLabel, int]:
    """Count postings per label.

    Args:
        postings: Postings to count.

    Returns:
        A count for every label in ``LABELS`` order, zero included.
    """
    counts = Counter(posting.label for posting in postings)
    return {label: counts[label] for label in LABELS}


def check_coverage(
    postings: Sequence[Posting],
    min_postings: int = MIN_POSTINGS,
    min_per_label: int = MIN_PER_LABEL,
) -> list[str]:
    """Check that a set is large and balanced enough to be meaningful.

    Args:
        postings: Postings to check.
        min_postings: Smallest acceptable number of postings.
        min_per_label: Smallest acceptable number of postings of each label.

    Returns:
        One message per shortfall; empty if the set meets both minimums.
    """
    problems = []
    if len(postings) < min_postings:
        problems.append(f"{len(postings)} postings, need at least {min_postings}")
    for label, count in label_counts(postings).items():
        if count < min_per_label:
            problems.append(f"{count} postings labeled {label}, need at least {min_per_label}")
    return problems


def heldout_size(count: int, heldout_share: float = HELDOUT_SHARE) -> int:
    """Decide how many of ``count`` postings of one label are held out.

    Args:
        count: Number of postings with the label.
        heldout_share: Share to hold out, between 0 and 1.

    Returns:
        The rounded share, but at least 1 as soon as there are two postings, so that
        every label appears on both sides of the split.
    """
    if count < 2:
        return 0
    return min(count - 1, max(1, round(count * heldout_share)))


# What the dev file of a complete set must still hold after the split.
MIN_DEV_POSTINGS = MIN_POSTINGS - heldout_size(MIN_POSTINGS)
MIN_DEV_PER_LABEL = MIN_PER_LABEL - heldout_size(MIN_PER_LABEL)


def split_postings(
    postings: Sequence[Posting],
    heldout_share: float = HELDOUT_SHARE,
    seed: int = SPLIT_SEED,
) -> tuple[list[Posting], list[Posting]]:
    """Split the set into a dev part and a held-out part, label by label.

    The choice is made by a seeded shuffle of the ids, not by a person, so nobody can
    steer easy postings to one side, and the same set always splits the same way.

    Args:
        postings: The complete labeled set.
        heldout_share: Share of every label to hold out.
        seed: Seed of the shuffle.

    Returns:
        The dev postings and the held-out postings, each in the order of the input.
    """
    heldout_ids: set[str] = set()
    for label in LABELS:
        ids = sorted(posting.id for posting in postings if posting.label == label)
        random.Random(f"{seed}:{label}").shuffle(ids)
        heldout_ids.update(ids[: heldout_size(len(ids), heldout_share)])
    dev = [posting for posting in postings if posting.id not in heldout_ids]
    heldout = [posting for posting in postings if posting.id in heldout_ids]
    return dev, heldout


def add_posting(path: Path, posting: Posting) -> None:
    """Append one posting to the test set as a correctly escaped JSON line.

    Args:
        path: JSONL file; created if it does not exist.
        posting: Posting to append.

    Raises:
        DatasetError: If the existing file is invalid or already has this id.
    """
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if existing.strip() and any(row.id == posting.id for row in load_postings(path)):
        raise DatasetError(f"{path}: id '{posting.id}' already exists")
    separator = "" if not existing or existing.endswith("\n") else "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(f"{separator}{posting.model_dump_json()}\n")


def load_label_log(path: Path) -> list[LabelRecord]:
    """Read the labeling decisions made so far.

    Args:
        path: JSONL file with one decision per line; a missing file means none yet.

    Returns:
        The decisions in the order they were made.

    Raises:
        DatasetError: If a line is not a valid record.
    """
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(LabelRecord.model_validate_json(line))
        except ValidationError as err:
            raise DatasetError(f"{path}:{line_number}: {_describe_problems(err)}") from err
    return records


def append_label_record(path: Path, record: LabelRecord) -> None:
    """Save one labeling decision right away, so a session can stop at any posting.

    Args:
        path: JSONL log file; created if missing.
        record: Decision to append.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(f"{record.model_dump_json()}\n")


def labeling_order(postings: Sequence[Posting], seed: int = LABEL_ORDER_SEED) -> list[Posting]:
    """Put postings into the order in which they are shown for labeling.

    Args:
        postings: Postings to label.
        seed: Seed of the shuffle.

    Returns:
        The same postings in a fixed pseudo-random order that does not follow labels.
    """
    ordered = sorted(postings, key=lambda posting: posting.id)
    random.Random(seed).shuffle(ordered)
    return ordered


def parse_label(answer: str) -> DecisionLabel | None:
    """Read a label typed by the candidate.

    Args:
        answer: What was typed, e.g. "a", "Apply" or "SKIP".

    Returns:
        The label, or None if the answer is not one.
    """
    return LABEL_ANSWERS.get(answer.strip().lower())


def label_session(
    postings: Sequence[Posting],
    done_ids: frozenset[str],
    *,
    ask: AskFn,
    show: ShowFn,
    save: SaveFn,
    show_suggestion: bool = False,
) -> bool:
    """Ask the candidate for a label for every posting not labeled yet.

    By default only the company, the role and the posting text are shown: never the
    label or the note the row carries, and never a model's output. With
    ``show_suggestion`` the row's label and note are shown too and an empty answer
    accepts them; that is quicker, and it is recorded, because a label chosen with the
    suggestion on screen is not independent of it.

    Args:
        postings: Rows to label, each carrying a suggested label.
        done_ids: Ids already labeled in an earlier session.
        ask: Shows a prompt and returns the typed answer.
        show: Prints text to the candidate.
        save: Stores one decision; called right after each answer.
        show_suggestion: True to show the suggested label and note and let Enter
            accept them.

    Returns:
        True if every posting is labeled, False if the candidate quit before the end.
    """
    pending = [posting for posting in labeling_order(postings) if posting.id not in done_ids]
    for position, posting in enumerate(pending, start=1):
        show(SEPARATOR)
        show(f"[{position} of {len(pending)} left] {posting.company} / {posting.role}")
        show(SEPARATOR)
        show(posting.text)
        show("")
        if show_suggestion:
            show(f"Suggested: {posting.label}" + (f" ({posting.note})" if posting.note else ""))
        own = None
        while own is None:
            answer = ask(CONFIRM_LABEL_PROMPT if show_suggestion else LABEL_PROMPT).strip()
            if answer.lower() == QUIT_ANSWER:
                return False
            own = posting.label if show_suggestion and not answer else parse_label(answer)
        note = ask(CONFIRM_NOTE_PROMPT if show_suggestion else NOTE_PROMPT).strip()
        if show_suggestion and not note:
            note = posting.note
        save(
            LabelRecord(
                id=posting.id,
                suggested=posting.label,
                own=own,
                note=note,
                saw_suggestion=show_suggestion,
            )
        )
    return True


def apply_labels(
    existing: Sequence[Posting], source: Sequence[Posting], records: Sequence[LabelRecord]
) -> list[Posting]:
    """Write the candidate's own labels into the test set.

    Args:
        existing: Postings already in the test set.
        source: The rows that were labeled. Rows whose id is not in the test set yet
            are appended to it.
        records: The candidate's decisions; the last one per id wins.

    Returns:
        The test set with the candidate's label and note on every labeled posting.
    """
    own = {record.id: record for record in records}
    known_ids = {posting.id for posting in existing}
    merged = [*existing, *(posting for posting in source if posting.id not in known_ids)]
    return [
        posting.model_copy(update={"label": own[posting.id].own, "note": own[posting.id].note})
        if posting.id in own
        else posting
        for posting in merged
    ]


def format_agreement(records: Sequence[LabelRecord]) -> str:
    """Report how far the candidate's labels agree with the suggested ones.

    Args:
        records: Labeling decisions.

    Returns:
        How many labels differ, Cohen's kappa between the two, and each difference.
    """
    if not records:
        return "no postings labeled yet"
    suggested = [record.suggested for record in records]
    own = [record.own for record in records]
    changed = [record for record in records if record.suggested != record.own]
    seen = sum(1 for record in records if record.saw_suggestion)
    lines = [
        f"labeled: {len(records)} postings, {len(changed)} differ from the suggested label",
        f"agreement with the suggestions: kappa {cohen_kappa(suggested, own):.2f}",
        f"suggestion on screen while labeling: {seen} of {len(records)}",
    ]
    lines += [
        f"  {record.id}: suggested {record.suggested}, yours {record.own}" for record in changed
    ]
    return "\n".join(lines)


def format_summary(postings: Sequence[Posting]) -> str:
    """Render the size and balance of the set for the terminal.

    Args:
        postings: Postings to summarize.

    Returns:
        The total, the count per label and any coverage shortfalls.
    """
    lines = [f"postings: {len(postings)}"]
    lines += [f"  {label:<8}{count}" for label, count in label_counts(postings).items()]
    problems = check_coverage(postings)
    lines += [f"incomplete: {problem}" for problem in problems]
    if not problems:
        lines.append("coverage: ok")
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Arguments without the program name; None reads ``sys.argv``.

    Returns:
        Parsed arguments; ``command`` is "add", "check", "label" or "split".
    """
    parser = argparse.ArgumentParser(description="Maintain the labeled test set.")
    commands = parser.add_subparsers(dest="command", required=True)

    add = commands.add_parser("add", help="Append one posting from a text file.")
    add.add_argument("--id", required=True, help="Unique id, e.g. acme-2026-11.")
    add.add_argument("--company", required=True, help="Hiring company.")
    add.add_argument("--role", required=True, help="Job title.")
    add.add_argument("--label", required=True, choices=LABELS, help="Your decision.")
    add.add_argument("--file", type=Path, required=True, help="Text file with the posting.")
    add.add_argument("--note", default="", help="Short reason for the label.")

    check = commands.add_parser("check", help="Validate the set and print its balance.")

    label = commands.add_parser(
        "label", help="Label postings yourself, without seeing the suggested labels."
    )
    label.add_argument(
        "--source",
        type=Path,
        required=True,
        help="JSONL rows to label; may be the test set itself or a file of new postings.",
    )
    label.add_argument("--log", type=Path, default=LABEL_LOG_PATH, help="Decision log (JSONL).")
    label.add_argument(
        "--show-suggestion",
        action="store_true",
        help="Show the suggested label; Enter accepts it. Quicker, but not independent.",
    )

    split = commands.add_parser("split", help="Write the dev and the held-out file.")
    split.add_argument("--dev", type=Path, default=DEV_DATA_PATH, help="Dev set to write.")
    split.add_argument("--heldout", type=Path, default=HELDOUT_DATA_PATH, help="Held-out set.")
    split.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing split. This moves postings between dev and held-out.",
    )
    split.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Split even if the set is smaller or less balanced than required.",
    )

    for command in (add, check, label, split):
        command.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="JSONL file.")
    return parser.parse_args(argv)


def _run_add(args: argparse.Namespace) -> int:
    """Append one posting and print the balance of the set.

    Args:
        args: Parsed arguments of the ``add`` command.

    Returns:
        Process exit code 0.
    """
    posting = Posting(
        id=args.id,
        company=args.company,
        role=args.role,
        text=args.file.read_text(encoding="utf-8"),
        label=args.label,
        note=args.note,
    )
    add_posting(args.data, posting)
    print(f"added '{posting.id}' ({posting.label}) to {args.data}")
    print(format_summary(load_postings(args.data)))
    return 0


def _run_check(args: argparse.Namespace) -> int:
    """Print the balance of the set.

    Args:
        args: Parsed arguments of the ``check`` command.

    Returns:
        Process exit code: 0 if the set is complete, 1 otherwise.
    """
    postings = load_postings(args.data)
    print(format_summary(postings))
    return 1 if check_coverage(postings) else 0


def _run_label(args: argparse.Namespace, ask: AskFn) -> int:
    """Run a labeling session and, once every row is labeled, update the test set.

    Args:
        args: Parsed arguments of the ``label`` command.
        ask: Shows a prompt and returns the typed answer.

    Returns:
        Process exit code 0.
    """
    source = load_postings(args.source)
    done_ids = frozenset(record.id for record in load_label_log(args.log))
    finished = label_session(
        source,
        done_ids,
        ask=ask,
        show=print,
        save=lambda record: append_label_record(args.log, record),
        show_suggestion=args.show_suggestion,
    )
    source_ids = {posting.id for posting in source}
    records = [record for record in load_label_log(args.log) if record.id in source_ids]
    if not finished:
        print(f"\nstopped: {len(records)} of {len(source)} labeled; run the command again to go on")
        return 0
    existing = load_postings(args.data) if args.data.exists() else []
    write_postings(args.data, apply_labels(existing, source, records))
    print(f"\nwrote your labels to {args.data}")
    print(format_agreement(records))
    print(format_summary(load_postings(args.data)))
    return 0


def _run_split(args: argparse.Namespace) -> int:
    """Write the dev and the held-out file from the complete set.

    Args:
        args: Parsed arguments of the ``split`` command.

    Returns:
        Process exit code 0.

    Raises:
        DatasetError: If the set is incomplete without ``--allow-incomplete``, or a
            split already exists and ``--force`` was not given.
    """
    postings = load_postings(args.data)
    problems = check_coverage(postings)
    if problems and not args.allow_incomplete:
        raise DatasetError(f"Test set is incomplete ({'; '.join(problems)}).")
    existing = [path for path in (args.dev, args.heldout) if path.exists()]
    if existing and not args.force:
        raise DatasetError(
            f"{existing[0]} already exists. Splitting again moves postings between dev "
            "and held-out, which leaks held-out postings into tuning; pass --force only "
            "before any tuning has started."
        )
    dev, heldout = split_postings(postings)
    write_postings(args.dev, dev)
    write_postings(args.heldout, heldout)
    for name, path, part in (("dev", args.dev, dev), ("held-out", args.heldout, heldout)):
        counts = ", ".join(f"{label} {count}" for label, count in label_counts(part).items())
        print(f"{name}: {len(part)} postings ({counts}) -> {path}")
    return 0


def main(argv: Sequence[str] | None = None, *, ask: AskFn = input) -> int:
    """Run the ``add``, ``check``, ``label`` or ``split`` command.

    Args:
        argv: Arguments without the program name; None reads ``sys.argv``.
        ask: Shows a prompt and returns the typed answer; tests pass a fake.

    Returns:
        Process exit code: 0 on success, 1 on a handled error or an incomplete set.
    """
    args = _parse_args(argv)
    # Postings contain characters the default Windows console encoding cannot print.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        if args.command == "add":
            return _run_add(args)
        if args.command == "label":
            return _run_label(args, ask)
        if args.command == "split":
            return _run_split(args)
        return _run_check(args)
    except ValidationError as err:
        print(f"error: {_describe_problems(err)}", file=sys.stderr)
        return 1
    except (OSError, DatasetError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
