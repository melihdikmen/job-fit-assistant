"""Labeled test set: load, validate and extend ``data/postings.jsonl``.

One posting per line. Labels are set by the candidate, never by a model.

Add a posting from a text file (one command, wrapped here for readability)::

    uv run python -m job_fit.dataset add --id acme-2026-11 --company Acme
        --role "Senior Mobile Engineer" --label APPLY --file posting.txt --note "..."

Check the size and balance of the set::

    uv run python -m job_fit.dataset check
"""

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

from job_fit.metrics import LABELS
from job_fit.schema import DecisionLabel

DEFAULT_DATA_PATH = Path("data/postings.jsonl")
MIN_POSTINGS = 20
MIN_PER_LABEL = 5

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


def label_counts(postings: Sequence[Posting]) -> dict[DecisionLabel, int]:
    """Count postings per label.

    Args:
        postings: Postings to count.

    Returns:
        A count for every label in ``LABELS`` order, zero included.
    """
    counts = Counter(posting.label for posting in postings)
    return {label: counts[label] for label in LABELS}


def check_coverage(postings: Sequence[Posting]) -> list[str]:
    """Check that the set is large and balanced enough to be meaningful.

    Args:
        postings: Postings to check.

    Returns:
        One message per shortfall; empty if the set has at least ``MIN_POSTINGS``
        postings and at least ``MIN_PER_LABEL`` of every label.
    """
    problems = []
    if len(postings) < MIN_POSTINGS:
        problems.append(f"{len(postings)} postings, need at least {MIN_POSTINGS}")
    for label, count in label_counts(postings).items():
        if count < MIN_PER_LABEL:
            problems.append(f"{count} postings labeled {label}, need at least {MIN_PER_LABEL}")
    return problems


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
        Parsed arguments; ``command`` is "add" or "check".
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

    for command in (add, check):
        command.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="JSONL file.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``add`` or ``check`` command.

    Args:
        argv: Arguments without the program name; None reads ``sys.argv``.

    Returns:
        Process exit code: 0 on success, 1 on a handled error or an incomplete set.
    """
    args = _parse_args(argv)
    try:
        if args.command == "add":
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
        postings = load_postings(args.data)
    except ValidationError as err:
        print(f"error: {_describe_problems(err)}", file=sys.stderr)
        return 1
    except (OSError, DatasetError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    print(format_summary(postings))
    if args.command == "check" and check_coverage(postings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
