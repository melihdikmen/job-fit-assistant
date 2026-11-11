"""Build the model comparison table from saved eval runs.

Run it with::

    uv run python -m job_fit.compare data/results/<run>.json data/results/<run>.json ...

It prints a Markdown table (one row per run, plus the majority baseline) and marks the
runs on the Pareto front of weighted error, cost and latency. With ``--table versions``
it prints the workflow-vs-agent table instead.
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from job_fit.eval import COST_REPORT_UNIT, DEFAULT_VERSION, EvalReport
from job_fit.workflow import DEFAULT_REASONING

VERTEX_PROVIDER = "vertex"
BASELINE_NAME = "Baseline"
PARETO_NOTE = "Pareto front"
UNKNOWN_COST = "unknown"
TABLE_HEADER = (
    "| Model | Accuracy | Weighted error (mean) | "
    f"Cost per {COST_REPORT_UNIT} postings | Avg. latency | Notes |"
)
TABLE_DIVIDER = "|---|---|---|---|---|---|"
VERSION_TABLE_HEADER = (
    "| Version | Accuracy | Weighted error (mean) | "
    f"Cost per {COST_REPORT_UNIT} postings | Avg. latency | Avg. model calls | Avg. tool calls |"
)
VERSION_TABLE_DIVIDER = "|---|---|---|---|---|---|---|"
MODELS_TABLE = "models"
VERSIONS_TABLE = "versions"
TABLES = (MODELS_TABLE, VERSIONS_TABLE)


class CompareError(RuntimeError):
    """The given runs cannot be put into one comparison table."""


@dataclass(frozen=True)
class Row:
    """One line of the comparison table.

    Attributes:
        name: Model id with its reasoning setting, or the baseline's name.
        accuracy: Share of postings decided correctly, between 0 and 1.
        weighted_error_mean: Error cost per posting; lower is better.
        cost_per_100_usd: List-price cost per 100 postings; None if unknown.
        mean_latency_s: Mean seconds per answered posting.
        notes: Remarks for the Notes column, e.g. the provider or failed postings.
    """

    name: str
    accuracy: float
    weighted_error_mean: float
    cost_per_100_usd: float | None
    mean_latency_s: float
    notes: tuple[str, ...] = ()


def load_runs(paths: Sequence[Path]) -> list[EvalReport]:
    """Read run files and check that they can be compared.

    Args:
        paths: Run files written by ``job_fit.eval``.

    Returns:
        The reports, in the order given.

    Raises:
        CompareError: If no path is given, a file is not a valid run file, or the runs
            differ in test set, prompt or number of postings.
        OSError: If a file cannot be read.
    """
    if not paths:
        raise CompareError("Give at least one run file.")
    reports = []
    for path in paths:
        try:
            reports.append(EvalReport.model_validate_json(path.read_text(encoding="utf-8")))
        except ValidationError as err:
            raise CompareError(f"{path} is not a valid run file: {err}") from err

    reference = reports[0]
    # The workflow and the agent have different prompts by design, so the prompt has to
    # match only among runs of the same version.
    prompt_by_version: dict[str, str] = {}
    for path, report in zip(paths, reports, strict=True):
        same_set = (report.dataset_sha256, report.dataset_size) == (
            reference.dataset_sha256,
            reference.dataset_size,
        )
        same_prompt = (
            prompt_by_version.setdefault(report.version, report.prompt_sha256)
            == report.prompt_sha256
        )
        if not (same_set and same_prompt):
            raise CompareError(
                f"{path} used a different test set, prompt or number of postings than "
                f"{paths[0]}; the runs are not comparable."
            )
    return reports


def _run_name(report: EvalReport) -> str:
    """Name a run by its model and, if it was set, its reasoning setting.

    Args:
        report: Report of the run.

    Returns:
        E.g. "gemini-2.5-flash (reasoning off)" or "gemini-2.5-flash (agent, reasoning on)".
    """
    details = [] if report.version == DEFAULT_VERSION else [report.version]
    if report.reasoning != DEFAULT_REASONING:
        details.append(f"reasoning {report.reasoning}")
    return f"{report.model} ({', '.join(details)})" if details else report.model


def _run_notes(report: EvalReport) -> tuple[str, ...]:
    """Collect the remarks a reader needs to interpret a run's numbers.

    Args:
        report: Report of the run.

    Returns:
        Provider (when it is not Vertex) and the number of failed postings, if any.
    """
    notes = []
    if report.provider != VERTEX_PROVIDER:
        notes.append(f"via {report.provider}")
    if report.final.failed:
        notes.append(f"{report.final.failed} failed")
    return tuple(notes)


def rows_from_reports(reports: Sequence[EvalReport]) -> list[Row]:
    """Turn comparable reports into table rows, baseline first.

    Runs that share a name (a repeated run) are numbered in the order given.

    Args:
        reports: Reports returned by ``load_runs``.

    Returns:
        The majority-baseline row followed by one row per run.

    Raises:
        CompareError: If there are no reports or the first has no majority baseline.
    """
    if not reports:
        raise CompareError("Give at least one run file.")
    majority = next((b for b in reports[0].baselines if b.is_majority), None)
    if majority is None:
        raise CompareError("The first run file has no majority baseline.")
    rows = [
        Row(
            name=BASELINE_NAME,
            accuracy=majority.scores.accuracy,
            weighted_error_mean=majority.scores.weighted_error_mean,
            cost_per_100_usd=0.0,
            mean_latency_s=0.0,
            notes=(f"always {majority.label}",),
        )
    ]
    seen: dict[str, int] = {}
    for report in reports:
        name = _run_name(report)
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name}, run {seen[name]}"
        rows.append(
            Row(
                name=name,
                accuracy=report.final.accuracy,
                weighted_error_mean=report.final.weighted_error_mean,
                cost_per_100_usd=report.cost_per_100_usd,
                mean_latency_s=report.mean_latency_s,
                notes=_run_notes(report),
            )
        )
    return rows


def _dominates(first: Row, second: Row) -> bool:
    """Tell whether one row is at least as good everywhere and better somewhere.

    Args:
        first: Candidate for the better row; its cost must be known.
        second: Row it is compared with; its cost must be known.

    Returns:
        True if ``first`` dominates ``second`` on weighted error, cost and latency.
    """
    first_values = (first.weighted_error_mean, first.cost_per_100_usd, first.mean_latency_s)
    second_values = (second.weighted_error_mean, second.cost_per_100_usd, second.mean_latency_s)
    pairs = list(zip(first_values, second_values, strict=True))
    return all(a <= b for a, b in pairs) and any(a < b for a, b in pairs)


def pareto_front(rows: Sequence[Row]) -> list[Row]:
    """Select the rows no other row beats on every measure at once.

    A row is on the front if no other row is at least as good on mean weighted error,
    cost per 100 postings and mean latency, and strictly better on one of them. Rows
    with an unknown cost cannot be compared and are left out.

    Args:
        rows: Candidate rows. Do not include the baseline: it costs nothing and takes
            no time, so nothing could ever dominate it.

    Returns:
        The rows on the front, in their original order.
    """
    priced = [row for row in rows if row.cost_per_100_usd is not None]
    return [row for row in priced if not any(_dominates(other, row) for other in priced)]


def format_markdown_table(rows: Sequence[Row]) -> str:
    """Render rows as the comparison table of ``docs/results.md``.

    Args:
        rows: Rows from ``rows_from_reports``; the baseline row is recognized by name
            and is not a Pareto candidate.

    Returns:
        A Markdown table with a Pareto mark in the Notes column.
    """
    front = pareto_front([row for row in rows if row.name != BASELINE_NAME])
    lines = [TABLE_HEADER, TABLE_DIVIDER]
    for row in rows:
        cost = UNKNOWN_COST if row.cost_per_100_usd is None else f"${row.cost_per_100_usd:.2f}"
        notes = list(row.notes) + ([PARETO_NOTE] if row in front else [])
        lines.append(
            f"| {row.name} | {row.accuracy:.1%} | {row.weighted_error_mean:.2f} | {cost} | "
            f"{row.mean_latency_s:.1f} s | {'; '.join(notes)} |"
        )
    return "\n".join(lines)


def format_version_table(reports: Sequence[EvalReport]) -> str:
    """Render the workflow-vs-agent table of ``docs/results.md``.

    Args:
        reports: Reports returned by ``load_runs``, one row each. Repeated runs of a
            version are numbered in the order given.

    Returns:
        A Markdown table with model calls and tool calls per posting.
    """
    lines = [VERSION_TABLE_HEADER, VERSION_TABLE_DIVIDER]
    seen: dict[str, int] = {}
    for report in reports:
        seen[report.version] = seen.get(report.version, 0) + 1
        name = f"{report.version.capitalize()}, run {seen[report.version]}"
        cost = (
            UNKNOWN_COST if report.cost_per_100_usd is None else f"${report.cost_per_100_usd:.2f}"
        )
        # Run files from before the field existed are workflow runs without retries.
        model_calls = 1.0 if report.mean_model_calls is None else report.mean_model_calls
        lines.append(
            f"| {name} | {report.final.accuracy:.1%} | {report.final.weighted_error_mean:.2f} | "
            f"{cost} | {report.mean_latency_s:.1f} s | {model_calls:.2f} | "
            f"{report.mean_tool_calls:.2f} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the comparison table for the given run files.

    Args:
        argv: Arguments without the program name; None reads ``sys.argv``.

    Returns:
        Process exit code: 0 on success, 1 on a handled error.
    """
    parser = argparse.ArgumentParser(description="Compare saved eval runs in one table.")
    parser.add_argument("runs", type=Path, nargs="+", help="Run files from data/results/.")
    parser.add_argument(
        "--table",
        choices=TABLES,
        default=MODELS_TABLE,
        help="models: one row per model run, with the Pareto front. "
        "versions: workflow vs. agent, with model and tool calls.",
    )
    args = parser.parse_args(argv)
    try:
        reports = load_runs(args.runs)
        rows = rows_from_reports(reports)
    except (OSError, CompareError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    if args.table == VERSIONS_TABLE:
        print(format_version_table(reports))
    else:
        print(format_markdown_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
