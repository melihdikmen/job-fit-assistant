"""Evaluate the workflow against the labeled test set.

Run it with::

    uv run python -m job_fit.eval --model <model-id>

It prints accuracy, the confusion matrix, the weighted error, the baselines, tokens,
estimated cost and latency, lists every wrong decision, and saves the run to
``data/results/``.
"""

import argparse
import hashlib
import re
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from job_fit.dataset import (
    DEFAULT_DATA_PATH,
    DatasetError,
    Posting,
    check_coverage,
    label_counts,
    load_postings,
)
from job_fit.llm import LLMError
from job_fit.metrics import LABELS, Scores, constant_baseline, majority_label, score
from job_fit.pricing import PRICES_CHECKED_ON, ModelPrice, call_cost_usd, price_for
from job_fit.schema import DecisionLabel
from job_fit.workflow import (
    DEFAULT_CV_PATH,
    DEFAULT_PROVIDER,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    DecisionError,
    DecisionResult,
    decide,
)

DecideFn = Callable[..., DecisionResult]
ProgressFn = Callable[[int, int, "PostingResult"], None]
Clock = Callable[[], datetime]

DEFAULT_RESULTS_DIR = Path("data/results")
# A run of failed calls means the setup is broken (wrong project, expired login), not
# that single postings are hard; stop instead of waiting for every timeout.
MAX_CONSECUTIVE_LLM_ERRORS = 3
COST_REPORT_UNIT = 100
LABEL_COLUMN_WIDTH = 9


class EvalError(RuntimeError):
    """The evaluation could not be completed."""


class PostingResult(BaseModel):
    """Outcome of the workflow for one posting.

    Attributes:
        id: Posting id.
        company: Hiring company from the test set.
        role: Job title from the test set.
        label: The candidate's label.
        note: The candidate's reason for the label.
        predicted: Final decision after the hard rules; None if the call failed.
        llm_decision: The LLM's decision before the rules; None if the call failed.
        fired_rules: Names of the rules that forced a SKIP.
        input_tokens: Prompt tokens used; 0 if the call failed.
        output_tokens: Generated tokens used; 0 if the call failed.
        latency_s: Seconds spent in LLM calls; 0 if the call failed.
        attempts: Number of LLM calls made; 0 if the call failed.
        rationale: The model's rationale, with rule notes appended.
        error: Why the call failed; empty on success.
    """

    id: str
    company: str
    role: str
    label: DecisionLabel
    note: str
    predicted: DecisionLabel | None = None
    llm_decision: DecisionLabel | None = None
    fired_rules: tuple[str, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    attempts: int = 0
    rationale: str = ""
    error: str = ""


class Baseline(BaseModel):
    """Scores of a "dumb model" that always predicts one label.

    Attributes:
        label: The label it always predicts.
        is_majority: True for the most common label, the baseline the model must beat.
        scores: Its metrics on the same postings as the model.
    """

    label: DecisionLabel
    is_majority: bool
    scores: Scores


class EvalReport(BaseModel):
    """Everything measured in one eval run; this is what gets saved.

    Attributes:
        model: Model id.
        provider: LLM provider.
        started_at: Local time the run started, ISO 8601.
        dataset_path: Test set file.
        dataset_sha256: Hash of the test set file. Runs are comparable only if equal.
        prompt_sha256: Hash of the prompt templates. Runs are comparable only if equal.
        dataset_size: Number of postings evaluated.
        label_counts: Postings evaluated per label.
        coverage_problems: Shortfalls of the full test set; empty if it is complete.
        final: Metrics of the final decision, after the hard rules.
        before_rules: Metrics of the LLM's own decision, before the rules.
        baselines: The three constant predictors.
        beats_baseline_accuracy: True if accuracy is above the majority baseline's.
        beats_baseline_weighted_error: True if the mean weighted error is below the
            majority baseline's.
        input_tokens: Prompt tokens, summed over the run.
        output_tokens: Generated tokens, summed over the run.
        price: List price used, or None if the model is not in the price table.
        prices_checked_on: Date the price table was last checked.
        cost_usd: Estimated cost of the run; None if the price is unknown.
        cost_per_100_usd: Estimated cost per 100 postings; None if the price is unknown.
        mean_latency_s: Mean seconds per answered posting.
        results: One record per posting, in test set order.
    """

    model: str
    provider: str
    started_at: str
    dataset_path: str
    dataset_sha256: str
    prompt_sha256: str
    dataset_size: int
    label_counts: dict[DecisionLabel, int]
    coverage_problems: list[str]
    final: Scores
    before_rules: Scores
    baselines: list[Baseline]
    beats_baseline_accuracy: bool
    beats_baseline_weighted_error: bool
    input_tokens: int
    output_tokens: int
    price: ModelPrice | None
    prices_checked_on: str
    cost_usd: float | None
    cost_per_100_usd: float | None
    mean_latency_s: float
    results: list[PostingResult]


def _evaluate_one(
    posting: Posting, cv: str, model: str, provider: str, decide_fn: DecideFn
) -> PostingResult:
    """Run the workflow on one posting.

    Args:
        posting: Labeled posting; only its text is passed on.
        cv: Full text of the candidate's CV.
        model: Model id.
        provider: LLM provider.
        decide_fn: Function with the signature of ``workflow.decide``.

    Returns:
        The outcome for this posting.

    Raises:
        LLMError: If the LLM call itself fails.
        DecisionError: If the model's answer stays invalid after the retry.
    """
    result = decide_fn(posting.text, cv, model=model, provider=provider)
    return PostingResult(
        id=posting.id,
        company=posting.company,
        role=posting.role,
        label=posting.label,
        note=posting.note,
        predicted=result.decision.decision,
        llm_decision=result.llm_decision,
        fired_rules=result.fired_rules,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_s=result.latency_s,
        attempts=result.attempts,
        rationale=result.decision.rationale,
    )


def _failed_result(posting: Posting, error: Exception) -> PostingResult:
    """Build the record of a posting for which no decision was produced.

    Args:
        posting: The posting that failed.
        error: What went wrong.

    Returns:
        A result without a prediction, carrying the error message.
    """
    return PostingResult(
        id=posting.id,
        company=posting.company,
        role=posting.role,
        label=posting.label,
        note=posting.note,
        error=f"{type(error).__name__}: {error}",
    )


def evaluate(
    postings: Sequence[Posting],
    cv: str,
    *,
    model: str,
    provider: str = DEFAULT_PROVIDER,
    decide_fn: DecideFn = decide,
    on_progress: ProgressFn | None = None,
) -> list[PostingResult]:
    """Run the workflow on every posting, one after the other.

    A posting that fails is recorded and the run goes on. The model sees only the
    posting text and the CV, never the label or the note.

    Args:
        postings: Labeled postings.
        cv: Full text of the candidate's CV.
        model: Model id.
        provider: LLM provider.
        decide_fn: Function with the signature of ``workflow.decide``; tests pass a fake.
        on_progress: Called after each posting with its position, the total and its
            result.

    Returns:
        One result per posting, in the same order.

    Raises:
        EvalError: If ``MAX_CONSECUTIVE_LLM_ERRORS`` LLM calls fail in a row.
    """
    results: list[PostingResult] = []
    consecutive_llm_errors = 0
    for position, posting in enumerate(postings, start=1):
        try:
            result = _evaluate_one(posting, cv, model, provider, decide_fn)
            consecutive_llm_errors = 0
        except DecisionError as err:
            result = _failed_result(posting, err)
            consecutive_llm_errors = 0
        except LLMError as err:
            result = _failed_result(posting, err)
            consecutive_llm_errors += 1
        results.append(result)
        if on_progress:
            on_progress(position, len(postings), result)
        if consecutive_llm_errors >= MAX_CONSECUTIVE_LLM_ERRORS:
            raise EvalError(
                f"{consecutive_llm_errors} LLM calls failed in a row; stopping. "
                f"Last error: {result.error}"
            )
    return results


def sha256_text(text: str) -> str:
    """Hash a text, to tell whether two runs used the same input.

    Args:
        text: Text to hash.

    Returns:
        The SHA-256 digest in hexadecimal.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_report(
    results: Sequence[PostingResult],
    *,
    model: str,
    provider: str,
    started_at: datetime,
    dataset_path: Path,
    dataset_sha256: str,
    coverage_problems: Sequence[str],
) -> EvalReport:
    """Compute all metrics for a finished run.

    Args:
        results: One result per posting.
        model: Model id.
        provider: LLM provider.
        started_at: When the run started.
        dataset_path: Test set file.
        dataset_sha256: Hash of the test set file.
        coverage_problems: Shortfalls of the full test set.

    Returns:
        The report to print and save.

    Raises:
        ValueError: If there are no results.
    """
    actual = [result.label for result in results]
    final = score(actual, [result.predicted for result in results])
    before_rules = score(actual, [result.llm_decision for result in results])

    majority = majority_label(actual)
    baselines = [
        Baseline(
            label=label, is_majority=label == majority, scores=constant_baseline(actual, label)
        )
        for label in LABELS
    ]
    majority_scores = constant_baseline(actual, majority)

    input_tokens = sum(result.input_tokens for result in results)
    output_tokens = sum(result.output_tokens for result in results)
    cost_usd = call_cost_usd(model, input_tokens, output_tokens)
    answered = [result for result in results if result.predicted is not None]
    mean_latency_s = (
        sum(result.latency_s for result in answered) / len(answered) if answered else 0.0
    )
    counts = {label: actual.count(label) for label in LABELS}

    return EvalReport(
        model=model,
        provider=provider,
        started_at=started_at.isoformat(timespec="seconds"),
        dataset_path=dataset_path.as_posix(),
        dataset_sha256=dataset_sha256,
        prompt_sha256=sha256_text(SYSTEM_PROMPT + USER_PROMPT_TEMPLATE),
        dataset_size=len(results),
        label_counts=counts,
        coverage_problems=list(coverage_problems),
        final=final,
        before_rules=before_rules,
        baselines=baselines,
        beats_baseline_accuracy=final.accuracy > majority_scores.accuracy,
        beats_baseline_weighted_error=(
            final.weighted_error_mean < majority_scores.weighted_error_mean
        ),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        price=price_for(model),
        prices_checked_on=PRICES_CHECKED_ON,
        cost_usd=cost_usd,
        cost_per_100_usd=None if cost_usd is None else cost_usd / len(results) * COST_REPORT_UNIT,
        mean_latency_s=mean_latency_s,
        results=list(results),
    )


def _format_confusion(scores: Scores) -> list[str]:
    """Render a confusion matrix as text rows.

    Args:
        scores: Metrics holding the matrix.

    Returns:
        A header row and one row per actual label.
    """
    width = LABEL_COLUMN_WIDTH
    header = " " * width + "".join(f"{label:>{width}}" for label in LABELS)
    rows = [
        f"{label:<{width}}" + "".join(f"{count:>{width}}" for count in row)
        for label, row in zip(LABELS, scores.confusion, strict=True)
    ]
    return [header, *rows]


def _format_score_row(name: str, scores: Scores) -> str:
    """Render one line of the comparison table.

    Args:
        name: What was scored, e.g. "model (final)".
        scores: Its metrics.

    Returns:
        Name, accuracy, total and mean weighted error, and failed count in columns.
    """
    return (
        f"{name:<28}{scores.accuracy:>9.1%}{scores.weighted_error_total:>8}"
        f"{scores.weighted_error_mean:>8.2f}{scores.failed:>8}"
    )


def _format_wrong(result: PostingResult) -> list[str]:
    """Render one wrongly decided posting for review.

    Args:
        result: A result whose prediction differs from the label.

    Returns:
        The lines describing what was expected, what came out and why.
    """
    lines = [
        f"- {result.id}: {result.company} / {result.role}",
        f"    label {result.label} -> predicted {result.predicted or 'FAILED'}"
        f" (LLM before rules: {result.llm_decision or 'n/a'};"
        f" rules fired: {', '.join(result.fired_rules) or 'none'})",
    ]
    if result.note:
        lines.append(f"    note: {result.note}")
    if result.error:
        lines.append(f"    error: {result.error}")
    if result.rationale:
        lines.append(f"    rationale: {result.rationale}")
    return lines


def _verdict(beats: bool) -> str:
    """Word a comparison with the baseline.

    Args:
        beats: True if the model is better than the baseline.

    Returns:
        "beats" or "does NOT beat".
    """
    return "beats" if beats else "does NOT beat"


def format_report(report: EvalReport) -> str:
    """Render a report for the terminal.

    Args:
        report: Report built by ``build_report``.

    Returns:
        Dataset summary, metrics, baselines, cost, latency and the wrong decisions.
    """
    counts = ", ".join(f"{label} {count}" for label, count in report.label_counts.items())
    lines = [
        f"model:    {report.model} ({report.provider})",
        f"dataset:  {report.dataset_path}, {report.dataset_size} postings ({counts})",
    ]
    lines += [f"WARNING:  incomplete test set: {problem}" for problem in report.coverage_problems]

    lines += ["", "Confusion matrix (rows: actual, columns: predicted)"]
    lines += _format_confusion(report.final)
    if report.final.failed:
        lines.append(f"failed postings (not in the matrix): {report.final.failed}")

    lines += [
        "",
        f"{'':<28}{'accuracy':>9}{'w.err':>8}{'mean':>8}{'failed':>8}",
        _format_score_row("model (final, after rules)", report.final),
        _format_score_row("model (LLM, before rules)", report.before_rules),
    ]
    for baseline in report.baselines:
        marker = " (majority)" if baseline.is_majority else ""
        lines.append(_format_score_row(f"always {baseline.label}{marker}", baseline.scores))
    lines += [
        "w.err = weighted error, lower is better; baselines are scored on the same postings.",
        "",
        f"vs. majority baseline: {_verdict(report.beats_baseline_accuracy)} it on accuracy, "
        f"{_verdict(report.beats_baseline_weighted_error)} it on mean weighted error.",
        "",
        f"tokens:   {report.input_tokens} input, {report.output_tokens} output",
    ]
    if report.cost_usd is None or report.cost_per_100_usd is None:
        lines.append(f"cost:     price unknown for '{report.model}' (add it to pricing.py)")
    else:
        lines.append(
            f"cost:     ${report.cost_usd:.4f} for this run, "
            f"${report.cost_per_100_usd:.4f} per {COST_REPORT_UNIT} postings "
            f"(list price, checked {report.prices_checked_on})"
        )
    lines.append(f"latency:  {report.mean_latency_s:.2f} s mean per posting")

    wrong = [result for result in report.results if result.predicted != result.label]
    lines += ["", f"Wrong decisions: {len(wrong)} of {report.dataset_size}"]
    for result in wrong:
        lines += _format_wrong(result)
    return "\n".join(lines)


def save_report(report: EvalReport, results_dir: Path, started_at: datetime) -> Path:
    """Write a report to ``<date>_<time>_<model>.json``.

    The time is part of the name so that a second run on the same day does not
    overwrite the first.

    Args:
        report: Report to save.
        results_dir: Folder for run files; created if missing.
        started_at: When the run started.

    Returns:
        The path of the file written.
    """
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", report.model)
    path = results_dir / f"{started_at:%Y-%m-%d_%H%M%S}_{safe_model}.json"
    results_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def _print_progress(position: int, total: int, result: PostingResult) -> None:
    """Print one line per finished posting.

    Args:
        position: 1-based position of the posting in the run.
        total: Number of postings in the run.
        result: Outcome of the posting.
    """
    print(
        f"[{position}/{total}] {result.id} label={result.label} "
        f"predicted={result.predicted or 'FAILED'} {result.latency_s:.1f}s",
        flush=True,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Arguments without the program name; None reads ``sys.argv``.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate a model on the labeled test set.")
    parser.add_argument("--model", required=True, help="Model id, as the provider names it.")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, help="LLM provider.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Test set (JSONL).")
    parser.add_argument("--cv", type=Path, default=DEFAULT_CV_PATH, help="Path to the CV.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N postings.")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Run even if the test set is smaller or less balanced than required.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace, decide_fn: DecideFn, clock: Clock) -> Path:
    """Load the inputs, evaluate, print the report and save it.

    Args:
        args: Parsed command-line arguments.
        decide_fn: Function with the signature of ``workflow.decide``.
        clock: Returns the current time.

    Returns:
        The path of the saved run file.

    Raises:
        DatasetError: If the test set is invalid, or incomplete without
            ``--allow-incomplete``.
        EvalError: If the run is aborted.
        OSError: If a file cannot be read or written.
        ValueError: If ``--limit`` is not positive or the CV is empty.
    """
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")
    postings = load_postings(args.data)
    coverage_problems = check_coverage(postings)
    if coverage_problems and not args.allow_incomplete:
        raise DatasetError(
            f"Test set is incomplete ({'; '.join(coverage_problems)}). "
            "Add postings or pass --allow-incomplete."
        )
    cv = args.cv.read_text(encoding="utf-8")
    if not cv.strip():
        raise ValueError(f"The CV file is empty: {args.cv}")
    selected = postings[: args.limit] if args.limit else postings

    started_at = clock()
    counts = ", ".join(f"{label} {count}" for label, count in label_counts(selected).items())
    print(f"evaluating {len(selected)} postings ({counts}) with {args.model}", flush=True)
    results = evaluate(
        selected,
        cv,
        model=args.model,
        provider=args.provider,
        decide_fn=decide_fn,
        on_progress=_print_progress,
    )
    report = build_report(
        results,
        model=args.model,
        provider=args.provider,
        started_at=started_at,
        dataset_path=args.data,
        dataset_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        coverage_problems=coverage_problems,
    )
    print()
    print(format_report(report))
    path = save_report(report, args.results_dir, started_at)
    print(f"\nsaved:    {path.as_posix()}")
    return path


def main(
    argv: Sequence[str] | None = None,
    *,
    decide_fn: DecideFn = decide,
    clock: Clock = datetime.now,
) -> int:
    """Evaluate a model on the test set and print all metrics.

    Args:
        argv: Arguments without the program name; None reads ``sys.argv``.
        decide_fn: Function with the signature of ``workflow.decide``; tests pass a fake.
        clock: Returns the current time; tests pass a fixed one.

    Returns:
        Process exit code: 0 when the run completes, 1 on a handled error. A model
        that loses to the baseline is a result, not an error.
    """
    args = _parse_args(argv)
    # Postings contain characters the default Windows console encoding cannot print.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        run(args, decide_fn, clock)
    except (OSError, ValueError, EvalError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
