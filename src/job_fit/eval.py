"""Evaluate the workflow or the agent against the labeled dev set.

Run it with::

    uv run python -m job_fit.eval --model <model-id> [--version agent] [--runs 3] [--judge]

It prints accuracy, the confusion matrix, precision and recall per class, Cohen's
kappa, the weighted error under three cost matrices, the baselines, tokens, estimated
cost and latency, lists every wrong decision, and saves the run to ``data/results/``.
With ``--runs`` it repeats the configuration and reports the mean and the spread of
every headline number; with ``--judge`` a second model rates the rationales.

The held-out set is refused unless ``--final-heldout`` is given: it is looked at once,
at the very end, and never while anything is being tuned.
"""

import argparse
import hashlib
import json
import re
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel

from job_fit.agent import agent_prompt_fingerprint, decide_with_agent
from job_fit.dataset import (
    DEV_DATA_PATH,
    HELDOUT_DATA_PATH,
    MIN_DEV_PER_LABEL,
    MIN_DEV_POSTINGS,
    DatasetError,
    Posting,
    check_coverage,
    label_counts,
    load_postings,
)
from job_fit.judge import (
    DEFAULT_JUDGE_MODEL,
    JudgeError,
    JudgeReport,
    format_judge_report,
    judge_file_for,
    judge_run_file,
)
from job_fit.llm import DEFAULT_TIMEOUT_S, LLMError, LLMResponse, vertex_location
from job_fit.metrics import (
    LABELS,
    Scores,
    Spread,
    constant_baseline,
    majority_label,
    score,
    spread,
)
from job_fit.pricing import PRICES_CHECKED_ON, ModelPrice, call_cost_usd, price_for
from job_fit.schema import DecisionLabel, JobDecision
from job_fit.tools import rule_check
from job_fit.tracing import flush, trace_scope
from job_fit.workflow import (
    DEFAULT_CV_PATH,
    DEFAULT_PROVIDER,
    DEFAULT_REASONING,
    RETRY_NOTE_TEMPLATE,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    DecideFn,
    DecisionError,
    ReasoningSetting,
    ToolCallRecord,
    add_llm_arguments,
    decide,
    reasoning_flag,
)

Version = Literal["workflow", "agent"]
ProgressFn = Callable[[int, int, "PostingResult"], None]
Clock = Callable[[], datetime]
SleepFn = Callable[[float], None]
LLMCall = Callable[..., LLMResponse]

VERSIONS: tuple[Version, ...] = get_args(Version)
DEFAULT_VERSION: Version = "workflow"
AGENT_VERSION: Version = "agent"
DECIDE_FUNCTIONS: dict[Version, DecideFn] = {"workflow": decide, "agent": decide_with_agent}
CHECK_RULES_TOOL_NAME = "check_rules"

DEFAULT_RESULTS_DIR = Path("data/results")
# A run of failed calls means the setup is broken (wrong project, expired login), not
# that single postings are hard; stop instead of waiting for every timeout.
MAX_CONSECUTIVE_LLM_ERRORS = 3
COST_REPORT_UNIT = 100
VERTEX_PROVIDER = "vertex"
LABEL_COLUMN_WIDTH = 9
NOT_APPLICABLE = "n/a"
SUMMARY_FILE_SUFFIX = "_summary.json"

ACCURACY = "accuracy"
WEIGHTED_ERROR = "weighted error (mean)"
KAPPA = "kappa"
COST_PER_100 = "cost per 100 postings (USD)"
LATENCY = "latency (s)"


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
        tool_calls: Tool calls the agent made, in order; empty for the workflow.
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
    tool_calls: tuple[ToolCallRecord, ...] = ()
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
        version: What made the decisions: the fixed "workflow" or the "agent".
        reasoning: Reasoning setting of the run: "on", "off" or the model's "default".
        location: Vertex AI location the calls went to; None for other providers.
        experiment: Name the run and its traces are tagged with.
        run_index: Number of the run within a repeated configuration, starting at 1.
        started_at: Local time the run started, ISO 8601.
        dataset_path: Test set file.
        dataset_sha256: Hash of the test set file. Runs are comparable only if equal.
        prompt_sha256: Hash of the prompt templates, the schema descriptions and, for
            the agent, the tool descriptions. Runs are comparable only if equal.
        cv_sha256: Hash of the CV; empty in run files written before the field existed.
        dataset_size: Number of postings evaluated.
        label_counts: Postings evaluated per label.
        coverage_problems: Shortfalls of the test set; empty if it is complete.
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
        mean_model_calls: Mean model calls per answered posting; None in run files
            written before the field existed.
        mean_tool_calls: Mean tool calls per answered posting; 0 for the workflow.
        tool_call_counts: Total calls per tool name over the run.
        check_rules_called: Postings for which the agent called ``check_rules``.
        check_rules_agreed: Of those, postings where the tool's answer for the agent's
            last call equals the rules that fired in code on the final output.
        results: One record per posting, in test set order.
    """

    model: str
    provider: str
    # Defaults keep run files written before these fields existed loadable.
    version: Version = DEFAULT_VERSION
    reasoning: ReasoningSetting = DEFAULT_REASONING
    location: str | None = None
    experiment: str = ""
    run_index: int = 1
    started_at: str
    dataset_path: str
    dataset_sha256: str
    prompt_sha256: str
    cv_sha256: str = ""
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
    mean_model_calls: float | None = None
    mean_tool_calls: float = 0.0
    tool_call_counts: dict[str, int] = {}
    check_rules_called: int = 0
    check_rules_agreed: int = 0
    results: list[PostingResult]


class RunSummary(BaseModel):
    """Mean and spread of the headline numbers over repeated runs of one configuration.

    Attributes:
        experiment: Name the runs are tagged with.
        model: Model id.
        provider: LLM provider.
        version: "workflow" or "agent".
        reasoning: Reasoning setting of the runs.
        dataset_sha256: Hash of the test set file all runs used.
        prompt_sha256: Hash of the prompts all runs used.
        run_files: The run files summarized, in run order.
        numbers: Spread of each headline number, by name.
        judge_model: Model that rated the rationales; None if they were not judged.
        judge_mean_score: Mean rationale score of the judged run; None if not judged.
    """

    experiment: str
    model: str
    provider: str
    version: Version
    reasoning: ReasoningSetting
    dataset_sha256: str
    prompt_sha256: str
    run_files: list[str]
    numbers: dict[str, Spread]
    judge_model: str | None = None
    judge_mean_score: float | None = None


def _evaluate_one(
    posting: Posting,
    cv: str,
    model: str,
    provider: str,
    reasoning: ReasoningSetting,
    timeout_s: float,
    decide_fn: DecideFn,
) -> PostingResult:
    """Run the workflow on one posting.

    Args:
        posting: Labeled posting; only its text is passed on.
        cv: Full text of the candidate's CV.
        model: Model id.
        provider: LLM provider.
        reasoning: Reasoning setting.
        timeout_s: Timeout for one HTTP request, in seconds.
        decide_fn: Function with the signature of ``workflow.decide``.

    Returns:
        The outcome for this posting.

    Raises:
        LLMError: If the LLM call itself fails.
        DecisionError: If the model's answer stays invalid after the retry.
    """
    result = decide_fn(
        posting.text,
        cv,
        model=model,
        provider=provider,
        reasoning=reasoning_flag(reasoning),
        timeout_s=timeout_s,
    )
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
        tool_calls=result.tool_calls,
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
    reasoning: ReasoningSetting = DEFAULT_REASONING,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    delay_s: float = 0.0,
    decide_fn: DecideFn = decide,
    on_progress: ProgressFn | None = None,
    sleep: SleepFn = time.sleep,
    version: Version = DEFAULT_VERSION,
    experiment: str = "",
    run_index: int = 1,
) -> list[PostingResult]:
    """Run the workflow on every posting, one after the other.

    A posting that fails is recorded and the run goes on. The model sees only the
    posting text and the CV, never the label or the note.

    Args:
        postings: Labeled postings.
        cv: Full text of the candidate's CV.
        model: Model id.
        provider: LLM provider.
        reasoning: Reasoning setting.
        timeout_s: Timeout for one HTTP request, in seconds.
        delay_s: Pause between two postings, to stay under a provider's rate limit.
            It is not part of the measured latency.
        decide_fn: Function with the signature of ``workflow.decide``; tests pass a fake.
        on_progress: Called after each posting with its position, the total and its
            result.
        sleep: Function that pauses for a number of seconds; tests pass a fake.
        version: "workflow" or "agent"; a tag on the traces.
        experiment: Experiment name; a tag on the traces.
        run_index: Number of the run within a repeated configuration; a tag on the
            traces.

    Returns:
        One result per posting, in the same order.

    Raises:
        EvalError: If ``MAX_CONSECUTIVE_LLM_ERRORS`` LLM calls fail in a row.
    """
    results: list[PostingResult] = []
    consecutive_llm_errors = 0
    for position, posting in enumerate(postings, start=1):
        if delay_s > 0 and position > 1:
            sleep(delay_s)
        try:
            with trace_scope(
                cv=cv,
                experiment=experiment,
                version=version,
                run_index=run_index,
                posting_id=posting.id,
            ):
                result = _evaluate_one(
                    posting, cv, model, provider, reasoning, timeout_s, decide_fn
                )
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


def schema_fingerprint() -> str:
    """Render the output schema as the model receives it.

    The field descriptions of ``JobDecision`` are sent to the model with every call,
    so they steer it as much as the prompt does.

    Returns:
        The JSON schema of ``JobDecision`` as text with a fixed key order.
    """
    return json.dumps(JobDecision.model_json_schema(), sort_keys=True)


def workflow_prompt_sha256() -> str:
    """Hash every text that steers the workflow's model.

    Returns:
        SHA-256 of the prompt templates, the retry note and the output schema.
    """
    return sha256_text(
        SYSTEM_PROMPT + USER_PROMPT_TEMPLATE + RETRY_NOTE_TEMPLATE + schema_fingerprint()
    )


def agent_prompt_sha256(cv: str) -> str:
    """Hash every text that steers the agent's model.

    Args:
        cv: Full text of the candidate's CV; one tool description lists its headings.

    Returns:
        SHA-256 of the agent prompts, the tool descriptions and the output schema.
    """
    return sha256_text(agent_prompt_fingerprint(cv) + schema_fingerprint())


def _check_rules_agreement(results: Sequence[PostingResult]) -> tuple[int, int]:
    """Compare what ``check_rules`` told the agent with the rules applied in code.

    The tool's answer is not stored, so it is recomputed from the arguments of the
    agent's last ``check_rules`` call. A mismatch means the facts the agent passed to
    the tool differ from the facts in its final answer.

    Args:
        results: One result per posting.

    Returns:
        The number of postings with a ``check_rules`` call, and how many of those
        match the rules that fired on the final output.
    """
    called = 0
    agreed = 0
    for result in results:
        calls = [call for call in result.tool_calls if call.name == CHECK_RULES_TOOL_NAME]
        if not calls:
            continue
        called += 1
        arguments = calls[-1].arguments
        told = rule_check(
            str(arguments.get("location", "")),
            str(arguments.get("work_mode", "")),
            bool(arguments.get("english_required", False)),
            str(arguments.get("seniority", "")),
        )
        agreed += tuple(told["violated_rules"]) == result.fired_rules
    return called, agreed


def build_report(
    results: Sequence[PostingResult],
    *,
    model: str,
    provider: str,
    started_at: datetime,
    dataset_path: Path,
    dataset_sha256: str,
    coverage_problems: Sequence[str],
    reasoning: ReasoningSetting = DEFAULT_REASONING,
    location: str | None = None,
    version: Version = DEFAULT_VERSION,
    prompt_sha256: str | None = None,
    experiment: str = "",
    run_index: int = 1,
    cv_sha256: str = "",
) -> EvalReport:
    """Compute all metrics for a finished run.

    Args:
        results: One result per posting.
        model: Model id.
        provider: LLM provider.
        started_at: When the run started.
        dataset_path: Test set file.
        dataset_sha256: Hash of the test set file.
        coverage_problems: Shortfalls of the test set.
        reasoning: Reasoning setting of the run.
        location: Vertex AI location the calls went to; None for other providers.
        version: What made the decisions: "workflow" or "agent".
        prompt_sha256: Hash of the prompts the run used; None for the workflow's.
        experiment: Name the run is tagged with.
        run_index: Number of the run within a repeated configuration.
        cv_sha256: Hash of the CV the run used.

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
    cost_usd = call_cost_usd(provider, model, input_tokens, output_tokens)
    answered = [result for result in results if result.predicted is not None]
    mean_latency_s = (
        sum(result.latency_s for result in answered) / len(answered) if answered else 0.0
    )
    counts = {label: actual.count(label) for label in LABELS}

    tool_call_counts: dict[str, int] = {}
    for result in results:
        for call in result.tool_calls:
            tool_call_counts[call.name] = tool_call_counts.get(call.name, 0) + 1
    answered_count = len(answered) or 1
    check_rules_called, check_rules_agreed = _check_rules_agreement(results)

    return EvalReport(
        model=model,
        provider=provider,
        version=version,
        reasoning=reasoning,
        location=location,
        experiment=experiment,
        run_index=run_index,
        started_at=started_at.isoformat(timespec="seconds"),
        dataset_path=dataset_path.as_posix(),
        dataset_sha256=dataset_sha256,
        prompt_sha256=prompt_sha256 or workflow_prompt_sha256(),
        cv_sha256=cv_sha256,
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
        price=price_for(provider, model),
        prices_checked_on=PRICES_CHECKED_ON,
        cost_usd=cost_usd,
        cost_per_100_usd=None if cost_usd is None else cost_usd / len(results) * COST_REPORT_UNIT,
        mean_latency_s=mean_latency_s,
        mean_model_calls=sum(result.attempts for result in answered) / answered_count,
        mean_tool_calls=sum(tool_call_counts.values()) / answered_count,
        tool_call_counts=tool_call_counts,
        check_rules_called=check_rules_called,
        check_rules_agreed=check_rules_agreed,
        results=list(results),
    )


def headline_numbers(report: EvalReport) -> dict[str, float]:
    """Collect the numbers of one run that are compared across runs.

    Args:
        report: Report of the run.

    Returns:
        Accuracy, mean weighted error (also under the alternative cost matrices),
        kappa, precision and recall per label, cost per 100 postings and latency, by
        name. A number that does not exist for this run (precision of a label that was
        never predicted, cost of an unpriced model) is left out.
    """
    final = report.final
    numbers = {ACCURACY: final.accuracy, WEIGHTED_ERROR: final.weighted_error_mean}
    if final.kappa is not None:
        numbers[KAPPA] = final.kappa
    for label, class_scores in (final.per_class or {}).items():
        if class_scores.recall is not None:
            numbers[f"recall {label}"] = class_scores.recall
        if class_scores.precision is not None:
            numbers[f"precision {label}"] = class_scores.precision
    for name, value in (final.alternative_weighted_error_mean or {}).items():
        numbers[f"weighted error, {name}"] = value
    if report.cost_per_100_usd is not None:
        numbers[COST_PER_100] = report.cost_per_100_usd
    numbers[LATENCY] = report.mean_latency_s
    return numbers


def summarize_runs(reports: Sequence[EvalReport]) -> dict[str, Spread]:
    """Compute mean and spread of every headline number over repeated runs.

    Args:
        reports: Reports of the runs of one configuration.

    Returns:
        The spread per number. A number missing from any run is left out, because a
        mean over fewer runs would not be comparable with the others.

    Raises:
        ValueError: If there are no reports.
    """
    if not reports:
        raise ValueError("Cannot summarize an empty list of runs.")
    per_run = [headline_numbers(report) for report in reports]
    return {
        name: spread([numbers[name] for numbers in per_run])
        for name in per_run[0]
        if all(name in numbers for numbers in per_run)
    }


def build_summary(
    reports: Sequence[EvalReport],
    run_files: Sequence[Path],
    judge_report: JudgeReport | None = None,
) -> RunSummary:
    """Summarize repeated runs of one configuration.

    Args:
        reports: Reports of the runs, in run order.
        run_files: The files the reports were saved to, in the same order.
        judge_report: The judge's report on one of the runs, if it was judged.

    Returns:
        The summary to print and save.

    Raises:
        ValueError: If there are no reports.
    """
    numbers = summarize_runs(reports)
    first = reports[0]
    return RunSummary(
        experiment=first.experiment,
        model=first.model,
        provider=first.provider,
        version=first.version,
        reasoning=first.reasoning,
        dataset_sha256=first.dataset_sha256,
        prompt_sha256=first.prompt_sha256,
        run_files=[path.as_posix() for path in run_files],
        numbers=numbers,
        judge_model=judge_report.judge_model if judge_report else None,
        judge_mean_score=judge_report.mean_score if judge_report else None,
    )


def format_summary(summary: RunSummary) -> str:
    """Render the summary of repeated runs for the terminal.

    Args:
        summary: Summary built by ``build_summary``.

    Returns:
        One line per headline number with mean, standard deviation, minimum and
        maximum, and the judge score if there is one.
    """
    runs = len(summary.run_files)
    lines = [
        f"Summary of {runs} run(s): {summary.model} ({summary.version}, "
        f"reasoning {summary.reasoning}), experiment '{summary.experiment}'",
        f"{'':<34}{'mean':>9}{'± stdev':>10}{'min':>9}{'max':>9}",
    ]
    lines += [
        f"{name:<34}{value.mean:>9.3f}{value.stdev:>10.3f}{value.minimum:>9.3f}"
        f"{value.maximum:>9.3f}"
        for name, value in summary.numbers.items()
    ]
    lines.append("A difference between two configurations that is smaller than the stdev is noise.")
    if summary.judge_model is not None:
        mean = NOT_APPLICABLE
        if summary.judge_mean_score is not None:
            mean = f"{summary.judge_mean_score:.2f}"
        lines.append(f"judge score (one run, by {summary.judge_model}): {mean}")
    return "\n".join(lines)


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


def _format_share(value: float | None) -> str:
    """Render a share as a percentage.

    Args:
        value: Share between 0 and 1, or None if it is undefined.

    Returns:
        E.g. "66.7%", or "n/a".
    """
    return NOT_APPLICABLE if value is None else f"{value:.1%}"


def _format_per_class(scores: Scores) -> list[str]:
    """Render precision and recall per label.

    Args:
        scores: Metrics of the final decisions.

    Returns:
        A header row and one row per label; empty for run files without the numbers.
    """
    if scores.per_class is None:
        return []
    lines = [f"{'':<9}{'precision':>11}{'recall':>9}{'labeled':>9}{'predicted':>11}"]
    lines += [
        f"{label:<9}{_format_share(row.precision):>11}{_format_share(row.recall):>9}"
        f"{row.support:>9}{row.predicted:>11}"
        for label, row in scores.per_class.items()
    ]
    lines.append(
        "precision: of the postings given this decision, the share that deserved it; "
        "recall: of the postings that deserved it, the share that got it."
    )
    return lines


def _format_kappa_and_costs(report: EvalReport) -> list[str]:
    """Render Cohen's kappa and the weighted error under the alternative cost matrices.

    Args:
        report: Report of the run.

    Returns:
        The lines, comparing the model with the majority baseline; empty for run
        files without the numbers.
    """
    final = report.final
    majority = next((b.scores for b in report.baselines if b.is_majority), None)
    if final.kappa is None or majority is None or majority.kappa is None:
        return []
    lines = [
        f"kappa:    {final.kappa:.2f} for the model, {majority.kappa:.2f} for the majority "
        "baseline (1 = perfect, 0 = no better than ignoring the posting)",
    ]
    alternatives = final.alternative_weighted_error_mean or {}
    majority_alternatives = majority.alternative_weighted_error_mean or {}
    for name, value in alternatives.items():
        lines.append(
            f"w.err if {name}: {value:.2f} for the model, "
            f"{majority_alternatives.get(name, 0.0):.2f} for the majority baseline"
        )
    return lines


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


def _describe_route(report: EvalReport) -> str:
    """Summarize where and how the model was called.

    Args:
        report: Report of the run.

    Returns:
        Version, provider, location if any, and the reasoning setting, e.g.
        "workflow, vertex, global, reasoning off".
    """
    parts = [report.version, report.provider]
    if report.location:
        parts.append(report.location)
    parts.append(f"reasoning {report.reasoning}")
    return ", ".join(parts)


def _format_agent_usage(report: EvalReport) -> list[str]:
    """Render how many steps and tool calls an agent run needed.

    Args:
        report: Report of an agent run.

    Returns:
        Lines for model calls, tool calls per tool, and the ``check_rules`` agreement.
    """
    per_tool = ", ".join(f"{name} {count}" for name, count in report.tool_call_counts.items())
    return [
        f"steps:    {report.mean_model_calls or 0.0:.2f} model calls mean per posting",
        f"tools:    {report.mean_tool_calls:.2f} calls mean per posting ({per_tool or 'none'})",
        f"rules:    check_rules called for {report.check_rules_called} postings; "
        f"its answer matched the rules applied in code for {report.check_rules_agreed}",
    ]


def format_report(report: EvalReport) -> str:
    """Render a report for the terminal.

    Args:
        report: Report built by ``build_report``.

    Returns:
        Dataset summary, metrics, baselines, cost, latency and the wrong decisions.
    """
    counts = ", ".join(f"{label} {count}" for label, count in report.label_counts.items())
    lines = [
        f"model:    {report.model} ({_describe_route(report)})",
        f"dataset:  {report.dataset_path}, {report.dataset_size} postings ({counts})",
    ]
    lines += [f"WARNING:  incomplete test set: {problem}" for problem in report.coverage_problems]

    lines += ["", "Confusion matrix (rows: actual, columns: predicted)"]
    lines += _format_confusion(report.final)
    if report.final.failed:
        lines.append(f"failed postings (not in the matrix): {report.final.failed}")
    per_class = _format_per_class(report.final)
    if per_class:
        lines += ["", *per_class]

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
    ]
    lines += _format_kappa_and_costs(report)
    lines += ["", f"tokens:   {report.input_tokens} input, {report.output_tokens} output"]
    if report.cost_usd is None or report.cost_per_100_usd is None:
        lines.append(f"cost:     price unknown for '{report.model}' (add it to pricing.py)")
    else:
        lines.append(
            f"cost:     ${report.cost_usd:.4f} for this run, "
            f"${report.cost_per_100_usd:.4f} per {COST_REPORT_UNIT} postings "
            f"(list price, checked {report.prices_checked_on})"
        )
    lines.append(f"latency:  {report.mean_latency_s:.2f} s mean per posting")
    if report.version == AGENT_VERSION:
        lines += _format_agent_usage(report)

    wrong = [result for result in report.results if result.predicted != result.label]
    lines += ["", f"Wrong decisions: {len(wrong)} of {report.dataset_size}"]
    for result in wrong:
        lines += _format_wrong(result)
    return "\n".join(lines)


def save_report(
    report: EvalReport, results_dir: Path, started_at: datetime, *, numbered: bool = False
) -> Path:
    """Write a report to ``<date>_<time>_<model>[_agent][_reasoning-<setting>][_run-<n>].json``.

    The time is part of the name so that a second run on the same day does not
    overwrite the first. The agent version and a reasoning setting other than the
    default are part of the name so that such runs are easy to tell apart.

    Args:
        report: Report to save.
        results_dir: Folder for run files; created if missing.
        started_at: When the run started.
        numbered: True to add the run number, for repeated runs of one configuration.

    Returns:
        The path of the file written.
    """
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", report.model)
    suffix = "" if report.version == DEFAULT_VERSION else f"_{report.version}"
    if report.reasoning != DEFAULT_REASONING:
        suffix += f"_reasoning-{report.reasoning}"
    if numbered:
        suffix += f"_run-{report.run_index}"
    path = results_dir / f"{started_at:%Y-%m-%d_%H%M%S}_{safe_model}{suffix}.json"
    results_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def summary_file_for(first_run_file: Path) -> Path:
    """Name the file the summary of repeated runs is saved to.

    Args:
        first_run_file: Run file of the first run.

    Returns:
        A path next to it, without the run number and ending in ``_summary.json``.
    """
    stem = re.sub(r"_run-\d+$", "", first_run_file.stem)
    return first_run_file.with_name(stem + SUMMARY_FILE_SUFFIX)


def check_heldout_access(
    postings: Sequence[Posting], heldout_path: Path, final_heldout: bool
) -> None:
    """Refuse to evaluate held-out postings before the final check.

    Args:
        postings: Postings about to be evaluated.
        heldout_path: The held-out file; if it does not exist there is nothing to guard.
        final_heldout: True if the caller explicitly asked for the final check.

    Raises:
        DatasetError: If any of the postings is held out and the final check was not
            asked for.
    """
    if final_heldout or not heldout_path.exists():
        return
    heldout_ids = {posting.id for posting in load_postings(heldout_path)}
    overlap = sorted(heldout_ids & {posting.id for posting in postings})
    if overlap:
        raise DatasetError(
            f"{len(overlap)} of these postings are held out (e.g. '{overlap[0]}'). The "
            "held-out set is evaluated once, at the very end; pass --final-heldout only "
            "for that run."
        )


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
    parser = argparse.ArgumentParser(description="Evaluate a model on the labeled dev set.")
    parser.add_argument("--model", required=True, help="Model id, as the provider names it.")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, help="LLM provider.")
    parser.add_argument(
        "--version",
        choices=VERSIONS,
        default=DEFAULT_VERSION,
        help="What decides: the fixed workflow or the tool-calling agent.",
    )
    parser.add_argument("--data", type=Path, default=DEV_DATA_PATH, help="Test set (JSONL).")
    parser.add_argument("--cv", type=Path, default=DEFAULT_CV_PATH, help="Path to the CV.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N postings.")
    add_llm_arguments(parser)
    parser.add_argument(
        "--delay-s",
        type=float,
        default=0.0,
        help="Pause between postings in seconds, for rate-limited providers.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Run even if the test set is smaller or less balanced than required.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Repeat the configuration N times and report mean and spread.",
    )
    parser.add_argument(
        "--experiment",
        default="",
        help="Name for the run files and the traces; default is built from the settings.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Let a second model rate the rationales of the last run.",
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Judge model id.")
    parser.add_argument(
        "--heldout",
        type=Path,
        default=HELDOUT_DATA_PATH,
        help="Held-out set; its postings are refused without --final-heldout.",
    )
    parser.add_argument(
        "--final-heldout",
        action="store_true",
        help="Allow held-out postings. Only for the one final check.",
    )
    return parser.parse_args(argv)


def run(
    args: argparse.Namespace,
    decide_fn: DecideFn,
    clock: Clock,
    judge_llm: LLMCall | None = None,
) -> list[Path]:
    """Load the inputs, evaluate, print the reports and save them.

    Args:
        args: Parsed command-line arguments.
        decide_fn: Function with the signature of ``workflow.decide``.
        clock: Returns the current time.
        judge_llm: Function with the signature of ``generate_json`` for the judge;
            None uses the real client.

    Returns:
        The paths of the saved run files, in run order.

    Raises:
        DatasetError: If the test set is invalid, held out, or incomplete without
            ``--allow-incomplete``.
        EvalError: If a run is aborted.
        JudgeError: If the judge cannot be run.
        OSError: If a file cannot be read or written.
        ValueError: If ``--limit`` or ``--runs`` is not positive, ``--delay-s`` is
            negative or the CV is empty.
    """
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")
    if args.runs < 1:
        raise ValueError("--runs must be at least 1.")
    if args.delay_s < 0:
        raise ValueError("--delay-s must not be negative.")
    postings = load_postings(args.data)
    check_heldout_access(postings, args.heldout, args.final_heldout)
    # The held-out file is small by design; its size is not a shortfall.
    coverage_problems = (
        [] if args.final_heldout else check_coverage(postings, MIN_DEV_POSTINGS, MIN_DEV_PER_LABEL)
    )
    if coverage_problems and not args.allow_incomplete:
        raise DatasetError(
            f"Test set is incomplete ({'; '.join(coverage_problems)}). "
            "Add postings or pass --allow-incomplete."
        )
    cv = args.cv.read_text(encoding="utf-8")
    if not cv.strip():
        raise ValueError(f"The CV file is empty: {args.cv}")
    selected = postings[: args.limit] if args.limit else postings
    experiment = args.experiment or f"{args.version}_{args.model}_reasoning-{args.reasoning}"
    counts = ", ".join(f"{label} {count}" for label, count in label_counts(selected).items())

    reports: list[EvalReport] = []
    paths: list[Path] = []
    for run_index in range(1, args.runs + 1):
        started_at = clock()
        print(
            f"run {run_index} of {args.runs}: evaluating {len(selected)} postings ({counts}) "
            f"with {args.model} ({args.version})",
            flush=True,
        )
        results = evaluate(
            selected,
            cv,
            model=args.model,
            provider=args.provider,
            reasoning=args.reasoning,
            timeout_s=args.timeout_s,
            delay_s=args.delay_s,
            decide_fn=decide_fn,
            on_progress=_print_progress,
            version=args.version,
            experiment=experiment,
            run_index=run_index,
        )
        report = build_report(
            results,
            model=args.model,
            provider=args.provider,
            started_at=started_at,
            dataset_path=args.data,
            dataset_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
            coverage_problems=coverage_problems,
            reasoning=args.reasoning,
            location=vertex_location() if args.provider == VERTEX_PROVIDER else None,
            version=args.version,
            prompt_sha256=agent_prompt_sha256(cv) if args.version == AGENT_VERSION else None,
            experiment=experiment,
            run_index=run_index,
            cv_sha256=sha256_text(cv),
        )
        print()
        print(format_report(report))
        path = save_report(report, args.results_dir, started_at, numbered=args.runs > 1)
        print(f"\nsaved:    {path.as_posix()}\n")
        reports.append(report)
        paths.append(path)

    judge_report = None
    if args.judge:
        print(f"judging the rationales of {paths[-1].name} with {args.judge_model}", flush=True)
        judge_report = judge_run_file(
            paths[-1],
            {posting.id: posting.text for posting in selected},
            cv,
            model=args.judge_model,
            llm=judge_llm,
        )
        print()
        print(format_judge_report(judge_report))
        print(f"saved:    {judge_file_for(paths[-1]).as_posix()}\n")
    if args.runs > 1 or judge_report is not None:
        summary = build_summary(reports, paths, judge_report)
        print(format_summary(summary))
        summary_path = summary_file_for(paths[0])
        summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        print(f"saved:    {summary_path.as_posix()}")
    flush()
    return paths


def main(
    argv: Sequence[str] | None = None,
    *,
    decide_fn: DecideFn | None = None,
    clock: Clock = datetime.now,
    judge_llm: LLMCall | None = None,
) -> int:
    """Evaluate a model on the test set and print all metrics.

    Args:
        argv: Arguments without the program name; None reads ``sys.argv``.
        decide_fn: Function with the signature of ``workflow.decide``. None picks the
            workflow or the agent according to ``--version``; tests pass a fake.
        clock: Returns the current time; tests pass a fixed one.
        judge_llm: Function with the signature of ``generate_json`` for the judge;
            tests pass a fake.

    Returns:
        Process exit code: 0 when the run completes, 1 on a handled error. A model
        that loses to the baseline is a result, not an error.
    """
    args = _parse_args(argv)
    # Postings contain characters the default Windows console encoding cannot print.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        run(args, decide_fn or DECIDE_FUNCTIONS[args.version], clock, judge_llm)
    except (OSError, ValueError, EvalError, JudgeError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
