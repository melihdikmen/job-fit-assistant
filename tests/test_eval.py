"""Tests for the eval runner. The workflow is always a fake; nothing goes to the network."""

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from job_fit import eval as eval_module
from job_fit.dataset import MIN_DEV_PER_LABEL, MIN_DEV_POSTINGS, Posting
from job_fit.eval import (
    EvalError,
    build_report,
    check_heldout_access,
    evaluate,
    format_report,
    headline_numbers,
    main,
    save_report,
    summarize_runs,
)
from job_fit.judge import CRITERIA
from job_fit.llm import DEFAULT_TIMEOUT_S, LLMError, LLMResponse
from job_fit.metrics import spread
from job_fit.pricing import call_cost_usd
from job_fit.schema import JobDecision
from job_fit.workflow import DecisionError, DecisionResult, ModelCall, ToolCallRecord

MODEL = "gemini-2.5-flash"
CV = "CV-MARKER seven years of React Native."
STARTED_AT = datetime(2026, 11, 10, 15, 30, 5)
LOCATION = "test-location"


@pytest.fixture(autouse=True)
def fixed_vertex_location(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the tests from reading the real environment for the Vertex location."""
    monkeypatch.setattr(eval_module, "vertex_location", lambda: LOCATION)


def _posting(posting_id: str, label: str, note: str = "") -> Posting:
    """Build a labeled posting whose text carries its id."""
    return Posting(
        id=posting_id,
        company="Acme",
        role="Engineer",
        text=f"TEXT-{posting_id}",
        label=label,
        note=note,
    )


def _decision_result(decision: str, llm_decision: str | None = None) -> DecisionResult:
    """Build a workflow result with fixed usage numbers."""
    llm = llm_decision or decision
    job_decision = JobDecision(
        rationale=f"Rationale for {decision}.",
        company="Acme",
        role="Engineer",
        location="Istanbul, Türkiye",
        work_mode="hybrid",
        english_risk=False,
        seniority="senior",
        matching_skills=["React Native"],
        missing_skills=[],
        decision=decision,
    )
    return DecisionResult(
        decision=job_decision,
        llm_decision=llm,
        fired_rules=("location",) if llm != decision else (),
        model=MODEL,
        input_tokens=1000,
        output_tokens=200,
        latency_s=2.0,
        attempts=1,
    )


class FakeDecide:
    """Callable with the signature of workflow.decide that replays canned outcomes."""

    def __init__(self, *outcomes: DecisionResult | Exception) -> None:
        """Store the outcomes to return or raise, one per call, in order."""
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        posting_text: str,
        cv: str,
        *,
        model: str,
        provider: str,
        reasoning: bool | None,
        timeout_s: float,
    ) -> DecisionResult:
        """Record the call, then return or raise the next canned outcome."""
        self.calls.append(
            {
                "posting_text": posting_text,
                "cv": cv,
                "model": model,
                "provider": provider,
                "reasoning": reasoning,
                "timeout_s": timeout_s,
            }
        )
        outcome = self.outcomes[len(self.calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _report(postings: list[Posting], fake: FakeDecide, model: str = MODEL) -> Any:
    """Evaluate postings with a fake workflow and build the report."""
    results = evaluate(postings, CV, model=model, decide_fn=fake)
    return build_report(
        results,
        model=model,
        provider="vertex",
        started_at=STARTED_AT,
        dataset_path=Path("data/postings.jsonl"),
        dataset_sha256="abc",
        coverage_problems=[],
    )


def test_evaluate_returns_one_result_per_posting_in_order() -> None:
    """Each posting gets the prediction, tokens and rationale of its own call."""
    postings = [_posting("a", "APPLY"), _posting("b", "SKIP")]
    fake = FakeDecide(_decision_result("APPLY"), _decision_result("PREPARE"))

    results = evaluate(postings, CV, model=MODEL, decide_fn=fake)

    assert [(result.id, result.predicted) for result in results] == [
        ("a", "APPLY"),
        ("b", "PREPARE"),
    ]
    assert results[1].rationale == "Rationale for PREPARE."
    assert results[1].input_tokens == 1000


def test_model_sees_only_the_posting_text_and_cv() -> None:
    """The label and the note never reach the workflow."""
    posting = _posting("a", "PREPARE", note="NOTE-MARKER")
    fake = FakeDecide(_decision_result("APPLY"))

    evaluate([posting], CV, model=MODEL, decide_fn=fake)

    assert fake.calls == [
        {
            "posting_text": "TEXT-a",
            "cv": CV,
            "model": MODEL,
            "provider": "vertex",
            "reasoning": None,
            "timeout_s": DEFAULT_TIMEOUT_S,
        }
    ]


def test_reasoning_and_timeout_reach_the_workflow() -> None:
    """The reasoning setting is translated to a switch and passed on with the timeout."""
    fake = FakeDecide(_decision_result("APPLY"))

    evaluate(
        [_posting("a", "APPLY")], CV, model=MODEL, reasoning="off", timeout_s=300.0, decide_fn=fake
    )

    assert fake.calls[0]["reasoning"] is False
    assert fake.calls[0]["timeout_s"] == 300.0


def test_delay_pauses_between_postings_but_not_before_the_first() -> None:
    """Three postings with a delay mean two pauses of that length."""
    postings = [_posting("a", "APPLY"), _posting("b", "SKIP"), _posting("c", "SKIP")]
    fake = FakeDecide(*[_decision_result("SKIP")] * 3)
    pauses: list[float] = []

    evaluate(postings, CV, model=MODEL, delay_s=4.0, decide_fn=fake, sleep=pauses.append)

    assert pauses == [4.0, 4.0]


def test_no_delay_means_no_pause() -> None:
    """Without a delay the sleep function is never called."""
    postings = [_posting("a", "APPLY"), _posting("b", "SKIP")]
    fake = FakeDecide(*[_decision_result("SKIP")] * 2)
    pauses: list[float] = []

    evaluate(postings, CV, model=MODEL, decide_fn=fake, sleep=pauses.append)

    assert pauses == []


def test_failed_posting_is_recorded_and_the_run_continues() -> None:
    """An invalid answer for one posting does not stop the others."""
    postings = [_posting("a", "APPLY"), _posting("b", "SKIP")]
    fake = FakeDecide(DecisionError("invalid twice"), _decision_result("SKIP"))

    results = evaluate(postings, CV, model=MODEL, decide_fn=fake)

    assert results[0].predicted is None
    assert results[0].error == "DecisionError: invalid twice"
    assert results[1].predicted == "SKIP"


def test_three_llm_errors_in_a_row_abort_the_run() -> None:
    """A broken setup stops the run instead of failing every posting."""
    postings = [_posting(str(index), "APPLY") for index in range(5)]
    fake = FakeDecide(*[LLMError("no credentials")] * 5)

    with pytest.raises(EvalError, match="3 LLM calls failed in a row.*no credentials"):
        evaluate(postings, CV, model=MODEL, decide_fn=fake)

    assert len(fake.calls) == 3


def test_llm_errors_that_are_not_consecutive_do_not_abort() -> None:
    """A success in between resets the failure count."""
    postings = [_posting(str(index), "APPLY") for index in range(5)]
    error = LLMError("timeout")
    fake = FakeDecide(error, error, _decision_result("APPLY"), error, error)

    results = evaluate(postings, CV, model=MODEL, decide_fn=fake)

    assert [result.predicted for result in results] == [None, None, "APPLY", None, None]


def test_progress_callback_gets_position_total_and_result() -> None:
    """The callback is called once per posting."""
    seen: list[tuple[int, int, str]] = []
    fake = FakeDecide(_decision_result("APPLY"), _decision_result("SKIP"))

    evaluate(
        [_posting("a", "APPLY"), _posting("b", "SKIP")],
        CV,
        model=MODEL,
        decide_fn=fake,
        on_progress=lambda position, total, result: seen.append((position, total, result.id)),
    )

    assert seen == [(1, 2, "a"), (2, 2, "b")]


def test_report_scores_final_and_pre_rule_decisions_separately() -> None:
    """A rule override shows up as a difference between the two score rows."""
    postings = [_posting("a", "APPLY"), _posting("b", "SKIP")]
    fake = FakeDecide(_decision_result("APPLY"), _decision_result("SKIP", llm_decision="APPLY"))

    report = _report(postings, fake)

    assert report.final.accuracy == 1.0
    assert report.before_rules.accuracy == 0.5
    assert report.before_rules.weighted_error_total == 1


def test_report_charges_a_failed_posting_the_worst_cost() -> None:
    """A failed APPLY posting costs 3 and counts as wrong."""
    postings = [_posting("a", "APPLY"), _posting("b", "SKIP")]
    fake = FakeDecide(DecisionError("invalid"), _decision_result("SKIP"))

    report = _report(postings, fake)

    assert report.final.accuracy == 0.5
    assert report.final.weighted_error_total == 3
    assert report.final.failed == 1


def test_report_has_three_baselines_with_the_majority_marked() -> None:
    """All constant predictors are scored; only the most common label is the majority."""
    postings = [_posting("a", "SKIP"), _posting("b", "SKIP"), _posting("c", "APPLY")]
    fake = FakeDecide(*[_decision_result("SKIP")] * 3)

    report = _report(postings, fake)

    assert [(baseline.label, baseline.is_majority) for baseline in report.baselines] == [
        ("APPLY", False),
        ("PREPARE", False),
        ("SKIP", True),
    ]
    assert report.baselines[2].scores.accuracy == pytest.approx(2 / 3)


def test_model_equal_to_the_baseline_does_not_beat_it() -> None:
    """Matching the majority baseline is not enough; the model must be better."""
    postings = [_posting("a", "SKIP"), _posting("b", "SKIP"), _posting("c", "APPLY")]
    fake = FakeDecide(*[_decision_result("SKIP")] * 3)

    report = _report(postings, fake)

    assert report.beats_baseline_accuracy is False
    assert report.beats_baseline_weighted_error is False


def test_model_better_than_the_baseline_beats_it() -> None:
    """A model that gets everything right beats the majority baseline on both measures."""
    postings = [_posting("a", "SKIP"), _posting("b", "SKIP"), _posting("c", "APPLY")]
    fake = FakeDecide(_decision_result("SKIP"), _decision_result("SKIP"), _decision_result("APPLY"))

    report = _report(postings, fake)

    assert report.beats_baseline_accuracy is True
    assert report.beats_baseline_weighted_error is True


def test_report_sums_tokens_and_estimates_cost_per_100_postings() -> None:
    """Cost uses the list price; latency is the mean over answered postings."""
    postings = [_posting("a", "APPLY"), _posting("b", "SKIP")]
    fake = FakeDecide(_decision_result("APPLY"), _decision_result("SKIP"))

    report = _report(postings, fake)

    assert (report.input_tokens, report.output_tokens) == (2000, 400)
    assert report.cost_usd == pytest.approx((2000 * 0.30 + 400 * 2.50) / 1_000_000)
    assert report.cost_per_100_usd == pytest.approx(report.cost_usd * 50)
    assert report.mean_latency_s == 2.0


def test_unknown_model_price_is_reported_not_invented() -> None:
    """A model without a list price has no cost, and the report says so."""
    fake = FakeDecide(_decision_result("APPLY"))

    report = _report([_posting("a", "APPLY")], fake, model="mystery-model")

    assert report.cost_usd is None
    assert "price unknown for 'mystery-model'" in format_report(report)


def test_formatted_report_shows_metrics_baseline_and_wrong_decisions() -> None:
    """The terminal report has every section and lists only the wrong postings."""
    postings = [_posting("right-one", "APPLY"), _posting("wrong-one", "APPLY", note="good fit")]
    fake = FakeDecide(_decision_result("APPLY"), _decision_result("SKIP", llm_decision="PREPARE"))

    text = format_report(_report(postings, fake))

    assert "Confusion matrix (rows: actual, columns: predicted)" in text
    assert "model (final, after rules)" in text
    assert "always APPLY (majority)" in text
    assert "vs. majority baseline: does NOT beat it on accuracy" in text
    assert "tokens:   2000 input, 400 output" in text
    assert "Wrong decisions: 1 of 2" in text
    assert (
        "label APPLY -> predicted SKIP (LLM before rules: PREPARE; rules fired: location)" in text
    )
    assert "note: good fit" in text
    assert "rationale: Rationale for SKIP." in text
    assert "right-one" not in text


def test_save_report_names_the_file_by_date_time_and_model(tmp_path: Path) -> None:
    """The file name is safe for model ids with slashes and unique per run."""
    fake = FakeDecide(_decision_result("APPLY"))
    report = _report([_posting("a", "APPLY")], fake, model="google/gemini:free")

    path = save_report(report, tmp_path / "results", STARTED_AT)

    assert path.name == "2026-11-10_153005_google-gemini-free.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["model"] == "google/gemini:free"
    assert saved["results"][0]["predicted"] == "APPLY"


def test_save_report_marks_a_non_default_reasoning_setting_in_the_name(tmp_path: Path) -> None:
    """Runs with reasoning on or off get a suffix, so they are not mistaken for each other."""
    results = evaluate(
        [_posting("a", "APPLY")], CV, model=MODEL, decide_fn=FakeDecide(_decision_result("APPLY"))
    )
    report = build_report(
        results,
        model=MODEL,
        provider="vertex",
        started_at=STARTED_AT,
        dataset_path=Path("data/postings.jsonl"),
        dataset_sha256="abc",
        coverage_problems=[],
        reasoning="off",
        location=LOCATION,
    )

    path = save_report(report, tmp_path, STARTED_AT)

    assert path.name == f"2026-11-10_153005_{MODEL}_reasoning-off.json"
    assert f"(workflow, vertex, {LOCATION}, reasoning off)" in format_report(report)


def test_local_model_run_costs_nothing() -> None:
    """A model served by Ollama has a price of zero, not an unknown price."""
    results = evaluate(
        [_posting("a", "APPLY")],
        CV,
        model="any-local-model",
        provider="ollama",
        decide_fn=FakeDecide(_decision_result("APPLY")),
    )

    report = build_report(
        results,
        model="any-local-model",
        provider="ollama",
        started_at=STARTED_AT,
        dataset_path=Path("data/postings.jsonl"),
        dataset_sha256="abc",
        coverage_problems=[],
    )

    assert report.cost_usd == 0.0
    assert report.location is None


def _write_inputs(tmp_path: Path, labels: list[str]) -> list[str]:
    """Write a test set and a CV, and return the CLI arguments pointing at them."""
    data = tmp_path / "set.jsonl"
    rows = [_posting(f"p-{index}", label).model_dump_json() for index, label in enumerate(labels)]
    data.write_text("\n".join(rows) + "\n", encoding="utf-8")
    cv = tmp_path / "cv.md"
    cv.write_text(CV, encoding="utf-8")
    results_dir = tmp_path / "results"
    return [
        "--model",
        MODEL,
        "--data",
        str(data),
        "--cv",
        str(cv),
        "--results-dir",
        str(results_dir),
    ]


# A dev set that meets the size and balance eval asks for.
COMPLETE_LABELS = (
    ["APPLY"] * MIN_DEV_PER_LABEL
    + ["PREPARE"] * MIN_DEV_PER_LABEL
    + ["SKIP"] * (MIN_DEV_POSTINGS - 2 * MIN_DEV_PER_LABEL)
)


def test_main_prints_metrics_and_saves_the_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One command evaluates the whole set, prints the report and writes the run file."""
    argv = _write_inputs(tmp_path, COMPLETE_LABELS)
    fake = FakeDecide(*[_decision_result(label) for label in COMPLETE_LABELS])

    exit_code = main(argv, decide_fn=fake, clock=lambda: STARTED_AT)

    output = capsys.readouterr().out
    assert exit_code == 0
    size = len(COMPLETE_LABELS)
    assert f"[{size}/{size}] p-{size - 1} label=SKIP predicted=SKIP" in output
    assert "always SKIP (majority)" in output
    run_file = tmp_path / "results" / f"2026-11-10_153005_{MODEL}.json"
    saved = json.loads(run_file.read_text(encoding="utf-8"))
    assert saved["final"]["accuracy"] == 1.0
    assert saved["dataset_size"] == len(COMPLETE_LABELS)
    assert len(saved["dataset_sha256"]) == 64
    assert {"final", "before_rules", "baselines", "cost_usd", "results"} <= set(saved)


def test_run_file_does_not_contain_the_cv_or_posting_texts(tmp_path: Path) -> None:
    """Only ids, labels, decisions and rationales are saved."""
    argv = _write_inputs(tmp_path, COMPLETE_LABELS)
    fake = FakeDecide(*[_decision_result(label) for label in COMPLETE_LABELS])

    main(argv, decide_fn=fake, clock=lambda: STARTED_AT)

    saved_text = next((tmp_path / "results").iterdir()).read_text(encoding="utf-8")
    assert "CV-MARKER" not in saved_text
    assert "TEXT-p-0" not in saved_text


def test_main_rejects_an_incomplete_set(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A set that is too small stops the run before any call is made."""
    argv = _write_inputs(tmp_path, ["APPLY", "SKIP"])
    fake = FakeDecide()

    exit_code = main(argv, decide_fn=fake, clock=lambda: STARTED_AT)

    assert exit_code == 1
    assert fake.calls == []
    assert "--allow-incomplete" in capsys.readouterr().err


def test_main_runs_an_incomplete_set_when_allowed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The flag lets a small set run and the report warns about it."""
    argv = _write_inputs(tmp_path, ["APPLY", "SKIP"]) + ["--allow-incomplete"]
    fake = FakeDecide(_decision_result("APPLY"), _decision_result("SKIP"))

    exit_code = main(argv, decide_fn=fake, clock=lambda: STARTED_AT)

    assert exit_code == 0
    warning = f"WARNING:  incomplete test set: 2 postings, need at least {MIN_DEV_POSTINGS}"
    assert warning in capsys.readouterr().out


def test_main_limit_evaluates_only_the_first_postings(tmp_path: Path) -> None:
    """With --limit the baseline and metrics cover just the evaluated postings."""
    argv = _write_inputs(tmp_path, COMPLETE_LABELS) + ["--limit", "3"]
    fake = FakeDecide(*[_decision_result("APPLY")] * 3)

    exit_code = main(argv, decide_fn=fake, clock=lambda: STARTED_AT)

    assert exit_code == 0
    assert [call["posting_text"] for call in fake.calls] == ["TEXT-p-0", "TEXT-p-1", "TEXT-p-2"]
    saved = json.loads(next((tmp_path / "results").iterdir()).read_text(encoding="utf-8"))
    assert saved["dataset_size"] == 3


def test_main_records_reasoning_and_location_in_the_run_file(tmp_path: Path) -> None:
    """The run file says which reasoning setting and Vertex location produced it."""
    argv = _write_inputs(tmp_path, COMPLETE_LABELS) + ["--reasoning", "on", "--timeout-s", "90"]
    fake = FakeDecide(*[_decision_result(label) for label in COMPLETE_LABELS])

    exit_code = main(argv, decide_fn=fake, clock=lambda: STARTED_AT)

    assert exit_code == 0
    assert fake.calls[0]["reasoning"] is True
    assert fake.calls[0]["timeout_s"] == 90.0
    run_file = tmp_path / "results" / f"2026-11-10_153005_{MODEL}_reasoning-on.json"
    saved = json.loads(run_file.read_text(encoding="utf-8"))
    assert (saved["reasoning"], saved["location"]) == ("on", LOCATION)


def test_main_rejects_a_negative_delay(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A negative pause is a usage error."""
    argv = _write_inputs(tmp_path, COMPLETE_LABELS) + ["--delay-s", "-1"]

    exit_code = main(argv, decide_fn=FakeDecide(), clock=lambda: STARTED_AT)

    assert exit_code == 1
    assert "--delay-s must not be negative" in capsys.readouterr().err


def test_main_rejects_a_non_positive_limit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A limit of zero is a usage error."""
    argv = _write_inputs(tmp_path, COMPLETE_LABELS) + ["--limit", "0"]

    exit_code = main(argv, decide_fn=FakeDecide(), clock=lambda: STARTED_AT)

    assert exit_code == 1
    assert "--limit must be at least 1" in capsys.readouterr().err


def test_main_reports_a_missing_test_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing file is a handled error, not a traceback."""
    argv = ["--model", MODEL, "--data", str(tmp_path / "missing.jsonl")]

    exit_code = main(argv, decide_fn=FakeDecide(), clock=lambda: STARTED_AT)

    assert exit_code == 1
    assert "not found" in capsys.readouterr().err


def test_main_returns_error_when_the_run_is_aborted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Three failed calls in a row end the run with exit code 1 and no run file."""
    argv = _write_inputs(tmp_path, COMPLETE_LABELS)
    fake = FakeDecide(*[LLMError("expired login")] * 3)

    exit_code = main(argv, decide_fn=fake, clock=lambda: STARTED_AT)

    assert exit_code == 1
    assert "expired login" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


def _agent_result(decision: str, check_rules_location: str = "Istanbul, Türkiye") -> DecisionResult:
    """Build an agent result: 3 model calls, two CV reads and one check_rules call."""
    check_rules_args = {
        "location": check_rules_location,
        "work_mode": "hybrid",
        "english_required": False,
        "seniority": "senior",
    }
    return replace(
        _decision_result(decision),
        attempts=3,
        tool_calls=(
            ToolCallRecord(name="get_cv_section", arguments={"topic": "experience"}),
            ToolCallRecord(name="get_cv_section", arguments={"topic": "skills"}),
            ToolCallRecord(name="check_rules", arguments=check_rules_args),
        ),
    )


def _agent_report(fake: FakeDecide, labels: list[str]) -> Any:
    """Evaluate one posting per label with a fake agent and build the agent report."""
    postings = [_posting(f"p-{index}", label) for index, label in enumerate(labels)]
    results = evaluate(postings, CV, model=MODEL, decide_fn=fake)
    return build_report(
        results,
        model=MODEL,
        provider="vertex",
        started_at=STARTED_AT,
        dataset_path=Path("data/postings.jsonl"),
        dataset_sha256="abc",
        coverage_problems=[],
        version="agent",
        prompt_sha256="agent-prompt",
    )


def test_agent_report_averages_model_and_tool_calls_over_answered_postings() -> None:
    """A failed posting has no calls and is left out of the means."""
    fake = FakeDecide(_agent_result("APPLY"), _agent_result("SKIP"), DecisionError("no steps left"))

    report = _agent_report(fake, ["APPLY", "SKIP", "SKIP"])

    assert report.version == "agent"
    assert report.mean_model_calls == 3.0
    assert report.mean_tool_calls == 3.0
    assert report.tool_call_counts == {"get_cv_section": 4, "check_rules": 2}
    assert report.results[0].tool_calls[0].arguments == {"topic": "experience"}


def test_agent_report_counts_when_check_rules_matches_the_rules_in_code() -> None:
    """The tool's answer is recomputed from its arguments and compared with the fired rules."""
    agrees = _agent_result("APPLY")
    told_no_violation_but_code_fired = replace(_agent_result("SKIP"), fired_rules=("location",))
    told_violation_and_code_fired = replace(
        _agent_result("SKIP", check_rules_location="Bursa, Türkiye"), fired_rules=("location",)
    )
    fake = FakeDecide(agrees, told_no_violation_but_code_fired, told_violation_and_code_fired)

    report = _agent_report(fake, ["APPLY", "SKIP", "SKIP"])

    assert (report.check_rules_called, report.check_rules_agreed) == (3, 2)


def test_workflow_report_has_one_model_call_and_no_tool_calls() -> None:
    """The workflow's numbers fill the same columns as the agent's."""
    report = _report([_posting("a", "APPLY")], FakeDecide(_decision_result("APPLY")))

    assert (report.version, report.mean_model_calls, report.mean_tool_calls) == ("workflow", 1.0, 0)
    assert report.tool_call_counts == {}
    assert report.prompt_sha256 == eval_module.workflow_prompt_sha256()


def test_formatted_agent_report_shows_steps_tools_and_rule_agreement() -> None:
    """The agent block appears only for agent runs."""
    agent_text = format_report(_agent_report(FakeDecide(_agent_result("APPLY")), ["APPLY"]))
    workflow_text = format_report(
        _report([_posting("a", "APPLY")], FakeDecide(_decision_result("APPLY")))
    )

    assert "(agent, vertex, reasoning default)" in agent_text
    assert "steps:    3.00 model calls mean per posting" in agent_text
    assert "tools:    3.00 calls mean per posting (get_cv_section 2, check_rules 1)" in agent_text
    assert "check_rules called for 1 postings" in agent_text
    assert "steps:" not in workflow_text


def test_save_report_marks_an_agent_run_in_the_name(tmp_path: Path) -> None:
    """An agent run cannot be mistaken for a workflow run of the same model."""
    report = _agent_report(FakeDecide(_agent_result("APPLY")), ["APPLY"])

    path = save_report(report, tmp_path, STARTED_AT)

    assert path.name == f"2026-11-10_153005_{MODEL}_agent.json"


def test_main_version_agent_runs_the_agent_and_hashes_its_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--version agent`` picks the agent's decide function and records the agent prompt."""
    argv = _write_inputs(tmp_path, COMPLETE_LABELS) + ["--version", "agent", "--reasoning", "on"]
    fake = FakeDecide(*[_agent_result(label) for label in COMPLETE_LABELS])
    monkeypatch.setitem(eval_module.DECIDE_FUNCTIONS, "agent", fake)

    exit_code = main(argv, clock=lambda: STARTED_AT)

    run_file = tmp_path / "results" / f"2026-11-10_153005_{MODEL}_agent_reasoning-on.json"
    saved = json.loads(run_file.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert len(fake.calls) == len(COMPLETE_LABELS)
    assert saved["version"] == "agent"
    assert saved["prompt_sha256"] != eval_module.workflow_prompt_sha256()
    assert saved["results"][0]["tool_calls"][0] == {
        "name": "get_cv_section",
        "arguments": {"topic": "experience"},
    }
    assert "CV-MARKER" not in run_file.read_text(encoding="utf-8")


def test_main_default_version_runs_the_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``--version`` the fixed workflow decides."""
    argv = _write_inputs(tmp_path, COMPLETE_LABELS)
    fake = FakeDecide(*[_decision_result(label) for label in COMPLETE_LABELS])
    monkeypatch.setitem(eval_module.DECIDE_FUNCTIONS, "workflow", fake)

    exit_code = main(argv, clock=lambda: STARTED_AT)

    saved_text = next((tmp_path / "results").iterdir()).read_text(encoding="utf-8")
    assert exit_code == 0
    assert len(fake.calls) == len(COMPLETE_LABELS)
    assert json.loads(saved_text)["version"] == "workflow"


def test_formatted_report_shows_precision_recall_and_kappa() -> None:
    """The report prints the per-class table and kappa next to the baseline's zero."""
    postings = [_posting("a", "APPLY"), _posting("b", "APPLY"), _posting("c", "SKIP")]
    fake = FakeDecide(_decision_result("APPLY"), _decision_result("SKIP"), _decision_result("SKIP"))

    text = format_report(_report(postings, fake))

    rows = [line.split() for line in text.splitlines()]
    assert ["precision", "recall", "labeled", "predicted"] in rows
    assert ["APPLY", "100.0%", "50.0%", "2", "1"] in rows
    assert ["PREPARE", "n/a", "n/a", "0", "0"] in rows
    assert ["SKIP", "50.0%", "100.0%", "1", "2"] in rows
    assert "kappa:    0.40 for the model, 0.00 for the majority baseline" in text
    assert "w.err if missed APPLY = 5: 1.67 for the model" in text


def test_headline_numbers_leave_out_what_does_not_exist() -> None:
    """Precision of a label that was never predicted is absent, not zero."""
    postings = [_posting("a", "APPLY"), _posting("b", "SKIP")]
    report = _report(postings, FakeDecide(_decision_result("SKIP"), _decision_result("SKIP")))

    numbers = headline_numbers(report)

    assert numbers["recall APPLY"] == 0.0
    assert "precision APPLY" not in numbers
    assert numbers["accuracy"] == 0.5
    assert numbers["weighted error (mean)"] == report.final.weighted_error_mean
    assert numbers["weighted error, missed APPLY = 5"] == 5 / 2


def test_summarize_runs_reports_the_spread_between_runs() -> None:
    """Two runs that differ on one posting show up as a non-zero standard deviation."""
    postings = [_posting("a", "APPLY"), _posting("b", "SKIP")]
    right = _report(postings, FakeDecide(_decision_result("APPLY"), _decision_result("SKIP")))
    wrong = _report(postings, FakeDecide(_decision_result("SKIP"), _decision_result("SKIP")))

    numbers = summarize_runs([right, wrong])

    assert numbers["accuracy"] == spread([1.0, 0.5])
    assert numbers["weighted error (mean)"].maximum == wrong.final.weighted_error_mean
    assert numbers["weighted error (mean)"].stdev > 0
    # Precision of APPLY exists only in the run that predicted APPLY, so it is dropped.
    assert "precision APPLY" not in numbers


def test_summarize_runs_needs_at_least_one_run() -> None:
    """An empty list of runs cannot be summarized."""
    with pytest.raises(ValueError, match="empty"):
        summarize_runs([])


def test_prompt_hash_covers_the_schema_descriptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Changing a field description changes the hash, for the workflow and the agent."""
    workflow_before = eval_module.workflow_prompt_sha256()
    agent_before = eval_module.agent_prompt_sha256(CV)

    monkeypatch.setattr(eval_module, "schema_fingerprint", lambda: "a different schema")

    assert eval_module.workflow_prompt_sha256() != workflow_before
    assert eval_module.agent_prompt_sha256(CV) != agent_before


def test_schema_fingerprint_contains_the_field_descriptions() -> None:
    """The fingerprint is the schema as sent to the model, descriptions included."""
    description = JobDecision.model_fields["decision"].description

    assert description is not None
    assert json.dumps(description)[1:-1] in eval_module.schema_fingerprint()


def test_main_records_experiment_and_cv_hash(tmp_path: Path) -> None:
    """The run file carries the experiment name and a hash of the CV, not the CV."""
    argv = _write_inputs(tmp_path, COMPLETE_LABELS) + ["--experiment", "week2-reference"]
    fake = FakeDecide(*[_decision_result(label) for label in COMPLETE_LABELS])

    main(argv, decide_fn=fake, clock=lambda: STARTED_AT)

    saved = json.loads(next((tmp_path / "results").iterdir()).read_text(encoding="utf-8"))
    assert saved["experiment"] == "week2-reference"
    assert saved["run_index"] == 1
    assert saved["cv_sha256"] == eval_module.sha256_text(CV)


def test_main_names_the_experiment_after_the_settings_by_default(tmp_path: Path) -> None:
    """Without ``--experiment`` the name says version, model and reasoning."""
    argv = _write_inputs(tmp_path, COMPLETE_LABELS) + ["--reasoning", "on"]
    fake = FakeDecide(*[_decision_result(label) for label in COMPLETE_LABELS])

    main(argv, decide_fn=fake, clock=lambda: STARTED_AT)

    saved = json.loads(next((tmp_path / "results").iterdir()).read_text(encoding="utf-8"))
    assert saved["experiment"] == f"workflow_{MODEL}_reasoning-on"


def test_main_repeats_the_configuration_and_saves_a_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--runs 3`` writes three numbered run files and one summary with the spread."""
    runs = 3
    argv = _write_inputs(tmp_path, COMPLETE_LABELS) + ["--runs", str(runs)]
    perfect = [_decision_result(label) for label in COMPLETE_LABELS]
    one_wrong = [_decision_result("SKIP"), *perfect[1:]]
    fake = FakeDecide(*perfect, *one_wrong, *perfect)

    exit_code = main(argv, decide_fn=fake, clock=lambda: STARTED_AT)

    results = tmp_path / "results"
    names = sorted(path.name for path in results.iterdir())
    stem = f"2026-11-10_153005_{MODEL}"
    summary = json.loads((results / f"{stem}_summary.json").read_text(encoding="utf-8"))
    size = len(COMPLETE_LABELS)
    assert exit_code == 0
    assert names == [f"{stem}_run-{n}.json" for n in range(1, runs + 1)] + [f"{stem}_summary.json"]
    assert len(fake.calls) == runs * size
    assert summary["numbers"]["accuracy"]["runs"] == runs
    assert summary["numbers"]["accuracy"]["minimum"] == (size - 1) / size
    assert summary["numbers"]["accuracy"]["maximum"] == 1.0
    assert summary["numbers"]["accuracy"]["stdev"] > 0
    assert len(summary["run_files"]) == runs
    output = capsys.readouterr().out
    assert f"Summary of {runs} run(s)" in output
    assert "run 2 of 3" in output


def test_main_rejects_a_non_positive_number_of_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--runs 0`` is a usage error."""
    argv = _write_inputs(tmp_path, COMPLETE_LABELS) + ["--runs", "0"]

    assert main(argv, decide_fn=FakeDecide(), clock=lambda: STARTED_AT) == 1
    assert "--runs must be at least 1" in capsys.readouterr().err


def _write_heldout(tmp_path: Path, *ids: str) -> Path:
    """Write a held-out file holding the given posting ids.

    Args:
        tmp_path: Folder to write into.
        *ids: Posting ids that are held out.

    Returns:
        The path of the file.
    """
    path = tmp_path / "heldout.jsonl"
    rows = [_posting(posting_id, "SKIP").model_dump_json() for posting_id in ids]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_main_refuses_held_out_postings(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A set that contains a held-out posting is not evaluated, and no model is called."""
    heldout = _write_heldout(tmp_path, "p-3", "other")
    argv = _write_inputs(tmp_path, COMPLETE_LABELS) + ["--heldout", str(heldout)]
    fake = FakeDecide()

    exit_code = main(argv, decide_fn=fake, clock=lambda: STARTED_AT)

    assert exit_code == 1
    assert fake.calls == []
    assert "held out" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


def test_main_evaluates_the_held_out_set_only_with_the_explicit_flag(tmp_path: Path) -> None:
    """``--final-heldout`` lifts the guard and does not ask for a full-size set."""
    labels = ["APPLY", "SKIP"]
    argv = _write_inputs(tmp_path, labels)
    heldout = _write_heldout(tmp_path, "p-0", "p-1")
    fake = FakeDecide(*[_decision_result(label) for label in labels])

    exit_code = main(
        [*argv, "--heldout", str(heldout), "--final-heldout"],
        decide_fn=fake,
        clock=lambda: STARTED_AT,
    )

    saved = json.loads(next((tmp_path / "results").iterdir()).read_text(encoding="utf-8"))
    assert exit_code == 0
    assert len(fake.calls) == len(labels)
    assert saved["coverage_problems"] == []


def test_check_heldout_access_allows_a_set_without_held_out_postings(tmp_path: Path) -> None:
    """Dev postings pass the guard; a missing held-out file means nothing to guard."""
    heldout = _write_heldout(tmp_path, "other")

    check_heldout_access([_posting("a", "APPLY")], heldout, False)
    check_heldout_access([_posting("other", "APPLY")], tmp_path / "missing.jsonl", False)


def _judge_answer(score: int) -> str:
    """Build a valid judge answer with every criterion met.

    Args:
        score: Overall score.

    Returns:
        The answer as JSON text.
    """
    criteria = {name: True for name in CRITERIA}
    return json.dumps({"reasoning": "Checked.", **criteria, "score": score})


def test_main_judges_the_last_run_and_reports_the_score(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--judge`` rates the rationales of the last run and adds the score to the summary."""
    argv = _write_inputs(tmp_path, COMPLETE_LABELS) + ["--judge", "--judge-model", "judge-model"]
    fake = FakeDecide(*[_decision_result(label) for label in COMPLETE_LABELS])
    judge_calls: list[str] = []

    def judge_llm(provider: str, model: str, system: str, user: str, schema: Any, **_: Any) -> Any:
        """Answer every judge call with a score of 4 and record the model asked.

        Args:
            provider: Provider of the judge.
            model: Judge model id.
            system: System prompt.
            user: User prompt.
            schema: Expected schema.
            **_: Further options.

        Returns:
            A canned judge response.
        """
        judge_calls.append(model)
        return LLMResponse(text=_judge_answer(4), input_tokens=10, output_tokens=5, latency_s=1.0)

    exit_code = main(argv, decide_fn=fake, clock=lambda: STARTED_AT, judge_llm=judge_llm)

    results = tmp_path / "results"
    stem = f"2026-11-10_153005_{MODEL}"
    summary = json.loads((results / f"{stem}_summary.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert judge_calls == ["judge-model"] * len(COMPLETE_LABELS)
    assert (results / f"{stem}.judge.json").exists()
    assert (summary["judge_model"], summary["judge_mean_score"]) == ("judge-model", 4.0)
    assert "judge score (one run, by judge-model): 4.00" in capsys.readouterr().out


def test_main_refuses_a_judge_that_is_the_judged_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Asking a model to judge its own rationales is an error, reported after the run."""
    argv = _write_inputs(tmp_path, COMPLETE_LABELS) + ["--judge", "--judge-model", MODEL]
    fake = FakeDecide(*[_decision_result(label) for label in COMPLETE_LABELS])

    exit_code = main(argv, decide_fn=fake, clock=lambda: STARTED_AT)

    assert exit_code == 1
    assert "its own output" in capsys.readouterr().err


EXTRACTOR_MODEL = "gemini-3.1-flash-lite"
EXTRACTOR_TOKENS = (400, 60)


def _extractor_result(decision: str, llm_decision: str | None = None) -> DecisionResult:
    """Build a result of the extractor version: main-model usage plus one sub-agent call."""
    return replace(
        _decision_result(decision, llm_decision),
        attempts=2,
        extra_calls=(
            ModelCall(
                model=EXTRACTOR_MODEL,
                input_tokens=EXTRACTOR_TOKENS[0],
                output_tokens=EXTRACTOR_TOKENS[1],
                latency_s=0.5,
            ),
        ),
    )


def _extractor_report(fake: FakeDecide, labels: list[str], extractor: str = EXTRACTOR_MODEL) -> Any:
    """Evaluate one posting per label with a fake extractor version and build its report."""
    postings = [_posting(f"p-{index}", label) for index, label in enumerate(labels)]
    results = evaluate(postings, CV, model=MODEL, decide_fn=fake)
    return build_report(
        results,
        model=MODEL,
        provider="vertex",
        started_at=STARTED_AT,
        dataset_path=Path("data/postings.jsonl"),
        dataset_sha256="abc",
        coverage_problems=[],
        version="extractor",
        prompt_sha256="extractor-prompt",
        extractor_model=extractor,
    )


class FakeExtractorDecide(FakeDecide):
    """A fake of ``decide_with_extractor``: it also takes the sub-agent's model."""

    def __call__(self, posting_text: str, cv: str, *, extractor_model: str, **options: Any) -> Any:
        """Record the sub-agent's model, then behave like the fake workflow."""
        self.extractor_models = [*getattr(self, "extractor_models", []), extractor_model]
        return super().__call__(posting_text, cv, **options)


def test_extractor_report_prices_each_model_at_its_own_rate() -> None:
    """The run's cost is the main model's tokens plus the sub-agent's, each at its price."""
    fake = FakeDecide(_extractor_result("APPLY"), _extractor_result("SKIP"))

    report = _extractor_report(fake, ["APPLY", "SKIP"])

    main_cost = call_cost_usd("vertex", MODEL, 2000, 400)
    extractor_cost = call_cost_usd(
        "vertex", EXTRACTOR_MODEL, 2 * EXTRACTOR_TOKENS[0], 2 * EXTRACTOR_TOKENS[1]
    )
    usage = report.extra_usage[EXTRACTOR_MODEL]
    assert (report.input_tokens, report.output_tokens) == (2000, 400)
    assert (usage.input_tokens, usage.output_tokens) == (800, 120)
    assert usage.cost_usd == pytest.approx(extractor_cost)
    assert report.cost_usd == pytest.approx(main_cost + extractor_cost)
    assert report.cost_usd > main_cost
    assert report.cost_per_100_usd == pytest.approx(report.cost_usd * 50)
    assert report.mean_model_calls == 2.0
    assert report.extractor_model == EXTRACTOR_MODEL


def test_unknown_sub_agent_price_makes_the_whole_cost_unknown() -> None:
    """A partial sum is never shown as the cost of a run."""
    unpriced = replace(
        _extractor_result("APPLY"),
        extra_calls=(
            ModelCall(model="mystery-extractor", input_tokens=1, output_tokens=1, latency_s=0.1),
        ),
    )

    report = _extractor_report(FakeDecide(unpriced), ["APPLY"], extractor="mystery-extractor")

    assert report.extra_usage["mystery-extractor"].cost_usd is None
    assert report.cost_usd is None
    assert report.cost_per_100_usd is None


def test_extractor_report_counts_what_skipping_rule_decided_postings_would_save() -> None:
    """Main-model calls on postings a rule decided are the cost a skip would save."""
    fake = FakeDecide(
        _extractor_result("APPLY"),
        _extractor_result("SKIP", llm_decision="APPLY"),
        _extractor_result("SKIP"),
    )

    report = _extractor_report(fake, ["APPLY", "SKIP", "SKIP"])

    assert report.rules_fired_postings == 1
    assert report.skippable_cost_usd == pytest.approx(call_cost_usd("vertex", MODEL, 1000, 200))


def test_workflow_report_has_no_sub_agent_usage() -> None:
    """A single-model run has nothing to add to its own tokens."""
    report = _report([_posting("a", "APPLY")], FakeDecide(_decision_result("APPLY")))

    assert report.extra_usage == {}
    assert report.extractor_model is None
    assert report.cost_usd == pytest.approx(call_cost_usd("vertex", MODEL, 1000, 200))


def test_formatted_extractor_report_shows_the_sub_agent_and_the_rule_saving() -> None:
    """The extractor block appears only for extractor runs."""
    fake = FakeDecide(_extractor_result("APPLY"), _extractor_result("SKIP", llm_decision="APPLY"))
    extractor_text = format_report(_extractor_report(fake, ["APPLY", "SKIP"]))
    workflow_text = format_report(
        _report([_posting("a", "APPLY")], FakeDecide(_decision_result("APPLY")))
    )

    assert f"(extractor {EXTRACTOR_MODEL}, vertex, reasoning default)" in extractor_text
    assert f"sub-agent: {EXTRACTOR_MODEL}: 800 input, 120 output tokens" in extractor_text
    assert "rules:    fired on 1 of 2 postings" in extractor_text
    assert "sub-agent:" not in workflow_text


def test_save_report_names_an_extractor_run_after_its_sub_agent(tmp_path: Path) -> None:
    """Runs with different sub-agent models get different file names."""
    report = _extractor_report(FakeDecide(_extractor_result("APPLY")), ["APPLY"])

    path = save_report(report, tmp_path, STARTED_AT)

    assert path.name == f"2026-11-10_153005_{MODEL}_extractor-{EXTRACTOR_MODEL}.json"


def test_main_version_extractor_binds_the_sub_agent_and_hashes_both_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--version extractor`` passes ``--extractor-model`` on and records the version's hash."""
    argv = _write_inputs(tmp_path, COMPLETE_LABELS)
    argv += ["--version", "extractor", "--extractor-model", "another-extractor"]
    fake = FakeExtractorDecide(*[_extractor_result(label) for label in COMPLETE_LABELS])
    monkeypatch.setitem(eval_module.DECIDE_FUNCTIONS, "extractor", fake)

    exit_code = main(argv, clock=lambda: STARTED_AT)

    run_file = tmp_path / "results" / f"2026-11-10_153005_{MODEL}_extractor-another-extractor.json"
    saved = json.loads(run_file.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert set(fake.extractor_models) == {"another-extractor"}
    assert saved["version"] == "extractor"
    assert saved["extractor_model"] == "another-extractor"
    assert saved["prompt_sha256"] == eval_module.extractor_prompt_sha256()
    assert saved["prompt_sha256"] != eval_module.workflow_prompt_sha256()
    assert saved["experiment"] == f"extractor-another-extractor_{MODEL}_reasoning-default"
    assert saved["results"][0]["extra_calls"][0]["model"] == EXTRACTOR_MODEL


def test_workflow_run_does_not_record_an_extractor_model(tmp_path: Path) -> None:
    """The option's default does not leak into runs of other versions."""
    argv = _write_inputs(tmp_path, COMPLETE_LABELS)
    fake = FakeDecide(*[_decision_result(label) for label in COMPLETE_LABELS])

    main(argv, decide_fn=fake, clock=lambda: STARTED_AT)

    saved = json.loads(next((tmp_path / "results").iterdir()).read_text(encoding="utf-8"))
    assert saved["extractor_model"] is None
    assert saved["experiment"] == f"workflow_{MODEL}_reasoning-default"
