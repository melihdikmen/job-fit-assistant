"""Tests for the rationale judge and its check against the candidate. No real LLM."""

import json
from pathlib import Path
from typing import Any

import pytest

from job_fit import judge
from job_fit.judge import (
    CRITERIA,
    Agreement,
    CalibrationItem,
    JudgeError,
    JudgeRecord,
    RationaleItem,
    agreement,
    build_judge_prompt,
    build_judge_report,
    calibration_agreement,
    items_from_run,
    judge_file_for,
    judge_items,
    judge_rationale,
    judge_run_file,
    load_calibration,
    load_judge_report,
    main,
    parse_score,
    pick_calibration_items,
    rubric_sha256,
    score_session,
    write_calibration,
)
from job_fit.llm import LLMError, LLMResponse
from job_fit.pricing import call_cost_usd
from job_fit.schema import MAX_RATIONALE_SCORE, MIN_RATIONALE_SCORE, RationaleScore

JUDGE_MODEL = "gemini-4-carbon"
JUDGED_MODEL = "gemini-2.5-flash"
CV = "CV-MARKER seven years of React Native."
POSTING = "POSTING-MARKER Senior React Native developer, Istanbul."
INPUT_TOKENS = 900
OUTPUT_TOKENS = 150


def _item(posting_id: str = "p-1", decision: str = "SKIP", **overrides: Any) -> RationaleItem:
    """Build a decision to judge.

    Args:
        posting_id: Posting id.
        decision: Final decision.
        **overrides: Fields that differ from the defaults.

    Returns:
        An item whose rationale carries its id.
    """
    fields: dict[str, Any] = {
        "id": posting_id,
        "rationale": f"RATIONALE-{posting_id} [rule: location]",
        "decision": decision,
        "llm_decision": "APPLY",
        "fired_rules": ("location",),
    }
    fields.update(overrides)
    return RationaleItem(**fields)


def _score_json(score: int = 4, **overrides: bool) -> str:
    """Build a valid judge answer.

    Args:
        score: Overall score.
        **overrides: Criteria that differ from "met".

    Returns:
        The answer as JSON text.
    """
    fields: dict[str, Any] = {name: True for name in CRITERIA}
    fields.update(overrides)
    return json.dumps({"reasoning": "Checked against posting and CV.", **fields, "score": score})


class FakeJudgeLLM:
    """Stands in for ``generate_json``: replays canned answers and records its calls."""

    def __init__(self, *outcomes: str | Exception) -> None:
        """Store the answers to return or the errors to raise, in order.

        Args:
            *outcomes: Response texts or exceptions; the last one repeats.
        """
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, provider: str, model: str, system: str, user: str, schema: Any, **options: Any
    ) -> LLMResponse:
        """Record the call, then return or raise the next canned outcome.

        Args:
            provider: Provider passed by the judge.
            model: Model passed by the judge.
            system: System prompt.
            user: User prompt.
            schema: Expected schema.
            **options: Keyword options passed by the judge.

        Returns:
            The canned response with fixed token counts.
        """
        self.calls.append(
            {"provider": provider, "model": model, "system": system, "user": user, **options}
        )
        outcome = self.outcomes[min(len(self.calls), len(self.outcomes)) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return LLMResponse(
            text=outcome, input_tokens=INPUT_TOKENS, output_tokens=OUTPUT_TOKENS, latency_s=2.0
        )


def _run_dict(*results: dict[str, Any], model: str = JUDGED_MODEL) -> dict[str, Any]:
    """Build the parts of a run file the judge reads.

    Args:
        *results: Per-posting results.
        model: Model that produced them.

    Returns:
        A run file as a dict.
    """
    return {"model": model, "results": list(results)}


def _result(posting_id: str, predicted: str | None = "SKIP") -> dict[str, Any]:
    """Build one per-posting result of a run file.

    Args:
        posting_id: Posting id.
        predicted: Final decision; None for a failed posting.

    Returns:
        The result as a dict.
    """
    return {
        "id": posting_id,
        "label": "APPLY",
        "note": "LABEL-NOTE",
        "predicted": predicted,
        "llm_decision": predicted,
        "fired_rules": [],
        "rationale": f"RATIONALE-{posting_id}" if predicted else "",
    }


def _write_run(folder: Path, *results: dict[str, Any], model: str = JUDGED_MODEL) -> Path:
    """Write a run file.

    Args:
        folder: Folder to write into.
        *results: Per-posting results.
        model: Model that produced them.

    Returns:
        The path of the run file.
    """
    path = folder / "2026-11-22_100000_run.json"
    path.write_text(json.dumps(_run_dict(*results, model=model)), encoding="utf-8")
    return path


def test_judge_prompt_shows_the_decision_but_not_the_label() -> None:
    """The judge sees posting, CV, decisions, rules and rationale, and no label."""
    prompt = build_judge_prompt(POSTING, CV, _item())

    assert "POSTING-MARKER" in prompt
    assert "CV-MARKER" in prompt
    assert "RATIONALE-p-1" in prompt
    assert "Final decision: SKIP" in prompt
    assert "Decision of the model before hard rules: APPLY" in prompt
    assert "Hard rules that fired in code: location" in prompt
    assert "label" not in prompt.lower()


def test_judge_prompt_says_none_when_no_rule_fired() -> None:
    """An empty rule list is spelled out instead of left blank."""
    prompt = build_judge_prompt(POSTING, CV, _item(fired_rules=()))

    assert "Hard rules that fired in code: none" in prompt


def test_valid_answer_becomes_a_scored_record() -> None:
    """The judge's JSON is validated and returned with its usage."""
    fake = FakeJudgeLLM(_score_json(4, mentions_english_risk=False))

    record = judge_rationale(POSTING, CV, _item(), model=JUDGE_MODEL, llm=fake)

    assert record.score is not None
    assert record.score.score == 4
    assert record.score.mentions_english_risk is False
    assert (record.input_tokens, record.output_tokens) == (INPUT_TOKENS, OUTPUT_TOKENS)
    assert record.error == ""
    assert fake.calls[0]["model"] == JUDGE_MODEL
    assert fake.calls[0]["system"] == judge.JUDGE_SYSTEM_PROMPT


@pytest.mark.parametrize("score", [MIN_RATIONALE_SCORE - 1, MAX_RATIONALE_SCORE + 1])
def test_score_outside_the_scale_is_rejected_by_the_schema(score: int) -> None:
    """A score beyond 1 to 5 is not a valid answer."""
    with pytest.raises(ValueError, match="score"):
        RationaleScore.model_validate_json(_score_json(score))


def test_invalid_answer_is_retried_once_and_usage_is_summed() -> None:
    """A second attempt follows an answer that does not match the schema."""
    fake = FakeJudgeLLM("not json", _score_json(5))

    record = judge_rationale(POSTING, CV, _item(), model=JUDGE_MODEL, llm=fake)

    assert record.score is not None
    assert record.score.score == 5
    assert len(fake.calls) == judge.MAX_ATTEMPTS
    assert record.input_tokens == INPUT_TOKENS * judge.MAX_ATTEMPTS
    assert fake.calls[1]["user"].endswith(judge.RETRY_NOTE)


def test_two_invalid_answers_are_recorded_as_a_failure() -> None:
    """The judge gets no third attempt; the record carries the error."""
    fake = FakeJudgeLLM("not json")

    record = judge_rationale(POSTING, CV, _item(), model=JUDGE_MODEL, llm=fake)

    assert record.score is None
    assert "invalid score" in record.error
    assert len(fake.calls) == judge.MAX_ATTEMPTS


def test_failed_call_is_recorded_not_raised() -> None:
    """A provider error on one rationale does not end the pass."""
    fake = FakeJudgeLLM(LLMError("429"), _score_json(3))
    items = [_item("p-1"), _item("p-2")]

    records = judge_items(items, {"p-1": POSTING, "p-2": POSTING}, CV, model=JUDGE_MODEL, llm=fake)

    assert records[0].score is None
    assert "429" in records[0].error
    assert records[1].score is not None


def test_item_without_a_posting_text_is_an_error() -> None:
    """A run that does not fit the test set file is refused."""
    with pytest.raises(JudgeError, match="p-1"):
        judge_items([_item("p-1")], {}, CV, model=JUDGE_MODEL, llm=FakeJudgeLLM(_score_json()))


def test_report_aggregates_scores_pass_rates_and_cost() -> None:
    """Mean score and pass rates cover the rated rationales; failures are counted apart."""
    passed = RationaleScore.model_validate_json(_score_json(5))
    partial = RationaleScore.model_validate_json(_score_json(2, names_missing_skills=False))
    records = [
        JudgeRecord(id="a", score=passed, input_tokens=100, output_tokens=10, latency_s=1.0),
        JudgeRecord(id="b", score=partial, input_tokens=100, output_tokens=10, latency_s=3.0),
        JudgeRecord(id="c", error="LLMError: 429"),
    ]

    report = build_judge_report(
        records, judge_model=JUDGE_MODEL, provider="vertex", judged_model=JUDGED_MODEL, run_file="r"
    )

    assert (report.judged, report.failed) == (2, 1)
    assert report.mean_score == (5 + 2) / 2
    assert report.criterion_pass_rates["names_missing_skills"] == 0.5
    assert report.criterion_pass_rates["consistent_with_decision"] == 1.0
    assert report.mean_latency_s == 2.0
    assert report.cost_usd == call_cost_usd("vertex", JUDGE_MODEL, 200, 20)
    assert report.rubric_sha256 == rubric_sha256()


def test_report_without_any_rating_has_no_mean() -> None:
    """A pass in which every call failed reports no score instead of zero."""
    report = build_judge_report(
        [JudgeRecord(id="a", error="boom")],
        judge_model=JUDGE_MODEL,
        provider="vertex",
        judged_model=JUDGED_MODEL,
        run_file="r",
    )

    assert report.mean_score is None
    assert report.criterion_pass_rates == {}
    assert "n/a" in judge.format_judge_report(report)


def test_items_from_run_skip_failed_postings() -> None:
    """A posting without a decision has no rationale to judge."""
    items = items_from_run(_run_dict(_result("a"), _result("b", predicted=None)))

    assert [item.id for item in items] == ["a"]
    assert items[0].rationale == "RATIONALE-a"


def test_calibration_sample_is_spread_over_the_decisions() -> None:
    """A run that mostly says SKIP still gives APPLY and PREPARE rationales to score."""
    items = (
        [_item(f"s-{n}", "SKIP") for n in range(20)]
        + [_item(f"a-{n}", "APPLY") for n in range(3)]
        + [_item(f"p-{n}", "PREPARE") for n in range(3)]
    )

    picked = pick_calibration_items(items, size=9)

    assert len(picked) == 9
    decisions = sorted(item.decision for item in picked)
    assert decisions == ["APPLY"] * 3 + ["PREPARE"] * 3 + ["SKIP"] * 3
    assert picked == pick_calibration_items(list(reversed(items)), size=9)


def test_calibration_sample_takes_everything_when_the_run_is_small() -> None:
    """Fewer rationales than the sample size are all taken, once each."""
    items = [_item("a", "APPLY"), _item("b", "SKIP")]

    picked = pick_calibration_items(items, size=10)

    assert sorted(item.id for item in picked) == ["a", "b"]


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("1", 1), (" 5 ", 5), ("0", None), ("6", None), ("4.5", None), ("good", None), ("", None)],
)
def test_parse_score_accepts_whole_numbers_on_the_scale(answer: str, expected: int | None) -> None:
    """Only 1 to 5 are scores."""
    assert parse_score(answer) == expected


def _scripted(*answers: str) -> tuple[Any, list[str]]:
    """Build an ``ask`` function that replays answers and a list capturing output.

    Args:
        *answers: What the candidate types, in order.

    Returns:
        The ``ask`` function and the list that ``show`` appends to.
    """
    remaining = list(answers)
    shown: list[str] = []

    def ask(prompt: str) -> str:
        """Return the next scripted answer.

        Args:
            prompt: Prompt shown to the candidate.

        Returns:
            The scripted answer.
        """
        shown.append(prompt)
        return remaining.pop(0)

    return ask, shown


def test_score_session_fills_in_missing_scores_only() -> None:
    """Already scored items are kept; the others are asked for, with the rubric shown."""
    calibration = [
        CalibrationItem(run_file="r", id="a", human_score=2),
        CalibrationItem(run_file="r", id="b"),
    ]
    items = {("r", "a"): _item("a"), ("r", "b"): _item("b")}
    ask, shown = _scripted("seven", "4")

    scored = score_session(
        calibration, items, {"a": POSTING, "b": POSTING}, ask=ask, show=shown.append
    )

    assert [entry.human_score for entry in scored] == [2, 4]
    printed = "\n".join(shown)
    assert "RATIONALE-b" in printed
    assert "RATIONALE-a" not in printed
    assert "Overall score:" in printed


def test_score_session_can_be_quit() -> None:
    """Quitting leaves the remaining items unscored."""
    calibration = [CalibrationItem(run_file="r", id="a"), CalibrationItem(run_file="r", id="b")]
    items = {("r", "a"): _item("a"), ("r", "b"): _item("b")}
    ask, shown = _scripted("3", "q")

    scored = score_session(
        calibration, items, {"a": POSTING, "b": POSTING}, ask=ask, show=shown.append
    )

    assert [entry.human_score for entry in scored] == [3, None]


def test_agreement_reports_exact_within_one_and_distance() -> None:
    """Three plain figures describe how close the judge is to the candidate."""
    result = agreement([5, 4, 2, 1], [5, 3, 4, 1])

    # Distances: 0, 1, 2, 0.
    assert result == Agreement(
        compared=4, exact=2 / 4, within_one=3 / 4, mean_absolute_difference=3 / 4
    )


@pytest.mark.parametrize(("human", "judged"), [([], []), ([1, 2], [1])])
def test_agreement_needs_matching_non_empty_scores(human: list[int], judged: list[int]) -> None:
    """Nothing to compare, or unequal lists, is an error."""
    with pytest.raises(ValueError, match="same"):
        agreement(human, judged)


def test_judge_run_file_saves_the_report_next_to_the_run(tmp_path: Path) -> None:
    """Judging a run writes ``<run>.judge.json`` that loads back."""
    run_file = _write_run(tmp_path, _result("a"), _result("b"))
    fake = FakeJudgeLLM(_score_json(4), _score_json(2))

    report = judge_run_file(run_file, {"a": POSTING, "b": POSTING}, CV, model=JUDGE_MODEL, llm=fake)

    assert report.mean_score == 3.0
    assert report.judged_model == JUDGED_MODEL
    assert judge_file_for(run_file).name == "2026-11-22_100000_run.judge.json"
    assert load_judge_report(run_file) == report


def test_judge_report_file_contains_no_cv_or_posting_text(tmp_path: Path) -> None:
    """Like run files, judge files hold scores and ids only."""
    run_file = _write_run(tmp_path, _result("a"))

    judge_run_file(run_file, {"a": POSTING}, CV, model=JUDGE_MODEL, llm=FakeJudgeLLM(_score_json()))

    saved = judge_file_for(run_file).read_text(encoding="utf-8")
    assert "CV-MARKER" not in saved
    assert "POSTING-MARKER" not in saved


def test_a_model_must_not_judge_its_own_rationales(tmp_path: Path) -> None:
    """The judge has to be a different model from the one being judged."""
    run_file = _write_run(tmp_path, _result("a"), model=JUDGE_MODEL)

    with pytest.raises(JudgeError, match="its own output"):
        judge_run_file(run_file, {"a": POSTING}, CV, model=JUDGE_MODEL, llm=FakeJudgeLLM("x"))


def test_unjudged_run_has_no_judge_report(tmp_path: Path) -> None:
    """A run file without a judge file next to it reports None."""
    assert load_judge_report(_write_run(tmp_path, _result("a"))) is None


def test_calibration_agreement_compares_only_items_scored_by_both(tmp_path: Path) -> None:
    """Unscored items and items the judge failed on are left out of the comparison."""
    run_file = _write_run(tmp_path, _result("a"), _result("b"), _result("c"))
    fake = FakeJudgeLLM(_score_json(4), _score_json(2), LLMError("boom"))
    judge_run_file(run_file, {"a": POSTING, "b": POSTING, "c": POSTING}, CV, llm=fake)
    name = run_file.as_posix()
    calibration = [
        CalibrationItem(run_file=name, id="a", human_score=4),
        CalibrationItem(run_file=name, id="b", human_score=4),
        CalibrationItem(run_file=name, id="c", human_score=1),
        CalibrationItem(run_file=name, id="d"),
    ]

    result = calibration_agreement(calibration)

    assert result == Agreement(compared=2, exact=0.5, within_one=0.5, mean_absolute_difference=1.0)


def test_calibration_agreement_without_any_pair_is_an_error(tmp_path: Path) -> None:
    """With no rationale scored by both there is nothing to report."""
    run_file = _write_run(tmp_path, _result("a"))

    scored_by_human_only = CalibrationItem(run_file=run_file.as_posix(), id="a", human_score=3)

    with pytest.raises(JudgeError, match="both"):
        calibration_agreement([scored_by_human_only])


def test_calibration_file_round_trips(tmp_path: Path) -> None:
    """Items written to the calibration file load back unchanged."""
    path = tmp_path / "data" / "calibration.jsonl"
    items = [
        CalibrationItem(run_file="r", id="a", human_score=3),
        CalibrationItem(run_file="r", id="b"),
    ]

    write_calibration(path, items)

    assert load_calibration(path) == items


def test_missing_calibration_file_names_the_command_to_run(tmp_path: Path) -> None:
    """The error tells the candidate how to create the file."""
    with pytest.raises(JudgeError, match="judge sample"):
        load_calibration(tmp_path / "missing.jsonl")


def test_cli_sample_score_and_agreement_work_together(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Sample picks rationales, score records the candidate's view, agreement compares."""
    run_file = _write_run(tmp_path, _result("a"), _result("b"))
    data = tmp_path / "dev.jsonl"
    rows = [
        {"id": name, "company": "Acme", "role": "Engineer", "text": POSTING, "label": "SKIP"}
        for name in ("a", "b")
    ]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    calibration = tmp_path / "calibration.jsonl"
    judge_run_file(
        run_file, {"a": POSTING, "b": POSTING}, CV, llm=FakeJudgeLLM(_score_json(4), _score_json(3))
    )
    ask, _ = _scripted("4", "5")
    shared = ["--calibration", str(calibration)]

    sampled = main(["sample", "--run", str(run_file), *shared])
    scored = main(["score", "--data", str(data), *shared], ask=ask)
    compared = main(["agreement", *shared])

    assert (sampled, scored, compared) == (0, 0, 0)
    assert sorted(entry.human_score for entry in load_calibration(calibration)) == [4, 5]
    output = capsys.readouterr().out
    assert "scored 2 of 2" in output
    assert "compared: 2 rationales scored by both" in output


def test_cli_reports_a_file_that_is_not_a_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A JSON file without results is refused with a readable message."""
    not_a_run = tmp_path / "other.json"
    not_a_run.write_text("{}", encoding="utf-8")

    exit_code = main(["sample", "--run", str(not_a_run), "--calibration", str(tmp_path / "c")])

    assert exit_code == 1
    assert "is not a run file" in capsys.readouterr().err
