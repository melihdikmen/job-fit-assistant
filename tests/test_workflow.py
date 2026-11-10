"""Tests for the fixed workflow. The LLM is always a fake; nothing goes to the network."""

import json
from pathlib import Path
from typing import Any

import pytest

from job_fit import workflow
from job_fit.llm import DEFAULT_TIMEOUT_S, LLMResponse
from job_fit.schema import JobDecision
from job_fit.workflow import DecisionError, build_user_prompt, decide, main, reasoning_flag

MODEL = "fake-model"
POSTING = "Senior React Native engineer wanted."
CV = "CV-MARKER seven years of React Native."


class FakeLLM:
    """Callable with the signature of generate_json that replays canned answers."""

    def __init__(self, *texts: str) -> None:
        """Store the answers to return, one per call, in order."""
        self.texts = list(texts)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        provider: str,
        model: str,
        system: str,
        user: str,
        schema: type,
        *,
        reasoning: bool | None,
        timeout_s: float,
    ) -> LLMResponse:
        """Record the call and return the next canned answer with fixed usage."""
        self.calls.append(
            {
                "provider": provider,
                "model": model,
                "system": system,
                "user": user,
                "schema": schema,
                "reasoning": reasoning,
                "timeout_s": timeout_s,
            }
        )
        text = self.texts[len(self.calls) - 1]
        return LLMResponse(text=text, input_tokens=100, output_tokens=20, latency_s=0.5)


def _answer(**overrides: Any) -> str:
    """Return a valid JobDecision JSON document, with overrides."""
    fields: dict[str, Any] = {
        "rationale": "Main stack matches.",
        "company": "Acme",
        "role": "Senior Mobile Engineer",
        "location": "Istanbul, Türkiye",
        "work_mode": "hybrid",
        "english_risk": False,
        "seniority": "senior",
        "matching_skills": ["React Native"],
        "missing_skills": [],
        "decision": "APPLY",
    }
    return json.dumps(fields | overrides)


def test_user_prompt_wraps_posting_and_cv_in_separate_blocks() -> None:
    """Posting and CV each sit inside their own tagged block."""
    prompt = build_user_prompt(POSTING, CV)

    assert f"<job_posting>\n{POSTING}\n</job_posting>" in prompt
    assert f"<candidate_cv>\n{CV}\n</candidate_cv>" in prompt


def test_valid_answer_on_first_attempt() -> None:
    """A valid answer needs one call and reports that call's usage."""
    llm = FakeLLM(_answer())

    result = decide(POSTING, CV, model=MODEL, llm=llm)

    assert result.decision.decision == "APPLY"
    assert result.llm_decision == "APPLY"
    assert result.fired_rules == ()
    assert result.attempts == 1
    assert (result.input_tokens, result.output_tokens) == (100, 20)
    assert result.latency_s == pytest.approx(0.5)
    assert result.model == MODEL


def test_llm_receives_prompts_schema_and_model() -> None:
    """The workflow passes the provider, model, prompts and schema to the LLM."""
    llm = FakeLLM(_answer())

    decide(POSTING, CV, model=MODEL, provider="vertex", llm=llm)

    call = llm.calls[0]
    assert call["provider"] == "vertex"
    assert call["model"] == MODEL
    assert call["system"] == workflow.SYSTEM_PROMPT
    assert POSTING in call["user"]
    assert CV in call["user"]
    assert call["schema"] is JobDecision


def test_reasoning_and_timeout_default_to_the_model_and_client_defaults() -> None:
    """Without options the model keeps its own reasoning default and the usual timeout."""
    llm = FakeLLM(_answer())

    decide(POSTING, CV, model=MODEL, llm=llm)

    assert llm.calls[0]["reasoning"] is None
    assert llm.calls[0]["timeout_s"] == DEFAULT_TIMEOUT_S


def test_reasoning_and_timeout_are_passed_to_the_llm() -> None:
    """The reasoning switch and the timeout reach the LLM call unchanged."""
    llm = FakeLLM(_answer())

    decide(POSTING, CV, model=MODEL, reasoning=False, timeout_s=300.0, llm=llm)

    assert llm.calls[0]["reasoning"] is False
    assert llm.calls[0]["timeout_s"] == 300.0


@pytest.mark.parametrize(("setting", "expected"), [("on", True), ("off", False), ("default", None)])
def test_reasoning_setting_maps_to_the_llm_switch(setting: str, expected: bool | None) -> None:
    """The command-line words translate to True, False and "leave the default"."""
    assert reasoning_flag(setting) is expected


def test_cli_passes_reasoning_and_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--reasoning and --timeout-s on the command line reach the LLM call."""
    posting_file = tmp_path / "posting.txt"
    cv_file = tmp_path / "cv.md"
    posting_file.write_text(POSTING, encoding="utf-8")
    cv_file.write_text(CV, encoding="utf-8")
    fake = FakeLLM(_answer())
    monkeypatch.setattr(workflow, "generate_json", fake)
    argv = ["--posting", str(posting_file), "--cv", str(cv_file), "--model", MODEL]

    exit_code = main([*argv, "--reasoning", "on", "--timeout-s", "90"])

    assert exit_code == 0
    assert fake.calls[0]["reasoning"] is True
    assert fake.calls[0]["timeout_s"] == 90.0


def test_invalid_answer_is_retried_once_and_usage_is_summed() -> None:
    """After one invalid answer the workflow retries and adds up tokens and latency."""
    llm = FakeLLM(_answer(decision="MAYBE"), _answer(decision="PREPARE"))

    result = decide(POSTING, CV, model=MODEL, llm=llm)

    assert result.decision.decision == "PREPARE"
    assert result.attempts == 2
    assert (result.input_tokens, result.output_tokens) == (200, 40)
    assert result.latency_s == pytest.approx(1.0)


def test_retry_prompt_tells_the_model_what_was_wrong() -> None:
    """The second prompt names the rejected field; the first prompt does not."""
    llm = FakeLLM(_answer(decision="MAYBE"), _answer())

    decide(POSTING, CV, model=MODEL, llm=llm)

    assert "rejected" not in llm.calls[0]["user"]
    assert "rejected" in llm.calls[1]["user"]
    assert "- decision:" in llm.calls[1]["user"]


def test_two_invalid_answers_raise_decision_error() -> None:
    """The workflow gives up after the retry and names the model, not the CV."""
    llm = FakeLLM("not json at all", _answer(decision="MAYBE"))

    with pytest.raises(DecisionError) as raised:
        decide(POSTING, CV, model=MODEL, llm=llm)

    message = str(raised.value)
    assert len(llm.calls) == 2
    assert MODEL in message
    assert "- decision:" in message
    assert "CV-MARKER" not in message


def test_rule_override_keeps_the_llm_decision_visible() -> None:
    """A location violation turns APPLY into SKIP but the LLM's choice is kept."""
    llm = FakeLLM(_answer(location="Bursa, Türkiye", work_mode="onsite"))

    result = decide(POSTING, CV, model=MODEL, llm=llm)

    assert result.llm_decision == "APPLY"
    assert result.decision.decision == "SKIP"
    assert result.fired_rules == ("location",)
    assert "[rule: location]" in result.decision.rationale


@pytest.mark.parametrize(("posting", "cv"), [("  ", CV), (POSTING, "")])
def test_empty_input_is_rejected_before_calling_the_llm(posting: str, cv: str) -> None:
    """An empty posting or CV fails fast and costs no LLM call."""
    llm = FakeLLM(_answer())

    with pytest.raises(ValueError, match="empty"):
        decide(posting, cv, model=MODEL, llm=llm)

    assert llm.calls == []


def test_cli_prints_decision_and_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command prints the JSON decision plus tokens and latency, and exits 0."""
    posting_file = tmp_path / "posting.txt"
    cv_file = tmp_path / "cv.md"
    posting_file.write_text(POSTING, encoding="utf-8")
    cv_file.write_text(CV, encoding="utf-8")
    monkeypatch.setattr(workflow, "generate_json", FakeLLM(_answer()))

    exit_code = main(["--posting", str(posting_file), "--cv", str(cv_file), "--model", MODEL])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert '"decision": "APPLY"' in output
    assert "input tokens:  100" in output
    assert "output tokens: 20" in output
    assert "latency:       0.50 s" in output


def test_cli_reports_missing_posting_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing posting file gives exit code 1 and an error on stderr."""
    missing = tmp_path / "missing.txt"

    exit_code = main(["--posting", str(missing), "--model", MODEL])

    assert exit_code == 1
    assert "error:" in capsys.readouterr().err
