"""Tests for the extractor version. The LLM is always a fake; nothing goes to the network."""

import json
from pathlib import Path
from typing import Any

import pytest

from job_fit import extract, workflow
from job_fit.extract import (
    DECIDER_SYSTEM_PROMPT,
    DEFAULT_EXTRACTOR_MODEL,
    EXTRACTOR_SYSTEM_PROMPT,
    build_extractor_prompt,
    decide_with_extractor,
    extract_facts,
    extractor_prompt_fingerprint,
    main,
)
from job_fit.llm import LLMError, LLMResponse
from job_fit.schema import (
    DECISION_DESCRIPTION,
    LOCATION_DESCRIPTION,
    FitDecision,
    PostingFacts,
)
from job_fit.workflow import DecisionError

MODEL = "fake-main-model"
EXTRACTOR = "fake-extractor-model"
POSTING = "POSTING-MARKER Senior React Native engineer wanted."
CV = "CV-MARKER seven years of React Native."
INVALID = '{"not": "the schema"}'

# Usage the fake reports per call, by the schema asked for: (input, output, latency).
EXTRACTOR_USAGE = (50, 10, 0.25)
DECIDER_USAGE = (100, 20, 0.5)


def _facts(**overrides: Any) -> str:
    """Return a valid PostingFacts JSON document, with overrides."""
    fields: dict[str, Any] = {
        "company": "Acme",
        "role": "Senior Mobile Engineer",
        "location": "Istanbul, Türkiye",
        "work_mode": "hybrid",
        "english_risk": False,
        "seniority": "senior",
    }
    return json.dumps(fields | overrides)


def _fit(**overrides: Any) -> str:
    """Return a valid FitDecision JSON document, with overrides."""
    fields: dict[str, Any] = {
        "rationale": "Main stack matches.",
        "matching_skills": ["React Native"],
        "missing_skills": ["Go"],
        "decision": "APPLY",
    }
    return json.dumps(fields | overrides)


class FakeLLM:
    """Callable with the signature of generate_json that answers per schema."""

    def __init__(self, facts: list[str | Exception], fits: list[str | Exception]) -> None:
        """Store the answers for the extractor and for the main model, in call order."""
        self.answers: dict[type, list[str | Exception]] = {
            PostingFacts: list(facts),
            FitDecision: list(fits),
        }
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
        """Record the call and return or raise the next answer for its schema."""
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
        answer = self.answers[schema].pop(0)
        if isinstance(answer, Exception):
            raise answer
        input_tokens, output_tokens, latency_s = (
            EXTRACTOR_USAGE if schema is PostingFacts else DECIDER_USAGE
        )
        return LLMResponse(
            text=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=latency_s,
        )

    def calls_for(self, schema: type) -> list[dict[str, Any]]:
        """Return the recorded calls that asked for the given schema."""
        return [call for call in self.calls if call["schema"] is schema]


def _decide(llm: FakeLLM, **options: Any) -> workflow.DecisionResult:
    """Run the extractor version on the fixed posting and CV with a fake LLM."""
    return decide_with_extractor(
        POSTING, CV, model=MODEL, extractor_model=EXTRACTOR, llm=llm, **options
    )


def test_facts_and_fit_are_joined_into_one_decision() -> None:
    """Fact fields come from the extractor, judgment fields from the main model."""
    result = _decide(FakeLLM([_facts(company="Globex")], [_fit(decision="PREPARE")]))

    decision = result.decision
    assert (decision.company, decision.location, decision.seniority) == (
        "Globex",
        "Istanbul, Türkiye",
        "senior",
    )
    assert (decision.rationale, decision.missing_skills) == ("Main stack matches.", ["Go"])
    assert decision.decision == "PREPARE"
    assert result.llm_decision == "PREPARE"
    assert result.fired_rules == ()


def test_hard_rules_fire_on_the_extractors_facts() -> None:
    """An on-site job outside the allowed cities is SKIP whatever the main model says."""
    llm = FakeLLM([_facts(location="Bursa, Türkiye", work_mode="onsite")], [_fit()])

    result = _decide(llm)

    assert result.llm_decision == "APPLY"
    assert result.decision.decision == "SKIP"
    assert result.fired_rules == ("location",)
    assert result.decision.rationale.endswith("[rule: location]")


def test_keyword_check_raises_the_english_flag_the_extractor_missed() -> None:
    """The English flag is the extractor's reading or the keyword match, as in the workflow."""
    posting = POSTING + " Fluent English is required."

    result = decide_with_extractor(
        posting, CV, model=MODEL, llm=FakeLLM([_facts(english_risk=False)], [_fit()])
    )

    assert result.decision.english_risk is True


def test_extractor_sees_the_posting_but_never_the_cv() -> None:
    """The call that needs no personal data does not get any."""
    llm = FakeLLM([_facts()], [_fit()])

    _decide(llm)

    call = llm.calls_for(PostingFacts)[0]
    assert call["model"] == EXTRACTOR
    assert call["system"] == EXTRACTOR_SYSTEM_PROMPT
    assert "POSTING-MARKER" in call["user"]
    assert "CV-MARKER" not in call["user"]
    assert "CV-MARKER" not in call["system"]


def test_main_model_sees_posting_and_cv_and_returns_only_the_fit() -> None:
    """The main model gets both texts and is asked for the judgment fields alone."""
    llm = FakeLLM([_facts()], [_fit()])

    _decide(llm)

    call = llm.calls_for(FitDecision)[0]
    assert call["model"] == MODEL
    assert call["system"] == DECIDER_SYSTEM_PROMPT
    assert "POSTING-MARKER" in call["user"]
    assert "CV-MARKER" in call["user"]


def test_extraction_runs_before_the_decision() -> None:
    """The facts are extracted first."""
    llm = FakeLLM([_facts()], [_fit()])

    _decide(llm)

    assert [call["schema"] for call in llm.calls] == [PostingFacts, FitDecision]


def test_reasoning_switch_reaches_only_the_main_model() -> None:
    """The extractor keeps its model's default; the switch is the main model's."""
    llm = FakeLLM([_facts()], [_fit()])

    _decide(llm, reasoning=True, timeout_s=90.0)

    assert llm.calls_for(PostingFacts)[0]["reasoning"] is None
    assert llm.calls_for(FitDecision)[0]["reasoning"] is True
    assert {call["timeout_s"] for call in llm.calls} == {90.0}


def test_usage_is_split_between_the_two_models() -> None:
    """Main-model tokens and extractor tokens are kept apart, so each can be priced."""
    result = _decide(FakeLLM([_facts()], [_fit()]))

    assert result.model == MODEL
    assert (result.input_tokens, result.output_tokens) == DECIDER_USAGE[:2]
    assert len(result.extra_calls) == 1
    extra = result.extra_calls[0]
    assert extra.model == EXTRACTOR
    assert (extra.input_tokens, extra.output_tokens) == EXTRACTOR_USAGE[:2]
    assert extra.latency_s == pytest.approx(EXTRACTOR_USAGE[2])
    assert result.latency_s == pytest.approx(EXTRACTOR_USAGE[2] + DECIDER_USAGE[2])
    assert result.attempts == 2


def test_invalid_facts_are_retried_once_and_the_retry_is_counted() -> None:
    """A schema retry of the extractor adds a call and its tokens to the extractor's usage."""
    llm = FakeLLM([INVALID, _facts()], [_fit()])

    result = _decide(llm)

    assert result.attempts == 3
    assert result.extra_calls[0].input_tokens == 2 * EXTRACTOR_USAGE[0]
    assert "did not match the required JSON schema" in llm.calls_for(PostingFacts)[1]["user"]
    assert result.input_tokens == DECIDER_USAGE[0]


def test_facts_invalid_twice_fail_the_posting_without_asking_the_main_model() -> None:
    """There is no fallback: a failed extraction is a failed posting."""
    llm = FakeLLM([INVALID, INVALID], [_fit()])

    with pytest.raises(DecisionError, match=EXTRACTOR):
        _decide(llm)

    assert llm.calls_for(FitDecision) == []


def test_fit_invalid_twice_fails_the_posting() -> None:
    """A main model that never returns a valid judgment fails the posting."""
    with pytest.raises(DecisionError, match=MODEL):
        _decide(FakeLLM([_facts()], [INVALID, INVALID]))


def test_a_failed_extractor_call_is_not_swallowed() -> None:
    """A provider error in the sub-agent reaches the caller as it is."""
    with pytest.raises(LLMError, match="quota"):
        _decide(FakeLLM([LLMError("quota")], [_fit()]))


@pytest.mark.parametrize(("posting", "cv"), [("  ", CV), (POSTING, "\n")])
def test_empty_posting_or_cv_is_rejected_before_any_call(posting: str, cv: str) -> None:
    """Nothing is sent to a model for an empty input."""
    llm = FakeLLM([_facts()], [_fit()])

    with pytest.raises(ValueError, match="empty"):
        decide_with_extractor(posting, cv, model=MODEL, llm=llm)

    assert llm.calls == []


def test_extract_facts_uses_the_cheap_model_by_default() -> None:
    """Without a choice the extractor is the cheapest measured model."""
    llm = FakeLLM([_facts()], [])

    extraction = extract_facts(POSTING, llm=llm)

    assert extraction.model == DEFAULT_EXTRACTOR_MODEL
    assert llm.calls[0]["model"] == DEFAULT_EXTRACTOR_MODEL
    assert extraction.facts.work_mode == "hybrid"
    assert extraction.usage.attempts == 1


def test_extract_facts_rejects_an_empty_posting() -> None:
    """An empty posting has no facts to extract."""
    with pytest.raises(ValueError, match="empty"):
        extract_facts(" ", llm=FakeLLM([_facts()], []))


def test_extractor_prompt_wraps_the_posting_in_a_block() -> None:
    """The posting sits inside its own tagged block."""
    assert build_extractor_prompt(f"  {POSTING}\n") == f"<job_posting>\n{POSTING}\n</job_posting>\n"


def test_decider_prompt_keeps_the_workflows_decision_definitions() -> None:
    """Only the extraction step is removed: the three definitions are word for word."""
    definitions = [line for line in workflow.SYSTEM_PROMPT.splitlines() if line.startswith("   - ")]

    assert len(definitions) == 3
    assert all(line in DECIDER_SYSTEM_PROMPT.splitlines() for line in definitions)
    assert "Extract the facts" not in DECIDER_SYSTEM_PROMPT
    assert "Extract the facts" in EXTRACTOR_SYSTEM_PROMPT


def test_fingerprint_covers_both_prompts_and_both_schemas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A change to either prompt changes the fingerprint; both schemas are part of it."""
    before = extractor_prompt_fingerprint()
    # The schema is rendered as JSON, where non-ASCII letters are escaped.
    assert json.dumps(LOCATION_DESCRIPTION)[1:-1] in before
    assert json.dumps(DECISION_DESCRIPTION)[1:-1] in before

    monkeypatch.setattr(extract, "EXTRACTOR_SYSTEM_PROMPT", "another extractor prompt")
    after_extractor = extractor_prompt_fingerprint()
    monkeypatch.setattr(extract, "DECIDER_SYSTEM_PROMPT", "another decider prompt")

    assert after_extractor != before
    assert extractor_prompt_fingerprint() != after_extractor


def test_cli_prints_the_decision_and_the_sub_agents_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command prints the joined decision and what the other model used."""
    posting_file = tmp_path / "posting.txt"
    cv_file = tmp_path / "cv.md"
    posting_file.write_text(POSTING, encoding="utf-8")
    cv_file.write_text(CV, encoding="utf-8")
    monkeypatch.setattr(extract, "generate_json", FakeLLM([_facts()], [_fit()]))

    exit_code = main(["--posting", str(posting_file), "--cv", str(cv_file), "--model", MODEL])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert '"decision": "APPLY"' in output
    assert '"company": "Acme"' in output
    assert f"other model:   {DEFAULT_EXTRACTOR_MODEL}: 50 input, 10 output tokens" in output
    assert "attempts:      2" in output
