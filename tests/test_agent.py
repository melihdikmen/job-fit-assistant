"""Tests for the agent. The chat model is a scripted fake; nothing goes to the network."""

from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from job_fit import agent as agent_module
from job_fit.agent import (
    AGENT_SYSTEM_PROMPT,
    MAX_AGENT_STEPS,
    agent_prompt_fingerprint,
    build_agent_prompt,
    decide_with_agent,
    main,
    summarize_run,
)
from job_fit.llm import CHAT_MODEL_ERRORS, DEFAULT_TIMEOUT_S, LLMError
from job_fit.workflow import DecisionError, ToolCallRecord

MODEL = "gemini-2.5-flash"
POSTING = "POSTING-MARKER Senior React Native developer, hybrid in Istanbul."
CV = """\
# Jane Doe

## Experience
CV-MARKER seven years of React Native.

## Skills
TypeScript
"""


def _decision_args(**overrides: Any) -> dict[str, Any]:
    """Build the arguments of a valid structured answer."""
    values: dict[str, Any] = {
        "rationale": "The CV covers the core stack.",
        "company": "Acme",
        "role": "Senior Developer",
        "location": "Istanbul, Türkiye",
        "work_mode": "hybrid",
        "english_risk": False,
        "seniority": "senior",
        "matching_skills": ["React Native"],
        "missing_skills": [],
        "decision": "APPLY",
    }
    return values | overrides


def _ai(
    calls: Sequence[tuple[str, dict[str, Any]]], tokens: tuple[int, int] = (100, 20)
) -> AIMessage:
    """Build a model message that makes the given tool calls and reports token usage."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": f"call-{index}", "type": "tool_call"}
            for index, (name, args) in enumerate(calls)
        ],
        usage_metadata={
            "input_tokens": tokens[0],
            "output_tokens": tokens[1],
            "total_tokens": sum(tokens),
        },
    )


def _answer(**overrides: Any) -> AIMessage:
    """Build the model message that carries the structured decision."""
    return _ai([("JobDecision", _decision_args(**overrides))])


READ_SKILLS = _ai([("get_cv_section", {"topic": "skills"})])
CHECK_RULES_ARGS = {
    "location": "Istanbul, Türkiye",
    "work_mode": "hybrid",
    "english_required": False,
    "seniority": "senior",
}


class ScriptedChatModel(BaseChatModel):
    """Chat model that replays canned messages and records what it was sent."""

    script: list[Any]
    seen: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        """Name the model type for LangChain."""
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        """Accept the agent's tools; the script decides what is called."""
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        """Return or raise the next scripted item; repeat the last one when exhausted."""
        self.seen.append(list(messages))
        item = self.script[min(len(self.seen), len(self.script)) - 1]
        if isinstance(item, Exception):
            raise item
        return ChatResult(generations=[ChatGeneration(message=item)])


class FakeFactory:
    """Stands in for ``llm.chat_model`` and remembers how it was called."""

    def __init__(self, *script: Any) -> None:
        """Create the scripted model the factory will hand out."""
        self.model = ScriptedChatModel(script=list(script), seen=[])
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, provider: str, model: str, *, reasoning: bool | None, timeout_s: float
    ) -> ScriptedChatModel:
        """Record the configuration asked for and return the scripted model."""
        self.calls.append(
            {"provider": provider, "model": model, "reasoning": reasoning, "timeout_s": timeout_s}
        )
        return self.model


def _decide(factory: FakeFactory, posting: str = POSTING, **kwargs: Any) -> Any:
    """Run the agent on the test posting and CV with a fake chat model."""
    return decide_with_agent(posting, CV, model=MODEL, chat_model_factory=factory, **kwargs)


def test_summarize_run_sums_tokens_over_all_model_calls() -> None:
    """Every model message counts as a step and adds its token usage."""
    messages = [
        HumanMessage("posting"),
        _ai([("get_cv_section", {"topic": "skills"})], tokens=(100, 20)),
        ToolMessage("TypeScript", tool_call_id="call-0"),
        _ai([("JobDecision", _decision_args())], tokens=(150, 30)),
    ]

    trace = summarize_run(messages)

    assert (trace.model_calls, trace.input_tokens, trace.output_tokens) == (2, 250, 50)


def test_summarize_run_lists_tool_calls_in_order_without_the_structured_answer() -> None:
    """The JobDecision call is the answer, not a tool the agent used."""
    messages = [
        _ai([("get_cv_section", {"topic": "skills"}), ("get_cv_section", {"topic": "experience"})]),
        _ai([("check_rules", CHECK_RULES_ARGS)]),
        _ai([("JobDecision", _decision_args())]),
    ]

    trace = summarize_run(messages)

    assert trace.tool_calls == (
        ToolCallRecord(name="get_cv_section", arguments={"topic": "skills"}),
        ToolCallRecord(name="get_cv_section", arguments={"topic": "experience"}),
        ToolCallRecord(name="check_rules", arguments=CHECK_RULES_ARGS),
    )


def test_summarize_run_counts_a_message_without_usage_as_zero_tokens() -> None:
    """A model message that reports no usage still counts as a step."""
    trace = summarize_run([AIMessage(content="done")])

    assert (trace.model_calls, trace.input_tokens, trace.output_tokens) == (1, 0, 0)


def test_agent_calls_tools_then_returns_the_decision() -> None:
    """Tool calls are executed, counted and followed by the structured answer."""
    factory = FakeFactory(READ_SKILLS, _ai([("check_rules", CHECK_RULES_ARGS)]), _answer())

    result = _decide(factory)

    assert result.decision.decision == "APPLY"
    assert result.llm_decision == "APPLY"
    assert result.attempts == 3
    assert [call.name for call in result.tool_calls] == ["get_cv_section", "check_rules"]
    assert (result.input_tokens, result.output_tokens) == (300, 60)
    assert result.latency_s > 0


def test_tool_result_reaches_the_model_on_the_next_step() -> None:
    """What get_cv_section returns is in the conversation of the following model call."""
    factory = FakeFactory(READ_SKILLS, _answer())

    _decide(factory)

    second_call = factory.model.seen[1]
    assert [message.content for message in second_call if isinstance(message, ToolMessage)] == [
        "TypeScript"
    ]


def test_first_model_call_sees_the_posting_but_not_the_cv() -> None:
    """The CV reaches the model only through get_cv_section (D23)."""
    factory = FakeFactory(_answer())

    _decide(factory)

    first_call = "\n".join(str(message.content) for message in factory.model.seen[0])
    assert "POSTING-MARKER" in first_call
    assert AGENT_SYSTEM_PROMPT in first_call
    assert "CV-MARKER" not in first_call


def test_rules_override_the_agents_decision_in_code() -> None:
    """An APPLY for an on-site job in Bursa becomes SKIP even if check_rules was not called."""
    factory = FakeFactory(_answer(location="Bursa, Türkiye", work_mode="onsite"))

    result = _decide(factory)

    assert result.llm_decision == "APPLY"
    assert result.decision.decision == "SKIP"
    assert result.fired_rules == ("location",)
    assert result.tool_calls == ()


def test_agent_that_keeps_calling_tools_hits_the_step_limit() -> None:
    """Running out of steps is a failed decision, not an endless loop (D25)."""
    factory = FakeFactory(READ_SKILLS)

    with pytest.raises(DecisionError, match=f"within {MAX_AGENT_STEPS} steps"):
        _decide(factory)

    assert len(factory.model.seen) == MAX_AGENT_STEPS


def test_step_limit_can_be_lowered() -> None:
    """The limit counts model calls."""
    factory = FakeFactory(READ_SKILLS)

    with pytest.raises(DecisionError, match="within 2 steps"):
        _decide(factory, max_steps=2)

    assert len(factory.model.seen) == 2


def test_invalid_structured_answer_is_retried_within_the_step_budget() -> None:
    """A decision outside the schema is sent back to the model, which can correct it."""
    factory = FakeFactory(_answer(decision="MAYBE"), _answer(decision="PREPARE"))

    result = _decide(factory)

    assert result.decision.decision == "PREPARE"
    assert result.attempts == 2


def test_agent_that_never_answers_validly_fails() -> None:
    """An answer that stays invalid uses up the steps and counts as failed."""
    factory = FakeFactory(_answer(decision="MAYBE"))

    with pytest.raises(DecisionError):
        _decide(factory)


def test_agent_that_ends_without_a_structured_answer_fails() -> None:
    """Plain text instead of a JobDecision is not a decision."""
    factory = FakeFactory(AIMessage(content="I think you should apply."))

    with pytest.raises(DecisionError):
        _decide(factory, max_steps=2)


def test_provider_error_becomes_llm_error() -> None:
    """A failed model call is reported as LLMError, as in the workflow."""
    error_type = next(kind for kind in CHAT_MODEL_ERRORS if kind.__name__ == "HTTPError")
    factory = FakeFactory(error_type("connection reset"))

    with pytest.raises(LLMError, match="connection reset"):
        _decide(factory)


def test_model_settings_reach_the_chat_model_factory() -> None:
    """Provider, model, reasoning and timeout are passed on unchanged."""
    factory = FakeFactory(_answer())

    _decide(factory, provider="vertex", reasoning=True, timeout_s=12.0)

    assert factory.calls == [
        {"provider": "vertex", "model": MODEL, "reasoning": True, "timeout_s": 12.0}
    ]


def test_default_timeout_is_the_workflows() -> None:
    """Without an explicit timeout the agent uses the shared default."""
    factory = FakeFactory(_answer())

    _decide(factory)

    assert factory.calls[0]["timeout_s"] == DEFAULT_TIMEOUT_S


@pytest.mark.parametrize(("posting", "cv"), [("  ", CV), (POSTING, "")])
def test_empty_posting_or_cv_is_rejected_before_any_model_call(posting: str, cv: str) -> None:
    """Bad input fails early."""
    factory = FakeFactory(_answer())

    with pytest.raises(ValueError, match="empty"):
        decide_with_agent(posting, cv, model=MODEL, chat_model_factory=factory)

    assert factory.calls == []


def test_tool_calls_are_logged_with_their_arguments(caplog: pytest.LogCaptureFixture) -> None:
    """Each tool call is logged by name and arguments."""
    factory = FakeFactory(READ_SKILLS, _answer())

    with caplog.at_level("INFO", logger=agent_module.__name__):
        _decide(factory)

    assert "tool call: get_cv_section {'topic': 'skills'}" in caplog.text
    assert "TypeScript" not in caplog.text


def test_agent_prompt_holds_the_posting_only() -> None:
    """The user message wraps the posting in a delimited block."""
    assert build_agent_prompt(" posting text ") == "<job_posting>\nposting text\n</job_posting>\n"


def test_prompt_fingerprint_changes_with_the_cv_headings() -> None:
    """A different topic list is a different prompt for the model."""
    other_cv = CV.replace("## Skills", "## Stack")

    assert agent_prompt_fingerprint(CV) != agent_prompt_fingerprint(other_cv)
    assert AGENT_SYSTEM_PROMPT in agent_prompt_fingerprint(CV)


def test_main_reports_a_missing_posting_file(
    tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing posting file gives exit code 1 and an error on stderr."""
    exit_code = main(["--posting", str(tmp_path / "missing.txt"), "--model", MODEL])

    assert exit_code == 1
    assert "error:" in capsys.readouterr().err
