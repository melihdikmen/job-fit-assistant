"""Tests for the page's handler. No server is started and no model is called."""

from pathlib import Path
from typing import Any

import pytest

from job_fit import ui
from job_fit.llm import LLMError
from job_fit.pricing import call_cost_usd, price_for
from job_fit.schema import JobDecision
from job_fit.workflow import DecisionError, DecisionResult, ToolCallRecord

POSTING_TEXT = "Senior Python Engineer at Acme, Istanbul. SECRET-POSTING-MARKER"
CV_TEXT = "## Experience\nSECRET-CV-MARKER"
INPUT_TOKENS = 2300
OUTPUT_TOKENS = 2100


def _decision(label: str = "APPLY") -> JobDecision:
    """Build a valid decision.

    Args:
        label: Final decision label.

    Returns:
        A decision for a senior role in Istanbul.
    """
    return JobDecision(
        rationale="The CV covers the core stack.",
        company="Acme",
        role="Senior Python Engineer",
        location="Istanbul, Türkiye",
        work_mode="hybrid",
        english_risk=True,
        seniority="senior",
        matching_skills=["Python", "FastAPI"],
        missing_skills=[],
        decision=label,
    )


def _result(**overrides: Any) -> DecisionResult:
    """Build a decision result with known token counts.

    Args:
        **overrides: Fields that differ from the defaults.

    Returns:
        A workflow-like result: one model call, no tool calls, no rule fired.
    """
    fields: dict[str, Any] = {
        "decision": _decision(),
        "llm_decision": "APPLY",
        "fired_rules": (),
        "model": "gemini-2.5-flash",
        "input_tokens": INPUT_TOKENS,
        "output_tokens": OUTPUT_TOKENS,
        "latency_s": 11.94,
        "attempts": 1,
    }
    fields.update(overrides)
    return DecisionResult(**fields)


class FakeDecide:
    """Stands in for ``decide`` or ``decide_with_agent`` and records its calls."""

    def __init__(self, outcome: DecisionResult | Exception) -> None:
        """Store what the fake returns or raises.

        Args:
            outcome: Result to return, or exception to raise.
        """
        self.outcome = outcome
        self.calls: list[dict[str, Any]] = []

    def __call__(self, posting_text: str, cv: str, **options: Any) -> DecisionResult:
        """Record the call, then return or raise the stored outcome.

        Args:
            posting_text: Posting passed by the handler.
            cv: CV passed by the handler.
            **options: Keyword options passed by the handler.

        Returns:
            The stored result.
        """
        self.calls.append({"posting_text": posting_text, "cv": cv, **options})
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _run(
    workflow: FakeDecide | None = None,
    agent: FakeDecide | None = None,
    *,
    posting_text: str = POSTING_TEXT,
    choice_label: str = ui.DEFAULT_MODEL_CHOICE.label,
    version: str = "workflow",
) -> ui.DecisionView:
    """Call the handler with fakes for both versions and a fixed CV.

    Args:
        workflow: Fake for the workflow; a succeeding one by default.
        agent: Fake for the agent; a succeeding one by default.
        posting_text: Text pasted into the page.
        choice_label: Chosen model configuration.
        version: Chosen version.

    Returns:
        The view the handler built.
    """
    return ui.run_decision(
        posting_text,
        choice_label,
        version,
        load_cv=lambda: CV_TEXT,
        decide_fns={
            "workflow": workflow or FakeDecide(_result()),
            "agent": agent or FakeDecide(_result()),
        },
    )


def _usage(view: ui.DecisionView) -> dict[str, str]:
    """Return the usage table of a view as a mapping.

    Args:
        view: View built by the handler.

    Returns:
        Measure name to displayed value.
    """
    return dict(view.usage)


@pytest.mark.parametrize("version", ui.VERSIONS)
def test_version_selects_the_matching_decide_function(version: str) -> None:
    """Only the function of the chosen version is called."""
    fakes = {"workflow": FakeDecide(_result()), "agent": FakeDecide(_result())}

    _run(fakes["workflow"], fakes["agent"], version=version)

    called = [name for name, fake in fakes.items() if fake.calls]
    assert called == [version]


@pytest.mark.parametrize("choice", ui.MODEL_CHOICES, ids=lambda choice: choice.label)
def test_chosen_configuration_reaches_the_decide_function(choice: ui.ModelChoice) -> None:
    """Provider, model and reasoning of the chosen row are passed on, with posting and CV."""
    fake = FakeDecide(_result())

    _run(fake, choice_label=choice.label)

    assert fake.calls == [
        {
            "posting_text": POSTING_TEXT,
            "cv": CV_TEXT,
            "model": choice.model,
            "provider": choice.provider,
            "reasoning": choice.reasoning,
        }
    ]


@pytest.mark.parametrize("choice", ui.MODEL_CHOICES, ids=lambda choice: choice.label)
def test_every_model_choice_is_a_priced_vertex_model(choice: ui.ModelChoice) -> None:
    """The agent supports only Vertex, and a choice without a price could show no cost."""
    assert choice.provider == "vertex"
    assert price_for(choice.provider, choice.model) is not None


def test_model_choice_labels_are_unique() -> None:
    """Labels are the lookup key of the dropdown."""
    labels = [choice.label for choice in ui.MODEL_CHOICES]

    assert len(labels) == len(set(labels))


def test_default_choice_is_the_model_chosen_in_phase_3() -> None:
    """The page starts on gemini-2.5-flash with reasoning on."""
    assert ui.DEFAULT_MODEL_CHOICE.model == "gemini-2.5-flash"
    assert ui.DEFAULT_MODEL_CHOICE.reasoning is True
    assert ui.DEFAULT_VERSION == "workflow"


def test_successful_decision_is_shown_with_rationale_and_facts() -> None:
    """The decision, the rationale and the extracted facts reach the view."""
    view = _run()

    assert view.decision == "APPLY"
    assert view.message == ui.RECOMMENDATION_MESSAGE
    assert view.rationale == "The CV covers the core stack."
    assert dict(view.facts) == {
        "Company": "Acme",
        "Role": "Senior Python Engineer",
        "Location": "Istanbul, Türkiye",
        "Work mode": "hybrid",
        "Seniority": "senior",
        "English risk": "yes",
        "Matching skills": "Python, FastAPI",
        "Missing skills": "none",
    }


def test_cost_is_computed_from_the_price_table() -> None:
    """The displayed cost is the list price of the tokens the call used."""
    choice = ui.DEFAULT_MODEL_CHOICE
    expected = call_cost_usd(choice.provider, choice.model, INPUT_TOKENS, OUTPUT_TOKENS)

    usage = _usage(_run())

    assert expected is not None
    assert usage["Cost (list price)"] == ui.format_cost(expected)
    assert usage["Input tokens"] == str(INPUT_TOKENS)
    assert usage["Output tokens"] == str(OUTPUT_TOKENS)
    assert usage["Latency"] == "11.9 s"
    assert usage["Model calls"] == "1"


def test_format_cost_keeps_fractions_of_a_cent() -> None:
    """A cost below one cent is not rounded to zero."""
    assert ui.format_cost(0.00639) == "$0.0064"


def test_format_cost_without_a_price_says_so() -> None:
    """An unknown price is reported, not shown as zero."""
    assert ui.format_cost(None) == ui.PRICE_UNKNOWN


def test_rule_override_shows_the_rule_and_the_models_own_decision() -> None:
    """When a hard rule forces SKIP, the page shows both decisions and the rule."""
    overridden = _result(
        decision=_decision("SKIP"), llm_decision="APPLY", fired_rules=("location",)
    )

    view = _run(FakeDecide(overridden))

    assert view.decision == "SKIP"
    assert _usage(view)["Model's decision before rules"] == "APPLY"
    assert _usage(view)["Hard rules fired"] == "location"


def test_no_rule_fired_is_shown_as_none() -> None:
    """Without an override the rules cell reads "none"."""
    assert _usage(_run())["Hard rules fired"] == "none"


def test_agent_tool_calls_are_listed_in_order() -> None:
    """Each tool call is one row: step number, tool name, arguments as JSON."""
    agent_result = _result(
        attempts=3,
        tool_calls=(
            ToolCallRecord(name="check_rules", arguments={"location": "İzmir, Türkiye"}),
            ToolCallRecord(name="get_cv_section", arguments={"topic": "skills"}),
        ),
    )

    view = _run(agent=FakeDecide(agent_result), version="agent")

    assert view.tool_calls == (
        ("1", "check_rules", '{"location": "İzmir, Türkiye"}'),
        ("2", "get_cv_section", '{"topic": "skills"}'),
    )
    assert _usage(view)["Model calls"] == "3"
    assert _usage(view)["Tool calls"] == "2"


def test_workflow_has_no_tool_call_rows() -> None:
    """The workflow calls no tools."""
    assert _run().tool_calls == ()


@pytest.mark.parametrize("posting_text", ["", "   \n\t"])
def test_empty_posting_makes_no_model_call(posting_text: str) -> None:
    """A blank posting is answered with a message and costs nothing."""
    fake = FakeDecide(_result())

    view = _run(fake, posting_text=posting_text)

    assert fake.calls == []
    assert view.decision == ""
    assert view.message == ui.EMPTY_POSTING_MESSAGE


@pytest.mark.parametrize(
    "error",
    [LLMError("Vertex call failed: 429"), DecisionError("invalid output 2 times")],
    ids=["llm-error", "decision-error"],
)
def test_handled_failure_becomes_a_message_without_personal_text(error: Exception) -> None:
    """A failed call shows its reason and neither the posting nor the CV."""
    view = _run(FakeDecide(error))

    assert view.decision == ""
    assert str(error) in view.message
    assert view.rationale == ""
    assert view.facts == view.usage == view.tool_calls == ()
    assert "SECRET-POSTING-MARKER" not in view.message
    assert "SECRET-CV-MARKER" not in view.message


def test_missing_cv_file_becomes_a_message(tmp_path: Path) -> None:
    """A CV that cannot be read is reported and no model call is made."""
    fake = FakeDecide(_result())
    missing = tmp_path / "cv.md"

    view = ui.run_decision(
        POSTING_TEXT,
        ui.DEFAULT_MODEL_CHOICE.label,
        "workflow",
        load_cv=lambda: missing.read_text(encoding="utf-8"),
        decide_fns={"workflow": fake},
    )

    assert fake.calls == []
    assert view.decision == ""
    assert view.message.startswith("No decision:")


def test_unknown_model_choice_becomes_a_message() -> None:
    """A label that is not in the table is refused before any call."""
    fake = FakeDecide(_result())

    view = _run(fake, choice_label="gpt-unknown")

    assert fake.calls == []
    assert "Unknown model choice" in view.message


def test_unknown_version_becomes_a_message() -> None:
    """A version other than workflow or agent is refused before any call."""
    fake = FakeDecide(_result())

    view = _run(fake, fake, version="swarm")

    assert fake.calls == []
    assert "Unknown version" in view.message


def test_view_outputs_follow_the_component_order() -> None:
    """The view maps to label, message, rationale and three tables of lists."""
    view = _run()

    decision, message, rationale, facts, usage, tool_calls = ui.view_outputs(view)

    assert decision == "APPLY"
    assert message == view.message
    assert rationale == view.rationale
    assert facts[0] == ["Company", "Acme"]
    assert usage[0] == ["Model", ui.DEFAULT_MODEL_CHOICE.label]
    assert tool_calls == []


def test_view_outputs_clear_the_label_when_there_is_no_decision() -> None:
    """An error view empties the recommendation instead of showing a blank label."""
    view = ui.DecisionView(decision="", message="No decision: boom")

    assert ui.view_outputs(view)[0] is None


def test_main_without_a_cv_file_exits_before_starting_a_server(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing CV is a configuration error: exit code 1 and no page."""
    exit_code = ui.main(["--cv", str(tmp_path / "missing.md")])

    assert exit_code == 1
    assert "CV file not found" in capsys.readouterr().err
