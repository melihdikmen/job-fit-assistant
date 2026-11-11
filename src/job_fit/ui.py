"""Local web page: paste a posting, choose model and version, see the decision and its cost.

Run it with::

    uv run python -m job_fit.ui

The page is a thin shell around ``workflow.decide`` and ``agent.decide_with_agent``. It
only recommends: it has no action that applies, sends or saves anything. It listens on
this machine only and has no option to publish a share link, because every request
sends the candidate's CV to a model.
"""

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

import gradio as gr

from job_fit.agent import decide_with_agent
from job_fit.llm import LLMError
from job_fit.pricing import call_cost_usd
from job_fit.workflow import (
    DEFAULT_CV_PATH,
    DecideFn,
    DecisionError,
    DecisionResult,
    decide,
)

Version = Literal["workflow", "agent"]
CvLoader = Callable[[], str]

VERSIONS: tuple[Version, ...] = get_args(Version)
DEFAULT_VERSION: Version = "workflow"

LOCAL_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
ANALYTICS_VARIABLE = "GRADIO_ANALYTICS_ENABLED"

APP_TITLE = "Job Fit Assistant"
PRICE_UNKNOWN = "price unknown"
NONE_TEXT = "none"
POSTING_LINES = 16
RATIONALE_LINES = 6

FACT_HEADERS = ("Field", "Value")
USAGE_HEADERS = ("Measure", "Value")
TOOL_CALL_HEADERS = ("Step", "Tool", "Arguments")

EMPTY_POSTING_MESSAGE = "Paste a job posting first. No model call was made."
RECOMMENDATION_MESSAGE = (
    "This is a recommendation only. Nothing was applied, sent or saved; the decision "
    "to apply is yours."
)
INTRO_TEXT = """\
# Job Fit Assistant

Paste a job posting, choose a model and a version, and press **Decide**. The posting and
the CV in `{cv_path}` are sent to the chosen model; the hard rules (location, seniority,
English) are then applied in code. The page only recommends: it never applies, sends
messages or fills in forms.
"""
VERSION_INFO = (
    "workflow: one model call, the version in use. agent: several calls with tools; "
    "measured only with the default model."
)


@dataclass(frozen=True)
class ModelChoice:
    """One model configuration the page offers.

    Attributes:
        label: Text shown in the dropdown.
        provider: LLM provider name.
        model: Model id as the provider names it.
        reasoning: True to enable model reasoning, False to disable it, None for the
            model default.
    """

    label: str
    provider: str
    model: str
    reasoning: bool | None


# Only configurations measured in Phase 3 (docs/results.md), all on Vertex AI because
# the agent supports no other provider.
MODEL_CHOICES: tuple[ModelChoice, ...] = (
    ModelChoice(
        label="gemini-2.5-flash, reasoning on (chosen model)",
        provider="vertex",
        model="gemini-2.5-flash",
        reasoning=True,
    ),
    ModelChoice(
        label="gemini-2.5-flash, reasoning off",
        provider="vertex",
        model="gemini-2.5-flash",
        reasoning=False,
    ),
    ModelChoice(
        label="gemini-3.1-flash-lite (cheapest, fastest)",
        provider="vertex",
        model="gemini-3.1-flash-lite",
        reasoning=None,
    ),
)
DEFAULT_MODEL_CHOICE = MODEL_CHOICES[0]

DECIDE_FUNCTIONS: Mapping[Version, DecideFn] = {
    "workflow": decide,
    "agent": decide_with_agent,
}


@dataclass(frozen=True)
class DecisionView:
    """Everything the page displays for one request, as plain text.

    Attributes:
        decision: "APPLY", "PREPARE" or "SKIP"; empty when no decision was made.
        message: Status line: the recommendation notice, or what went wrong.
        rationale: The model's reasoning; empty when no decision was made.
        facts: (field, value) rows of the remaining ``JobDecision`` fields.
        usage: (measure, value) rows: model, rules, tokens, cost, latency, calls.
        tool_calls: (step, tool, arguments) rows; empty for the workflow.
    """

    decision: str
    message: str
    rationale: str = ""
    facts: tuple[tuple[str, str], ...] = ()
    usage: tuple[tuple[str, str], ...] = ()
    tool_calls: tuple[tuple[str, str, str], ...] = ()


def find_model_choice(label: str) -> ModelChoice:
    """Look up a model configuration by its dropdown label.

    Args:
        label: Label as shown in the dropdown.

    Returns:
        The matching configuration.

    Raises:
        ValueError: If no configuration has that label.
    """
    for choice in MODEL_CHOICES:
        if choice.label == label:
            return choice
    raise ValueError(f"Unknown model choice: '{label}'.")


def format_cost(cost_usd: float | None) -> str:
    """Render the cost of one decision.

    One decision costs a fraction of a cent, so two decimals would print "$0.00".

    Args:
        cost_usd: Cost in USD, or None when the model has no known price.

    Returns:
        E.g. "$0.0064", or "price unknown".
    """
    if cost_usd is None:
        return PRICE_UNKNOWN
    return f"${cost_usd:.4f}"


def _join(items: Sequence[str]) -> str:
    """Join a list for one table cell.

    Args:
        items: Values to show.

    Returns:
        The values separated by commas, or "none" for an empty list.
    """
    return ", ".join(items) or NONE_TEXT


def _fact_rows(result: DecisionResult) -> tuple[tuple[str, str], ...]:
    """Build the table of facts the model extracted from the posting.

    Args:
        result: Result of a decision.

    Returns:
        (field, value) rows.
    """
    decision = result.decision
    return (
        ("Company", decision.company),
        ("Role", decision.role),
        ("Location", decision.location),
        ("Work mode", decision.work_mode),
        ("Seniority", decision.seniority),
        ("English risk", "yes" if decision.english_risk else "no"),
        ("Matching skills", _join(decision.matching_skills)),
        ("Missing skills", _join(decision.missing_skills)),
    )


def _usage_rows(
    result: DecisionResult, choice: ModelChoice, version: Version
) -> tuple[tuple[str, str], ...]:
    """Build the table of what the decision cost and how the rules affected it.

    Args:
        result: Result of a decision.
        choice: Model configuration that produced it.
        version: Version that produced it.

    Returns:
        (measure, value) rows.
    """
    cost_usd = call_cost_usd(
        choice.provider, choice.model, result.input_tokens, result.output_tokens
    )
    return (
        ("Model", choice.label),
        ("Version", version),
        ("Model's decision before rules", result.llm_decision),
        ("Hard rules fired", _join(result.fired_rules)),
        ("Input tokens", str(result.input_tokens)),
        ("Output tokens", str(result.output_tokens)),
        ("Cost (list price)", format_cost(cost_usd)),
        ("Latency", f"{result.latency_s:.1f} s"),
        ("Model calls", str(result.attempts)),
        ("Tool calls", str(len(result.tool_calls))),
    )


def _tool_call_rows(result: DecisionResult) -> tuple[tuple[str, str, str], ...]:
    """Build the table of the agent's tool calls, in order.

    Args:
        result: Result of a decision.

    Returns:
        (step, tool, arguments) rows; arguments as JSON text.
    """
    return tuple(
        (str(step), call.name, json.dumps(call.arguments, ensure_ascii=False))
        for step, call in enumerate(result.tool_calls, start=1)
    )


def run_decision(
    posting_text: str,
    choice_label: str,
    version: str,
    *,
    load_cv: CvLoader,
    decide_fns: Mapping[Version, DecideFn] = DECIDE_FUNCTIONS,
) -> DecisionView:
    """Decide one posting and describe the outcome for the page.

    Handled failures (empty posting, unreadable CV, failed model call, no valid
    decision) become a view with a message instead of an exception, so the browser
    shows one readable line and no traceback.

    Args:
        posting_text: Text pasted into the page.
        choice_label: Dropdown label of the chosen model configuration.
        version: "workflow" or "agent".
        load_cv: Returns the CV text; called once per request.
        decide_fns: Decision function per version. Tests pass fakes.

    Returns:
        What the page displays.
    """
    if not posting_text.strip():
        return DecisionView(decision="", message=EMPTY_POSTING_MESSAGE)
    try:
        choice = find_model_choice(choice_label)
        if version not in decide_fns:
            raise ValueError(f"Unknown version: '{version}'.")
        result = decide_fns[version](
            posting_text,
            load_cv(),
            model=choice.model,
            provider=choice.provider,
            reasoning=choice.reasoning,
        )
    except (OSError, ValueError, LLMError, DecisionError) as err:
        return DecisionView(decision="", message=f"No decision: {err}")
    return DecisionView(
        decision=result.decision.decision,
        message=RECOMMENDATION_MESSAGE,
        rationale=result.decision.rationale,
        facts=_fact_rows(result),
        usage=_usage_rows(result, choice, version),
        tool_calls=_tool_call_rows(result),
    )


def view_outputs(view: DecisionView) -> tuple[Any, ...]:
    """Turn a view into the values of the page's output components, in order.

    Args:
        view: What to display.

    Returns:
        Decision label (None clears it), message, rationale and the three tables.
    """
    return (
        view.decision or None,
        view.message,
        view.rationale,
        [list(row) for row in view.facts],
        [list(row) for row in view.usage],
        [list(row) for row in view.tool_calls],
    )


def build_app(
    *,
    cv_path: Path = DEFAULT_CV_PATH,
    decide_fns: Mapping[Version, DecideFn] = DECIDE_FUNCTIONS,
) -> gr.Blocks:
    """Build the page: three inputs, one button, the decision and its cost.

    Text that comes from the model is shown in plain-text components, never as
    Markdown or HTML: a posting could talk the model into emitting a link or an image
    that the browser would then load.

    Args:
        cv_path: CV file, read again on every request.
        decide_fns: Decision function per version.

    Returns:
        The Gradio app, not yet launched.
    """

    def on_decide(posting_text: str, choice_label: str, version: str) -> tuple[Any, ...]:
        """Handle a click on "Decide".

        Args:
            posting_text: Text pasted into the page.
            choice_label: Chosen model configuration.
            version: Chosen version.

        Returns:
            Values for the output components.
        """
        view = run_decision(
            posting_text,
            choice_label,
            version,
            load_cv=lambda: cv_path.read_text(encoding="utf-8"),
            decide_fns=decide_fns,
        )
        return view_outputs(view)

    with gr.Blocks(analytics_enabled=False, title=APP_TITLE) as app:
        gr.Markdown(INTRO_TEXT.format(cv_path=cv_path.as_posix()))
        posting = gr.Textbox(label="Job posting", lines=POSTING_LINES, max_lines=POSTING_LINES)
        with gr.Row():
            choice = gr.Dropdown(
                choices=[item.label for item in MODEL_CHOICES],
                value=DEFAULT_MODEL_CHOICE.label,
                label="Model",
            )
            version = gr.Radio(
                choices=list(VERSIONS), value=DEFAULT_VERSION, label="Version", info=VERSION_INFO
            )
        button = gr.Button("Decide", variant="primary")
        decision = gr.Label(label="Recommendation")
        message = gr.Textbox(label="Status", interactive=False)
        rationale = gr.Textbox(label="Rationale", lines=RATIONALE_LINES, interactive=False)
        facts = gr.Dataframe(
            headers=list(FACT_HEADERS), label="Posting facts", interactive=False, wrap=True
        )
        usage = gr.Dataframe(
            headers=list(USAGE_HEADERS), label="Cost and rules", interactive=False, wrap=True
        )
        tool_calls = gr.Dataframe(
            headers=list(TOOL_CALL_HEADERS),
            label="Agent tool calls",
            interactive=False,
            wrap=True,
        )
        button.click(
            on_decide,
            inputs=[posting, choice, version],
            outputs=[decision, message, rationale, facts, usage, tool_calls],
            api_name="decide",
        )
    return app


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Arguments without the program name; None reads ``sys.argv``.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Open the Job Fit Assistant page locally.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Local port to listen on.")
    parser.add_argument("--cv", type=Path, default=DEFAULT_CV_PATH, help="Path to the CV.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Serve the page on this machine until interrupted.

    Args:
        argv: Arguments without the program name; None reads ``sys.argv``.

    Returns:
        Process exit code: 0 after a normal stop, 1 if the CV file is missing.
    """
    args = _parse_args(argv)
    if not args.cv.is_file():
        print(f"error: CV file not found: {args.cv}", file=sys.stderr)
        return 1
    # Gradio reports usage to its maker unless told otherwise; this tool handles a CV.
    os.environ[ANALYTICS_VARIABLE] = "False"
    app = build_app(cv_path=args.cv)
    # Never a share link, a remote-callable MCP endpoint or an installable web app.
    app.launch(
        server_name=LOCAL_HOST,
        server_port=args.port,
        share=False,
        ssr_mode=False,
        mcp_server=False,
        pwa=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
