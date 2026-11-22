"""Tests for tracing: what is recorded, how it is tagged, and that the CV never is.

Traces are exported to memory. No test sends anything to Langfuse.
"""

from collections.abc import Iterator
from typing import Any

import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from test_agent import CV as AGENT_CV
from test_agent import POSTING as AGENT_POSTING
from test_agent import READ_SKILLS, FakeFactory, _answer

from job_fit import llm, tracing
from job_fit.agent import decide_with_agent
from job_fit.llm import LLMResponse, generate_json
from job_fit.pricing import call_cost_usd
from job_fit.schema import JobDecision

CV = """\
# Jane Example

## Experience
Seven years of React Native at Example Bank, SECRET-CV-LINE.

## Skills
React Native, TypeScript, SECRET-SKILL-LINE
"""
POSTING = "Senior React Native developer wanted. POSTING-MARKER"
MODEL = "gemini-2.5-flash"
PROVIDER = "vertex"


@pytest.fixture(scope="module")
def memory_client() -> Iterator[tuple[Langfuse, InMemorySpanExporter]]:
    """Create one Langfuse client for the module that keeps its spans in memory.

    Yields:
        The client and the exporter holding what it would have sent.
    """
    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key="pk-lf-test",
        secret_key="sk-lf-test",
        base_url="http://127.0.0.1:9",
        mask=tracing._mask,
        span_exporter=exporter,
    )
    yield client, exporter
    exporter.clear()


@pytest.fixture
def exported(
    memory_client: tuple[Langfuse, InMemorySpanExporter], monkeypatch: pytest.MonkeyPatch
) -> InMemorySpanExporter:
    """Switch tracing on for one test, exporting to memory.

    Args:
        memory_client: The module's client and exporter.
        monkeypatch: Pytest's patching helper.

    Returns:
        The exporter, emptied before the test.
    """
    client, exporter = memory_client
    exporter.clear()
    monkeypatch.setattr(tracing, "_client", lambda: client)
    return exporter


def _spans(exporter: InMemorySpanExporter) -> list[Any]:
    """Flush the client and return the spans it produced.

    Args:
        exporter: The in-memory exporter.

    Returns:
        The finished spans.
    """
    tracing.flush()
    return list(exporter.get_finished_spans())


def _recorded_text(spans: list[Any]) -> str:
    """Join everything the spans carry into one text to search.

    Args:
        spans: Finished spans.

    Returns:
        All attribute values, as text.
    """
    return "\n".join(str(value) for span in spans for value in span.attributes.values())


def _attribute(span: Any, suffix: str) -> Any:
    """Read one attribute of a span by the end of its key.

    Args:
        span: A finished span.
        suffix: End of the attribute key, e.g. "observation.output".

    Returns:
        The attribute value.
    """
    return next(value for key, value in span.attributes.items() if key.endswith(suffix))


def _fake_vertex(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    """Replace the Vertex call by a canned response.

    Args:
        monkeypatch: Pytest's patching helper.
        text: Response text to return.
    """
    monkeypatch.setattr(llm, "_vertex_client", lambda: None)
    monkeypatch.setattr(
        llm,
        "_generate_vertex",
        lambda *args: LLMResponse(text=text, input_tokens=1200, output_tokens=300, latency_s=1.5),
    )


def test_cv_fragments_start_with_the_whole_cv_then_its_long_lines() -> None:
    """The whole CV is looked for first, then each line that is long enough to matter."""
    fragments = tracing.cv_fragments(CV)

    assert fragments[0] == CV.strip()
    assert "Seven years of React Native at Example Bank, SECRET-CV-LINE." in fragments
    assert "## Skills" in fragments
    assert all(len(fragment) >= tracing.MIN_MASKED_LINE_CHARS for fragment in fragments)
    assert list(fragments[1:]) == sorted(fragments[1:], key=lambda line: (-len(line), line))


def test_mask_data_replaces_cv_text_at_any_depth() -> None:
    """Strings inside lists and dicts are masked; other values pass through."""
    fragments = tracing.cv_fragments(CV)
    marker = tracing.cv_marker(CV)
    data = {
        "prompt": f"<posting>{POSTING}</posting><cv>{CV.strip()}</cv>",
        "messages": [{"content": "React Native, TypeScript, SECRET-SKILL-LINE"}, 7, None],
    }

    masked = tracing.mask_data(data, fragments, marker)

    assert masked["prompt"] == f"<posting>{POSTING}</posting><cv>{marker}</cv>"
    assert masked["messages"] == [{"content": marker}, 7, None]
    assert data["messages"][0]["content"].endswith("SECRET-SKILL-LINE")


def test_cv_marker_identifies_the_cv_without_showing_it() -> None:
    """The marker is the same for the same CV and differs for another one."""
    assert tracing.cv_marker(CV) == tracing.cv_marker(f"  {CV}\n")
    assert tracing.cv_marker(CV) != tracing.cv_marker(CV + "one more line")
    assert "SECRET" not in tracing.cv_marker(CV)


def test_mask_without_a_scope_records_nothing_of_the_value() -> None:
    """A value that reaches the mask outside a scope is replaced wholesale."""
    assert tracing._mask(data=f"anything, even {CV}") == tracing.NO_SCOPE_TEXT


def test_tracing_is_off_without_keys() -> None:
    """With no client configured the scope and the recorder do nothing."""
    assert tracing.is_enabled() is False
    assert tracing.langchain_callbacks() == []
    with (
        tracing.trace_scope(cv=CV, experiment="e", version="workflow"),
        tracing.traced_generation(
            provider=PROVIDER, model=MODEL, system="s", user="u", reasoning=None
        ) as recorder,
    ):
        recorder.finish("out", 1, 1)

    assert recorder.observation is None


def test_generation_is_recorded_with_prompt_output_tokens_cost_and_tags(
    exported: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model call inside a scope is traced with everything PROCESS.md asks for."""
    _fake_vertex(monkeypatch, '{"decision": "APPLY"}')

    with tracing.trace_scope(
        cv=CV, experiment="week2-reference", version="workflow", run_index=2, posting_id="p-7"
    ):
        generate_json(PROVIDER, MODEL, "SYSTEM-PROMPT", f"{POSTING}\n{CV}", JobDecision)

    spans = _spans(exported)
    generation = next(span for span in spans if span.name == tracing.GENERATION_NAME)
    cost = call_cost_usd(PROVIDER, MODEL, 1200, 300)
    assert "SYSTEM-PROMPT" in _attribute(generation, "observation.input")
    assert "POSTING-MARKER" in _attribute(generation, "observation.input")
    assert _attribute(generation, "observation.output") == '{"decision": "APPLY"}'
    assert _attribute(generation, "model.name") == MODEL
    assert '"input": 1200' in _attribute(generation, "usage_details")
    assert '"output": 300' in _attribute(generation, "usage_details")
    assert f'"total": {cost}' in _attribute(generation, "cost_details")
    assert set(_attribute(generation, "trace.tags")) == {
        "experiment:week2-reference",
        "version:workflow",
        "run:2",
    }
    # The fake call returns at once and the Windows clock is coarse, so equal is allowed.
    assert generation.end_time >= generation.start_time


def test_traced_prompt_holds_a_marker_instead_of_the_cv(
    exported: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No line of the CV reaches the trace, in the prompt or in the output."""
    _fake_vertex(monkeypatch, "The CV says: React Native, TypeScript, SECRET-SKILL-LINE")

    with tracing.trace_scope(cv=CV, experiment="e", version="workflow"):
        generate_json(PROVIDER, MODEL, "system", f"{POSTING}\n<cv>{CV}</cv>", JobDecision)

    recorded = _recorded_text(_spans(exported))
    assert "SECRET-CV-LINE" not in recorded
    assert "SECRET-SKILL-LINE" not in recorded
    assert tracing.cv_marker(CV) in recorded
    assert "POSTING-MARKER" in recorded


def test_model_call_outside_a_scope_is_not_traced(
    exported: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a scope there is no CV to mask with, so nothing is recorded."""
    _fake_vertex(monkeypatch, "{}")

    generate_json(PROVIDER, MODEL, "system", f"{POSTING}\n{CV}", JobDecision)

    assert _spans(exported) == []
    assert tracing.langchain_callbacks() == []


def test_scope_ends_with_the_block(exported: InMemorySpanExporter) -> None:
    """After the block the next call is outside any scope again, even after an error."""
    with pytest.raises(RuntimeError), tracing.trace_scope(cv=CV, experiment="e", version="v"):
        raise RuntimeError("decision failed")

    assert tracing._current_scope.get() is None


def test_agent_trace_never_contains_cv_text(exported: InMemorySpanExporter) -> None:
    """The agent reads CV sections through a tool; their text is masked in its trace."""
    factory = FakeFactory(READ_SKILLS, _answer())

    with tracing.trace_scope(cv=AGENT_CV, experiment="e", version="agent", posting_id="p-1"):
        decide_with_agent(AGENT_POSTING, AGENT_CV, model=MODEL, chat_model_factory=factory)

    spans = _spans(exported)
    recorded = _recorded_text(spans)
    long_lines = [
        line.strip()
        for line in AGENT_CV.splitlines()
        if len(line.strip()) >= tracing.MIN_MASKED_LINE_CHARS
    ]
    assert len(spans) > 1
    assert long_lines
    assert not [line for line in long_lines if line in recorded]
    assert "POSTING-MARKER" in recorded
    assert tracing.cv_marker(AGENT_CV) in recorded
