"""Tests for the LLM client. No network: the Vertex client is replaced by a fake."""

from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import errors as genai_errors

from job_fit import llm
from job_fit.llm import LLMError, LLMResponse, generate_json
from job_fit.schema import JobDecision

MODEL = "fake-model"


class FakeModels:
    """Stand-in for ``client.models`` that records the call and returns a canned reply."""

    def __init__(self, reply: Any = None, error: Exception | None = None) -> None:
        """Store the reply to return or the error to raise."""
        self.reply = reply
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        """Record the keyword arguments, then return the reply or raise the error."""
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.reply


def _fake_client(reply: Any = None, error: Exception | None = None) -> Any:
    """Build an object shaped like a genai client around FakeModels."""
    return SimpleNamespace(models=FakeModels(reply=reply, error=error))


def _reply(
    text: str | None, prompt: int | None, candidates: int | None, thoughts: int | None
) -> Any:
    """Build an object shaped like a Vertex response."""
    usage = SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        thoughts_token_count=thoughts,
    )
    return SimpleNamespace(text=text, usage_metadata=usage)


def _call_vertex(client: Any, reasoning: bool | None = None) -> LLMResponse:
    """Call the Vertex code path with fixed prompts and the given client."""
    return llm._generate_vertex(
        client, MODEL, "system prompt", "user prompt", JobDecision, reasoning, timeout_s=30.0
    )


def test_vertex_response_is_mapped_to_llm_response() -> None:
    """Text is passed through and thinking tokens are counted as output."""
    client = _fake_client(reply=_reply('{"a": 1}', prompt=120, candidates=40, thoughts=15))

    result = _call_vertex(client)

    assert result.text == '{"a": 1}'
    assert result.input_tokens == 120
    assert result.output_tokens == 55
    assert result.latency_s >= 0


def test_missing_token_counts_become_zero() -> None:
    """Token counts the API leaves empty are reported as zero, not None."""
    client = _fake_client(reply=_reply("{}", prompt=None, candidates=None, thoughts=None))

    result = _call_vertex(client)

    assert (result.input_tokens, result.output_tokens) == (0, 0)


def test_vertex_request_asks_for_schema_constrained_json() -> None:
    """The request carries the prompts, the schema, JSON mode and a timeout."""
    client = _fake_client(reply=_reply("{}", 1, 1, 0))

    _call_vertex(client)

    call = client.models.calls[0]
    config = call["config"]
    assert call["model"] == MODEL
    assert call["contents"] == "user prompt"
    assert config.system_instruction == "system prompt"
    assert config.response_mime_type == "application/json"
    assert config.response_schema is JobDecision
    assert config.temperature == 0.0
    assert config.http_options.timeout == 30_000
    assert config.http_options.retry_options.attempts == llm.MAX_HTTP_ATTEMPTS


@pytest.mark.parametrize(
    ("reasoning", "expected_budget"),
    [(False, llm.THINKING_DISABLED), (True, llm.THINKING_AUTOMATIC)],
)
def test_reasoning_switch_sets_thinking_budget(reasoning: bool, expected_budget: int) -> None:
    """Reasoning off and on map to the disabled and automatic thinking budgets."""
    client = _fake_client(reply=_reply("{}", 1, 1, 0))

    _call_vertex(client, reasoning=reasoning)

    config = client.models.calls[0]["config"]
    assert config.thinking_config.thinking_budget == expected_budget


def test_reasoning_default_leaves_thinking_config_unset() -> None:
    """Without a reasoning choice the model's default thinking behavior is kept."""
    client = _fake_client(reply=_reply("{}", 1, 1, 0))

    _call_vertex(client)

    assert client.models.calls[0]["config"].thinking_config is None


def test_empty_response_text_raises_llm_error() -> None:
    """A response without text is an error, not an empty string."""
    client = _fake_client(reply=_reply(None, 10, 0, 0))

    with pytest.raises(LLMError, match="no text"):
        _call_vertex(client)


def test_api_error_is_wrapped_in_llm_error() -> None:
    """SDK errors surface as LLMError and keep the original as the cause."""
    api_error = genai_errors.APIError(503, {"error": {"message": "boom", "status": "UNAVAILABLE"}})
    client = _fake_client(error=api_error)

    with pytest.raises(LLMError, match=MODEL) as raised:
        _call_vertex(client)

    assert raised.value.__cause__ is api_error


def test_missing_project_raises_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Creating the Vertex client without a project id fails with a clear message."""
    monkeypatch.setattr(llm, "load_dotenv", lambda: None)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    llm._vertex_client.cache_clear()

    with pytest.raises(LLMError, match="GOOGLE_CLOUD_PROJECT"):
        llm._vertex_client()

    llm._vertex_client.cache_clear()


def test_generate_json_routes_vertex_to_the_vertex_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public entry point uses the Vertex client for the vertex provider."""
    client = _fake_client(reply=_reply('{"ok": true}', 5, 3, 0))
    monkeypatch.setattr(llm, "_vertex_client", lambda: client)

    result = generate_json("vertex", MODEL, "system prompt", "user prompt", JobDecision)

    assert result.text == '{"ok": true}'
    assert len(client.models.calls) == 1


@pytest.mark.parametrize("provider", ["openrouter", "ollama"])
def test_providers_planned_for_phase_3_raise_llm_error(provider: str) -> None:
    """Providers that are not implemented yet fail loudly instead of silently."""
    with pytest.raises(LLMError, match="Phase 3"):
        generate_json(provider, MODEL, "system", "user", JobDecision)


def test_unknown_provider_raises_value_error() -> None:
    """A provider name outside the supported set is rejected."""
    with pytest.raises(ValueError, match="Unknown provider"):
        generate_json("bedrock", MODEL, "system", "user", JobDecision)
