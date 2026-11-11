"""Tests for the LLM client. No network: provider clients are replaced by fakes."""

from types import SimpleNamespace
from typing import Any

import httpx
import openai
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


@pytest.mark.parametrize(
    "transport_error",
    [httpx.ReadTimeout("timed out"), httpx.ConnectError("no route to host")],
)
def test_timeout_and_connection_errors_are_wrapped_in_llm_error(
    transport_error: httpx.HTTPError,
) -> None:
    """A timeout fails one posting through LLMError instead of crashing the eval run.

    The SDK re-raises httpx errors as they are once its retries are used up; they are
    not APIError.
    """
    client = _fake_client(error=transport_error)

    with pytest.raises(LLMError, match=MODEL) as raised:
        _call_vertex(client)

    assert raised.value.__cause__ is transport_error


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


class FakeCompletions:
    """Stand-in for ``client.chat.completions`` that records the call."""

    def __init__(self, reply: Any = None, error: Exception | None = None) -> None:
        """Store the reply to return or the error to raise."""
        self.reply = reply
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        """Record the keyword arguments, then return the reply or raise the error."""
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.reply


def _fake_chat_client(reply: Any = None, error: Exception | None = None) -> Any:
    """Build an object shaped like an OpenAI client around FakeCompletions."""
    completions = FakeCompletions(reply=reply, error=error)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def _chat_reply(text: str | None, prompt: int = 0, completion: int = 0) -> Any:
    """Build an object shaped like a chat completion with one choice."""
    usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)
    choice = SimpleNamespace(message=SimpleNamespace(content=text))
    return SimpleNamespace(choices=[choice], usage=usage)


def _call_chat(client: Any, extra_body: dict[str, Any] | None = None) -> LLMResponse:
    """Call the OpenAI-compatible code path with fixed prompts and the given client."""
    return llm._generate_openai_compatible(
        client, MODEL, "system prompt", "user prompt", JobDecision, 30.0, extra_body or {}
    )


@pytest.fixture
def pinned_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the fake model to a fake OpenRouter provider."""
    monkeypatch.setattr(llm, "OPENROUTER_PINNED_PROVIDERS", {MODEL: "FakeHost"})


def test_chat_response_is_mapped_to_llm_response() -> None:
    """Text and token counts of a chat completion are passed through."""
    client = _fake_chat_client(reply=_chat_reply('{"a": 1}', prompt=120, completion=55))

    result = _call_chat(client)

    assert result.text == '{"a": 1}'
    assert (result.input_tokens, result.output_tokens) == (120, 55)
    assert result.latency_s >= 0


def test_chat_request_asks_for_schema_constrained_json() -> None:
    """The request carries both prompts, a strict JSON schema, a timeout and the extras."""
    client = _fake_chat_client(reply=_chat_reply("{}"))

    _call_chat(client, extra_body={"provider": {"order": ["FakeHost"]}})

    call = client.chat.completions.calls[0]
    assert call["model"] == MODEL
    assert call["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user prompt"},
    ]
    assert call["temperature"] == 0.0
    assert call["timeout"] == 30.0
    assert call["extra_body"] == {"provider": {"order": ["FakeHost"]}}
    json_schema = call["response_format"]["json_schema"]
    assert call["response_format"]["type"] == "json_schema"
    assert json_schema["strict"] is True
    assert json_schema["schema"]["additionalProperties"] is False
    assert next(iter(json_schema["schema"]["properties"])) == "rationale"


def test_chat_response_without_choices_raises_llm_error() -> None:
    """An error body that arrives with HTTP 200 and no choices is an error."""
    client = _fake_chat_client(reply=SimpleNamespace(choices=None, usage=None))

    with pytest.raises(LLMError, match="No text"):
        _call_chat(client)


def test_chat_response_with_empty_text_raises_llm_error() -> None:
    """A choice without content is an error, not an empty string."""
    client = _fake_chat_client(reply=_chat_reply(None))

    with pytest.raises(LLMError, match="No text"):
        _call_chat(client)


def test_chat_sdk_error_is_wrapped_in_llm_error() -> None:
    """OpenAI SDK errors surface as LLMError and keep the original as the cause."""
    sdk_error = openai.OpenAIError("rate limited")
    client = _fake_chat_client(error=sdk_error)

    with pytest.raises(LLMError, match=MODEL) as raised:
        _call_chat(client)

    assert raised.value.__cause__ is sdk_error


@pytest.mark.usefixtures("pinned_model")
def test_openrouter_request_pins_the_provider_without_fallbacks() -> None:
    """Only the pinned provider may answer, and it must honor the JSON schema."""
    body = llm._extra_body("openrouter", MODEL, None)

    assert body == {
        "provider": {"order": ["FakeHost"], "allow_fallbacks": False, "require_parameters": True}
    }


@pytest.mark.usefixtures("pinned_model")
@pytest.mark.parametrize("reasoning", [True, False])
def test_openrouter_reasoning_switch_is_sent_in_the_body(reasoning: bool) -> None:
    """Reasoning on and off are passed in OpenRouter's reasoning field."""
    body = llm._extra_body("openrouter", MODEL, reasoning)

    assert body["reasoning"] == {"enabled": reasoning}


def test_openrouter_model_without_a_pinned_provider_is_rejected() -> None:
    """An unpinned model fails before any request, instead of using any provider."""
    with pytest.raises(LLMError, match="No pinned provider"):
        llm._extra_body("openrouter", "vendor/unpinned-model", None)


def test_ollama_sends_no_extra_fields() -> None:
    """A local model needs no routing fields."""
    assert llm._extra_body("ollama", MODEL, None) == {}


def test_ollama_rejects_the_reasoning_switch() -> None:
    """The reasoning switch is not silently ignored for local models."""
    with pytest.raises(LLMError, match="not supported"):
        llm._extra_body("ollama", MODEL, True)


def test_missing_openrouter_key_raises_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Creating the OpenRouter client without a key fails with a clear message."""
    monkeypatch.setattr(llm, "load_dotenv", lambda: None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    llm._openai_compatible_client.cache_clear()

    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        llm._openai_compatible_client("openrouter")

    llm._openai_compatible_client.cache_clear()


def test_ollama_client_uses_the_configured_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """The local endpoint comes from OLLAMA_BASE_URL."""
    monkeypatch.setattr(llm, "load_dotenv", lambda: None)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:9999/v1")
    llm._openai_compatible_client.cache_clear()

    client = llm._openai_compatible_client("ollama")

    assert str(client.base_url).rstrip("/") == "http://localhost:9999/v1"
    llm._openai_compatible_client.cache_clear()


def test_openrouter_client_uses_the_openrouter_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """The OpenRouter client points at OpenRouter and is built from the key in the environment."""
    monkeypatch.setattr(llm, "load_dotenv", lambda: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    llm._openai_compatible_client.cache_clear()

    client = llm._openai_compatible_client("openrouter")

    assert str(client.base_url).rstrip("/") == llm.OPENROUTER_BASE_URL
    llm._openai_compatible_client.cache_clear()


@pytest.mark.usefixtures("pinned_model")
@pytest.mark.parametrize("provider", ["openrouter", "ollama"])
def test_generate_json_routes_chat_providers_to_the_chat_client(
    provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public entry point uses the OpenAI-compatible client for both providers."""
    client = _fake_chat_client(reply=_chat_reply('{"ok": true}', 5, 3))
    requested: list[str] = []

    def fake_factory(name: str) -> Any:
        """Record which provider's client was asked for and return the fake."""
        requested.append(name)
        return client

    monkeypatch.setattr(llm, "_openai_compatible_client", fake_factory)

    result = generate_json(provider, MODEL, "system prompt", "user prompt", JobDecision)

    assert result.text == '{"ok": true}'
    assert requested == [provider]


def test_vertex_location_falls_back_to_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without GOOGLE_CLOUD_LOCATION the default location is reported."""
    monkeypatch.setattr(llm, "load_dotenv", lambda: None)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)

    assert llm.vertex_location() == llm.DEFAULT_VERTEX_LOCATION


def test_unknown_provider_raises_value_error() -> None:
    """A provider name outside the supported set is rejected."""
    with pytest.raises(ValueError, match="Unknown provider"):
        generate_json("bedrock", MODEL, "system", "user", JobDecision)


@pytest.fixture
def recorded_chat_model(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the LangChain Gemini class and the environment; return what it is built with."""
    recorded: dict[str, Any] = {}
    monkeypatch.setattr(llm, "ChatGoogleGenerativeAI", lambda **kwargs: recorded.update(kwargs))
    monkeypatch.setattr(llm, "load_dotenv", lambda: None)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "test-location")
    return recorded


def test_chat_model_is_configured_like_the_workflow_calls(
    recorded_chat_model: dict[str, Any],
) -> None:
    """The agent's model uses the same project, location, temperature and retries."""
    llm.chat_model("vertex", MODEL, timeout_s=30.0)

    assert recorded_chat_model == {
        "model": MODEL,
        "vertexai": True,
        "project": "test-project",
        "location": "test-location",
        "temperature": llm.TEMPERATURE,
        "timeout": 30.0,
        "max_retries": llm.MAX_HTTP_ATTEMPTS - 1,
    }


@pytest.mark.parametrize(
    ("reasoning", "expected_budget"),
    [(True, llm.THINKING_AUTOMATIC), (False, llm.THINKING_DISABLED)],
)
def test_chat_model_reasoning_switch_sets_the_thinking_budget(
    recorded_chat_model: dict[str, Any], reasoning: bool, expected_budget: int
) -> None:
    """Reasoning on and off map to the same thinking budgets as in the workflow."""
    llm.chat_model("vertex", MODEL, reasoning=reasoning)

    assert recorded_chat_model["thinking_budget"] == expected_budget


@pytest.mark.parametrize("provider", ["openrouter", "ollama", "bedrock"])
def test_chat_model_supports_only_vertex(provider: str) -> None:
    """The agent cannot run on a provider other than Vertex."""
    with pytest.raises(LLMError, match="supports only the vertex provider"):
        llm.chat_model(provider, MODEL)


def test_chat_model_without_a_project_raises_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing project id is reported before any model is created."""
    monkeypatch.setattr(llm, "load_dotenv", lambda: None)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

    with pytest.raises(LLMError, match="GOOGLE_CLOUD_PROJECT"):
        llm.chat_model("vertex", MODEL)
