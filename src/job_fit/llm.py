"""Single entry point for LLM calls.

This is the only module that imports a provider SDK or reads ``.env``. The rest of the
code calls ``generate_json`` and does not care which provider answers.
"""

import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import openai
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

Provider = Literal["vertex", "openrouter", "ollama"]

DEFAULT_TIMEOUT_S = 60.0
DEFAULT_VERTEX_LOCATION = "us-central1"
MAX_HTTP_ATTEMPTS = 3
JSON_MIME_TYPE = "application/json"
TEMPERATURE = 0.0
THINKING_DISABLED = 0
THINKING_AUTOMATIC = -1

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
# Ollama ignores the key, but the OpenAI SDK refuses to start without one.
OLLAMA_PLACEHOLDER_KEY = "ollama"
OPENAI_COMPATIBLE_PROVIDERS = frozenset({"openrouter", "ollama"})

# OpenRouter can serve one model through several providers that differ in price,
# quantization and speed. Each model used here is pinned to one provider so that a run
# measures one thing. Key: OpenRouter model id. Value: OpenRouter provider name.
OPENROUTER_PINNED_PROVIDERS: dict[str, str] = {
    "nvidia/nemotron-3.5-super-vl:free": "nvidia",
    "liquid/lfm-3-1.7b:free": "liquid",
}


class LLMError(RuntimeError):
    """An LLM call could not be made or returned nothing usable."""


@dataclass(frozen=True)
class LLMResponse:
    """Raw result of one LLM call.

    Attributes:
        text: Response body, expected to be a JSON document. Not validated here.
        input_tokens: Tokens in the prompt.
        output_tokens: Tokens generated, including thinking tokens.
        latency_s: Wall-clock seconds the call took, including HTTP retries.
    """

    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float


def vertex_location() -> str:
    """Return the Vertex AI location calls are sent to, for recording in run files.

    Returns:
        The value of ``GOOGLE_CLOUD_LOCATION``, or the default location.
    """
    load_dotenv()
    return os.environ.get("GOOGLE_CLOUD_LOCATION") or DEFAULT_VERTEX_LOCATION


@lru_cache(maxsize=1)
def _vertex_client() -> genai.Client:
    """Create the Vertex AI client from environment variables.

    Returns:
        A client bound to the configured Google Cloud project and location.

    Raises:
        LLMError: If ``GOOGLE_CLOUD_PROJECT`` is not set.
    """
    load_dotenv()
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise LLMError("GOOGLE_CLOUD_PROJECT is not set. Copy .env.example to .env and fill it in.")
    return genai.Client(vertexai=True, project=project, location=vertex_location())


def _thinking_config(reasoning: bool | None) -> types.ThinkingConfig | None:
    """Translate the reasoning switch into a Gemini thinking configuration.

    Args:
        reasoning: True to let the model think, False to disable thinking, None to
            keep the model's default.

    Returns:
        The thinking configuration, or None for the model default.
    """
    if reasoning is None:
        return None
    budget = THINKING_AUTOMATIC if reasoning else THINKING_DISABLED
    return types.ThinkingConfig(thinking_budget=budget)


def _generate_vertex(
    client: genai.Client,
    model: str,
    system: str,
    user: str,
    schema: type[BaseModel],
    reasoning: bool | None,
    timeout_s: float,
) -> LLMResponse:
    """Call a Gemini model on Vertex AI and ask for JSON matching a schema.

    Args:
        client: Vertex AI client.
        model: Model id, e.g. a Gemini Flash model.
        system: System prompt.
        user: User prompt.
        schema: Pydantic model the response must follow.
        reasoning: See ``_thinking_config``.
        timeout_s: Timeout for one HTTP request, in seconds.

    Returns:
        The response text with token counts and latency.

    Raises:
        LLMError: If the API call fails or the response has no text.
    """
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=TEMPERATURE,
        response_mime_type=JSON_MIME_TYPE,
        response_schema=schema,
        thinking_config=_thinking_config(reasoning),
        http_options=types.HttpOptions(
            timeout=int(timeout_s * 1000),
            retry_options=types.HttpRetryOptions(attempts=MAX_HTTP_ATTEMPTS),
        ),
    )
    started = time.perf_counter()
    try:
        response = client.models.generate_content(model=model, contents=user, config=config)
    except genai_errors.APIError as err:
        raise LLMError(f"Vertex call failed for model '{model}': {err}") from err
    latency_s = time.perf_counter() - started

    if not response.text:
        raise LLMError(f"Vertex returned no text for model '{model}'.")
    usage = response.usage_metadata
    input_tokens = (usage.prompt_token_count or 0) if usage else 0
    output_tokens = (
        (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0) if usage else 0
    )
    return LLMResponse(
        text=response.text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_s=latency_s,
    )


@lru_cache(maxsize=2)
def _openai_compatible_client(provider: str) -> openai.OpenAI:
    """Create the client for a provider that speaks the OpenAI chat API.

    Args:
        provider: "openrouter" or "ollama".

    Returns:
        A client pointed at the provider's base URL.

    Raises:
        LLMError: If the provider needs an API key and none is set.
    """
    load_dotenv()
    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise LLMError("OPENROUTER_API_KEY is not set. Add it to .env.")
        base_url = OPENROUTER_BASE_URL
    else:
        api_key = OLLAMA_PLACEHOLDER_KEY
        base_url = os.environ.get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL
    return openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=MAX_HTTP_ATTEMPTS - 1)


def _json_schema_response_format(schema: type[BaseModel]) -> dict[str, Any]:
    """Describe the expected JSON in the OpenAI ``response_format`` shape.

    Args:
        schema: Pydantic model the response must follow. Only its top level is closed
            against extra keys, which is enough for a flat model such as JobDecision.

    Returns:
        The ``response_format`` value for a chat completion request.
    """
    json_schema = schema.model_json_schema()
    json_schema["additionalProperties"] = False
    return {
        "type": "json_schema",
        "json_schema": {"name": schema.__name__, "strict": True, "schema": json_schema},
    }


def _openrouter_extra_body(model: str, reasoning: bool | None) -> dict[str, Any]:
    """Build the OpenRouter-only request fields: provider pinning and reasoning.

    Args:
        model: OpenRouter model id.
        reasoning: True to enable reasoning, False to disable it, None for the default.

    Returns:
        Fields to merge into the request body.

    Raises:
        LLMError: If the model has no pinned provider.
    """
    pinned = OPENROUTER_PINNED_PROVIDERS.get(model)
    if pinned is None:
        raise LLMError(
            f"No pinned provider for OpenRouter model '{model}'. "
            "Add it to OPENROUTER_PINNED_PROVIDERS in llm.py."
        )
    body: dict[str, Any] = {
        # No fallbacks: a different provider would silently change what is measured.
        # require_parameters: fail instead of ignoring the JSON schema.
        "provider": {"order": [pinned], "allow_fallbacks": False, "require_parameters": True}
    }
    if reasoning is not None:
        body["reasoning"] = {"enabled": reasoning}
    return body


def _extra_body(provider: str, model: str, reasoning: bool | None) -> dict[str, Any]:
    """Build the provider-specific request fields of an OpenAI-compatible call.

    Args:
        provider: "openrouter" or "ollama".
        model: Model id as the provider names it.
        reasoning: True to enable reasoning, False to disable it, None for the default.

    Returns:
        Fields to merge into the request body; empty for Ollama.

    Raises:
        LLMError: If the reasoning switch is used with Ollama, or an OpenRouter model
            has no pinned provider.
    """
    if provider == "openrouter":
        return _openrouter_extra_body(model, reasoning)
    if reasoning is not None:
        raise LLMError("The reasoning switch is not supported for the ollama provider.")
    return {}


def _generate_openai_compatible(
    client: openai.OpenAI,
    model: str,
    system: str,
    user: str,
    schema: type[BaseModel],
    timeout_s: float,
    extra_body: dict[str, Any],
) -> LLMResponse:
    """Call a model through an OpenAI-compatible chat endpoint and ask for JSON.

    Args:
        client: Client pointed at OpenRouter or Ollama.
        model: Model id as the provider names it.
        system: System prompt.
        user: User prompt.
        schema: Pydantic model the response must follow.
        timeout_s: Timeout for one HTTP request, in seconds.
        extra_body: Provider-specific request fields.

    Returns:
        The response text with token counts and latency.

    Raises:
        LLMError: If the API call fails or the response has no text.
    """
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=TEMPERATURE,
            response_format=_json_schema_response_format(schema),
            timeout=timeout_s,
            extra_body=extra_body,
        )
    except openai.OpenAIError as err:
        raise LLMError(f"Call failed for model '{model}': {err}") from err
    latency_s = time.perf_counter() - started

    # OpenRouter can answer HTTP 200 with an error body and no choices.
    text = response.choices[0].message.content if response.choices else None
    if not text:
        raise LLMError(f"No text returned for model '{model}'.")
    usage = response.usage
    return LLMResponse(
        text=text,
        input_tokens=(usage.prompt_tokens or 0) if usage else 0,
        # Reasoning tokens are already included in completion_tokens.
        output_tokens=(usage.completion_tokens or 0) if usage else 0,
        latency_s=latency_s,
    )


def generate_json(
    provider: str,
    model: str,
    system: str,
    user: str,
    schema: type[BaseModel],
    *,
    reasoning: bool | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> LLMResponse:
    """Ask a model for a JSON document that follows a schema.

    The text is returned as is; validating it against the schema is the caller's job.

    Args:
        provider: "vertex", "openrouter" or "ollama".
        model: Model id as the provider names it.
        system: System prompt.
        user: User prompt.
        schema: Pydantic model describing the expected JSON.
        reasoning: True to enable model reasoning, False to disable it, None for the
            model default.
        timeout_s: Timeout for one HTTP request, in seconds.

    Returns:
        The response text with token counts and latency.

    Raises:
        ValueError: If the provider name is unknown.
        LLMError: If the provider is misconfigured or the call fails.
    """
    if provider == "vertex":
        return _generate_vertex(_vertex_client(), model, system, user, schema, reasoning, timeout_s)
    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        extra_body = _extra_body(provider, model, reasoning)
        client = _openai_compatible_client(provider)
        return _generate_openai_compatible(
            client, model, system, user, schema, timeout_s, extra_body
        )
    raise ValueError(f"Unknown provider '{provider}'. Use 'vertex', 'openrouter' or 'ollama'.")
