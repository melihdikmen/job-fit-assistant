"""Single entry point for LLM calls.

This is the only module that imports a provider SDK or reads ``.env``. The rest of the
code calls ``generate_json`` and does not care which provider answers.
"""

import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

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
PROVIDERS_PLANNED_FOR_PHASE_3 = frozenset({"openrouter", "ollama"})


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
    location = os.environ.get("GOOGLE_CLOUD_LOCATION") or DEFAULT_VERTEX_LOCATION
    return genai.Client(vertexai=True, project=project, location=location)


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
        LLMError: If the provider is not implemented yet or the call fails.
    """
    if provider == "vertex":
        return _generate_vertex(_vertex_client(), model, system, user, schema, reasoning, timeout_s)
    if provider in PROVIDERS_PLANNED_FOR_PHASE_3:
        raise LLMError(f"Provider '{provider}' is not implemented until Phase 3.")
    raise ValueError(f"Unknown provider '{provider}'. Use 'vertex', 'openrouter' or 'ollama'.")
