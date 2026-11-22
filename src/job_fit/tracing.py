"""Optional tracing of model calls to Langfuse.

Tracing is off unless ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are set in
``.env``. When it is on, every model call made inside a ``trace_scope`` is recorded
with its prompt, output, tokens, latency, cost and model, tagged with the experiment.

The CV never leaves the machine through a trace: every piece of traced text passes
through a mask that replaces the CV, and any line of it, by a short hash. A model call
made outside a ``trace_scope`` is not traced at all, because without the CV at hand
there is nothing to mask with.

This module reads only the ``LANGFUSE_*`` variables.
"""

import contextvars
import hashlib
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler

from job_fit.pricing import TOKENS_PER_MILLION, price_for

PUBLIC_KEY_VARIABLE = "LANGFUSE_PUBLIC_KEY"
SECRET_KEY_VARIABLE = "LANGFUSE_SECRET_KEY"
BASE_URL_VARIABLE = "LANGFUSE_BASE_URL"

TRACE_NAME = "decide"
GENERATION_NAME = "generate_json"
HASH_PREFIX_CHARS = 12
# CV lines shorter than this are left alone: masking a heading such as "Skills" would
# blank the same word wherever a posting uses it, and would hide nothing personal.
MIN_MASKED_LINE_CHARS = 8
NO_SCOPE_TEXT = "[masked: recorded outside a trace scope]"


@dataclass(frozen=True)
class TraceScope:
    """What one decision is tagged with and what must be masked in its trace.

    Attributes:
        experiment: Name of the experiment the call belongs to.
        version: "workflow" or "agent".
        run_index: Number of the run within a repeated configuration, starting at 1.
        posting_id: Id of the posting being decided; empty outside an eval.
        fragments: Pieces of CV text to mask, longest first.
        marker: Text that replaces a masked piece.
    """

    experiment: str
    version: str
    run_index: int
    posting_id: str
    fragments: tuple[str, ...]
    marker: str


# The scope of the decision being made right now. A context variable, not a module
# global: it is set for the duration of one ``trace_scope`` block and is invisible to
# everything outside it.
_current_scope: contextvars.ContextVar[TraceScope | None] = contextvars.ContextVar(
    "job_fit_trace_scope", default=None
)


def cv_marker(cv: str) -> str:
    """Build the text that stands in for the CV in a trace.

    Args:
        cv: Full text of the candidate's CV.

    Returns:
        A marker carrying the start of the CV's SHA-256, so two traces can be seen to
        have used the same CV without showing it.
    """
    digest = hashlib.sha256(cv.strip().encode("utf-8")).hexdigest()[:HASH_PREFIX_CHARS]
    return f"[cv sha256:{digest}]"


def cv_fragments(cv: str) -> tuple[str, ...]:
    """List the pieces of the CV to look for in traced text.

    The whole CV comes first, so a prompt that embeds it gets one marker. Its single
    lines follow, because the agent reads the CV section by section.

    Args:
        cv: Full text of the candidate's CV.

    Returns:
        The stripped CV and its distinct lines of at least ``MIN_MASKED_LINE_CHARS``
        characters, longest first.
    """
    whole = cv.strip()
    lines = {line.strip() for line in whole.splitlines()}
    long_lines = sorted(
        (line for line in lines if len(line) >= MIN_MASKED_LINE_CHARS),
        key=lambda line: (-len(line), line),
    )
    return tuple(fragment for fragment in (whole, *long_lines) if fragment)


def mask_data(data: Any, fragments: tuple[str, ...], marker: str) -> Any:
    """Replace every piece of CV text inside a value by the marker.

    Args:
        data: A string, or lists, tuples and dicts nested to any depth. Other values
            are returned unchanged.
        fragments: Pieces of text to replace, longest first.
        marker: Replacement text.

    Returns:
        A copy of the value with the pieces replaced.
    """
    if isinstance(data, str):
        for fragment in fragments:
            data = data.replace(fragment, marker)
        return data
    if isinstance(data, dict):
        return {key: mask_data(value, fragments, marker) for key, value in data.items()}
    if isinstance(data, list | tuple):
        return [mask_data(value, fragments, marker) for value in data]
    return data


def _mask(*, data: Any, **_: Any) -> Any:
    """Mask a traced value with the CV of the current scope.

    Langfuse calls this for every input, output and metadata value before it is sent.

    Args:
        data: Value about to be recorded.
        **_: Further arguments Langfuse may pass; unused.

    Returns:
        The masked value; a fixed text if no scope is active, so nothing unmasked can
        slip through.
    """
    scope = _current_scope.get()
    if scope is None:
        return NO_SCOPE_TEXT
    return mask_data(data, scope.fragments, scope.marker)


@lru_cache(maxsize=1)
def _client() -> Langfuse | None:
    """Create the Langfuse client if tracing is configured.

    Returns:
        The client, or None when the keys are not set.
    """
    load_dotenv()
    public_key = os.environ.get(PUBLIC_KEY_VARIABLE)
    secret_key = os.environ.get(SECRET_KEY_VARIABLE)
    if not public_key or not secret_key:
        return None
    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=os.environ.get(BASE_URL_VARIABLE) or None,
        mask=_mask,
    )


def is_enabled() -> bool:
    """Tell whether model calls are being traced.

    Returns:
        True if the Langfuse keys are configured.
    """
    return _client() is not None


@contextmanager
def trace_scope(
    *,
    cv: str,
    experiment: str,
    version: str,
    run_index: int = 1,
    posting_id: str = "",
    client: Langfuse | None = None,
) -> Iterator[None]:
    """Trace the model calls of one decision, tagged and with the CV masked.

    Does nothing when tracing is not configured.

    Args:
        cv: Full text of the candidate's CV; it is masked in everything recorded.
        experiment: Name of the experiment, used as a tag to find the run later.
        version: "workflow" or "agent".
        run_index: Number of the run within a repeated configuration.
        posting_id: Id of the posting being decided.
        client: Langfuse client to use; None takes the configured one. Tests pass a
            client that exports to memory.

    Yields:
        Nothing; model calls made inside the block are traced.
    """
    langfuse = client or _client()
    if langfuse is None:
        yield
        return
    scope = TraceScope(
        experiment=experiment,
        version=version,
        run_index=run_index,
        posting_id=posting_id,
        fragments=cv_fragments(cv),
        marker=cv_marker(cv),
    )
    token = _current_scope.set(scope)
    try:
        with (
            propagate_attributes(
                trace_name=TRACE_NAME,
                tags=[f"experiment:{experiment}", f"version:{version}", f"run:{run_index}"],
                metadata={"posting_id": posting_id, "experiment": experiment},
            ),
            langfuse.start_as_current_observation(
                name=TRACE_NAME, as_type="span", input={"posting_id": posting_id}
            ),
        ):
            yield
    finally:
        _current_scope.reset(token)


@dataclass
class GenerationRecorder:
    """Handle for one traced model call; filled in when the call returns.

    Attributes:
        observation: The Langfuse generation, or None when the call is not traced.
        provider: LLM provider, for the price lookup.
        model: Model id, for the price lookup.
    """

    observation: Any
    provider: str
    model: str

    def finish(self, output: str, input_tokens: int, output_tokens: int) -> None:
        """Record what the model returned and what it cost.

        Args:
            output: Response text.
            input_tokens: Tokens in the prompt.
            output_tokens: Tokens generated, including thinking tokens.
        """
        if self.observation is None:
            return
        details: dict[str, Any] = {
            "output": output,
            "usage_details": {"input": input_tokens, "output": output_tokens},
        }
        price = price_for(self.provider, self.model)
        if price is not None:
            input_cost = input_tokens * price.input_usd_per_million / TOKENS_PER_MILLION
            output_cost = output_tokens * price.output_usd_per_million / TOKENS_PER_MILLION
            details["cost_details"] = {
                "input": input_cost,
                "output": output_cost,
                "total": input_cost + output_cost,
            }
        self.observation.update(**details)


@contextmanager
def traced_generation(
    *,
    provider: str,
    model: str,
    system: str,
    user: str,
    reasoning: bool | None,
    client: Langfuse | None = None,
) -> Iterator[GenerationRecorder]:
    """Trace one model call. Latency is the time spent inside the block.

    Outside a ``trace_scope``, or when tracing is not configured, the call is not
    traced and the recorder does nothing.

    Args:
        provider: LLM provider.
        model: Model id.
        system: System prompt.
        user: User prompt.
        reasoning: Reasoning setting of the call.
        client: Langfuse client to use; None takes the configured one.

    Yields:
        A recorder; call ``finish`` on it with the response.
    """
    langfuse = client or _client()
    if langfuse is None or _current_scope.get() is None:
        yield GenerationRecorder(observation=None, provider=provider, model=model)
        return
    with langfuse.start_as_current_observation(
        name=GENERATION_NAME,
        as_type="generation",
        model=model,
        input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        model_parameters={"provider": provider, "reasoning": str(reasoning)},
    ) as observation:
        yield GenerationRecorder(observation=observation, provider=provider, model=model)


def langchain_callbacks() -> list[Any]:
    """Return the callbacks that trace a LangChain run, for the agent.

    Returns:
        A Langfuse callback handler inside an active ``trace_scope`` with tracing
        configured; otherwise an empty list.
    """
    if _client() is None or _current_scope.get() is None:
        return []
    return [CallbackHandler()]


def flush() -> None:
    """Send traces that are still queued; call it before the process ends."""
    langfuse = _client()
    if langfuse is not None:
        langfuse.flush()
