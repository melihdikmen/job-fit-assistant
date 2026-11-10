"""List prices per token, used to estimate what an eval run costs.

Costs always use the public list price, even when free credits pay for the calls, so
that models stay comparable. Pure functions only: no I/O and no network.
"""

from dataclasses import dataclass

TOKENS_PER_MILLION = 1_000_000

# Date the table below was last compared with the providers' pricing pages.
PRICES_CHECKED_ON = "2026-11-10"

# A model run on the candidate's own machine has no price per token (electricity and
# hardware are not counted).
LOCAL_PROVIDER = "ollama"


@dataclass(frozen=True)
class ModelPrice:
    """List price of one model.

    Attributes:
        input_usd_per_million: USD per million prompt tokens.
        output_usd_per_million: USD per million generated tokens, thinking included.
    """

    input_usd_per_million: float
    output_usd_per_million: float


FREE = ModelPrice(input_usd_per_million=0.0, output_usd_per_million=0.0)

# Standard (non-batch) text prices, keyed by (provider, model id): the same model can
# cost differently on another provider. Gemini prices are tiered by prompt size; the
# tier below 200k prompt tokens applies, since prompts here are about 2.3k tokens.
# Source for gemini-2.5-flash: https://cloud.google.com/vertex-ai/generative-ai/pricing
# Source for the other Gemini models: https://aistudio.google.com/docs/pricing (Gemini
# Developer API, Standard tier). The Vertex AI page could not be read on the day of the
# check, so these two prices are not confirmed to be identical on Vertex AI.
# gemini-4-carbon is at an introductory price through 2026-11-30; from 2026-12-01 it
# is 4.00 / 20.00.
# Source for OpenRouter: https://openrouter.ai/models (free variants are listed at 0).
PRICES_USD_PER_MILLION_TOKENS: dict[tuple[str, str], ModelPrice] = {
    ("vertex", "gemini-2.5-flash"): ModelPrice(
        input_usd_per_million=0.30, output_usd_per_million=2.50
    ),
    ("vertex", "gemini-4-carbon"): ModelPrice(
        input_usd_per_million=2.00, output_usd_per_million=10.00
    ),
    ("vertex", "gemini-3.1-flash-lite"): ModelPrice(
        input_usd_per_million=0.25, output_usd_per_million=1.50
    ),
    ("openrouter", "nvidia/nemotron-3.5-super-vl:free"): FREE,
    ("openrouter", "liquid/lfm-3-1.7b:free"): FREE,
}


def price_for(provider: str, model: str) -> ModelPrice | None:
    """Look up the list price of a model on a provider.

    Args:
        provider: "vertex", "openrouter" or "ollama".
        model: Model id as the provider names it.

    Returns:
        The price, or None if the model is not in the table. Local models are free.
    """
    if provider == LOCAL_PROVIDER:
        return FREE
    return PRICES_USD_PER_MILLION_TOKENS.get((provider, model))


def call_cost_usd(provider: str, model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimate the list-price cost of the given token usage.

    Args:
        provider: "vertex", "openrouter" or "ollama".
        model: Model id as the provider names it.
        input_tokens: Prompt tokens.
        output_tokens: Generated tokens, thinking included.

    Returns:
        Cost in USD, or None if the model has no known price. None is returned rather
        than zero so that an unknown price is never mistaken for a free model.
    """
    price = price_for(provider, model)
    if price is None:
        return None
    return (
        input_tokens * price.input_usd_per_million + output_tokens * price.output_usd_per_million
    ) / TOKENS_PER_MILLION
