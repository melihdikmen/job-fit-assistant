"""List prices per token, used to estimate what an eval run costs.

Costs always use the public list price, even when free credits pay for the calls, so
that models stay comparable. Pure functions only: no I/O and no network.
"""

from dataclasses import dataclass

TOKENS_PER_MILLION = 1_000_000

# Date the table below was last compared with the providers' pricing pages.
PRICES_CHECKED_ON = "2026-11-10"


@dataclass(frozen=True)
class ModelPrice:
    """List price of one model.

    Attributes:
        input_usd_per_million: USD per million prompt tokens.
        output_usd_per_million: USD per million generated tokens, thinking included.
    """

    input_usd_per_million: float
    output_usd_per_million: float


# Standard (non-batch) text prices. Source for Gemini on Vertex AI:
# https://cloud.google.com/vertex-ai/generative-ai/pricing
PRICES_USD_PER_MILLION_TOKENS: dict[str, ModelPrice] = {
    "gemini-2.5-flash": ModelPrice(input_usd_per_million=0.30, output_usd_per_million=2.50),
}


def price_for(model: str) -> ModelPrice | None:
    """Look up the list price of a model.

    Args:
        model: Model id as the provider names it.

    Returns:
        The price, or None if the model is not in the table.
    """
    return PRICES_USD_PER_MILLION_TOKENS.get(model)


def call_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimate the list-price cost of the given token usage.

    Args:
        model: Model id as the provider names it.
        input_tokens: Prompt tokens.
        output_tokens: Generated tokens, thinking included.

    Returns:
        Cost in USD, or None if the model has no known price. None is returned rather
        than zero so that an unknown price is never mistaken for a free model.
    """
    price = price_for(model)
    if price is None:
        return None
    return (
        input_tokens * price.input_usd_per_million + output_tokens * price.output_usd_per_million
    ) / TOKENS_PER_MILLION
