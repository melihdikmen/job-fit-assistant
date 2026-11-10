"""Tests for the list-price table and cost estimate."""

import pytest

from job_fit import pricing
from job_fit.pricing import ModelPrice, call_cost_usd, price_for


@pytest.fixture
def fake_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the price table with one fixed model, so tests survive price changes."""
    table = {
        ("vertex", "fake-model"): ModelPrice(input_usd_per_million=1.0, output_usd_per_million=4.0)
    }
    monkeypatch.setattr(pricing, "PRICES_USD_PER_MILLION_TOKENS", table)


@pytest.mark.usefixtures("fake_prices")
def test_cost_adds_input_and_output_at_their_own_price() -> None:
    """Input and output tokens are priced separately, per million."""
    cost = call_cost_usd("vertex", "fake-model", input_tokens=2_000_000, output_tokens=500_000)

    assert cost == pytest.approx(4.0)


@pytest.mark.usefixtures("fake_prices")
def test_unknown_model_has_no_cost_rather_than_zero() -> None:
    """A model missing from the table returns None, never a made-up number."""
    assert call_cost_usd("vertex", "not-in-table", input_tokens=1000, output_tokens=1000) is None
    assert price_for("vertex", "not-in-table") is None


@pytest.mark.usefixtures("fake_prices")
def test_price_belongs_to_a_provider_not_just_a_model() -> None:
    """The same model id on another provider is not silently given the same price."""
    assert price_for("openrouter", "fake-model") is None


def test_local_model_is_free_without_a_table_entry() -> None:
    """Any model run through Ollama costs zero per token."""
    assert call_cost_usd("ollama", "whatever:4b", input_tokens=1000, output_tokens=1000) == 0.0


def test_phase_one_model_is_in_the_price_table() -> None:
    """The model used so far has a list price."""
    assert price_for("vertex", "gemini-2.5-flash") is not None
