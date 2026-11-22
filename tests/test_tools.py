"""Tests for the agent's tools. Pure logic and local tool calls; no LLM, no network."""

from typing import Any

import pytest
from langchain_core.tools import BaseTool

from job_fit.rules import FX_RATES_CHECKED_ON, FX_RATES_TO_TRY
from job_fit.tools import (
    build_tools,
    find_cv_section,
    rule_check,
    split_cv_sections,
    tool_descriptions,
)

ENGLISH_POSTING = "Fluent English is required; the team works across Europe."
NEUTRAL_POSTING = "We build mobile apps with React Native."

CV = """\
# Jane Doe

Intro line before any section.

## Experience

### Senior Developer | Acme
Seven years of React Native.

## Projects
A delivery app.

## Skills
TypeScript, React Native
"""


def _tool(name: str, cv: str = CV, posting_text: str = "") -> BaseTool:
    """Return one of the tools built for a CV and a posting, by name."""
    return next(item for item in build_tools(cv, posting_text) if item.name == name)


def _check(
    location: str,
    work_mode: str,
    seniority: str,
    english: bool = False,
    posting_text: str = "",
) -> dict[str, Any]:
    """Call the check_rules tool the way the agent does."""
    return _tool("check_rules", posting_text=posting_text).invoke(
        {
            "location": location,
            "work_mode": work_mode,
            "english_required": english,
            "seniority": seniority,
        }
    )


def test_build_tools_returns_the_three_tools_of_the_process() -> None:
    """The agent gets exactly the tools PROCESS.md names."""
    names = [item.name for item in build_tools(CV)]

    assert names == ["check_rules", "get_cv_section", "convert_currency"]


def test_check_rules_onsite_outside_allowed_city_forces_skip() -> None:
    """An on-site job in Bursa violates the location rule."""
    answer = _check("Bursa, Türkiye", "onsite", "senior")

    assert answer == {
        "violated_rules": ["location"],
        "forced_decision": "SKIP",
        "english_risk": False,
    }


def test_check_rules_junior_role_forces_skip() -> None:
    """A junior role violates the seniority rule even in an allowed city."""
    answer = _check("Ankara, Türkiye", "hybrid", "junior")

    assert answer["violated_rules"] == ["seniority"]
    assert answer["forced_decision"] == "SKIP"


def test_check_rules_allowed_job_forces_nothing() -> None:
    """A senior hybrid job in Istanbul passes every rule."""
    answer = _check("İstanbul, Türkiye", "hybrid", "senior")

    assert answer["violated_rules"] == []
    assert answer["forced_decision"] is None


def test_check_rules_english_requirement_is_a_flag_not_a_skip() -> None:
    """Required English is reported back but never forces a decision."""
    answer = _check("Istanbul, Türkiye", "hybrid", "senior", english=True)

    assert answer["english_risk"] is True
    assert answer["forced_decision"] is None


def test_check_rules_raises_the_english_flag_from_the_posting_text() -> None:
    """A posting that demands fluent English is flagged even if the model said no."""
    answer = _check("Istanbul, Türkiye", "hybrid", "senior", posting_text=ENGLISH_POSTING)

    assert answer["english_risk"] is True
    assert answer["forced_decision"] is None


def test_check_rules_leaves_the_english_flag_down_for_a_neutral_posting() -> None:
    """Without a keyword match and without the model's flag there is no English risk."""
    answer = _check("Istanbul, Türkiye", "hybrid", "senior", posting_text=NEUTRAL_POSTING)

    assert answer["english_risk"] is False


def test_the_posting_text_does_not_change_what_the_model_is_told() -> None:
    """Tool descriptions, and so the agent's prompt hash, do not depend on the posting."""
    assert tool_descriptions(build_tools(CV, ENGLISH_POSTING)) == tool_descriptions(build_tools(CV))


def test_rule_check_accepts_values_in_any_letter_case() -> None:
    """Facts written as 'Onsite' or 'JUNIOR' are still recognized."""
    answer = rule_check("Bursa, Türkiye", "Onsite", False, "JUNIOR")

    assert answer["violated_rules"] == ["location", "seniority"]


def test_split_cv_sections_keys_sections_by_lowercase_heading() -> None:
    """Each second-level heading becomes one section, in document order."""
    sections = split_cv_sections(CV)

    assert list(sections) == ["experience", "projects", "skills"]
    assert sections["skills"] == "TypeScript, React Native"


def test_split_cv_sections_keeps_sub_headings_inside_their_section() -> None:
    """A third-level heading does not start a new section."""
    sections = split_cv_sections(CV)

    assert "### Senior Developer | Acme" in sections["experience"]
    assert "A delivery app." not in sections["experience"]


def test_split_cv_sections_of_a_cv_without_headings_is_empty() -> None:
    """Text without section headings yields no sections."""
    assert split_cv_sections("Just one paragraph.") == {}


@pytest.mark.parametrize("topic", ["skills", "Skills", "  SKILLS ", "skill", "technical skills"])
def test_find_cv_section_matches_exact_and_partial_topics(topic: str) -> None:
    """Case, surrounding spaces and a partial heading do not prevent a match."""
    assert find_cv_section(split_cv_sections(CV), topic) == "TypeScript, React Native"


@pytest.mark.parametrize("topic", ["hobbies", "", "   "])
def test_find_cv_section_returns_none_when_nothing_matches(topic: str) -> None:
    """An unknown or empty topic matches no section."""
    assert find_cv_section(split_cv_sections(CV), topic) is None


def test_get_cv_section_returns_the_section_text() -> None:
    """The tool hands back the text of the section asked for."""
    answer = _tool("get_cv_section").invoke({"topic": "experience"})

    assert "Seven years of React Native." in answer
    assert "TypeScript" not in answer


def test_get_cv_section_lists_the_topics_for_an_unknown_topic() -> None:
    """An unknown topic is answered with the valid topics, not with an error."""
    answer = _tool("get_cv_section").invoke({"topic": "hobbies"})

    assert answer == (
        "No CV section matches 'hobbies'. Available topics: experience, projects, skills."
    )


def test_get_cv_section_description_names_the_topics_of_this_cv() -> None:
    """The model learns the valid topics from the tool description."""
    description = _tool("get_cv_section").description

    assert "experience, projects, skills" in description


def test_convert_currency_tool_returns_the_amount_in_try() -> None:
    """The tool returns a number, as its signature in PROCESS.md says."""
    answer = _tool("convert_currency").invoke({"amount": 2, "currency": "EUR"})

    assert answer == 2 * FX_RATES_TO_TRY["EUR"]


def test_convert_currency_tool_reports_a_bad_currency_to_the_model() -> None:
    """An unsupported currency comes back as a message instead of ending the run."""
    answer = _tool("convert_currency").invoke({"amount": 2, "currency": "GBP"})

    assert "Unsupported currency 'GBP'" in answer


def test_convert_currency_description_says_the_rate_is_fixed_and_dated() -> None:
    """The model is told that the rate is not live, and when it was checked."""
    description = _tool("convert_currency").description

    assert "fixed exchange rate" in description
    assert FX_RATES_CHECKED_ON in description


def test_tool_descriptions_has_one_line_per_tool() -> None:
    """The fingerprint text names every tool once."""
    lines = tool_descriptions(build_tools(CV)).splitlines()

    assert [line.split(":")[0] for line in lines] == [
        "check_rules",
        "get_cv_section",
        "convert_currency",
    ]
