"""Tests for the hard rules that override the LLM's decision."""

import unicodedata
from typing import Any

import pytest

from job_fit.rules import (
    RULE_LOCATION,
    RULE_SENIORITY,
    apply_rules,
    has_english_risk,
    violated_rules,
)
from job_fit.schema import JobDecision

NEUTRAL_POSTING = "We build mobile apps with React Native."


def _decision(**overrides: Any) -> JobDecision:
    """Build an APPLY decision for a senior hybrid Istanbul role, with overrides."""
    fields: dict[str, Any] = {
        "rationale": "Strong match on the main stack.",
        "company": "Acme",
        "role": "Senior Mobile Engineer",
        "location": "Istanbul, Türkiye",
        "work_mode": "hybrid",
        "english_risk": False,
        "seniority": "senior",
        "matching_skills": ["React Native"],
        "missing_skills": [],
        "decision": "APPLY",
    }
    return JobDecision(**(fields | overrides))


@pytest.mark.parametrize(
    ("location", "work_mode"),
    [
        ("Bursa, Türkiye", "onsite"),
        ("Izmir, Türkiye", "onsite"),
        ("İzmir, Türkiye", "hybrid"),
        ("Berlin, Germany", "hybrid"),
        ("Remote (US only)", "remote"),
        ("Remote (Germany)", "remote"),
    ],
)
def test_location_outside_allowed_area_is_violation(location: str, work_mode: str) -> None:
    """Office work outside Ankara/Istanbul and remote closed to Türkiye violate."""
    assert violated_rules(location, work_mode, "senior") == [RULE_LOCATION]


@pytest.mark.parametrize(
    ("location", "work_mode"),
    [
        ("Ankara, Türkiye", "onsite"),
        ("İstanbul, Türkiye", "hybrid"),
        ("ISTANBUL", "onsite"),
        ("Remote (Türkiye)", "remote"),
        ("Remote (Turkey)", "remote"),
        ("Remote (Worldwide)", "remote"),
    ],
)
def test_location_inside_allowed_area_passes(location: str, work_mode: str) -> None:
    """Ankara, Istanbul in any spelling, and remote open to Türkiye pass."""
    assert violated_rules(location, work_mode, "senior") == []


@pytest.mark.parametrize(
    ("location", "work_mode"),
    [
        # Decomposed forms: a base letter followed by a combining mark, as text copied
        # from a PDF or lowercased with str.lower() can be.
        (unicodedata.normalize("NFD", "İstanbul, Türkiye"), "onsite"),
        ("İstanbul".lower() + ", Türkiye", "hybrid"),
        (unicodedata.normalize("NFD", "Remote (Türkiye)"), "remote"),
    ],
)
def test_decomposed_unicode_spellings_pass(location: str, work_mode: str) -> None:
    """An allowed place is recognized however its accented letters are encoded."""
    assert violated_rules(location, work_mode, "senior") == []


def test_english_requirement_is_detected_in_decomposed_text() -> None:
    """The keyword check reads decomposed Turkish text like composed text."""
    posting_text = unicodedata.normalize("NFD", "İleri seviye İngilizce bilgisi")

    assert has_english_risk(posting_text) is True


@pytest.mark.parametrize(
    ("location", "work_mode"),
    [
        ("unknown", "onsite"),
        ("", "hybrid"),
        ("Bursa, Türkiye", "unknown"),
        ("Remote", "remote"),
        ("Remote (unknown)", "remote"),
    ],
)
def test_unknown_location_or_work_mode_never_violates(location: str, work_mode: str) -> None:
    """Missing information is not enough to skip a job."""
    assert violated_rules(location, work_mode, "senior") == []


@pytest.mark.parametrize("seniority", ["intern", "junior"])
def test_intern_and_junior_roles_are_violation(seniority: str) -> None:
    """Internship and junior roles violate the seniority rule."""
    assert violated_rules("Ankara, Türkiye", "onsite", seniority) == [RULE_SENIORITY]


@pytest.mark.parametrize("seniority", ["mid", "senior", "lead", "unknown"])
def test_other_seniorities_pass(seniority: str) -> None:
    """Mid and above, and unknown seniority, pass the seniority rule."""
    assert violated_rules("Ankara, Türkiye", "onsite", seniority) == []


@pytest.mark.parametrize(
    "posting_text",
    [
        "Fluent English is a must.",
        "Strong written and verbal communication skills in English",
        "Advanced level of English, spoken and written.",
        "English: fluent",
        "İleri seviye İngilizce bilgisi",
        "Çok iyi derecede İngilizce bilen",
        "İngilizce: ileri",
    ],
)
def test_strong_english_requirement_is_detected(posting_text: str) -> None:
    """Common English and Turkish phrasings of a strong requirement are detected."""
    assert has_english_risk(posting_text) is True


@pytest.mark.parametrize(
    "posting_text",
    [
        NEUTRAL_POSTING,
        "Strong TypeScript skills. Documentation is written in English.",
        "Temel seviye İngilizce yeterlidir.",
    ],
)
def test_weak_or_absent_english_requirement_is_not_detected(posting_text: str) -> None:
    """Postings without a strong English requirement are not flagged."""
    assert has_english_risk(posting_text) is False


def test_location_violation_overrides_apply_and_tags_rationale() -> None:
    """An APPLY for an on-site job in Bursa becomes SKIP with a rule tag."""
    llm_decision = _decision(location="Bursa, Türkiye", work_mode="onsite")

    outcome = apply_rules(llm_decision, NEUTRAL_POSTING)

    assert outcome.decision.decision == "SKIP"
    assert outcome.fired_rules == (RULE_LOCATION,)
    assert outcome.decision.rationale.endswith("[rule: location]")


def test_seniority_violation_alone_overrides_apply_and_tags_rationale() -> None:
    """An APPLY for a junior role in an allowed city becomes SKIP with a rule tag."""
    llm_decision = _decision(seniority="junior")

    outcome = apply_rules(llm_decision, NEUTRAL_POSTING)

    assert outcome.decision.decision == "SKIP"
    assert outcome.fired_rules == (RULE_SENIORITY,)
    assert outcome.decision.rationale.endswith("[rule: seniority]")


def test_remote_role_closed_to_turkiye_overrides_apply() -> None:
    """An APPLY for a remote role open only to another country becomes SKIP (D3)."""
    llm_decision = _decision(location="Remote (US only)", work_mode="remote")

    outcome = apply_rules(llm_decision, NEUTRAL_POSTING)

    assert outcome.decision.decision == "SKIP"
    assert outcome.fired_rules == (RULE_LOCATION,)


def test_two_violations_are_both_recorded() -> None:
    """Location and seniority violations are both reported and tagged."""
    llm_decision = _decision(location="Bursa, Türkiye", work_mode="onsite", seniority="junior")

    outcome = apply_rules(llm_decision, NEUTRAL_POSTING)

    assert outcome.fired_rules == (RULE_LOCATION, RULE_SENIORITY)
    assert "[rule: location]" in outcome.decision.rationale
    assert "[rule: seniority]" in outcome.decision.rationale


def test_passing_job_keeps_llm_decision_and_rationale() -> None:
    """When no rule fires, decision and rationale are returned unchanged."""
    llm_decision = _decision(decision="PREPARE")

    outcome = apply_rules(llm_decision, NEUTRAL_POSTING)

    assert outcome.fired_rules == ()
    assert outcome.decision.decision == "PREPARE"
    assert outcome.decision.rationale == llm_decision.rationale


def test_english_keyword_raises_flag_without_changing_decision() -> None:
    """A keyword match sets english_risk but leaves the decision alone."""
    llm_decision = _decision(english_risk=False)

    outcome = apply_rules(llm_decision, "Fluent English is required.")

    assert outcome.decision.english_risk is True
    assert outcome.decision.decision == "APPLY"
    assert outcome.fired_rules == ()


def test_llm_english_flag_is_kept_without_keyword() -> None:
    """The LLM's flag survives when the keyword check finds nothing."""
    llm_decision = _decision(english_risk=True)

    outcome = apply_rules(llm_decision, NEUTRAL_POSTING)

    assert outcome.decision.english_risk is True


def test_input_decision_is_not_mutated() -> None:
    """apply_rules returns a new object and leaves its input as it was."""
    llm_decision = _decision(location="Bursa, Türkiye", work_mode="onsite")

    apply_rules(llm_decision, "Fluent English is required.")

    assert llm_decision.decision == "APPLY"
    assert llm_decision.english_risk is False
    assert llm_decision.rationale == "Strong match on the main stack."
