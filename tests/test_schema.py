"""Tests for the JobDecision schema."""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from job_fit.schema import FitDecision, JobDecision, PostingFacts, join_decision


def _valid_payload() -> dict[str, Any]:
    """Return a payload that satisfies every JobDecision field."""
    return {
        "rationale": "The stack matches the CV. Go is missing. Location is acceptable.",
        "company": "Acme",
        "role": "Senior Mobile Engineer",
        "location": "Istanbul, Türkiye",
        "work_mode": "hybrid",
        "english_risk": False,
        "seniority": "senior",
        "matching_skills": ["React Native", "TypeScript"],
        "missing_skills": ["Go"],
        "decision": "PREPARE",
    }


def test_valid_json_is_parsed() -> None:
    """A complete, well-typed JSON document becomes a JobDecision."""
    decision = JobDecision.model_validate_json(json.dumps(_valid_payload()))

    assert decision.decision == "PREPARE"
    assert decision.missing_skills == ["Go"]


def test_rationale_is_the_first_field() -> None:
    """Rationale stays first so the model reasons before it decides."""
    field_names = list(JobDecision.model_fields)

    assert field_names[0] == "rationale"
    assert field_names[-1] == "decision"


def test_malformed_json_is_rejected() -> None:
    """Text that is not valid JSON raises a validation error."""
    with pytest.raises(ValidationError):
        JobDecision.model_validate_json('{"rationale": "cut off')


@pytest.mark.parametrize("document", ["[]", "null", '"APPLY"', "42", ""])
def test_json_of_the_wrong_shape_is_rejected(document: str) -> None:
    """Valid JSON that is not an object, and an empty answer, are rejected."""
    with pytest.raises(ValidationError):
        JobDecision.model_validate_json(document)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("decision", "MAYBE"),
        ("work_mode", "office"),
        ("seniority", "principal"),
        ("english_risk", "sometimes"),
        ("matching_skills", "React Native"),
    ],
)
def test_invalid_field_value_is_rejected(field: str, bad_value: Any) -> None:
    """A value outside the allowed set or of the wrong type is rejected."""
    payload = _valid_payload() | {field: bad_value}

    with pytest.raises(ValidationError):
        JobDecision.model_validate(payload)


@pytest.mark.parametrize("field", ["rationale", "location", "decision"])
def test_missing_field_is_rejected(field: str) -> None:
    """Every field is required; leaving one out is rejected."""
    payload = _valid_payload()
    del payload[field]

    with pytest.raises(ValidationError):
        JobDecision.model_validate(payload)


def test_facts_and_fit_together_have_exactly_the_fields_of_a_decision() -> None:
    """The two-call schemas split the single-call schema without a gap or an overlap."""
    facts = set(PostingFacts.model_fields)
    fit = set(FitDecision.model_fields)

    assert facts & fit == set()
    assert facts | fit == set(JobDecision.model_fields)


@pytest.mark.parametrize("part", [PostingFacts, FitDecision])
def test_split_schemas_describe_each_field_as_the_decision_does(part: type) -> None:
    """A field means the same thing whichever model fills it."""
    for name, field in part.model_fields.items():
        assert field.description == JobDecision.model_fields[name].description
        assert field.annotation == JobDecision.model_fields[name].annotation


def test_fit_decision_reasons_first_and_decides_last() -> None:
    """The main model still writes the rationale before it commits to a decision."""
    names = list(FitDecision.model_fields)

    assert (names[0], names[-1]) == ("rationale", "decision")


def test_posting_facts_reject_an_unknown_work_mode() -> None:
    """The extractor's answer is validated like the workflow's."""
    with pytest.raises(ValidationError):
        PostingFacts(
            company="Acme",
            role="Engineer",
            location="unknown",
            work_mode="from the moon",
            english_risk=False,
            seniority="senior",
        )


def test_join_decision_takes_every_field_from_one_of_the_two_parts() -> None:
    """Joining facts and fit gives a complete decision."""
    facts = PostingFacts(
        company="Acme",
        role="Engineer",
        location="Remote (Türkiye)",
        work_mode="remote",
        english_risk=True,
        seniority="senior",
    )
    fit = FitDecision(
        rationale="Fits.", matching_skills=["Python"], missing_skills=[], decision="APPLY"
    )

    decision = join_decision(facts, fit)

    assert decision.model_dump() == facts.model_dump() | fit.model_dump()
    assert list(decision.model_dump()) == list(JobDecision.model_fields)
