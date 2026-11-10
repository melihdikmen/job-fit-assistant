"""Tests for the JobDecision schema."""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from job_fit.schema import JobDecision


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
