"""Pydantic models describing the structured output expected from the LLM."""

from typing import Literal

from pydantic import BaseModel, Field

WorkMode = Literal["remote", "hybrid", "onsite", "unknown"]
Seniority = Literal["intern", "junior", "mid", "senior", "lead", "unknown"]
DecisionLabel = Literal["APPLY", "PREPARE", "SKIP"]


MIN_RATIONALE_SCORE = 1
MAX_RATIONALE_SCORE = 5


class RationaleScore(BaseModel):
    """A judge model's rating of one decision's rationale against the rubric.

    Field order is intentional, as in ``JobDecision``: the judge explains first, then
    answers the yes/no criteria, and gives the overall score last.
    """

    reasoning: str = Field(
        description=(
            "Think first: 2 to 4 sentences on how well the rationale explains the "
            "decision, checked against the posting and the CV."
        )
    )
    names_matching_skills: bool = Field(
        description=(
            "True if the rationale names at least one concrete requirement of the posting "
            "that the CV covers, or correctly says that there is none."
        )
    )
    names_missing_skills: bool = Field(
        description=(
            "True if the rationale names the concrete gaps between the posting and the CV, "
            "or correctly says that there are none."
        )
    )
    reflects_hard_rules: bool = Field(
        description=(
            "True if every hard rule listed as fired is stated in the rationale, or no rule fired."
        )
    )
    mentions_english_risk: bool = Field(
        description=(
            "True if the posting demands fluent or advanced English and the rationale "
            "says so, or the posting makes no such demand."
        )
    )
    consistent_with_decision: bool = Field(
        description="True if the reasoning given leads to the final decision that was made."
    )
    score: int = Field(
        ge=MIN_RATIONALE_SCORE,
        le=MAX_RATIONALE_SCORE,
        description=(
            "Overall quality from 1 to 5. 5: specific, accurate and complete. 3: correct "
            "but generic or missing one criterion. 1: wrong about the posting or the CV, "
            "or contradicts the decision."
        ),
    )


# One wording per field, shared by every model that carries the field, so that the
# single-call schema (``JobDecision``) and the two-call schemas (``PostingFacts`` and
# ``FitDecision``) cannot drift apart.
RATIONALE_DESCRIPTION = (
    "Think first: 3 to 5 sentences comparing the posting's requirements with the "
    "candidate's CV. Written in English."
)
COMPANY_DESCRIPTION = "Hiring company name as written in the posting, or 'unknown' if not stated."
ROLE_DESCRIPTION = "Job title as written in the posting, or 'unknown' if not stated."
LOCATION_DESCRIPTION = (
    "Where the job is done. On-site or hybrid: 'City, Country' (e.g. 'Istanbul, "
    "Türkiye'). Remote: 'Remote (<region the role is open to>)' (e.g. 'Remote "
    "(Türkiye)', 'Remote (Worldwide)', 'Remote (US only)'). Use 'unknown' if the "
    "posting does not say. Never guess."
)
WORK_MODE_DESCRIPTION = "Working arrangement stated in the posting; 'unknown' if not stated."
ENGLISH_RISK_DESCRIPTION = (
    "True if the posting requires fluent or advanced English, or the team works in English."
)
SENIORITY_DESCRIPTION = "Seniority level the posting targets; 'unknown' if it cannot be told."
MATCHING_SKILLS_DESCRIPTION = "Requirements from the posting that the CV clearly covers."
MISSING_SKILLS_DESCRIPTION = "Requirements from the posting that the CV does not cover."
DECISION_DESCRIPTION = (
    "APPLY: good fit, apply now. PREPARE: mostly a fit with 1 or 2 gaps that a "
    "short preparation can close. SKIP: not a fit."
)


class JobDecision(BaseModel):
    """Structured decision about one job posting for one candidate.

    Field order is intentional. The model generates fields in this order, so
    ``rationale`` comes first to make it reason before it commits to a decision,
    and ``decision`` comes last so it can rely on everything extracted above it.
    """

    rationale: str = Field(description=RATIONALE_DESCRIPTION)
    company: str = Field(description=COMPANY_DESCRIPTION)
    role: str = Field(description=ROLE_DESCRIPTION)
    location: str = Field(description=LOCATION_DESCRIPTION)
    work_mode: WorkMode = Field(description=WORK_MODE_DESCRIPTION)
    english_risk: bool = Field(description=ENGLISH_RISK_DESCRIPTION)
    seniority: Seniority = Field(description=SENIORITY_DESCRIPTION)
    matching_skills: list[str] = Field(description=MATCHING_SKILLS_DESCRIPTION)
    missing_skills: list[str] = Field(description=MISSING_SKILLS_DESCRIPTION)
    decision: DecisionLabel = Field(description=DECISION_DESCRIPTION)


class PostingFacts(BaseModel):
    """Facts stated in a job posting, extracted without looking at the candidate.

    The extraction sub-agent returns this. It holds the fields of ``JobDecision`` that
    need no judgment about the fit, in the same order and with the same descriptions.
    """

    company: str = Field(description=COMPANY_DESCRIPTION)
    role: str = Field(description=ROLE_DESCRIPTION)
    location: str = Field(description=LOCATION_DESCRIPTION)
    work_mode: WorkMode = Field(description=WORK_MODE_DESCRIPTION)
    english_risk: bool = Field(description=ENGLISH_RISK_DESCRIPTION)
    seniority: Seniority = Field(description=SENIORITY_DESCRIPTION)


class FitDecision(BaseModel):
    """The judgment about the fit: what the main model returns when a sub-agent extracts.

    It holds the remaining fields of ``JobDecision``. ``rationale`` still comes first
    and ``decision`` last, for the reason given there.
    """

    rationale: str = Field(description=RATIONALE_DESCRIPTION)
    matching_skills: list[str] = Field(description=MATCHING_SKILLS_DESCRIPTION)
    missing_skills: list[str] = Field(description=MISSING_SKILLS_DESCRIPTION)
    decision: DecisionLabel = Field(description=DECISION_DESCRIPTION)


def join_decision(facts: PostingFacts, fit: FitDecision) -> JobDecision:
    """Combine extracted facts and a fit judgment into one decision.

    Args:
        facts: Facts extracted from the posting.
        fit: The main model's judgment about the fit.

    Returns:
        A ``JobDecision`` with every field taken from one of the two parts.
    """
    return JobDecision(**facts.model_dump(), **fit.model_dump())
