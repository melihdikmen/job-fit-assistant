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


class JobDecision(BaseModel):
    """Structured decision about one job posting for one candidate.

    Field order is intentional. The model generates fields in this order, so
    ``rationale`` comes first to make it reason before it commits to a decision,
    and ``decision`` comes last so it can rely on everything extracted above it.
    """

    rationale: str = Field(
        description=(
            "Think first: 3 to 5 sentences comparing the posting's requirements with the "
            "candidate's CV. Written in English."
        )
    )
    company: str = Field(
        description="Hiring company name as written in the posting, or 'unknown' if not stated."
    )
    role: str = Field(
        description="Job title as written in the posting, or 'unknown' if not stated."
    )
    location: str = Field(
        description=(
            "Where the job is done. On-site or hybrid: 'City, Country' (e.g. 'Istanbul, "
            "Türkiye'). Remote: 'Remote (<region the role is open to>)' (e.g. 'Remote "
            "(Türkiye)', 'Remote (Worldwide)', 'Remote (US only)'). Use 'unknown' if the "
            "posting does not say. Never guess."
        )
    )
    work_mode: WorkMode = Field(
        description="Working arrangement stated in the posting; 'unknown' if not stated."
    )
    english_risk: bool = Field(
        description=(
            "True if the posting requires fluent or advanced English, or the team works in English."
        )
    )
    seniority: Seniority = Field(
        description="Seniority level the posting targets; 'unknown' if it cannot be told."
    )
    matching_skills: list[str] = Field(
        description="Requirements from the posting that the CV clearly covers."
    )
    missing_skills: list[str] = Field(
        description="Requirements from the posting that the CV does not cover."
    )
    decision: DecisionLabel = Field(
        description=(
            "APPLY: good fit, apply now. PREPARE: mostly a fit with 1 or 2 gaps that a "
            "short preparation can close. SKIP: not a fit."
        )
    )
