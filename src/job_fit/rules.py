"""Hard candidate rules enforced in code.

The LLM extracts facts and proposes a decision; these rules have the final word.
Everything here is a pure function: values in, values out, no I/O and no LLM.
"""

import re
from dataclasses import dataclass

from job_fit.schema import JobDecision

RULE_LOCATION = "location"
RULE_SENIORITY = "seniority"

UNKNOWN = "unknown"
REMOTE = "remote"
OFFICE_WORK_MODES = frozenset({"onsite", "hybrid"})
BLOCKED_SENIORITIES = frozenset({"intern", "junior"})

# Match data below is compared against normalized text (see _normalize), so it is
# written lowercase and without Turkish diacritics.
ALLOWED_CITIES = ("ankara", "istanbul")
REMOTE_ALLOWED_REGIONS = ("turkiye", "turkey", "worldwide", "global", "anywhere", "emea")

# A strength word close to the language name, in either order, within one sentence.
# English and Turkish phrasings are both covered because postings come in both.
_ENGLISH_RISK_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"(fluent|fluency|advanced|excellent|strong|proficien)[^.\n]{0,60}\benglish\b",
        r"\benglish\b[^.\n]{0,40}(fluent|fluency|advanced|excellent|proficien)",
        r"(ileri|akici|cok iyi|iyi derecede)[^.\n]{0,40}ingilizce",
        r"ingilizce[^.\n]{0,40}(ileri|akici|cok iyi)",
    )
)

_TURKISH_CHAR_MAP = str.maketrans(
    {
        "İ": "i",
        "I": "i",
        "ı": "i",
        "Ş": "s",
        "ş": "s",
        "Ğ": "g",
        "ğ": "g",
        "Ü": "u",
        "ü": "u",
        "Ö": "o",
        "ö": "o",
        "Ç": "c",
        "ç": "c",
    }
)


@dataclass(frozen=True)
class RuleOutcome:
    """Result of applying the rules to an LLM decision.

    Attributes:
        decision: The decision after rules were applied.
        fired_rules: Names of the rules that forced a SKIP, in evaluation order.
    """

    decision: JobDecision
    fired_rules: tuple[str, ...]


def _normalize(text: str) -> str:
    """Lowercase text and replace Turkish letters with their ASCII look-alikes.

    Args:
        text: Any text, e.g. a location or a posting.

    Returns:
        The text in a form where "İstanbul", "ISTANBUL" and "istanbul" are equal.
    """
    return text.translate(_TURKISH_CHAR_MAP).casefold()


def _remote_region(normalized_location: str) -> str:
    """Extract the region a remote role is open to.

    Args:
        normalized_location: Normalized location, e.g. "remote (us only)".

    Returns:
        The region words without the word "remote" and punctuation, e.g. "us only".
        Empty if the location names no region.
    """
    without_remote = normalized_location.replace(REMOTE, " ")
    letters_only = re.sub(r"[^a-z]+", " ", without_remote)
    return letters_only.strip()


def _violates_location(location: str, work_mode: str) -> bool:
    """Check the location rule.

    Args:
        location: Location as extracted from the posting.
        work_mode: One of "remote", "hybrid", "onsite", "unknown".

    Returns:
        True if the job is on-site or hybrid outside the allowed cities, or remote
        but not open to Türkiye. Unknown location or work mode never violates.
    """
    place = _normalize(location).strip()
    if not place or UNKNOWN in place:
        return False
    if work_mode in OFFICE_WORK_MODES:
        return not any(city in place for city in ALLOWED_CITIES)
    if work_mode == REMOTE:
        region = _remote_region(place)
        if not region:
            return False
        accepted = (*REMOTE_ALLOWED_REGIONS, *ALLOWED_CITIES)
        return not any(name in region for name in accepted)
    return False


def violated_rules(location: str, work_mode: str, seniority: str) -> list[str]:
    """List the hard rules a job violates.

    Args:
        location: Location as extracted from the posting.
        work_mode: One of "remote", "hybrid", "onsite", "unknown".
        seniority: One of "intern", "junior", "mid", "senior", "lead", "unknown".

    Returns:
        Rule names that force a SKIP; empty if the job passes every rule.
    """
    violations: list[str] = []
    if _violates_location(location, work_mode):
        violations.append(RULE_LOCATION)
    if seniority in BLOCKED_SENIORITIES:
        violations.append(RULE_SENIORITY)
    return violations


def has_english_risk(posting_text: str) -> bool:
    """Detect a strong English requirement by keyword.

    Args:
        posting_text: Full text of the job posting.

    Returns:
        True if the posting asks for fluent, advanced or otherwise strong English.
    """
    text = _normalize(posting_text)
    return any(pattern.search(text) for pattern in _ENGLISH_RISK_PATTERNS)


def apply_rules(decision: JobDecision, posting_text: str) -> RuleOutcome:
    """Apply the hard rules on top of the LLM's decision.

    A violated rule turns the decision into SKIP and appends "[rule: <name>]" to the
    rationale. The English flag is raised if either the LLM or the keyword check
    says so; it never changes the decision by itself.

    Args:
        decision: Validated decision produced by the LLM.
        posting_text: Full text of the job posting.

    Returns:
        The final decision and the names of the rules that fired. The input object
        is left untouched.
    """
    fired = violated_rules(decision.location, decision.work_mode, decision.seniority)
    rule_notes = "".join(f" [rule: {name}]" for name in fired)
    final = decision.model_copy(
        update={
            "decision": "SKIP" if fired else decision.decision,
            "rationale": decision.rationale + rule_notes,
            "english_risk": decision.english_risk or has_english_risk(posting_text),
        }
    )
    return RuleOutcome(decision=final, fired_rules=tuple(fired))
