"""Hard candidate rules enforced in code.

The LLM extracts facts and proposes a decision; these rules have the final word.
Everything here is a pure function: values in, values out, no I/O and no LLM.
"""

import re
import unicodedata
from dataclasses import dataclass

from job_fit.schema import JobDecision

RULE_LOCATION = "location"
RULE_SENIORITY = "seniority"

UNKNOWN = "unknown"
REMOTE = "remote"
OFFICE_WORK_MODES = frozenset({"onsite", "hybrid"})
BLOCKED_SENIORITIES = frozenset({"intern", "junior"})

# Match data below is compared against normalized text (see _normalize), so it is
# written lowercase and without Turkish diacritics. Names are matched as whole words:
# "istanbul" must not match "Istanbulska", nor "turkey" a longer word.
ALLOWED_CITIES = frozenset({"ankara", "istanbul"})
TURKIYE_NAMES = frozenset({"turkiye", "turkey", "tr"})
# Regions a remote role can be limited to that still include the candidate. "europe"
# is ambiguous for a candidate in Türkiye; an ambiguous region does not force a SKIP.
REMOTE_ALLOWED_REGIONS = TURKIYE_NAMES | frozenset({"emea", "europe"})
# Words that say Türkiye is left out when they appear next to its name.
EXCLUSION_WORDS = frozenset({"excluding", "except", "excl", "not", "outside", "no", "without"})
# Words that name no place. A location made only of these is unknown or unrestricted
# ("Fully remote", "Remote (Worldwide)", "Not specified"), and unknown never violates.
NON_PLACE_WORDS = frozenset(
    {
        "remote", "remotely", "uzaktan", "hybrid", "onsite", "fully", "full", "part",
        "time", "first", "home", "office", "work", "working", "based", "only", "from",
        "in", "the", "of", "within", "and", "or", "available", "not", "specified",
        "unspecified", "n", "a", "na", "tbd", "worldwide", "global", "globally",
        "anywhere",
    }
)  # fmt: skip

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
    # Decompose first, so that a letter typed as one character and the same letter
    # stored as base letter + combining mark (common in text copied from a PDF) end up
    # equal; then drop the marks. The map still handles "ı", which has no decomposition.
    decomposed = unicodedata.normalize("NFKD", text)
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return without_marks.translate(_TURKISH_CHAR_MAP).casefold()


def _place_words(location: str) -> frozenset[str]:
    """Reduce a location to the words that can name a place.

    Args:
        location: Location as extracted from the posting, e.g. "Remote (US only)".

    Returns:
        The normalized words of the location without filler such as "remote", "only"
        or "worldwide", e.g. {"us"}. Empty if the location names no place.
    """
    words = re.findall(r"[a-z]+", _normalize(location))
    return frozenset(words) - NON_PLACE_WORDS


def _excludes_turkiye(location: str) -> bool:
    """Tell whether a location says that Türkiye is left out.

    Args:
        location: Location as extracted from the posting.

    Returns:
        True for wording such as "Global, excluding Turkey".
    """
    words = frozenset(re.findall(r"[a-z]+", _normalize(location)))
    return bool(words & TURKIYE_NAMES) and bool(words & EXCLUSION_WORDS)


def _violates_location(location: str, work_mode: str) -> bool:
    """Check the location rule.

    Args:
        location: Location as extracted from the posting.
        work_mode: One of "remote", "hybrid", "onsite", "unknown".

    Returns:
        True if the job is on-site or hybrid in a named place other than the allowed
        cities, or remote and limited to a named region that leaves Türkiye out. A
        location that names no place, an office location that names only the country,
        and an unknown work mode never violate.
    """
    if UNKNOWN in _normalize(location):
        return False
    places = _place_words(location)
    if work_mode in OFFICE_WORK_MODES:
        named_places = places - TURKIYE_NAMES
        return bool(named_places) and not named_places & ALLOWED_CITIES
    if work_mode == REMOTE:
        if _excludes_turkiye(location):
            return True
        return bool(places) and not places & (REMOTE_ALLOWED_REGIONS | ALLOWED_CITIES)
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
