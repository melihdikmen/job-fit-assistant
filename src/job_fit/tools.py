"""Tools the agent can call: the hard rules, the CV by section, and a currency converter.

The business logic lives in ``rules.py`` (the hard rules, the English check, the
currency conversion); the functions here only adapt it (``rule_check``) or read the CV
(``split_cv_sections``, ``find_cv_section``), and ``build_tools`` wraps them as LangChain
tools. The tool descriptions below are what the model reads, so they are part of the
agent's prompt.
"""

from typing import Any

from langchain_core.tools import BaseTool, ToolException, tool

from job_fit.rules import (
    FX_RATES_CHECKED_ON,
    FX_RATES_TO_TRY,
    english_risk,
    to_try,
    violated_rules,
)

CV_SECTION_HEADING_PREFIX = "## "
FORCED_DECISION = "SKIP"

CHECK_RULES_DESCRIPTION = (
    "Check a job against the candidate's hard rules. Pass the facts exactly as the "
    "posting states them, and 'unknown' for a fact it does not state. "
    "location: 'City, Country' for on-site or hybrid roles, 'Remote (<region>)' for "
    "remote roles. work_mode: remote, hybrid, onsite or unknown. seniority: intern, "
    "junior, mid, senior, lead or unknown. english_required: true if the posting asks "
    "for fluent English or the team works in English. Returns the violated rules, the "
    "decision they force (SKIP, or null if none) and the English risk flag, which never "
    "forces a decision."
)
GET_CV_SECTION_DESCRIPTION = (
    "Return one section of the candidate's CV. The CV is not in the conversation; this "
    "tool is the only way to read it. Available topics: {topics}."
)
CONVERT_CURRENCY_DESCRIPTION = (
    "Convert a salary in USD or EUR to Turkish lira (TRY). Uses a fixed exchange rate "
    f"checked on {FX_RATES_CHECKED_ON} (not a live rate): "
    + ", ".join(f"1 {code} = {rate} TRY" for code, rate in FX_RATES_TO_TRY.items())
    + "."
)


def rule_check(
    location: str,
    work_mode: str,
    english_required: bool,
    seniority: str,
    posting_text: str = "",
) -> dict[str, Any]:
    """Run the hard rules on facts extracted from a posting.

    Args:
        location: Location as the posting states it.
        work_mode: One of "remote", "hybrid", "onsite", "unknown".
        english_required: True if the posting asks for strong English.
        seniority: One of "intern", "junior", "mid", "senior", "lead", "unknown".
        posting_text: Full text of the posting, for the keyword check on English;
            empty if it is not available.

    Returns:
        ``violated_rules`` (rule names), ``forced_decision`` ("SKIP" or None) and
        ``english_risk`` (the flag as given or a keyword match in the posting, the
        same way the rules compute it on the final output; it never forces a decision).
    """
    violations = violated_rules(location, work_mode.strip().lower(), seniority.strip().lower())
    return {
        "violated_rules": violations,
        "forced_decision": FORCED_DECISION if violations else None,
        "english_risk": english_risk(english_required, posting_text),
    }


def split_cv_sections(cv: str) -> dict[str, str]:
    """Split a Markdown CV into its second-level sections.

    Args:
        cv: Full text of the CV, with sections introduced by ``## Heading`` lines.

    Returns:
        Section text by lowercase heading, in document order. Sub-headings stay inside
        their section. Text before the first heading is left out.
    """
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in cv.splitlines():
        if line.startswith(CV_SECTION_HEADING_PREFIX):
            current = sections.setdefault(
                line[len(CV_SECTION_HEADING_PREFIX) :].strip().lower(), []
            )
        elif current is not None:
            current.append(line)
    return {heading: "\n".join(lines).strip() for heading, lines in sections.items()}


def find_cv_section(sections: dict[str, str], topic: str) -> str | None:
    """Look up a CV section by topic.

    Args:
        sections: Sections from ``split_cv_sections``.
        topic: What the caller asks for, e.g. "skills" or "Work Experience".

    Returns:
        The section whose heading equals the topic, or else the first whose heading
        contains the topic or is contained in it. None if nothing matches.
    """
    wanted = topic.strip().lower()
    if not wanted:
        return None
    if wanted in sections:
        return sections[wanted]
    for heading, text in sections.items():
        if wanted in heading or heading in wanted:
            return text
    return None


def build_tools(cv: str, posting_text: str = "") -> list[BaseTool]:
    """Create the agent's tools for one candidate CV and one posting.

    The tools are closures over the CV and the posting, so nothing is kept in
    module-level state and tests can build them from a small fake CV.

    Args:
        cv: Full text of the candidate's CV.
        posting_text: Full text of the posting being decided; ``check_rules`` runs the
            keyword check on English against it. It does not change any description.

    Returns:
        The tools ``check_rules``, ``get_cv_section`` and ``convert_currency``.
    """
    sections = split_cv_sections(cv)
    topics = ", ".join(sections) or "none"

    @tool(description=CHECK_RULES_DESCRIPTION)
    def check_rules(
        location: str, work_mode: str, english_required: bool, seniority: str
    ) -> dict[str, Any]:
        """Check a job against the candidate's hard rules."""
        return rule_check(location, work_mode, english_required, seniority, posting_text)

    @tool(description=GET_CV_SECTION_DESCRIPTION.format(topics=topics))
    def get_cv_section(topic: str) -> str:
        """Return one section of the candidate's CV."""
        section = find_cv_section(sections, topic)
        if section is None:
            # A message instead of an error lets the model pick a valid topic and go on.
            return f"No CV section matches '{topic}'. Available topics: {topics}."
        return section

    @tool(description=CONVERT_CURRENCY_DESCRIPTION)
    def convert_currency(amount: float, currency: str) -> float:
        """Convert a USD or EUR salary to TRY at the fixed rate."""
        try:
            return to_try(amount, currency)
        except ValueError as err:
            raise ToolException(str(err)) from err

    # Hand a ToolException back to the model as the tool's answer instead of ending
    # the run, so a wrong currency code costs one step, not the whole posting.
    convert_currency.handle_tool_error = True
    return [check_rules, get_cv_section, convert_currency]


def tool_descriptions(tools: list[BaseTool]) -> str:
    """Join tool names and descriptions, to fingerprint what the model is told.

    Args:
        tools: Tools from ``build_tools``.

    Returns:
        One "name: description" line per tool.
    """
    return "\n".join(f"{item.name}: {item.description}" for item in tools)
