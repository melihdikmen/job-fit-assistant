"""Tests for the guard rails around the project, not for its logic.

They cover two promises: tests never reach the network, and the tool only recommends.
"""

import re
import socket
import tomllib
from pathlib import Path

import pytest
from conftest import NetworkAccessError

from job_fit.tools import build_tools

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# Everything the package may import at runtime. All of it serves one purpose: asking a
# model for a decision and validating the answer. A library that could act for the
# candidate (browser automation, e-mail, form filling, job-board clients) must not be
# added without changing this list on purpose.
ALLOWED_RUNTIME_DEPENDENCIES = frozenset(
    {
        "google-genai",
        "langchain",
        "langchain-google-genai",
        "openai",
        "pydantic",
        "python-dotenv",
    }
)

# The three read-only tools of PROCESS.md Phase 4.
ALLOWED_AGENT_TOOLS = frozenset({"check_rules", "get_cv_section", "convert_currency"})


def _requirement_name(requirement: str) -> str:
    """Return the package name of a requirement string such as "openai>=3.49.0"."""
    match = re.match(r"[A-Za-z0-9._-]+", requirement.strip())
    assert match is not None, f"Unreadable requirement: {requirement!r}"
    return match.group(0).lower()


def test_network_connections_are_refused_in_tests() -> None:
    """The autouse guard turns any connection attempt into an error."""
    with socket.socket() as sock, pytest.raises(NetworkAccessError):
        sock.connect(("example.invalid", 443))


def test_runtime_dependencies_match_the_allow_list() -> None:
    """A new runtime dependency has to be added to the allow-list deliberately."""
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]

    declared = {_requirement_name(requirement) for requirement in project["dependencies"]}

    assert declared == ALLOWED_RUNTIME_DEPENDENCIES


def test_agent_has_only_the_read_only_tools() -> None:
    """The agent cannot be handed a tool that applies, sends or fills in anything."""
    tool_names = {tool.name for tool in build_tools("## Skills\nPython")}

    assert tool_names == ALLOWED_AGENT_TOOLS
