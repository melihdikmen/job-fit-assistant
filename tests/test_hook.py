"""Tests for the pure parts of the Claude Code post-edit hook."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

HOOK_PATH = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "post_edit_check.py"


def _load_hook() -> ModuleType:
    """Import the hook script, which lives outside the package, from its file path."""
    spec = importlib.util.spec_from_file_location("post_edit_check", HOOK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hook = _load_hook()


@pytest.mark.parametrize(
    "file_path",
    [
        "d:\\Projects\\JobDecisionAsisstant\\src\\job_fit\\rules.py",
        "/home/user/project/tests/test_rules.py",
        "C:\\repo\\pyproject.toml",
        "src/job_fit/RULES.PY",
    ],
)
def test_code_files_need_a_check(file_path: str) -> None:
    """Python files and pyproject.toml trigger the checks, on any path style."""
    assert hook.needs_check(file_path) is True


@pytest.mark.parametrize(
    "file_path",
    ["d:\\Projects\\JobDecisionAsisstant\\README.md", "docs/results.md", "data/postings.jsonl", ""],
)
def test_other_files_need_no_check(file_path: str) -> None:
    """Documentation and data edits cannot break lint or tests, so they are skipped."""
    assert hook.needs_check(file_path) is False


def test_edited_file_is_read_from_the_tool_input() -> None:
    """The path comes from tool_input.file_path of the hook payload."""
    payload = {"tool_name": "Edit", "tool_input": {"file_path": "C:\\repo\\src\\a.py"}}

    assert hook.edited_file(json.dumps(payload)) == "C:\\repo\\src\\a.py"


@pytest.mark.parametrize(
    "hook_input",
    ["", "not json", "[]", "{}", '{"tool_input": null}', '{"tool_input": {"file_path": 7}}'],
)
def test_unreadable_payload_yields_no_file(hook_input: str) -> None:
    """A payload the hook cannot read means "nothing to check", not a crash."""
    assert hook.edited_file(hook_input) == ""
